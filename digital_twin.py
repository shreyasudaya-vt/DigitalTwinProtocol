import socket
import struct
import time
import csv
import numpy as np
from Crypto.Cipher import AES
from Crypto.Util import Counter
from filterpy.kalman import KalmanFilter
from collections import deque
import sys

class DigitalTwinServer:
    def __init__(self, scenario_name="Scenario_A", listen_ip="127.0.0.1", listen_port=5000, feedback_ip="127.0.0.1", feedback_port=9001, master_key=b'\x00'*16, salt=b'\x01'*4):
        self.rx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.rx_sock.bind((listen_ip, listen_port))
        
        self.feedback_address = (feedback_ip, feedback_port)
        self.tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.blackout_recovery = False
        self.master_key = master_key
        self.salt = salt
        self.received_packets = 0
        self.window_size = 200
        self.pdr_window = deque(maxlen=self.window_size)
        self.last_seq = -1
        self.is_locked_out = False

        self.consecutive_anomalies = 0
        self.is_enrolled = False
        self.baseline_k_auth_bits = None
        self.baseline_k_health = None

        self.received_packets_since_tier_drop = 0
        self.bp_graph = []       
        self.resolved_bits = {}
        
        # --- Kalman Filter Setup ---
        self.kf = KalmanFilter(dim_x=2, dim_z=1)
        self.kf.x = np.array([[0.0], [0.0]])
        dt = 0.01
        self.kf.F = np.array([[1.0, dt], [0.0, 1.0]])
        self.kf.H = np.array([[1.0, 0.0]])
        
        self.kf.P = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.kf.R = np.array([[0.005**2]]) 
        
        # Optimized Q Matrix: Locks in a slow, steady degradation slope
        self.kf.Q = np.array([
            [1e-4, 0.0], 
            [0.0, 1e-6]
        ])
    
        self.warmup_count = 0
        self.WARMUP_PACKETS = 25
        self.start_time = time.time()
        
        self.log_filename = f"telemetry_{scenario_name}.csv"
        self.log_file = open(self.log_filename, "w", newline="")
        self.csv_writer = csv.writer(self.log_file)
        
        self.csv_writer.writerow([
            "Time", "Seq", "Tier", "PDR", "Hamming_Distance",
            "Raw_Measurement", "Kalman_State", "Kalman_P", "Innovation",
            "Dynamic_Threshold", "Alarm_Active",
            "Sensor_Vel", "Sensor_Accel", "Expected_Accel", "Spatial_Residual", "Spatial_Alarm"
        ])
        self.log_file.flush()

    def _generate_crypto_masks(self, seq):
        ctr_n = Counter.new(128, initial_value=seq)
        cipher_n = AES.new(self.master_key, AES.MODE_CTR, counter=ctr_n)
        n_i = cipher_n.encrypt(b'\x00' * 16)

        derived_s_key = AES.new(self.master_key, AES.MODE_ECB).encrypt(self.salt.ljust(16, b'\x00'))
        ctr_s = Counter.new(128, initial_value=seq)
        cipher_s = AES.new(derived_s_key, AES.MODE_CTR, counter=ctr_s)
        s_i = cipher_s.encrypt(b'\x00' * 4)
        return n_i, s_i

    def _fountain_compress(self, data_bytes, mask_bytes):
        seed = int.from_bytes(mask_bytes, byteorder='big')
        rng = np.random.default_rng(seed)
        bits = np.unpackbits(np.frombuffer(data_bytes, dtype=np.uint8))
        encoded_bits = np.zeros(32, dtype=np.uint8)
        degree_choices = [1, 2, 3, 4]
        degree_probs = [0.20, 0.50, 0.15, 0.15]
        for i in range(32):
            d = rng.choice(degree_choices, p=degree_probs)
            indices = rng.choice(128, size=d, replace=False)
            val = 0
            for idx in indices:
                val ^= bits[idx]
            encoded_bits[i] = val
        return np.packbits(encoded_bits).tobytes()

    def _fountain_decompress_graph(self, mask_bytes):
        seed = int.from_bytes(mask_bytes, byteorder="big")
        rng = np.random.default_rng(seed)
        degree_choices = [1, 2, 3, 4]
        degree_probs = [0.20, 0.50, 0.15, 0.15]
        graph = []
        for _ in range(32):
            d = rng.choice(degree_choices, p=degree_probs)
            indices = rng.choice(128, size=d, replace=False)
            graph.append(set(indices))
        return graph

    def process_packet(self, data):
        if len(data) < 21: return

        timestamp, seq, tier_id, telemetry, sens_vel, sens_accel = struct.unpack(">IIBfff", data[:21])
        
        if self.last_seq != -1 and seq <= self.last_seq:
            print(f"   🚨 ALARM: Replay Attack Detected! (Seq {seq} <= {self.last_seq})")
            return
            
        payload = data[21:]
        self.warmup_count += 1
        self.received_packets += 1
        
        # Calculate PDR
        if self.last_seq == -1:
            self.pdr_window.append(1)
        else:
            diff = seq - self.last_seq
            if diff > 0:
                missed = min(diff - 1, self.window_size)
                for _ in range(missed): self.pdr_window.append(0)
                self.pdr_window.append(1)
        self.last_seq = max(self.last_seq, seq)
        
        pdr = sum(self.pdr_window) / max(1, len(self.pdr_window))
        self.tx_sock.sendto(struct.pack(">f", pdr), self.feedback_address)

        # --- FIXED dt CALCULATION: Track true simulation time ---
        prev_telemetry_time = getattr(self, 'last_telemetry_time', None)
        if prev_telemetry_time is not None:
            dt = telemetry - prev_telemetry_time
            # Guard against out-of-order packets causing negative time
            if dt <= 0: 
                dt = 0.01
        else:
            dt = 0.01
        if dt > 5.0:
            self.blackout_recovery = True
            dt = 0.01
        self.last_telemetry_time = telemetry

        self.last_packet_timestamp = timestamp
        R_base = 0.002**2
        alpha_noise = 0.01
        self.kf.R = np.array([[R_base + alpha_noise * (1.0 - pdr)]])

        n_i, s_i = self._generate_crypto_masks(seq)
        spatial_alarm = 0
        spatial_residual = 0.0
        expected_accel = 0.0

        
        if hasattr(self, 'last_vel') and prev_telemetry_time is not None:
            dt_spatial = telemetry - prev_telemetry_time
            if dt_spatial > 0.001:
                raw_expected_accel = (sens_vel - self.last_vel) / dt_spatial

                # EWMA: smooths residual noise without lagging the hijack step response
                alpha = 0.5
                self._smooth_exp_accel = alpha * raw_expected_accel + (1 - alpha) * getattr(
                    self, '_smooth_exp_accel', raw_expected_accel
                )
                expected_accel = self._smooth_exp_accel

                spatial_residual = abs(expected_accel - sens_accel)
                
                if spatial_residual > 5.0 and telemetry > 25.0:
                    self._spatial_counter = getattr(self, '_spatial_counter', 0) + 1
                else:
                    self._spatial_counter = max(0, getattr(self, '_spatial_counter', 0) - 1)

                if getattr(self, '_spatial_counter', 0) >= 5:
                    spatial_alarm = 1
                    print(f"   🚨 PHASE 3 ALARM: Transducer Hijacking! "
                        f"Expected: {expected_accel:.2f}, Reported: {sens_accel:.2f}")                   

        self.last_vel = sens_vel

        if tier_id == 1:
            self.bp_graph = []
            self.resolved_bits = {}
            self.received_packets_since_tier_drop = 0
            p_auth = payload[:16]
            p_health_bytes = payload[16:144]

            # 1. Safe Float Deserialization & Sanity Check
            if len(p_health_bytes) == 128:
                try:
                    k_health_rec = np.array(struct.unpack(">32f", p_health_bytes))
                    
                    # Prevent mangled channel floats from poisoning the tracker
                    if np.any(np.isnan(k_health_rec)) or np.any(np.isinf(k_health_rec)) or np.any(np.abs(k_health_rec) > 100.0):
                        raise ValueError("Mangled floats detected due to RF channel noise.")
                except (struct.error, ValueError):
                    # Coast gracefully using prior state if frame is corrupted
                    k_health_rec = self.baseline_k_health if self.baseline_k_health is not None else np.zeros(32)
            else:
                k_health_rec = self.baseline_k_health if self.baseline_k_health is not None else np.zeros(32)

            k_auth_rec = bytes(a ^ b for a, b in zip(p_auth, n_i))
            rec_bits = np.unpackbits(np.frombuffer(k_auth_rec, dtype=np.uint8))

            if not self.is_enrolled:
                self.baseline_k_auth_bits = np.copy(rec_bits)
                self.baseline_k_health = np.copy(k_health_rec)
                self.is_enrolled = True
                print("   [Enrollment] Identity and Low-Variance Health Baselines Locked!")

            hd = int(np.sum(rec_bits != self.baseline_k_auth_bits))
            health_drift = float(np.linalg.norm(k_health_rec - self.baseline_k_health))
            
            self.kf.F[0, 1] = dt
            self.kf.predict()
            
            innovation = float(health_drift - self.kf.x[0, 0])
            S_pre = float(self.kf.P[0, 0] + self.kf.R[0, 0])
            
            # Dynamic Sigma for FAR/TDR optimization
            sigma_multiplier = 7.0 if pdr >= 0.90 else 8.0
            dyn_threshold = float(sigma_multiplier * np.sqrt(S_pre))
            threshold = max(0.015, dyn_threshold)
            alarm_triggered = 0
            
            if telemetry < 25.0 or self.warmup_count < self.WARMUP_PACKETS:
                self.kf.update(np.array([[health_drift]]))
                self.consecutive_anomalies = 0
                alarm_triggered = 0
                self.is_locked_out = False
            else:
                # 1. Identity Spoofing Check (Independent of tracking health)
                if hd > 12 and pdr > 0.85:
                    self._id_anomalies = getattr(self, '_id_anomalies', 0) + 1
                    if self._id_anomalies >= 5:
                        print(f"    🚨 ALARM: Confirmed Identity Spoofing! (HD: {hd} | PDR: {pdr:.2f})")
                        alarm_triggered = 1
                else:
                    self._id_anomalies = max(0, getattr(self, '_id_anomalies', 0) - 1)
                    alarm_triggered = 0
                # 2. Network Blackout State Recovery
                if self.blackout_recovery and not alarm_triggered:
                    self.kf.x = np.array([[health_drift], [0.0006]])
                    self.kf.P = np.array([[1e-4, 0.0], [0.0, 1e-4]])
                    self.consecutive_anomalies = 0
                    alarm_triggered = 0
                    self.blackout_recovery = False
                    
                # 3. Structural Innovation Breach Evaluation
                elif abs(innovation) > threshold and not alarm_triggered:
                    if pdr < 0.80:
                        # Coast defensively through wireless dropouts without dropping filter state
                        alarm_triggered = 0
                    else:
                        # Increment anomaly counter (cap at 30 to prevent integer windup)
                        self.consecutive_anomalies = min(30, self.consecutive_anomalies + 1)
                        print(f"   ⚠️ Physical Deviation Detected (Innov: {abs(innovation):.4f}) | Counter: {self.consecutive_anomalies}/12")
                        
                        # Requires persistent consecutive breaches to flag a true stealth ramp attack
                        if self.consecutive_anomalies >= 25:
                            alarm_triggered = 1
                        else:
                            self.kf.update(np.array([[health_drift]]))
                            alarm_triggered = 0  
                elif not alarm_triggered:
                    # 4. Clean, Verified State: Gracefully cool down the anomaly accumulator
                    self.consecutive_anomalies = max(0, self.consecutive_anomalies - 1)
                    self.kf.update(np.array([[health_drift]]))
                    

            self.csv_writer.writerow([
                telemetry, seq, tier_id, pdr, hd,
                health_drift, float(self.kf.x[0, 0]), float(self.kf.P[0, 0]),
                innovation, threshold, alarm_triggered,
                sens_vel, sens_accel, expected_accel, spatial_residual, spatial_alarm 
                ])
            self.log_file.flush()

        elif tier_id == 2:
            self.received_packets_since_tier_drop += 1
            p_fountain = payload[:4]
            
            fountain_graph = self._fountain_decompress_graph(s_i)
            n_i_comp = self._fountain_compress(n_i, s_i)
            received_lt_symbols = bytes(a ^ b for a, b in zip(p_fountain, n_i_comp))
            received_bits = np.unpackbits(np.frombuffer(received_lt_symbols, dtype=np.uint8))

            for i in range(32):
                self.bp_graph.append({
                    'value': received_bits[i],
                    'indices': fountain_graph[i]
                })

            for node in self.bp_graph:
                to_remove = []
                for idx in node['indices']:
                    if idx in self.resolved_bits:
                        node['value'] ^= self.resolved_bits[idx]
                        to_remove.append(idx)
                for idx in to_remove:
                    node['indices'].remove(idx)

            progress = True
            while progress:
                progress = False
                degree_1_nodes = [n for n in self.bp_graph if len(n['indices']) == 1]
                
                for node in degree_1_nodes:
                    if len(node['indices']) != 1:
                        continue
                        
                    idx = list(node['indices'])[0]
                    if idx not in self.resolved_bits:
                        val = node['value']
                        self.resolved_bits[idx] = val
                        progress = True
                        for other_node in self.bp_graph:
                            if idx in other_node['indices']:
                                other_node['indices'].remove(idx)
                                other_node['value'] ^= val
                                
                self.bp_graph = [n for n in self.bp_graph if len(n['indices']) > 0]

            self.kf.F[0, 1] = dt
            self.kf.predict()
            estimated_health = float(self.kf.x[0, 0])
            threshold =  float(5.0 * np.sqrt(float(self.kf.P[0, 0] + self.kf.R[0, 0])))

            unobserved = 128 - len(self.resolved_bits)
            if unobserved > 0:
                self.csv_writer.writerow([
                    telemetry, seq, tier_id, pdr, -1, 
                    np.nan, estimated_health, float(self.kf.P[0, 0]), 0.0, threshold, 0,
                    sens_vel, sens_accel, expected_accel, spatial_residual, spatial_alarm # <-- ADDED
                ])
                self.log_file.flush()
                return

            print(f"📊 LT Belief Propagation Avalanche Complete! Packets required: {self.received_packets_since_tier_drop}")
            
            final_bits = np.zeros(128, dtype=np.uint8)
            for idx in range(128):
                final_bits[idx] = self.resolved_bits[idx]

            if not self.is_enrolled:
                self.baseline_k_auth_bits = np.copy(final_bits)
                self.is_enrolled = True

            hd = int(np.sum(final_bits != self.baseline_k_auth_bits))
            alarm_triggered = 1 if (self.warmup_count >= self.WARMUP_PACKETS and hd > 15) else 0
            if not alarm_triggered and self.is_enrolled:
                pass
            self.csv_writer.writerow([
                telemetry, seq, tier_id, pdr, hd,
                np.nan, estimated_health, float(self.kf.P[0, 0]), 0.0, threshold, alarm_triggered,
                sens_vel, sens_accel, expected_accel, spatial_residual, spatial_alarm # <-- ADDED
            ])
            self.log_file.flush()
            
            self.bp_graph = []
            self.resolved_bits = {}
            self.received_packets_since_tier_drop = 0

    def start(self):
        print("🟢 Digital Twin Monitoring Station Active...")
        try:
            while True:
                data, _ = self.rx_sock.recvfrom(2048)
                self.process_packet(data)
        except KeyboardInterrupt:
            self.log_file.close()
            self.rx_sock.close()
            self.tx_sock.close()

if __name__ == "__main__":
    scenario = sys.argv[1] if len(sys.argv) > 1 else "Scenario_A"
    server = DigitalTwinServer(scenario_name=scenario)
    server.start()
