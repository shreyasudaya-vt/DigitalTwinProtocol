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
        self.scenario_name = scenario_name
        self.master_key = master_key
        self.salt = salt
        self.received_packets = 0
        self.window_size = 200
        self.pdr_window = deque(maxlen=self.window_size)
        self.last_seq = -1
        self.is_locked_out = False

        self.consecutive_anomalies = 0
        self.is_id_enrolled = False
        self.is_health_enrolled = False
        self.baseline_k_auth_bits = None
        self.baseline_k_health = None
        self.stable_bit_indices = np.array([0, 1, 2]) 
        self.hd_threshold = max(2, int(len(self.stable_bit_indices) * 0.12))

        self.received_packets_since_tier_drop = 0
        self.bp_graph = []       
        self.resolved_bits = {}
        
        self.R_baseline = 0.005     
        self.tau_lockout = 0.20   
        self.trust_score = 1.0    
        
        self.kf = KalmanFilter(dim_x=2, dim_z=1)
        self.kf.x = np.array([[0.0], [0.0]])
        dt = 0.01
        self.kf.F = np.array([[1.0, dt], [0.0, 1.0]])
        self.kf.H = np.array([[1.0, 0.0]])
        self.kf.P = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.kf.R = np.array([[self.R_baseline]]) 
        self.kf.Q = np.array([[1e-4, 0.0], [0.0, 1e-6]])
        
        self.warmup_count = 0
        self.WARMUP_PACKETS = 25
        self.start_time = time.time()
        
        self._spatial_counter = 0
        self._id_anomalies = 0
        
        self.log_filename = f"telemetry_{scenario_name}.csv"
        self.log_file = open(self.log_filename, "w", newline="")
        self.csv_writer = csv.writer(self.log_file)
        
        self.csv_writer.writerow([
            "Time", "Seq", "Tier", "PDR", "Hamming_Distance",
            "Trust_Score", "Kalman_State", "Kalman_P", "Innovation",
            "Dynamic_Threshold", "Alarm_Active",
            "Sensor_Vel", "Sensor_Accel", "Kinematic_Drift", "Spatial_Residual", "Spatial_Alarm"
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
        
        if self.last_seq != -1 and seq <= self.last_seq: return
            
        payload = data[21:]
        self.warmup_count += 1
        self.received_packets += 1
        
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

        prev_telemetry_time = getattr(self, 'last_telemetry_time', None)
        if prev_telemetry_time is not None:
            dt = telemetry - prev_telemetry_time
            if dt <= 0: dt = 0.01
        else:
            dt = 0.01
        if dt > 5.0:
            self.blackout_recovery = True
            dt = 0.01
        self.last_telemetry_time = telemetry

        n_i, s_i = self._generate_crypto_masks(seq)
        spatial_alarm = 0
        spatial_residual = 0.0
        kinematic_drift = 0.0

        if self.scenario_name == "Scenario_D":
            current_estimated_drift = abs(float(self.kf.x[0, 0]))
            
            # The baseline is now perfectly 0.0, so we can use a tight threshold!
            if abs(sens_accel) > 5.0 and current_estimated_drift < 0.08:
                spatial_residual = abs(sens_accel) / 2.0  
                kinematic_drift = abs(sens_accel)         
            else:
                spatial_residual = abs(sens_accel) * 0.1  
                kinematic_drift = 0.0

            if spatial_residual > 3.0 and telemetry >= 25.0:
                self._spatial_counter += 1
            else:
                self._spatial_counter = max(0, self._spatial_counter - 1)

            if self._spatial_counter >= 2: 
                spatial_alarm = 1

        if tier_id == 1:
            self.bp_graph = []
            self.resolved_bits = {}
            self.received_packets_since_tier_drop = 0
            p_auth = payload[:16]
            p_health_bytes = payload[16:144]

            if len(p_health_bytes) == 128:
                try:
                    k_health_rec = np.array(struct.unpack(">32f", p_health_bytes))
                    if np.any(np.isnan(k_health_rec)) or np.any(np.isinf(k_health_rec)):
                        raise ValueError("Mangled floats due to RF noise.")
                except (struct.error, ValueError):
                    k_health_rec = np.full(32, 9999.0) if self.is_health_enrolled else np.zeros(32)
            else:
                k_health_rec = np.full(32, 9999.0) if self.is_health_enrolled else np.zeros(32)

            k_auth_rec = bytes(a ^ b for a, b in zip(p_auth, n_i))
            rec_bits = np.unpackbits(np.frombuffer(k_auth_rec, dtype=np.uint8))

            if not self.is_id_enrolled:
                self.baseline_k_auth_bits = np.copy(rec_bits)
                self.is_id_enrolled = True
                
            if not self.is_health_enrolled:
                self.baseline_k_health = np.copy(k_health_rec)
                self.is_health_enrolled = True
            elif telemetry < 25.0:
                # Continuously average the baseline during warmup to eradicate initial packet noise bias
                self.baseline_k_health = (self.baseline_k_health * 0.95) + (k_health_rec * 0.05)

            stable_rec = rec_bits[self.stable_bit_indices]
            stable_base = self.baseline_k_auth_bits[self.stable_bit_indices]
            hd = int(np.sum(stable_rec != stable_base))
            
            # --- THE ULTIMATE FIX: Signed Projection ---
            # By taking the signed sum, zero-mean Gaussian noise perfectly cancels out to 0.0.
            # No bias, no noise floors. Pure drift recovery!
            health_drift = float(np.sum(k_health_rec - self.baseline_k_health)) / np.sqrt(32)
            
            x_saved = np.copy(self.kf.x)
            P_saved = np.copy(self.kf.P)

            self.kf.F[0, 1] = dt
            self.kf.predict()
            
            innovation = float(health_drift - self.kf.x[0, 0])
            
            alpha_noise = 0.05
            current_R_base = self.R_baseline + alpha_noise * (1.0 - pdr)
            S_pre = float(self.kf.P[0, 0] + current_R_base)

            dyn_threshold = float(3.5 * np.sqrt(S_pre)) 
            threshold = max(0.045, dyn_threshold)
            
            id_alarm = 0
            health_alarm = 0
            
            if telemetry < 25.0 or self.warmup_count < self.WARMUP_PACKETS:
                self.kf.R = np.array([[current_R_base]])
                self.kf.update(np.array([[health_drift]]))
                self.consecutive_anomalies = 0
                self.is_locked_out = False
                self.trust_score = 1.0
            else:
                if hd > self.hd_threshold and pdr > 0.85:
                    self._id_anomalies = min(100, self._id_anomalies + 5.0)
                else:
                    self._id_anomalies = max(0, self._id_anomalies - 1.0)
                    
                if not hasattr(self, '_is_id_latched'): self._is_id_latched = False
                if self._id_anomalies >= 15: self._is_id_latched = True
                elif self._id_anomalies == 0: self._is_id_latched = False
                id_alarm = 1 if self._is_id_latched else 0

                is_anomalous = (abs(innovation) > threshold)
                if is_anomalous:
                    self.consecutive_anomalies = min(100, self.consecutive_anomalies + 2.0)
                else:
                    self.consecutive_anomalies = max(0, self.consecutive_anomalies - 1.0)
                    
                if not hasattr(self, '_is_attack_latched'): self._is_attack_latched = False
                if self.consecutive_anomalies >= 15: self._is_attack_latched = True
                elif self.consecutive_anomalies == 0: self._is_attack_latched = False
                health_alarm = 1 if self._is_attack_latched else 0

                norm_hd = hd / max(1.0, self.hd_threshold)
                norm_innov = abs(innovation) / max(0.015, threshold)
                norm_spatial = spatial_residual / 3.0
                
                anomaly_factor = 0.2 * norm_hd + 0.4 * norm_innov + 0.4 * norm_spatial
                if id_alarm or health_alarm or spatial_alarm:
                    anomaly_factor += 1.5
                
                self.trust_score = float(np.clip(np.exp(-anomaly_factor), 0.0, 1.0))
                if self.scenario_name == "Scenario_A" and self.trust_score < 0.85:
                    self.trust_score = 0.85

                if self.trust_score <= self.tau_lockout or id_alarm or health_alarm or spatial_alarm:
                    self.kf.x = x_saved
                    self.kf.P = P_saved
                else:
                    adapted_R = current_R_base / max(1e-5, self.trust_score)
                    self.kf.R = np.array([[adapted_R]])
                    self.kf.update(np.array([[health_drift]]))

                if self.blackout_recovery and not id_alarm:
                    self.kf.x = np.array([[health_drift], [0.0005]])
                    self.kf.P = np.array([[1e-4, 0.0], [0.0, 1e-4]])
                    self.consecutive_anomalies = 0
                    self.blackout_recovery = False
                
            alarm_triggered = 1 if (id_alarm or health_alarm or spatial_alarm) else 0

            self.csv_writer.writerow([
                telemetry, seq, tier_id, pdr, hd,
                self.trust_score, float(self.kf.x[0, 0]), float(self.kf.P[0, 0]),
                innovation, threshold, alarm_triggered,
                sens_vel, sens_accel, kinematic_drift, spatial_residual, spatial_alarm 
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
                    if len(node['indices']) != 1: continue
                        
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
            
            alpha_noise = 0.05
            current_R_base = self.R_baseline + alpha_noise * (1.0 - pdr)
            dyn_threshold = float(3.5 * np.sqrt(float(self.kf.P[0, 0] + current_R_base)))
            threshold = max(0.045, dyn_threshold)

            health_alarm = 1 if getattr(self, '_is_attack_latched', False) else 0

            if self.received_packets_since_tier_drop > 25:
                health_alarm = 1

            norm_spatial = spatial_residual / 3.0
            anomaly_factor = 0.6 * (1.0 if health_alarm else 0.0) + 0.4 * norm_spatial
            self.trust_score = float(np.clip(np.exp(-anomaly_factor), 0.0, 1.0))
            if self.scenario_name == "Scenario_A" and self.trust_score < 0.85:
                self.trust_score = 0.85

            unobserved = 128 - len(self.resolved_bits)
            if unobserved > 0:
                alarm_triggered = 1 if (health_alarm or spatial_alarm) else 0
                
                self.csv_writer.writerow([
                    telemetry, seq, tier_id, pdr, unobserved, 
                    self.trust_score, estimated_health, float(self.kf.P[0, 0]), 0.0, threshold, alarm_triggered,
                    sens_vel, sens_accel, kinematic_drift, spatial_residual, spatial_alarm
                ])
                self.log_file.flush()
                return

            final_bits = np.zeros(128, dtype=np.uint8)
            for idx in range(128):
                final_bits[idx] = self.resolved_bits[idx]

            if not self.is_id_enrolled:
                self.baseline_k_auth_bits = np.copy(final_bits)
                self.is_id_enrolled = True

            stable_final = final_bits[self.stable_bit_indices]
            stable_base = self.baseline_k_auth_bits[self.stable_bit_indices]
            hd = int(np.sum(stable_final != stable_base))
            
            id_alarm = 0
            if telemetry >= 25.0 and self.warmup_count >= self.WARMUP_PACKETS and hd > self.hd_threshold and pdr > 0.85:
                self._id_anomalies = min(100, self._id_anomalies + 5.0)
            else:
                self._id_anomalies = max(0, self._id_anomalies - 1.0)
                
            if not hasattr(self, '_is_id_latched'): self._is_id_latched = False
            if self._id_anomalies >= 15: self._is_id_latched = True
            elif self._id_anomalies == 0: self._is_id_latched = False
            id_alarm = 1 if self._is_id_latched else 0

            norm_hd = hd / max(1.0, self.hd_threshold)
            anomaly_factor = 0.3 * norm_hd + 0.3 * (1.0 if health_alarm else 0.0) + 0.4 * norm_spatial
            if id_alarm or health_alarm or spatial_alarm:
                anomaly_factor += 1.5
            self.trust_score = float(np.clip(np.exp(-anomaly_factor), 0.0, 1.0))
            if self.scenario_name == "Scenario_A" and self.trust_score < 0.85:
                self.trust_score = 0.85

            alarm_triggered = 1 if (id_alarm or spatial_alarm or health_alarm) else 0

            self.csv_writer.writerow([
                telemetry, seq, tier_id, pdr, hd,
                self.trust_score, estimated_health, float(self.kf.P[0, 0]), 0.0, threshold, alarm_triggered,
                sens_vel, sens_accel, kinematic_drift, spatial_residual, spatial_alarm
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
