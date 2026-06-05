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
        
        self.master_key = master_key
        self.salt = salt
        self.received_packets = 0
        self.window_size = 40
        self.pdr_window = deque(maxlen=self.window_size)
        self.last_seq = -1
        
        self.is_enrolled = False
        self.baseline_k_auth_bits = None
        self.baseline_k_health = None

        self.reconstruction_buffer = np.zeros(128, dtype=np.float32)
        self.hit_counts = np.zeros(128, dtype=np.int32)
        self.kf = KalmanFilter(dim_x=2, dim_z=1)
        self.kf.x = np.array([[0.0], [0.005]])
        self.kf.F = np.array([[1.0, 1.0],
                              [0.0, 1.0]])
        self.kf.H = np.array([[1.0, 0.0]])
        
        # 1. Start with a tight initial covariance so there is no initial "megaphone" cone
        self.kf.P = np.array([[0.01, 0.0],
                              [0.0, 1e-4]])
                              
        self.kf.R = np.array([[0.002]])  # Base measurement noise (tuned for good tracking without alarms)
        self.kf.Q = np.diag([1e-4, 1e-7])
        # self.kf.Q = (sigma_a**2) * np.array([
        #     [dt**4 / 4,  dt**3 / 2],
        #     [dt**3 / 2,  dt**2    ]
        # ])
 
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

    def _fountain_decompress_indices(self, mask_bytes):
        seed = int.from_bytes(mask_bytes, byteorder="big")
        rng = np.random.default_rng(seed)
        return rng.choice(128, size=32, replace=False)

    def process_packet(self, data):
        if len(data) < 14:
            return

        timestamp, seq, tier_id, telemetry, reserved = struct.unpack(">IIBfB", data[:14])
        payload = data[14:]

        self.received_packets += 1
        if self.last_seq == -1:
            self.pdr_window.append(1)
        else:
            diff = seq - self.last_seq
            if diff > 0:
                missed = min(diff - 1, self.window_size)
                for _ in range(missed):
                    self.pdr_window.append(0)
                self.pdr_window.append(1)
        self.last_seq = max(self.last_seq, seq)

        pdr = sum(self.pdr_window) / max(1, len(self.pdr_window))
        self.tx_sock.sendto(struct.pack(">f", pdr), self.feedback_address)

        R_base = 0.002
        alpha_noise = 5.0
        R_base = 0.0001 
        alpha_noise = 5.0
        self.kf.R = np.array([[R_base + alpha_noise * (1.0 - pdr)]])

        n_i, s_i = self._generate_crypto_masks(seq)
        hd = 0
        innovation = 0.0
        alarm_triggered = 0

        if tier_id == 1:
            p_auth = payload[:16]
            p_health_bytes = payload[16:144]

            k_auth_rec = bytes(a ^ b for a, b in zip(p_auth, n_i))
            rec_bits = np.unpackbits(np.frombuffer(k_auth_rec, dtype=np.uint8))

            if len(p_health_bytes) == 128:
                k_health_rec = np.array(struct.unpack(">32f", p_health_bytes))
            else:
                k_health_rec = self.baseline_k_health if self.baseline_k_health is not None else np.zeros(32)

            if not self.is_enrolled:
                self.baseline_k_auth_bits = np.copy(rec_bits)
                self.baseline_k_health = np.copy(k_health_rec)
                self.is_enrolled = True
                print("   [Enrollment] Identity and Low-Variance Health Baselines Locked!")

            hd = int(np.sum(rec_bits != self.baseline_k_auth_bits))
            health_drift = float(np.linalg.norm(k_health_rec - self.baseline_k_health))

            self.kf.predict()

            innovation = float(health_drift - self.kf.x[0, 0])
            S_pre = float(self.kf.P[0, 0] + self.kf.R[0, 0])
            nis = (innovation ** 2) / S_pre
            threshold = float(3.0 * np.sqrt(S_pre))

            if self.warmup_count < self.WARMUP_PACKETS:
                self.warmup_count += 1
                self.kf.update(np.array([[health_drift]]))
                alarm_triggered = 0
            else:
                # FIXED: Restored hybrid HD/NIS gating
                if hd > 8:
                    print(f"   🚨 ALARM: Identity Spoofing Detected! (Hamming Distance: {hd})")
                    alarm_triggered = 1
                    innovation = threshold + 2.0  
                elif nis > 9.0:
                    print(f"   🚨 ALARM: Physical Health Anomaly! (NIS: {nis:.2f} > 9.0)")
                    alarm_triggered = 1
                    innovation = threshold + 2.0  
                else:
                    self.kf.update(np.array([[health_drift]]))

            t_elapsed = time.time() - self.start_time
            self.csv_writer.writerow([
                t_elapsed, seq, tier_id, pdr, hd,
                health_drift, float(self.kf.x[0, 0]), float(self.kf.P[0, 0]),
                innovation, threshold, alarm_triggered,
            ])
            self.log_file.flush()

        elif tier_id == 2:
            p_fountain = payload[:4]
            fountain_indices = self._fountain_decompress_indices(s_i)
            n_i_bits = np.unpackbits(np.frombuffer(n_i, dtype=np.uint8))
            p_fountain_bits = np.unpackbits(np.frombuffer(p_fountain, dtype=np.uint8))

            for local_idx, global_idx in enumerate(fountain_indices):
                received_bit = p_fountain_bits[local_idx]
                mask_bit = n_i_bits[global_idx]
                self.reconstruction_buffer[global_idx] += received_bit ^ mask_bit
                self.hit_counts[global_idx] += 1

            unobserved = np.sum(self.hit_counts == 0)
            if unobserved > 0:
                self.kf.predict()
                return
            if unobserved == 0 and self.hit_counts.max() == 1: # Just hit full coverage
                print(f"📊 Tier 2 Reconstruction Successful! Packets required: {self.received_packets_since_tier_drop}")

            final_bits = np.where((self.reconstruction_buffer / self.hit_counts) >= 0.5, 1, 0).astype(np.uint8)

            if not self.is_enrolled:
                self.baseline_k_auth_bits = np.copy(final_bits)
                self.is_enrolled = True

            hd = int(np.sum(final_bits != self.baseline_k_auth_bits))

            self.kf.predict()
            estimated_health = float(self.kf.x[0, 0])
            threshold = float(3.0 * np.sqrt(float(self.kf.P[0, 0])))
            innovation = 0.0

            if self.warmup_count >= self.WARMUP_PACKETS and hd > 8:
                alarm_triggered = 1

            t_elapsed = time.time() - self.start_time
            self.csv_writer.writerow([
                t_elapsed, seq, tier_id, pdr, hd,
                estimated_health, estimated_health, float(self.kf.P[0, 0]),
                innovation, threshold, alarm_triggered,
            ])
            self.log_file.flush()
            
            # FIXED: Reset reconstruction buffer to prevent infinite P inflation
            self.reconstruction_buffer.fill(0)
            self.hit_counts.fill(0)

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