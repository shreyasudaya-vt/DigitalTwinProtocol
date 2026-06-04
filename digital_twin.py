import socket
import struct
import time
import csv
import numpy as np
from Crypto.Cipher import AES
from Crypto.Util import Counter
from filterpy.kalman import KalmanFilter

class DigitalTwinServer:
    def __init__(self, listen_ip="127.0.0.1", listen_port=5000, feedback_ip="127.0.0.1", feedback_port=9001, master_key=b'\x00'*16, salt=b'\x01'*4):
        self.rx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx_sock.bind((listen_ip, listen_port))
        
        self.feedback_address = (feedback_ip, feedback_port)
        self.tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        self.master_key = master_key
        self.salt = salt
        
        # PDR Measurement Parameters
        self.received_packets = 0
        self.expected_packets = 0
        self.last_seq = -1
        
        # Enrollment System Configuration
        self.is_enrolled = False
        self.baseline_k_auth_bits = None
        self.baseline_k_health = None
        
        # Fountain Erasure Matrix Spaces
        self.reconstruction_buffer = np.zeros(128, dtype=np.float32)
        self.hit_counts = np.zeros(128, dtype=np.int32)
        
        # Kalman Tracking Setup (Physics-Constrained State Estimation)
        self.kf = KalmanFilter(dim_x=1, dim_z=1)
        self.kf.x = np.array([[0.0]])   
        self.kf.F = np.array([[1.0]])   
        self.kf.H = np.array([[1.0]])   
        self.kf.P *= 0.5                
        self.kf.R = np.array([[0.1]])   
        self.kf.Q = np.array([[0.01]])  

        self.start_time = time.time()
        self.log_file = open("thesis_telemetry_log.csv", "w", newline='')
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow(["Time", "Seq", "Tier", "PDR", "Hamming_Distance", "Kalman_State", "Innovation", "Threshold_Alarm"])
        self.log_file.flush()

    def _generate_crypto_masks(self, seq):
        # Tied securely to sequence index to prevent keystream desynchronization
        CRYPTO_MODE = "SEQUENCE" 
        
        val_n = seq
        val_s = seq

        ctr_n = Counter.new(128, initial_value=val_n)
        cipher_n = AES.new(self.master_key, AES.MODE_CTR, counter=ctr_n)
        n_i = cipher_n.encrypt(b'\x00' * 16)
        
        derived_s_key = AES.new(self.master_key, AES.MODE_ECB).encrypt(self.salt.ljust(16, b'\x00'))
        ctr_s = Counter.new(128, initial_value=val_s)
        cipher_s = AES.new(derived_s_key, AES.MODE_CTR, counter=ctr_s)
        s_i = cipher_s.encrypt(b'\x00' * 4)
        
        return n_i, s_i

    def _fountain_decompress_indices(self, mask_bytes):
        seed = int.from_bytes(mask_bytes, byteorder='big')
        rng = np.random.default_rng(seed)
        return rng.choice(128, size=32, replace=False)

    def process_packet(self, data):
        if len(data) < 14:
            return

        timestamp, seq, tier_id, telemetry, reserved = struct.unpack(">IIBfB", data[:14])
        payload = data[14:]
        
        self.received_packets += 1
        if self.last_seq == -1:
            self.expected_packets = 1
        else:
            diff = seq - self.last_seq
            if diff > 0:
                self.expected_packets += diff
            else:
                self.expected_packets += 1
        self.last_seq = seq
        
        # Calculate Packet Delivery Ratio (PDR)
        pdr = self.received_packets / self.expected_packets
        self.tx_sock.sendto(struct.pack(">f", pdr), self.feedback_address)
        print(f"\n[Twin Engine] Seq: {seq} | Tier: {tier_id} | Channel PDR: {pdr:.2f}")

        # ---------------------------------------------------------
        # SECTION 5.2: PDR-COUPLED STATE ESTIMATION INFLATION
        # Inflate observation noise (R_t) when PDR drops to widen trust boundaries
        # ---------------------------------------------------------
        R_base = 0.1   
        alpha = 5.0    
        self.kf.R = np.array([[R_base + alpha * (1.0 - pdr)]])

        n_i, s_i = self._generate_crypto_masks(seq)
        hd = 0
        innovation = 0.0
        alarm_triggered = 0

        if tier_id == 1:
            p_auth = payload[:16]
            p_health_bytes = payload[16:144]  # Unpack the 32 low-variance PCA floats
            
            k_auth_rec = bytes(a ^ b for a, b in zip(p_auth, n_i))
            rec_bits = np.unpackbits(np.frombuffer(k_auth_rec, dtype=np.uint8))
            
            k_health_rec = np.array(struct.unpack(">32f", p_health_bytes))
            
            if not self.is_enrolled:
                self.baseline_k_auth_bits = np.copy(rec_bits)
                self.baseline_k_health = np.copy(k_health_rec)
                self.is_enrolled = True
                print("   [Enrollment] Identity and Low-Variance Health Baselines Locked!")
            
            # 1. Measure Identity Integrity (High-Variance)
            hd = int(np.sum(rec_bits != self.baseline_k_auth_bits))
            
            # 2. Measure Gradual Degradation (Low-Variance Euclidean Drift)
            health_drift = float(np.linalg.norm(k_health_rec - self.baseline_k_health))
            
            # 3. Kalman Filter tracks the LOW-VARIANCE degradation
            self.kf.predict()
            innovation = float(health_drift - self.kf.x[0, 0])
            self.kf.update(np.array([[health_drift]]))
            
            print(f"   [Identity Check] Direct Tier 1 Hamming Distance = {hd}")
            
            # Alarm Logic: If Identity spikes (Jammer) OR if Innovation spikes wildly
            threshold = float(3.0 * np.sqrt(self.kf.S[0, 0]))
            if hd > 5 or abs(innovation) > threshold:
                print(f"   🚨 ALARM: Active Cyber-Payload Injection Spoofing Attack Detected!")
                alarm_triggered = 1
            else:
                print(f"   [Health Tracker] Low-Variance Drift: {health_drift:.4f} | Kalman Track: {self.kf.x[0,0]:.4f}")
            
            t_elapsed = time.time() - self.start_time
            self.csv_writer.writerow([t_elapsed, seq, tier_id, pdr, hd, self.kf.x[0,0], innovation, alarm_triggered])
            self.log_file.flush()
                
        elif tier_id == 2:
            p_fountain = payload[:4]
            fountain_indices = self._fountain_decompress_indices(s_i)
            n_i_bits = np.unpackbits(np.frombuffer(n_i, dtype=np.uint8))
            p_fountain_bits = np.unpackbits(np.frombuffer(p_fountain, dtype=np.uint8))
            
            for local_idx, global_idx in enumerate(fountain_indices):
                received_bit = p_fountain_bits[local_idx]
                mask_bit = n_i_bits[global_idx]
                decrypted_bit = received_bit ^ mask_bit
                
                self.reconstruction_buffer[global_idx] += decrypted_bit
                self.hit_counts[global_idx] += 1
                
            unobserved = np.sum(self.hit_counts == 0)
            print(f"   [Fountain Pipeline] Matrix Aggregating. Missing structural indexes: {unobserved}/128")
            
            if unobserved == 0:
                final_bits = np.where((self.reconstruction_buffer / self.hit_counts) >= 0.5, 1, 0).astype(np.uint8)
                
                if not self.is_enrolled:
                    self.baseline_k_auth_bits = np.copy(final_bits)
                    self.is_enrolled = True
                    print("   [Enrollment] Physical Device Baseline Identity Profile Locked via Fountain Reconstruction Matrix!")

                hd = int(np.sum(final_bits != self.baseline_k_auth_bits))
                print(f"   🎉 [Fountain Complete] Full Eigenspace Key Reassembled! Hamming Distance = {hd}")
                
                self.kf.predict()
                innovation = float(hd - self.kf.x[0, 0])
                self.kf.update(np.array([[hd]]))
                
                threshold = float(3.0 * np.sqrt(self.kf.S[0, 0]))
                if abs(innovation) > threshold:
                    alarm_triggered = 1

                t_elapsed = time.time() - self.start_time
                self.csv_writer.writerow([t_elapsed, seq, tier_id, pdr, hd, self.kf.x[0,0], innovation, alarm_triggered])
                self.log_file.flush()

    def start(self):
        print("🟢 Digital Twin Monitoring Station Active. Listening for telemetry...")
        try:
            while True:
                data, _ = self.rx_sock.recvfrom(2048)
                self.process_packet(data)
        except KeyboardInterrupt:
            print("\nShutting down Digital Twin Server.")
            self.log_file.close()
            self.rx_sock.close()
            self.tx_sock.close()

if __name__ == "__main__":
    server = DigitalTwinServer()
    server.start()