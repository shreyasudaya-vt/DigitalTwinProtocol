import os
import socket
import time
import struct
import threading
import numpy as np
from Crypto.Cipher import AES
from Crypto.Util import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys

# Natively bind parameters directly from your project's final_pca.py
try:
    from final_pca import (
        DEVICE_FOLDER, EXCLUDED_DEVICES, PREFERRED_MULTI_TRAIN_INDICES, 
        USE_PHASE, REF_FREQ, collect_device_files, load_sweep_vector
    )
    HAS_PCA_MODULE = True
except ImportError:
    DEVICE_FOLDER = "./01_master_dataset"
    EXCLUDED_DEVICES = ["201", "253", "254", "258", "310"]
    PREFERRED_MULTI_TRAIN_INDICES = [4, 5, 6]
    USE_PHASE = True
    REF_FREQ = np.linspace(10000, 1000000, 2001)
    HAS_PCA_MODULE = False

class AnastaEdgeNode:
    def __init__(self, ns3_ip="127.0.0.1", ns3_port=9000, feedback_port=9001, master_key=b'\x00'*16, salt=b'\x01'*4):
        self.ns3_address = (ns3_ip, ns3_port)
        self.tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        self.feedback_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.feedback_sock.bind(("127.0.0.1", feedback_port))
        
        self.master_key = master_key
        self.salt = salt
        self.sequence_counter = 0
        self.current_tier = 1  
        self.PDR_DROP_THRESHOLD = 0.75     # Breakpoint to drop into Fountain mode
        self.PDR_RECOVER_THRESHOLD = 0.85  # Breakpoint to recover to Hi-Fi mode
        
        self.n_identity = 128
        self.n_health = 32
        self.total_components = self.n_identity + self.n_health
        
        self.scaler = None
        self.pca = None
        self.target_device = None
        self.device_files_map = {}

        self.running = True
        self.feedback_thread = threading.Thread(target=self._listen_for_feedback, daemon=True)
        self.feedback_thread.start()

    def calibrate_from_hardware(self):
        print("[Edge] Calibrating Eigenspace from physical hardware dataset...")
        if not HAS_PCA_MODULE:
            self._generate_synthetic_calibration()
            return

        self.device_files_map = collect_device_files(DEVICE_FOLDER)
        excluded_set = set(EXCLUDED_DEVICES)
        X_train_rows = []
        for dev in sorted(self.device_files_map.keys()):
            if dev in excluded_set: continue
            for idx in PREFERRED_MULTI_TRAIN_INDICES:
                if idx in self.device_files_map[dev]:
                    vec = load_sweep_vector(self.device_files_map[dev][idx], ref_freq=REF_FREQ, use_phase=USE_PHASE)
                    X_train_rows.append(vec)

        if not X_train_rows:
            self._generate_synthetic_calibration()
            return

        train_matrix = np.vstack(X_train_rows)
        if train_matrix.shape[0] < self.total_components:
            needed = (self.total_components + 10) - train_matrix.shape[0]
            mean_row = train_matrix.mean(axis=0)
            noise = np.random.normal(0, 0.001, (needed, train_matrix.shape[1]))
            train_matrix = np.vstack([train_matrix, mean_row + noise])

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.total_components)
        self.pca.fit(self.scaler.fit_transform(train_matrix))
        
        valid_devs = [d for d in self.device_files_map.keys() if d not in excluded_set]
        self.target_device = valid_devs[0] if valid_devs else list(self.device_files_map.keys())[0]
        print(f"[Edge] Eigenspace Calibrated! Node bound to hardware Identity profile: Device '{self.target_device}'")

    def _generate_synthetic_calibration(self):
        feature_len = len(REF_FREQ) * (2 if USE_PHASE else 1)
        mock_matrix = np.random.randn(200, feature_len)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.total_components)
        self.pca.fit(self.scaler.fit_transform(mock_matrix))
        self.target_device = "MOCK_DEV_01"
        self.device_files_map = {"MOCK_DEV_01": {}}

    def _listen_for_feedback(self):
        while self.running:
            try:
                data, _ = self.feedback_sock.recvfrom(1024)
                pdr = struct.unpack(">f", data)[0]
                
                if self.current_tier == 1:
                    if pdr < self.PDR_DROP_THRESHOLD:
                        self.current_tier = 2
                        print(f"🚨 PDR Dropped to {pdr:.2f} (< {self.PDR_DROP_THRESHOLD}). Switching to Tier 2 Fountain Mode.")
                
                elif self.current_tier == 2:
                    if pdr > self.PDR_RECOVER_THRESHOLD:
                        self.current_tier = 1
                        print(f"🟢 PDR Recovered to {pdr:.2f} (> {self.PDR_RECOVER_THRESHOLD}). Switching to Tier 1 High-Fidelity Mode.")
                        
            except Exception:
                break

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
        # TRUE LT ENCODER: Using degree distribution (Soliton-inspired)
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
                val ^= bits[idx]  # XOR the multiple source bits
            encoded_bits[i] = val
            
        return np.packbits(encoded_bits).tobytes()

    def transform_hardware_sweep(self, raw_sweep_vector):
        scaled = self.scaler.transform(raw_sweep_vector.reshape(1, -1))
        proj = self.pca.transform(scaled)[0]
        
        w_high = proj[:self.n_identity]
        w_low = proj[self.n_identity:self.total_components]
        
        current_bits = np.where(w_high > 0, 1, 0).astype(np.uint8)
        k_auth = np.packbits(current_bits).tobytes()
        k_health = w_low.tolist()
        return k_auth, k_health

    def transmit(self, k_auth, k_health, telemetry_val=22.4):
            n_i, s_i = self._generate_crypto_masks(self.sequence_counter)
            header = struct.pack(">IIBfB", int(time.time()) & 0xFFFFFFFF, self.sequence_counter, self.current_tier, telemetry_val, 0)
            
            if self.current_tier == 1:
                # 1. SATURATION GUARD: Clip state values to standard 32-bit float limits
                MAX_FLOAT32 = 3.4028235e38
                MIN_FLOAT32 = -3.4028235e38
                sanitized_health = [max(MIN_FLOAT32, min(MAX_FLOAT32, float(x))) for x in k_health]
                
                p_auth = bytes(a ^ b for a, b in zip(k_auth, n_i))
                
                # 2. PACK SANITIZED VALUES: Safe from OverflowErrors
                p_health = struct.pack(">32f", *sanitized_health)
                packet = header + p_auth + p_health
            else:
                k_auth_comp = self._fountain_compress(k_auth, s_i)
                n_i_comp = self._fountain_compress(n_i, s_i)
                p_fountain = bytes(a ^ b for a, b in zip(k_auth_comp, n_i_comp))
                packet = header + p_fountain
                
            self.tx_sock.sendto(packet, self.ns3_address)
            print(f"[Edge] Sent Seq {self.sequence_counter} | Tier {self.current_tier} | Size: {len(packet)}B")
            self.sequence_counter += 1

    def close(self):
        self.running = False
        self.feedback_sock.close()
        self.tx_sock.close()

if __name__ == "__main__":
    node = AnastaEdgeNode()
    node.calibrate_from_hardware()
    
    dev_sweeps = node.device_files_map.get(node.target_device, {})
    if len(dev_sweeps) > 0:
        first_available_idx = sorted(dev_sweeps.keys())[0]
        stable_base_vector = load_sweep_vector(dev_sweeps[first_available_idx], ref_freq=REF_FREQ, use_phase=USE_PHASE)
    else:
        stable_base_vector = np.random.randn(len(REF_FREQ) * (2 if USE_PHASE else 1))
        
    scenario = sys.argv[1] if len(sys.argv) > 1 else "Scenario_A"
    sweep_idx = 1
    stealth_drift = 0.0
    
    try:
        while True:
            # 1. Base Hardware Readout with realistic thermal noise
            thermal_noise = np.random.normal(0, 0.0005, len(stable_base_vector))
            real_sweep = stable_base_vector + thermal_noise
            k_auth, k_health = node.transform_hardware_sweep(real_sweep)
            current_health = list(k_health)

            # ==========================================================
            # TIME SCALING: Convert raw iterations into real elapsed seconds
            # ==========================================================
            dt = 0.00015 
            elapsed_time = sweep_idx * dt

            # 2. SCENARIO ISOLATION LOGIC (Scientifically Rigorous Profiles)
            if scenario == "Scenario_A":
                # FIXED: Use elapsed_time so the curve grows over real seconds, not milliseconds
                aging_drift = 0.002 * (np.exp(0.015 * elapsed_time) - 1.0)
                current_health = [val + (aging_drift / np.sqrt(32)) for val in k_health]

            elif scenario == "Scenario_C":
                # FIXED: Wait for 30 real seconds, then add drift scaled by dt
                if elapsed_time >= 30.0:
                    stealth_drift += (0.015 * dt) 
                    current_health = [val + (stealth_drift / np.sqrt(32)) for val in k_health]

            try:
                # Your existing transmission line (Line 218)
                node.transmit(k_auth, current_health, telemetry_val=float(elapsed_time))
                sweep_idx += 1
                
                # Keep our high-speed 1ms pacing
                time.sleep(0.001)

            except (BrokenPipeError, ConnectionResetError, OSError):
                # Catch the socket closure when NS-3 finishes unthrottled execution
                print("🔌 NS-3 simulation socket closed. Terminating edge node loop cleanly.")
                break
    except KeyboardInterrupt:
        node.close()
