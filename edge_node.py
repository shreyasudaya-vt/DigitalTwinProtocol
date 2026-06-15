import os
import socket
import time
import struct
import threading
import numpy as np
import pandas as pd
import glob
from Crypto.Cipher import AES
from Crypto.Util import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import deque
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

def load_pzt_csv_vector(file_path, use_phase=True):
    """
    Natively parses the exact laboratory EMIS schema from the 01_master_dataset.
    Extracts Real Impedance (Rs) and Phase (th) as high-dimensional features.
    """
    try:
        df = pd.read_csv(file_path)
        # Ensure frequency sorting for dimensional alignment
        if "Frequency (Hz)" in df.columns:
            df = df.sort_values("Frequency (Hz)")
        
        rs_vector = df["Trace Rs (Ohm)"].values
        if use_phase and "Trace th (deg)" in df.columns:
            phase_vector = df["Trace th (deg)"].values
            return np.concatenate([rs_vector, phase_vector])
        return rs_vector
    except Exception as e:
        print(f"[Parser Error] Could not read file {file_path}: {e}")
        return None

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
        self.received_packets = 0
        self.window_size = 200
        self.pdr_window = deque(maxlen=self.window_size)
        self.scaler = None
        self.pca = None
        self.target_device = None
        self.smoothed_pdr = 1.0
        self.device_files_map = {}
        self.last_switch_time = 0
        self.running = True
        self.feedback_thread = threading.Thread(target=self._listen_for_feedback, daemon=True)
        self.feedback_thread.start()

    def calibrate_from_hardware(self):
        print("[Edge] Calibrating Eigenspace from physical hardware dataset...")
        
        # Native collection scan over 01_master_dataset searching for real CSVs
        csv_files = sorted(glob.glob(os.path.join(DEVICE_FOLDER, "**/*.csv"), recursive=True)) + \
                    sorted(glob.glob(os.path.join(DEVICE_FOLDER, "*.csv")))
        
        X_train_rows = []
        excluded_set = set(EXCLUDED_DEVICES)
        
        if csv_files:
            print(f"[Edge] Discovered {len(csv_files)} files matching the PZT schema in '{DEVICE_FOLDER}'")
            for f in csv_files:
                # Basic string filter to bypass excluded profiles
                if any(ex in os.path.basename(f) for ex in excluded_set):
                    continue
                vec = load_pzt_csv_vector(f, use_phase=USE_PHASE)
                if vec is not None:
                    X_train_rows.append(vec)
                    
        if not X_train_rows:
            print("[Edge] No physical hardware data found. Falling back to high-fidelity simulation baseline.")
            self._generate_synthetic_calibration()
            return

        train_matrix = np.vstack(X_train_rows)
        # Ensure mathematical dimensionality satisfies PCA requirements via standard noise expansion
        if train_matrix.shape[0] < self.total_components:
            needed = (self.total_components + 10) - train_matrix.shape[0]
            mean_row = train_matrix.mean(axis=0)
            noise = np.random.normal(0, 0.001, (needed, train_matrix.shape[1]))
            train_matrix = np.vstack([train_matrix, mean_row + noise])

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.total_components)
        self.pca.fit(self.scaler.fit_transform(train_matrix))
        
        self.target_device = "PZT_NODE_01"
        self.device_files_map = {"PZT_NODE_01": csv_files if csv_files else {}}
        print(f"[Edge] Eigenspace Calibrated! Node successfully bound to PZT Hardware Schema via Real Resistance Channels.")

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
                self.smoothed_pdr = (0.80 * self.smoothed_pdr) + (0.20 * pdr)
                current_time = time.time()
                cooldown_seconds = 1.0
                if current_time - self.last_switch_time > cooldown_seconds:
                    if self.current_tier == 1:
                        if self.smoothed_pdr < self.PDR_DROP_THRESHOLD:
                            self.current_tier = 2
                            print(f"🚨 PDR Dropped to {pdr:.2f} (< {self.PDR_DROP_THRESHOLD}). Switching to Tier 2 Fountain Mode.")
                    
                    elif self.current_tier == 2:
                        if self.smoothed_pdr > self.PDR_RECOVER_THRESHOLD:
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

    def transform_hardware_sweep(self, raw_sweep_vector):
        scaled = self.scaler.transform(raw_sweep_vector.reshape(1, -1))
        proj = self.pca.transform(scaled)[0]
        
        w_high = proj[:self.n_identity]
        w_low = proj[self.n_identity:self.total_components]
        
        current_bits = np.where(w_high > 0, 1, 0).astype(np.uint8)
        k_auth = np.packbits(current_bits).tobytes()
        k_health = w_low.tolist()
        return k_auth, k_health

    def transmit(self, k_auth, k_health, telemetry_val1=0.0, telemetry_val2=0.0, telemetry_val3=0.0):
        n_i, s_i = self._generate_crypto_masks(self.sequence_counter)
        header = struct.pack(">IIBfff", int(time.time()), self.sequence_counter, self.current_tier, float(telemetry_val1), float(telemetry_val2), float(telemetry_val3))
        
        if self.current_tier == 1:
            MAX_FLOAT32 = 3.4028235e38
            MIN_FLOAT32 = -3.4028235e38
            sanitized_health = [max(MIN_FLOAT32, min(MAX_FLOAT32, float(x))) for x in k_health]
            
            p_auth = bytes(a ^ b for a, b in zip(k_auth, n_i))
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
    
    # Extract real CSV sweep catalogs
    all_real_csvs = node.device_files_map.get(node.target_device, [])
    
    if len(all_real_csvs) > 0:
        print(f"[Main] Ingesting real hardware base vector from baseline file: {os.path.basename(all_real_csvs[0])}")
        stable_base_vector = load_pzt_csv_vector(all_real_csvs[0], use_phase=USE_PHASE)
    else:
        feature_len = len(REF_FREQ) * (2 if USE_PHASE else 1)
        stable_base_vector = np.random.randn(feature_len)
        
    scenario = sys.argv[1] if len(sys.argv) > 1 else "Scenario_D"
    scaled_base = node.scaler.transform(stable_base_vector.reshape(1, -1))
    proj_base = node.pca.transform(scaled_base)[0]
    w_high_base = proj_base[:node.n_identity]
    stable_indices = np.where(np.abs(w_high_base) > 0.05)[0]

    sweep_idx = 0
    stealth_drift = 0.0
    start_time = time.time()
    
    print(f"\n🚀 SHM Vehicle Verification Environment Active. Running: {scenario}\n")
    
    try:
        while True:
            elapsed_time = time.time() - start_time

            # ------------------------------------------------------------------
            # STEP 1: PHYSICAL HARDWARE IMPEDANCE INGESTION
            # ------------------------------------------------------------------
            if len(all_real_csvs) > 0:
                # Dynamically index files sequentially to pull genuine PZT sweep vectors
                target_file = all_real_csvs[sweep_idx % len(all_real_csvs)]
                real_sweep = load_pzt_csv_vector(target_file, use_phase=USE_PHASE)
                if real_sweep is None:
                    real_sweep = stable_base_vector
            else:
                # Maintain mathematical noise variance floor if dataset files are absent
                thermal_noise = np.random.normal(0, 0.0005, len(stable_base_vector))
                real_sweep = stable_base_vector + thermal_noise
                
            k_auth, k_health = node.transform_hardware_sweep(real_sweep)
            current_health = list(k_health)

            # ------------------------------------------------------------------
            # STEP 2: PURE SHM CROSS-LAYER SCENARIO ENGINE
            # ------------------------------------------------------------------
            if scenario == "Scenario_A":
                if elapsed_time >= 25.0:
                    elapsed_attack = elapsed_time - 25.0
                    stealth_drift = 0.0005 * elapsed_attack
                    for i in range(len(k_health)):
                        k_health[i] += (stealth_drift / np.sqrt(32))
                current_health = k_health

            elif scenario == "Scenario_C":
                if elapsed_time >= 30.0:
                    current_health = [val + 0.25 for val in k_health]

            elif scenario == "Scenario_D":
                # SIMULATING SHM VEHICLE OPERATIONAL VIBRATIONS (Chassis bouncing on terrain)
                true_structural_accel = 1.8 * np.sin(0.4 * elapsed_time)
                
                # Raw sensor readings under normal environmental noise
                sensor_accel = true_structural_accel + np.random.normal(0, 0.15)
                
                # TRANSDUCER HIJACKING ATTACK: Exactly at t=150s, attacker spoofs fake collision/damage
                if elapsed_time >= 150.0:
                    sensor_accel += 15.0  # Spikes kinematic telemetry line maliciously
                
                # THE CROSS-MODAL CONTRADDICTION: Maintain a completely clean, uncompromised material health matrix.
                # If a real structural collapse occurred, 'current_health' would warp drastically.
                # Keeping it flat forces the Digital Twin to detect the telemetry mismatch instantly.
                current_health = [val for val in k_health]

            # ------------------------------------------------------------------
            # STEP 3: TELEMETRY ENCAPSULATION AND PACKET EMISSION
            # ------------------------------------------------------------------
            try:
                if scenario == "Scenario_D":
                    # val1 = clock, val2 = 0.0 (Velocity stripped), val3 = Structural Acceleration
                    node.transmit(k_auth, current_health, telemetry_val1=elapsed_time, telemetry_val2=0.0, telemetry_val3=sensor_accel)
                else:
                    node.transmit(k_auth, current_health, telemetry_val1=elapsed_time, telemetry_val2=0.0, telemetry_val3=0.0)
                
                sweep_idx += 1
                time.sleep(0.01)  # Strict 10ms framework pacing

            except (BrokenPipeError, ConnectionResetError, OSError):
                print("🔌 NS-3 simulation socket closed. Terminating edge node loop cleanly.")
                break
    except KeyboardInterrupt:
        node.close()
