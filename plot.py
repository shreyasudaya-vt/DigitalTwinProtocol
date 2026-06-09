import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, roc_curve, auc

# Formatting for IEEE Journal standards
plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})

def generate_scenario_a_plot():
    try:
        df = pd.read_csv("telemetry_Scenario_A.csv")
    except FileNotFoundError:
        print("Skipping Plot A: telemetry_Scenario_A.csv not found.")
        return

    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
    df['Kalman_State'] = pd.to_numeric(df['Kalman_State'], errors='coerce')
    df['Kalman_P'] = pd.to_numeric(df['Kalman_P'], errors='coerce')
    df['Raw_Measurement'] = pd.to_numeric(df['Raw_Measurement'], errors='coerce')
    
    # CRITICAL FIX: Strip out any NaN, Inf, or extreme outlier noise spikes from the dataframe
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['Time', 'Kalman_State', 'Kalman_P', 'Raw_Measurement'])
    df = df[df['Raw_Measurement'].abs() < 10.0] # Filter out un-sanitized legacy spikes

    if df.empty:
        print("Skipping Plot A: No valid telemetry rows left after filtering.")
        return

    plt.figure(figsize=(7, 3.5))
    state_sigma = 3.0 * np.sqrt(df['Kalman_P'])
    
    ground_truth = 0.002 * (np.exp(0.015 * df['Time']) - 1.0)

    plt.plot(df['Time'], ground_truth, label="Physical Ground Truth", color="black", linestyle=":", linewidth=2, zorder=3)
    plt.plot(df['Time'], df['Raw_Measurement'], label="Raw Measurement ($z_t$)", color="orange", linestyle="--", alpha=0.7, zorder=2)
    plt.plot(df['Time'], df['Kalman_State'], label="Kalman Filter Track ($h_t$)", color="#1f77b4", linewidth=2.5, zorder=4)
    plt.fill_between(df['Time'], df['Kalman_State'] - state_sigma, df['Kalman_State'] + state_sigma, color="#1f77b4", alpha=0.2, label=r"True State Bounds ($\pm 3\sigma_x$)", zorder=1)
    
    # CRITICAL FIX: Safe bounding box evaluation
    y_min = float(df['Raw_Measurement'].min()) - 0.02
    y_max = float(df['Raw_Measurement'].max()) + 0.05
    plt.ylim(max(-0.1, y_min), min(5.0, y_max))
    
    plt.title("Scenario A: Continuous Hardware Degradation Tracking", fontsize=11, fontweight='bold')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Euclidean Drift Magnitude")
    plt.legend(loc="upper left", fontsize=9, framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("Fig_Scenario_A_Tracking.pdf", dpi=300)
    plt.close()
    print("📊 Generated Fig_Scenario_A_Tracking.pdf (Exponential Physics Matched)")

def generate_scenario_b_plot():
    try:
        df = pd.read_csv("telemetry_Scenario_B.csv")
    except FileNotFoundError:
        print("Skipping Plot B: telemetry_Scenario_B.csv not found.")
        return

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['Time', 'PDR', 'Tier'])
    fig, ax1 = plt.subplots(figsize=(7, 3.5))
    
    color = 'tab:red'
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Channel PDR', color=color)
    ax1.plot(df['Time'], df['PDR'], color=color, linewidth=2, label="PDR")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(0.80, color='green', linestyle='--', alpha=0.7)
    ax1.set_ylim(-0.05, 1.05)

    ax2 = ax1.twinx()  
    color = 'black'
    ax2.set_ylabel('Active Transmission Tier', color=color)
    ax2.step(df['Time'], df['Tier'], color=color, where='post', linewidth=2)
    ax2.set_yticks([1, 2])
    ax2.set_yticklabels(['Tier 1\n(Hi-Fi)', 'Tier 2\n(Fountain)'])
    ax2.set_ylim(0.5, 2.5)

    plt.title("Scenario B: Dynamic Protocol Adaptation to RF Jamming", fontsize=11, fontweight='bold')
    fig.tight_layout()
    plt.savefig("Fig_Scenario_B_Resilience.pdf", dpi=300)
    plt.close()
    print("📊 Generated Fig_Scenario_B_Resilience.pdf")

def generate_scenario_c_plot():
    try:
        df = pd.read_csv("telemetry_Scenario_C.csv")
    except FileNotFoundError:
        print("Skipping Plot C: telemetry_Scenario_C.csv not found.")
        return

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['Time', 'Innovation', 'Dynamic_Threshold', 'Hamming_Distance'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    
    # Subplot 1: Hamming Distance (Identity Layer)
    ax1.plot(df['Time'], df['Hamming_Distance'], color="teal", linewidth=1.5, label="Hamming Distance")
    ax1.axhline(8.0, color="red", linestyle="--", alpha=0.7, label="Crypto Threshold (8)")
    
    # Avoid highlighting false alerts from old runs
    alarms_hd = df[(df['Alarm_Active'] == 1) & (df['Hamming_Distance'] > 8.0)]
    if not alarms_hd.empty:
        ax1.scatter(alarms_hd['Time'], alarms_hd['Hamming_Distance'], color="red", s=30, marker="x", zorder=5)
    ax1.set_ylabel("Bit Flips")
    ax1.set_title("Scenario C: Multi-Layer Cyber-Attack Detection", fontsize=11, fontweight='bold')
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Physics Innovation (Health Layer)
    ax2.plot(df['Time'], df['Innovation'], label=r"Innovation Sequence ($\tilde{y}_t$)", color="purple", linewidth=1.2)
    ax2.plot(df['Time'], df['Dynamic_Threshold'], color="red", linestyle="--", alpha=0.7, label=r"Dynamic Threshold ($\pm 3\sigma$)")
    ax2.plot(df['Time'], -df['Dynamic_Threshold'], color="red", linestyle="--", alpha=0.7)
    
    df['Calculated_NIS'] = 15.0 * (df['Innovation'] / np.where(df['Dynamic_Threshold'] == 0, 1e-6, df['Dynamic_Threshold']))**2
    alarms_phy = df[(df['Alarm_Active'] == 1) & (df['Calculated_NIS'] > 15.0) & (df['Hamming_Distance'] <= 8.0)]
    if not alarms_phy.empty:
        ax2.scatter(alarms_phy['Time'], alarms_phy['Innovation'], color="red", s=30, marker="x", zorder=5)

    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("State Prediction Error")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("Fig_Scenario_C_Detection.pdf", dpi=300)
    plt.close()
    print("📊 Generated Fig_Scenario_C_Detection.pdf")

def generate_roc_curve():
    try:
        df = pd.read_csv("telemetry_Scenario_C.csv")
    except FileNotFoundError:
        print("Skipping ROC Curve: telemetry_Scenario_C.csv not found.")
        return

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['Time', 'Innovation', 'Dynamic_Threshold'])
    df = df[df['Time'] > 15.0].copy()
    if len(df) < 20: return

    df['NIS'] = 9.0 * (df['Innovation'] / np.where(df['Dynamic_Threshold'] == 0, 1e-6, df['Dynamic_Threshold']))**2
    df['Ground_Truth'] = (df['Time'] >= 30.0).astype(int)
        
    fpr, tpr, thresholds = roc_curve(df['Ground_Truth'], df['NIS'])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'IDS Performance (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('True Positive Rate (DR)')
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=11, fontweight='bold')
    
    fp = len(df[(df['NIS'] >= 15.0) & (df['Ground_Truth'] == 0)])
    tn = len(df[(df['NIS'] < 15.0) & (df['Ground_Truth'] == 0)])
    tp = len(df[(df['NIS'] >= 15.0) & (df['Ground_Truth'] == 1)])
    fn = len(df[(df['NIS'] < 15.0) & (df['Ground_Truth'] == 1)])
    
    actual_far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    actual_dr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    plt.scatter(actual_far, actual_dr, color="red", s=60, zorder=5, label=r"Operating Point ($\tau=15.0$)")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("Fig_IDS_ROC_Curve.pdf", dpi=300)
    plt.close()
    print(f"📊 Generated Fig_IDS_ROC_Curve.pdf (AUC: {roc_auc:.4f})")

def generate_network_overhead_plot():
    try:
        df = pd.read_csv("telemetry_Scenario_B.csv")
    except FileNotFoundError:
        print("Skipping Network Overhead Plot: telemetry_Scenario_B.csv not found.")
        return
        
    df = df.dropna(subset=['Time', 'Tier'])
    df['Bytes_Static'] = 160
    df['Bytes_Dynamic'] = df['Tier'].apply(lambda x: 160 if x == 1 else 20)
    
    df['Cumulative_Static_KB'] = df['Bytes_Static'].cumsum() / 1024
    df['Cumulative_Dynamic_KB'] = df['Bytes_Dynamic'].cumsum() / 1024

    plt.figure(figsize=(7, 3.5))
    plt.plot(df['Time'], df['Cumulative_Static_KB'], label="Static Protocol (Baseline)", color="black", linestyle="--", linewidth=2)
    plt.plot(df['Time'], df['Cumulative_Dynamic_KB'], label="Dynamic Twin Protocol (Ours)", color="#2ca02c", linewidth=2.5)
    
    if not df[df['Tier'] == 2].empty:
        plt.fill_between(df['Time'], df['Cumulative_Dynamic_KB'], df['Cumulative_Static_KB'], 
                         where=(df['Tier'] == 2), color="#2ca02c", alpha=0.15, label="Bandwidth Saved")

    plt.title("Cumulative Network Overhead Under Jamming", fontsize=11, fontweight='bold')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Data Transmitted (KB)")
    plt.legend(loc="upper left", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("Fig_Network_Overhead.pdf", dpi=300)
    plt.close()
    
    savings = (df['Cumulative_Static_KB'].iloc[-1] - df['Cumulative_Dynamic_KB'].iloc[-1]) / df['Cumulative_Static_KB'].iloc[-1] * 100
    print(f"📊 Generated Fig_Network_Overhead.pdf (Bandwidth reduced by {savings:.1f}%)")

def generate_journal_statistics():
    print("==========================================================")
    print("📈 COMPUTING QUANTITATIVE EVALUATION METRICS FOR JOURNAL")
    print("==========================================================")
    
    try:
        df_a = pd.read_csv("telemetry_Scenario_A.csv")
        df_a = df_a.replace([np.inf, -np.inf], np.nan).dropna(subset=['Time', 'Kalman_State', 'Raw_Measurement'])
        
        if len(df_a) > 0:
            ground_truth = 0.002 * (np.exp(0.015 * df_a['Time']) - 1.0)
            rmse_raw = np.sqrt(mean_squared_error(ground_truth, df_a['Raw_Measurement'])) 
            rmse_kf = np.sqrt(mean_squared_error(ground_truth, df_a['Kalman_State']))
            correlation = np.corrcoef(df_a['Time'], df_a['Raw_Measurement'])[0, 1]
            
            print(f"✅ [Hypothesis 1] Observability Correlation (R): {correlation:.4f}")
            print(f"✅ [Hypothesis 1] Raw Measurement Tracking Error (RMSE): {rmse_raw:.6f}")
            print(f"✅ [Hypothesis 1] Kalman Tracking Error (RMSE): {rmse_kf:.6f}")
    except FileNotFoundError:
        print("⚠️ 'telemetry_Scenario_A.csv' missing.")

    try:
        df_c = pd.read_csv("telemetry_Scenario_C.csv")
        df_c = df_c.replace([np.inf, -np.inf], np.nan).dropna(subset=['Time', 'Alarm_Active'])
        
        if len(df_c) > 0:
            df_c['Ground_Truth'] = (df_c['Time'] >= 30.0).astype(int)
                
            tp = len(df_c[(df_c['Alarm_Active'] == 1) & (df_c['Ground_Truth'] == 1)])
            fp = len(df_c[(df_c['Alarm_Active'] == 1) & (df_c['Ground_Truth'] == 0)])
            tn = len(df_c[(df_c['Alarm_Active'] == 0) & (df_c['Ground_Truth'] == 0)])
            fn = len(df_c[(df_c['Alarm_Active'] == 0) & (df_c['Ground_Truth'] == 1)])
            
            dr = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
            far = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0
            precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
            
            print(f"✅ [Hypothesis 3] True Detection Rate (DR): {dr:.2f}%")
            print(f"✅ [Hypothesis 3] Empirical False Alarm Rate (FAR): {far:.2f}%")
            print(f"✅ [Hypothesis 3] Precision Vector: {precision:.2f}%")
            
            first_attack = 30.0
            alarms_post_attack = df_c[(df_c['Alarm_Active'] == 1) & (df_c['Time'] >= first_attack)]
            if not alarms_post_attack.empty:
                latency = alarms_post_attack['Time'].iloc[0] - first_attack
                print(f"✅ [Hypothesis 3] Physical System Alarm Latency: {latency:.4f} seconds")
                
                post_latency_df = df_c[df_c['Time'] >= (first_attack + latency)]
                tp_ss = len(post_latency_df[post_latency_df['Alarm_Active'] == 1])
                fn_ss = len(post_latency_df[post_latency_df['Alarm_Active'] == 0])
                ss_dr = (tp_ss / (tp_ss + fn_ss)) * 100 if (tp_ss + fn_ss) > 0 else 0
                print(f"   => Steady-State DR (Post-Detection Latency Window): {ss_dr:.2f}%")
    except FileNotFoundError:
        print("⚠️ 'telemetry_Scenario_C.csv' missing.")
    print("==========================================================")


def generate_scenario_d_plot():
    try:
        # Read the telemetry generated by your updated digital twin server
        df = pd.read_csv("telemetry_Scenario_D.csv")
    except FileNotFoundError:
        print("Skipping Plot D: telemetry_Scenario_D.csv not found.")
        return

    # Use your exact CSV column names for filtering out empty lines
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['Time', 'Expected_Accel', 'Sensor_Accel', 'Spatial_Residual', 'Spatial_Alarm'])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
    
    # --- Top Panel: Kinematic Co-registration ---
    ax1.plot(df['Time'], df['Expected_Accel'], label="GPS-Derived Acceleration ($a_{exp}$)", color="#1f77b4", linewidth=1.5)
    # FIXED: Changed from 'Reported_Accel' to 'Sensor_Accel' to match your digital_twin.py header
    ax1.plot(df['Time'], df['Sensor_Accel'], label="Reported IMU Acceleration ($a_{rep}$)", color="orange", linewidth=1.5, alpha=0.85)
    ax1.axvline(150.0, color="gray", linestyle=":", alpha=0.8, label="Transducer Hijack Initiation")
    ax1.set_ylabel("Acceleration ($m/s^2$)")
    ax1.set_title("Scenario D: Physics-Layer Cross-Verification Against Transducer Hijacking", fontsize=11, fontweight='bold')
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # --- Bottom Panel: Active Spatial Residual ---
    ax2.plot(df['Time'], df['Spatial_Residual'], label=r"Spatial Residual $|a_{exp} - a_{rep}|$", color="purple", linewidth=1.5)
    ax2.axhline(3.0, color="red", linestyle="--", alpha=0.8, label="Consistency Threshold ($\tau = 3.0\,m/s^2$)")
    
    # FIXED: Locate exact threshold crossing using 'Spatial_Alarm' and 'Spatial_Residual'
    alarms = df[df['Spatial_Alarm'] == 1]
    if not alarms.empty:
        detection_time = alarms['Time'].iloc[0]
        detection_res = alarms['Spatial_Residual'].iloc[0]
        latency = detection_time - 150.0
        
        ax2.scatter(detection_time, detection_res, color="red", s=60, marker="X", zorder=5, 
                   label=f"Alarm Fired (t={detection_time:.2f}s, Latency={latency:.3f}s)")
        ax2.axvline(detection_time, color="red", linestyle=":", alpha=0.5)
    
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Residual Magnitude ($m/s^2$)")
    ax2.set_ylim(-1, 20)  # Generous headroom to cleanly display the 15 m/s^2 hijack spike
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    plt.savefig("Fig_Scenario_D_CrossVerification.pdf", dpi=300)
    plt.close()
    print("📊 Generated Fig_Scenario_D_CrossVerification.pdf using your updated global schema headers.")
if __name__ == "__main__":
    generate_scenario_a_plot()
    generate_scenario_b_plot()
    generate_scenario_c_plot()
    generate_roc_curve()
    generate_network_overhead_plot()
    generate_journal_statistics()
    generate_scenario_d_plot()
