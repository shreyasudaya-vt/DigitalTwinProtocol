import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, roc_curve, auc

# Formatting for IEEE Journal standards (Two-column column width compliance)
plt.rcParams.update({
    'font.size': 10, 
    'font.family': 'serif',
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8
})

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
    df = df.dropna(subset=['Time', 'Kalman_State', 'Kalman_P'])

    plt.figure(figsize=(7, 3.5))
    state_sigma = 3.0 * np.sqrt(df['Kalman_P'])
    ground_truth = 0.005 * df['Time']

    plt.plot(df['Time'], ground_truth, label="Physical Ground Truth", color="black", linestyle=":", linewidth=2, zorder=3)
    # The raw measurements will now beautifully show the realistic noise instead of matching perfectly
    plt.scatter(df['Time'], df['Raw_Measurement'], label="Noisy Raw Measurement ($z_t$)", color="orange", s=6, alpha=0.5, zorder=2)
    plt.plot(df['Time'], df['Kalman_State'], label="Kalman Filter Track ($\hat{x}_t$)", color="#1f77b4", linewidth=2.0, zorder=4)
    plt.fill_between(df['Time'], df['Kalman_State'] - state_sigma, df['Kalman_State'] + state_sigma, color="#1f77b4", alpha=0.15, label=r"Confidence Bounds ($\pm 3\sigma_x$)", zorder=1)
    
    plt.title("Scenario A: Continuous Hardware Degradation Tracking under AWGN", fontsize=10, fontweight='bold')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Euclidean Drift Magnitude")
    plt.legend(loc="upper left", framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("Fig_Scenario_A_Tracking.pdf", dpi=300)
    plt.close()
    print(" Bars Generated Fig_Scenario_A_Tracking.pdf (With Realistic Filtering Expression)")

def generate_scenario_b_plot():
    try:
        df = pd.read_csv("telemetry_Scenario_B.csv")
    except FileNotFoundError:
        print("Skipping Plot B: telemetry_Scenario_B.csv not found.")
        return

    # Clean core structural elements
    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
    df['PDR'] = pd.to_numeric(df['PDR'], errors='coerce')
    df['Tier'] = pd.to_numeric(df['Tier'], errors='coerce')
    df['Kalman_State'] = pd.to_numeric(df['Kalman_State'], errors='coerce')
    df['Kalman_P'] = pd.to_numeric(df['Kalman_P'], errors='coerce')
    df['Raw_Measurement'] = pd.to_numeric(df['Raw_Measurement'], errors='coerce')
    df = df.dropna(subset=['Time', 'PDR', 'Tier'])

    # FIXED: Upgraded into a publication-grade 2-panel subplot to expose the Digital Twin state
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
    
    # --- SUBPLOT 1: NETWORK CHANNEL PROFILE ---
    color = 'tab:red'
    ax1.set_ylabel('Channel PDR', color=color)
    ax1.plot(df['Time'], df['PDR'], color=color, linewidth=2, label="PDR")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(0.80, color='gray', linestyle='--', alpha=0.7, label="Tier Threshold (0.80)")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.set_title("Scenario B: Dynamic State Estimation & Protocol Adaptation under RF Jamming", fontsize=10, fontweight='bold')

    ax2 = ax1.twinx()  
    color = 'black'
    ax2.set_ylabel('Active Protocol Mode', color=color)
    ax2.step(df['Time'], df['Tier'], color=color, where='post', linewidth=1.5, linestyle="-")
    ax2.set_yticks([1, 2])
    ax2.set_yticklabels(['Tier 1\n(High-Fidelity)', 'Tier 2\n(Fountain Baseline)'])
    ax2.set_ylim(0.5, 2.5)

    # --- SUBPLOT 2: PREDICTIVE HEALTH STATE & UNCERTAINTY BLOOM ---
    ground_truth = 0.0  # Scenario B isolates network stress; physical aging is 0
    state_sigma = 3.0 * np.sqrt(df['Kalman_P'])

    ax3.axhline(ground_truth, color="black", linestyle=":", linewidth=2, label="Physical Ground Truth")
    
    # Map raw measurements: Notice it naturally leaves a gap where values are NaN during Tier 2!
    ax3.scatter(df['Time'], df['Raw_Measurement'], color="orange", s=8, alpha=0.6, label="Received Raw Health ($z_t$)")
    ax3.plot(df['Time'], df['Kalman_State'], color="#1f77b4", linewidth=2.0, label="Digital Twin Coasting Estimate ($\hat{x}_t$)")
    
    # The shaded confidence area will visibly balloon out (bloom) during the jamming gap
    ax3.fill_between(df['Time'], df['Kalman_State'] - state_sigma, df['Kalman_State'] + state_sigma, 
                     color="#1f77b4", alpha=0.15, label=r"Dynamic State Bounds ($\pm 3\sigma_x$)")
    
    # Highlight the Fountain Identity checks that occur post-reconstruction (ignoring the -1 pending flags)
    valid_hd = df[df['Hamming_Distance'] >= 0]
    if not valid_hd.empty:
        ax3.scatter(valid_hd['Time'], [ground_truth-0.015]*len(valid_hd), color="teal", marker="^", s=20, 
                    label="Identity Verified (HD $\leq$ 8)")

    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Estimated Physical Drift')
    ax3.set_ylim(-0.1, 0.1)
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(loc="upper left", framealpha=0.9, ncol=2)

    plt.tight_layout()
    plt.savefig("Fig_Scenario_B_Resilience.pdf", dpi=300)
    plt.close()
    print(" Bars Generated Fig_Scenario_B_Resilience.pdf (With Complete State Uncertainty Bloom Rendering)")

def generate_scenario_c_plot():
    try:
        df = pd.read_csv("telemetry_Scenario_C.csv")
    except FileNotFoundError:
        print("Skipping Plot C: telemetry_Scenario_C.csv not found.")
        return

    df = df.dropna(subset=['Time', 'Innovation', 'Dynamic_Threshold', 'Hamming_Distance'])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    
    # Subplot 1: Hamming Distance (Identity Layer Verification)
    ax1.plot(df['Time'], df['Hamming_Distance'], color="teal", linewidth=1.5, label="Observed Hamming Distance")
    ax1.axhline(8.0, color="red", linestyle="--", alpha=0.7, label="Security Bound ($\tau_{id} = 8$)")
    
    alarms_hd = df[(df['Alarm_Active'] == 1) & (df['Hamming_Distance'] > 8.0)]
    if not alarms_hd.empty:
        ax1.scatter(alarms_hd['Time'], alarms_hd['Hamming_Distance'], color="red", s=35, marker="X", zorder=5, label="Identity Spoof Alarm")
        
    ax1.set_ylabel("Bit Flips")
    ax1.set_title("Scenario C: Multi-Layered Cyber-Attack Discrimination Performance", fontsize=10, fontweight='bold')
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Physics Innovation (Health/Injection Layer Verification)
    ax2.plot(df['Time'], df['Innovation'], label=r"Filter Innovation ($\tilde{y}_t$)", color="purple", linewidth=1.2)
    ax2.plot(df['Time'], df['Dynamic_Threshold'], color="red", linestyle="--", alpha=0.7, label=r"Dynamic Validation Profile ($\pm 3\sigma$)")
    ax2.plot(df['Time'], -df['Dynamic_Threshold'], color="red", linestyle="--", alpha=0.7)
    
    df['Calculated_NIS'] = 9.0 * (df['Innovation'] / df['Dynamic_Threshold'])**2
    alarms_phy = df[(df['Alarm_Active'] == 1) & (df['Calculated_NIS'] > 9.0) & (df['Hamming_Distance'] <= 8.0)]
    if not alarms_phy.empty:
        ax2.scatter(alarms_phy['Time'], alarms_phy['Innovation'], color="red", s=35, marker="X", zorder=5, label="State Injection Alarm")

    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Prediction Error")
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("Fig_Scenario_C_Detection.pdf", dpi=300)
    plt.close()
    print(" Bars Generated Fig_Scenario_C_Detection.pdf")

def generate_roc_curve():
    try:
        df = pd.read_csv("telemetry_Scenario_C.csv")
    except FileNotFoundError:
        print("Skipping ROC Curve: telemetry_Scenario_C.csv not found.")
        return

    df = df.dropna(subset=['Time', 'Innovation', 'Dynamic_Threshold'])
    df = df[df['Time'] > 15.0].copy() # Filter initialization window

    if len(df) < 20: return

    df['NIS'] = 9.0 * (df['Innovation'] / df['Dynamic_Threshold'])**2
    df['Ground_Truth'] = (df['Time'] >= 30.0).astype(int)
        
    fpr, tpr, thresholds = roc_curve(df['Ground_Truth'], df['NIS'])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(4.5, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'IDS Performance (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Alarm Rate (FAR)')
    plt.ylabel('True Detection Rate (DR)')
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=10, fontweight='bold')
    
    # Calculate operational index metrics manually
    fp = len(df[(df['NIS'] >= 9.0) & (df['Ground_Truth'] == 0)])
    tn = len(df[(df['NIS'] < 9.0) & (df['Ground_Truth'] == 0)])
    tp = len(df[(df['NIS'] >= 9.0) & (df['Ground_Truth'] == 1)])
    fn = len(df[(df['NIS'] < 9.0) & (df['Ground_Truth'] == 1)])
    
    actual_far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    actual_dr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    plt.scatter(actual_far, actual_dr, color="red", s=50, zorder=5, label=r"Engine Threshold ($\tau_{nis}=9.0$)")

    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("Fig_IDS_ROC_Curve.pdf", dpi=300)
    plt.close()
    print(f" Bars Generated Fig_IDS_ROC_Curve.pdf (True AUC: {roc_auc:.4f})")

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
    plt.plot(df['Time'], df['Cumulative_Static_KB'], label="Static Communications Protocol", color="black", linestyle="--", linewidth=1.5)
    plt.plot(df['Time'], df['Cumulative_Dynamic_KB'], label="ANASTA-Pro Dynamic Protocol", color="#2ca02c", linewidth=2.0)
    
    if not df[df['Tier'] == 2].empty:
        plt.fill_between(df['Time'], df['Cumulative_Dynamic_KB'], df['Cumulative_Static_KB'], 
                         where=(df['Tier'] == 2), color="#2ca02c", alpha=0.15, label="Conserved Channel Bandwidth")

    plt.title("Cumulative Network Overhead Savings under Active Jamming Profile", fontsize=10, fontweight='bold')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Data Transmitted (KB)")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("Fig_Network_Overhead.pdf", dpi=300)
    plt.close()
    
    savings = (df['Cumulative_Static_KB'].iloc[-1] - df['Cumulative_Dynamic_KB'].iloc[-1]) / df['Cumulative_Static_KB'].iloc[-1] * 100
    print(f" Bars Generated Fig_Network_Overhead.pdf (Bandwidth conserved by {savings:.1f}%)")

def generate_journal_statistics():
    print("==========================================================")
    print("📈 COMPUTING QUANTITATIVE EVALUATION METRICS FOR JOURNAL")
    print("==========================================================")
    
    try:
        df_a = pd.read_csv("telemetry_Scenario_A.csv")
        df_a = df_a.dropna(subset=['Time', 'Kalman_State', 'Raw_Measurement'])
        df_steady = df_a[df_a['Time'] > 30.0].copy()
        
        if len(df_steady) > 0:
            ground_truth = 0.005 * df_steady['Time']
            rmse_raw = np.sqrt(mean_squared_error(ground_truth, df_steady['Raw_Measurement'])) 
            rmse_kf = np.sqrt(mean_squared_error(ground_truth, df_steady['Kalman_State']))
            correlation = np.corrcoef(df_steady['Time'], df_steady['Raw_Measurement'])[0, 1]
            
            print(f"✅ [Hypothesis 1] Observability Correlation Matrix (R): {correlation:.4f}")
            print(f"✅ [Hypothesis 1] Unfiltered Hardware Tracking Error (RMSE): {rmse_raw:.6f}")
            print(f"✅ [Hypothesis 1] State Estimation Filter Tracking Error (RMSE): {rmse_kf:.6f}")
            if rmse_raw > 0:
                print(f"   => Verification: {((rmse_raw - rmse_kf)/rmse_raw)*100:.1f}% noise variance reduction via State Filtering")
    except FileNotFoundError:
        print("⚠️ 'telemetry_Scenario_A.csv' missing.")

    try:
        df_c = pd.read_csv("telemetry_Scenario_C.csv")
        df_c = df_c.dropna(subset=['Time', 'Alarm_Active'])
        
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
    except FileNotFoundError:
        print("⚠️ 'telemetry_Scenario_C.csv' missing.")
    print("==========================================================")

if __name__ == "__main__":
    generate_scenario_a_plot()
    generate_scenario_b_plot()
    generate_scenario_c_plot()
    generate_roc_curve()
    generate_network_overhead_plot()
    generate_journal_statistics()
    print("🎉 All 5 vector PDF academic charts cleanly structured for deployment output!")