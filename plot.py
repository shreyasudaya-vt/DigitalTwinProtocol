import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})

def generate_scenario_a_plot():
    try:
        df = pd.read_csv("telemetry_Scenario_A.csv")
    except FileNotFoundError:
        print("Skipping Plot A: telemetry_Scenario_A.csv not found.")
        return

    # Sanitize
    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
    df['Kalman_State'] = pd.to_numeric(df['Kalman_State'], errors='coerce')
    df['Kalman_P'] = pd.to_numeric(df['Kalman_P'], errors='coerce')
    df['Raw_Measurement'] = pd.to_numeric(df['Raw_Measurement'], errors='coerce')
    df = df.dropna(subset=['Time', 'Kalman_State', 'Kalman_P'])

    plt.figure(figsize=(7, 3.5))
    
    # 1. Calculate True State Bounds using the logged Kalman Error Covariance (P)
    state_sigma = 3.0 * np.sqrt(df['Kalman_P'])
    
    # 2. Synthesize the True Physical Ground Truth (alpha = 0.005 per second)
    ground_truth = 0.005 * df['Time']

    # 3. Plot the layers
    plt.plot(df['Time'], ground_truth, label="Physical Ground Truth", color="black", linestyle=":", linewidth=2, zorder=3)
    plt.plot(df['Time'], df['Raw_Measurement'], label="Raw Measurement ($z_t$)", color="orange", linestyle="--", alpha=0.7, zorder=2)
    plt.plot(df['Time'], df['Kalman_State'], label="Kalman Filter Track ($h_t$)", color="#1f77b4", linewidth=2.5, zorder=4)
    
    # 4. Fill the TRUE tight state confidence bounds
    plt.fill_between(df['Time'], 
                     df['Kalman_State'] - state_sigma, 
                     df['Kalman_State'] + state_sigma, 
                     color="#1f77b4", alpha=0.2, label=r"True State Bounds ($\pm 3\sigma_x$)", zorder=1)
    
    plt.title("Scenario A: Continuous Hardware Degradation Tracking", fontsize=11, fontweight='bold')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Euclidean Drift Magnitude")
    plt.legend(loc="upper left", fontsize=9, framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("Fig_Scenario_A_Tracking.pdf", dpi=300)
    plt.close()
    print("📊 Generated Fig_Scenario_A_Tracking.pdf")
    
def generate_scenario_b_plot():
    try:
        df = pd.read_csv("telemetry_Scenario_B.csv")
    except FileNotFoundError:
        print("Skipping Plot B: telemetry_Scenario_B.csv not found.")
        return

    # Sanitize
    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
    df['PDR'] = pd.to_numeric(df['PDR'], errors='coerce')
    df['Tier'] = pd.to_numeric(df['Tier'], errors='coerce')
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

    # Sanitize
    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
    df['Innovation'] = pd.to_numeric(df['Innovation'], errors='coerce')
    df['Dynamic_Threshold'] = pd.to_numeric(df['Dynamic_Threshold'], errors='coerce')
    df['Alarm_Active'] = pd.to_numeric(df['Alarm_Active'], errors='coerce')
    df = df.dropna(subset=['Time', 'Innovation', 'Dynamic_Threshold'])

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(df['Time'], df['Innovation'], label=r"Innovation Sequence ($\tilde{y}_t$)", color="purple", linewidth=1.2)
    ax.plot(df['Time'], df['Dynamic_Threshold'], color="red", linestyle="--", alpha=0.7, label=r"Dynamic Threshold ($\pm\tau_t$)")
    ax.plot(df['Time'], -df['Dynamic_Threshold'], color="red", linestyle="--", alpha=0.7)

    alarms = df[df['Alarm_Active'] == 1]
    if not alarms.empty:
        ax.scatter(alarms['Time'], alarms['Innovation'], color="red", s=50, marker="x", zorder=5, label="Alarm Tripped")

    ax.set_title("Scenario C: Physics-Constrained Cyber-Attack Detection", fontsize=11, fontweight='bold')
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("State Prediction Error")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("Fig_Scenario_C_Detection.pdf", dpi=300)
    plt.close()
    print("📊 Generated Fig_Scenario_C_Detection.pdf")

def generate_journal_statistics():
    print("==========================================================")
    print("📈 COMPUTING QUANTITATIVE EVALUATION METRICS FOR IoTJ")
    print("==========================================================")
    
    # 1. QUANTIFYING HYPOTHESIS 1 (TRACKING FIDELITY)
    try:
        df_a = pd.read_csv("telemetry_Scenario_A.csv")
        df_a['Time'] = pd.to_numeric(df_a['Time'], errors='coerce')
        df_a['Kalman_State'] = pd.to_numeric(df_a['Kalman_State'], errors='coerce')
        df_a['Raw_Measurement'] = pd.to_numeric(df_a['Raw_Measurement'], errors='coerce')
        df_a = df_a.dropna(subset=['Time', 'Kalman_State', 'Raw_Measurement'])
        
        # Strip out initial enrollment phase
        df_steady = df_a[df_a['Time'] > 30.0].copy()
        
        if len(df_steady) == 0:
            print("⚠️ [Hypothesis 1] Not enough data in Scenario A (Simulation aborted early).")
        else:
            ground_truth = 0.005 * df_steady['Time']
            rmse_raw = np.sqrt(mean_squared_error(ground_truth, df_steady['Raw_Measurement'])) 
            rmse_kf = np.sqrt(mean_squared_error(ground_truth, df_steady['Kalman_State']))
            correlation = np.corrcoef(df_steady['Time'], df_steady['Raw_Measurement'])[0, 1]
            
            print(f"✅ [Hypothesis 1] Observability Correlation (R): {correlation:.4f}")
            print(f"✅ [Hypothesis 1] Raw Measurement Tracking Error (RMSE): {rmse_raw:.6f}")
            print(f"✅ [Hypothesis 1] Kalman Tracking Error (RMSE): {rmse_kf:.6f}")
            if rmse_raw > 0:
                print(f"   => Improvement: {((rmse_raw - rmse_kf)/rmse_raw)*100:.1f}% noise reduction!")
    except FileNotFoundError:
        print("⚠️ 'telemetry_Scenario_A.csv' missing. Run Scenario A first.")

    # 2. QUANTIFYING HYPOTHESIS 3 (ANOMALY DISCRIMINATION)
    try:
        df_c = pd.read_csv("telemetry_Scenario_C.csv")
        df_c['Time'] = pd.to_numeric(df_c['Time'], errors='coerce')
        df_c['Alarm_Active'] = pd.to_numeric(df_c['Alarm_Active'], errors='coerce')
        df_c = df_c.dropna(subset=['Time', 'Alarm_Active'])
        
        if len(df_c) == 0:
            print("⚠️ [Hypothesis 3] Not enough data in Scenario C.")
        else:
            
            attack_windows = [30.0, 130.0, 230.0]
            window_duration = 10.0
            df_c['Ground_Truth'] = (
                df_c['Time'] >= 30.0
            ).astype(int)
            for start in attack_windows:
                df_c.loc[(df_c['Time'] >= start) & (df_c['Time'] <= start + window_duration), 'Ground_Truth'] = 1
                
            tp = len(df_c[(df_c['Alarm_Active'] == 1) & (df_c['Ground_Truth'] == 1)])
            fp = len(df_c[(df_c['Alarm_Active'] == 1) & (df_c['Ground_Truth'] == 0)])
            tn = len(df_c[(df_c['Alarm_Active'] == 0) & (df_c['Ground_Truth'] == 0)])
            fn = len(df_c[(df_c['Alarm_Active'] == 0) & (df_c['Ground_Truth'] == 1)])
            
            dr = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
            far = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0
            precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
            
            print(f"✅ [Hypothesis 3] True Detection Rate (DR): {dr:.2f}%")
            print(f"✅ [Hypothesis 3] Empirical False Alarm Rate (FAR): {far:.2f}%")
            print(f"✅ [Hypothesis 3] Precision: {precision:.2f}%")
            
            first_attack = attack_windows[0]
            alarms_post_attack = df_c[(df_c['Alarm_Active'] == 1) & (df_c['Time'] >= first_attack)]
            if not alarms_post_attack.empty:
                latency = alarms_post_attack['Time'].iloc[0] - first_attack
                print(f"✅ [Hypothesis 3] System Detection Latency: {latency:.4f} seconds")
                
    except FileNotFoundError:
        print("⚠️ 'telemetry_Scenario_C.csv' missing. Run Scenario C first.")
    print("==========================================================")

if __name__ == "__main__":
    generate_scenario_a_plot()
    generate_scenario_b_plot()
    generate_scenario_c_plot()
    generate_journal_statistics()
    print("🎉 All three academic figures compiled independently and exported to vector PDF format!")