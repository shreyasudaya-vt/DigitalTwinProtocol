import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})

def generate_scenario_a_plot():
    try:
        df = pd.read_csv("telemetry_Scenario_A.csv")
    except FileNotFoundError:
        print("Skipping Plot A: telemetry_Scenario_A.csv not found.")
        return

    plt.figure(figsize=(7, 3.5))
    plt.plot(df['Time'], df['Raw_Measurement'], label="Raw Measurement ($z_t$)", color="orange", linestyle="--")
    plt.plot(df['Time'], df['Kalman_State'], label="Kalman Filter Track ($h_t$)", color="#1f77b4", linewidth=2)
    
    # Render safety boundaries around the active track
    plt.fill_between(df['Time'], 
                     df['Kalman_State'] - df['Dynamic_Threshold'], 
                     df['Kalman_State'] + df['Dynamic_Threshold'], 
                     color="#aec7e8", alpha=0.4, label="State Bounds ($\pm 3\sigma$)")
    
    plt.title("Scenario A: Continuous Hardware Degradation Tracking", fontsize=11, fontweight='bold')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Euclidean Drift")
    plt.legend(loc="upper left", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("Fig_Scenario_A_Tracking.pdf", dpi=300)
    plt.close()

def generate_scenario_b_plot():
    try:
        df = pd.read_csv("telemetry_Scenario_B.csv")
    except FileNotFoundError:
        print("Skipping Plot B: telemetry_Scenario_B.csv not found.")
        return

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

def generate_scenario_c_plot():
    try:
        df = pd.read_csv("telemetry_Scenario_C.csv")
    except FileNotFoundError:
        print("Skipping Plot C: telemetry_Scenario_C.csv not found.")
        return

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

if __name__ == "__main__":
    generate_scenario_a_plot()
    generate_scenario_b_plot()
    generate_scenario_c_plot()
    print("🎉 All three academic figures compiled independently and exported to vector PDF format!")