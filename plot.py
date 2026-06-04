import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the telemetry data
try:
    df = pd.read_csv("thesis_telemetry_log.csv")
except FileNotFoundError:
    print("Error: thesis_telemetry_log.csv not found. Run the simulation first.")
    exit()

# Set up global plot styling for academic papers
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

# =====================================================================
# SCENARIO A: Low-Variance PCA Degradation Tracking (Hypothesis 1 & 3)
# =====================================================================
plt.figure(figsize=(10, 5))
plt.plot(df['Time'], df['Kalman_State'], label="Kalman Filter Track ($h_t$)", color="blue", linewidth=2)

# Recreate the dynamic variance bounds (3-sigma) based on the inflated Rt logic
R_base = 0.1
alpha = 5.0
dynamic_Rt = R_base + alpha * (1.0 - df['PDR'])
sigma_bounds = 3.0 * np.sqrt(dynamic_Rt) # Approximation for plotting

plt.fill_between(df['Time'], 
                 df['Kalman_State'] - sigma_bounds, 
                 df['Kalman_State'] + sigma_bounds, 
                 color="lightblue", alpha=0.4, label="State Variance Bounds ($\pm 3\sigma$)")

plt.title("Scenario A: Continuous Low-Variance Hardware Degradation Tracking")
plt.xlabel("Time (seconds)")
plt.ylabel("Euclidean Drift (Trailing PCA Components)")
plt.legend(loc="upper left")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("Scenario_A_Health.png", dpi=300)

# =====================================================================
# SCENARIO B: Adaptive Fountain Transmission Under Jamming (Hypothesis 2)
# =====================================================================
fig, ax1 = plt.subplots(figsize=(10, 5))

color = 'tab:red'
ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Channel PDR', color=color)
ax1.plot(df['Time'], df['PDR'], color=color, label="Packet Delivery Ratio", linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.axhline(0.80, color='green', linestyle='--', alpha=0.7, label="Tier Switch Threshold (80%)")
ax1.set_ylim(-0.1, 1.1)

ax2 = ax1.twinx()  
color = 'black'
ax2.set_ylabel('Active Transmission Tier', color=color)  
ax2.step(df['Time'], df['Tier'], color=color, where='post', label="Protocol Tier State", linewidth=2.5)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_yticks([1, 2])
ax2.set_yticklabels(['Tier 1\n(158B High-Fidelity)', 'Tier 2\n(18B Fountain Mode)'])
ax2.set_ylim(0.5, 2.5)

fig.suptitle("Scenario B: Dynamic Protocol Adaptation to RF Jamming")
fig.tight_layout()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("Scenario_B_Jamming.png", dpi=300)

# =====================================================================
# SCENARIO C: Physics-Constrained Injection Detection (Hypothesis 3)
# =====================================================================
plt.figure(figsize=(10, 5))
plt.plot(df['Time'], df['Innovation'], label="Innovation Sequence ($\\tilde{y}_t$)", color="purple", linewidth=1.5)

# Plot dynamic alarm thresholds
plt.plot(df['Time'], sigma_bounds, color="red", linestyle="--", label="Dynamic Alarm Threshold ($\\pm\\tau_t$)")
plt.plot(df['Time'], -sigma_bounds, color="red", linestyle="--")

# Highlight alarms with red X markers
alarms = df[df['Threshold_Alarm'] == 1]
if not alarms.empty:
    plt.scatter(alarms['Time'], alarms['Innovation'], color="red", s=100, marker="x", zorder=5, label="Spoofing Alarm Triggered")

plt.title("Scenario C: Anomaly Detection via Physics-Constrained Innovation")
plt.xlabel("Time (seconds)")
plt.ylabel("State Prediction Error")
plt.legend(loc="upper right")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("Scenario_C_Injection.png", dpi=300)

print("✅ Plots generated successfully! Check your folder for the new PNG files.")