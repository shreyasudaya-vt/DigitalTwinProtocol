import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# Load the telemetry data
try:
    df = pd.read_csv("thesis_telemetry_log.csv")
except FileNotFoundError:
    print("Error: thesis_telemetry_log.csv not found. Run the simulation first.")
    exit()

# Set up global plot styling for academic papers
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

# Recreate the dynamic variance bounds (3-sigma) based on the inflated Rt logic
R_base = 0.1
alpha = 5.0
dynamic_Rt = R_base + alpha * (1.0 - df['PDR'])
sigma_bounds = 3.0 * np.sqrt(dynamic_Rt)

# =====================================================================
# SCENARIO A: Continuous Low-Variance Hardware Degradation Tracking
# =====================================================================
plt.figure(figsize=(10, 5))
plt.plot(df['Time'], df['Kalman_State'], label="Kalman Filter Track ($h_t$)", color="#1f77b4", linewidth=2)

plt.fill_between(df['Time'], 
                 df['Kalman_State'] - sigma_bounds, 
                 df['Kalman_State'] + sigma_bounds, 
                 color="#aec7e8", alpha=0.5, label="State Variance Bounds ($\pm 3\sigma$)")

plt.title("Scenario A: Hardware Degradation Tracking via Low-Variance Subspace")
plt.xlabel("Time (seconds)")
plt.ylabel("Euclidean Drift")
plt.legend(loc="upper left")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("Scenario_A_Health.png", dpi=300)

# =====================================================================
# SCENARIO B: Adaptive Protocol Transmission Under Jamming
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
ax2.set_ylabel('Active Transmission Tier', color=color, labelpad=15)  # Added padding here
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
# SCENARIO C: Physics-Constrained Cyber-Attack Detection (With Inset Zoom)
# =====================================================================
fig, ax = plt.subplots(figsize=(10, 5))

# 1. Primary Canvas (Macro View: Captures Attack Magnitude)
ax.plot(df['Time'], df['Innovation'], label=r"Innovation Sequence ($\tilde{y}_t$)", color="purple", linewidth=1.5)
ax.plot(df['Time'], sigma_bounds, color="red", linestyle="--", alpha=0.7, label=r"Dynamic Threshold ($\pm\tau_t$)")
ax.plot(df['Time'], -sigma_bounds, color="red", linestyle="--", alpha=0.7)

# Highlight alarms with red X markers
alarms = df[df['Threshold_Alarm'] == 1]
if not alarms.empty:
    ax.scatter(alarms['Time'], alarms['Innovation'], color="red", s=60, marker="x", zorder=5, label="Spoofing Alarm Triggered")

ax.set_title("Scenario C: Anomaly Detection via Physics-Constrained Innovation")
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("State Prediction Error")
ax.legend(loc="upper right", framealpha=1.0)
ax.grid(True, linestyle="--", alpha=0.6)

# 2. Embedded Canvas (Micro View: Magnified Steady State Boundaries)
# This zooms in on the first 40 seconds to show the normal bounds inflating during the first jammer encounter
ax_inset = inset_axes(ax, width="35%", height="35%", loc="lower left", borderpad=4)
ax_inset.plot(df['Time'], df['Innovation'], color="purple", linewidth=1.2)
ax_inset.plot(df['Time'], sigma_bounds, color="red", linestyle="--", alpha=0.8)
ax_inset.plot(df['Time'], -sigma_bounds, color="red", linestyle="--", alpha=0.8)

# Zoom limits: Look at t=0 to t=45 seconds, focusing closely on the Y-axis near zero
ax_inset.set_xlim(0, 45)
ax_inset.set_ylim(-5, 5) 
ax_inset.grid(True, linestyle=':', alpha=0.6)
ax_inset.set_title("Close-up: Dynamic Bounds Inflating", fontsize=9, style='italic')
ax_inset.tick_params(axis='both', which='major', labelsize=8)

# Draw geometric connectors highlighting where the zoom window comes from
mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5", linestyle=":")

plt.tight_layout()
plt.savefig("Scenario_C_Injection.png", dpi=300)

print("✅ Publication-ready plots generated successfully!")