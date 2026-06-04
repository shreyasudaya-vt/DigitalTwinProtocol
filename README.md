```markdown
# Adaptive Network Authentication and State Tracking Architecture (ANASTA)
### A Dual-Tier Digital Twin Co-Simulation Framework for UAV Safeguarding

---

## 1. Protocol Architecture & Operational Overview

The **ANASTA Protocol** is an adaptive, cyber-physical security framework designed to maintain Unmanned Aerial Vehicle (UAV) identity authentication and state tracking under severe channel degradation or active Electronic Counter-Measures (ECM). Operating at the intersection of a physical edge asset and its cloud-hosted Digital Twin, the architecture continuously assesses channel state parameters to balance cryptographic payload volume against tracking resolution.


```

```
              +-----------------------------------+
              |           UAV Edge Node           |
              +-----------------------------------+
                                |
      +-------------------------+-------------------------+
      | (PDR >= 0.80)                                     | (PDR < 0.80)
      v                                                   v

```

+-------------------------------+                   +-------------------------------+
|       Tier 1: Nominal         |                   |      Tier 2: Fountain Mode    |
| - Full Telemetry Data         |                   | - Erasure-Tolerant Unification|
| - Kalman Filter Tracking      |                   | - Multi-Bit Compression       |
| - Tight Variance Bounds       |                   | - Covariance (Rt) Inflation   |
+-------------------------------+                   +-------------------------------+
|                                                   |
+-------------------------+-------------------------+
v
+-----------------------------------+
|        Digital Twin Server        |
+-----------------------------------+

```

### Operational Modes

#### Tier 1: Nominal Tracking Mode ($\text{PDR} \ge 0.80$)
Under clean channel conditions, the Edge Node encapsulates high-dimensional telemetry—comprising Principal Component Analysis (PCA) hardware sweep matrices ($\mathbf{k}_{\text{health}}$)—alongside cryptographic identity tokens ($\mathbf{k}_{\text{auth}}$) masked by a sequence-dependent pseudorandom vector ($\mathbf{n}_i$). The Digital Twin executes continuous tracking via a predictive Kalman Filter. Under this tier, observation noise variance ($R_t$) remains minimal, mapping tight, high-fidelity anomaly threshold boundaries ($\pm 3\sigma$) around the state estimation vector to rapidly detect subtle spoofing vectors or kinematic deviations.

#### Tier 2: Degraded/Fountain Tracking Mode ($\text{PDR} < 0.80$)
Upon encountering localized jamming or path loss, the measured Packet Delivery Ratio (PDR) falls below the critical threshold ($0.80$). The protocol dynamically transitions into a compressed, erasure-resilient state:
1. **Dynamic Covariance Inflation:** The Digital Twin monotonically increases the observation covariance matrix parameter ($R_t$). This step structurally relaxes the Kalman filter tracking bounds, preventing false anomaly classification triggered by sparse, non-contiguous updates.
2. **Fountain-Compressed Authentication:** The Edge Node discards the heavy floating-point physical telemetry payload ($\mathbf{k}_{\text{health}}$). The 128-bit core identity bitmask ($\mathbf{k}_{\text{auth}}$) is compressed down to a compact representation using a random-sampling fountain matrix governed by seed $\mathbf{s}_i$. The resulting compressed byte array minimizes the transmission footprint (e.g., 18 bytes), maximizing the probability of reception over heavily degraded physical-layer channels.

---

## 2. System Prerequisites & Environment Setup

The co-simulation requires a split C++11 (Network Simulation) and Python 3 (Analytical Processing and Digital Twin Engine) execution environment.

### 2.1 Native System Requirements
* **Operating System:** Linux (Ubuntu 20.04 LTS / 22.04 LTS recommended)
* **Compiler:** GCC g++ v9.3.0 or higher
* **Build Systems:** CMake v3.16+, Make v4.3+

### 2.2 Python Dependency Matrix
Install the necessary analytical, mathematical, and cryptographic emulation modules via `pip`:

```bash
pip install numpy pandas matplotlib

```

| Library | Version Requirement | Purpose |
| --- | --- | --- |
| **NumPy** | $\ge$ 1.20.0 | High-performance matrix manipulation, PCA transformations, vector serialization. |
| **Pandas** | $\ge$ 1.3.0 | Structural parsing of telemetry `.csv` logs and experimental time-series evaluation. |
| **Matplotlib** | $\ge$ 3.4.0 | Generation of empirical performance plots, variance bounds, and PDR summary distributions. |

### 2.3 Network Simulator 3 (ns-3) Environment

The RF-layer physical jamming dynamics are simulated inside the discrete-event network simulator **ns-3** (tested on versions `ns-3.35` through `ns-3.40`). Ensure your ns-3 installation has the `network`, `internet`, and `mobility` modules built properly.

---

## 3. Directory Layout

Ensure your project environment follows this directory architecture for automated relative-path processing:

```
anasta-framework/
├── ns-3-dev/scratch/
│   └── anasta_simulation.cc       # ns-3 C++ Discrete-Event RF Jamming Loop
├── src/
│   ├── digital_twin.py           # Twin Server & Kalman Filter State Machine
│   ├── edge_node.py              # UAV Client, PCA Encoder, & Fountain Engine
│   └── final_pca.py              # Auxiliary Mathematical Modeling Utilities
├── tests/
│   └── experiment_harness.py     # Automated Parametric Sweep & Verification Harness
└── python_reports/               # Target Directory for Exported Figures & Data

```

---

## 4. Execution Directives

### 4.1 Running the In-Process Parametric Experiment Suite

To execute a deterministic PDR evaluation sweep ($0.2 \to 1.0$) and generate statistical proof of protocol resilience under variance inflation, execute the experimental harness script:

```bash
python3 tests/experiment_harness.py

```

* **Expected Output:** The script performs a multi-round evaluation, processing sequence-driven packet streams against dynamic PDR drops. Upon termination, it updates `telemetry_log.csv` and outputs publication-grade visualization graphs to `python_reports/experiment_summary.png`.

### 4.2 Running the Full Network-Coupled Co-Simulation

To execute the live socket-bound co-simulation where python instances interface natively with the ns-3 discrete-event network engine over local loopback interfaces:

#### Step 1: Initialize the Digital Twin Server Core

Launch the processing engine to bind to incoming network packets:

```bash
python3 src/digital_twin.py

```

*The server will initialize internal Kalman filter coefficients and open a UDP listening interface on port `5000` while mapping the telemetry feedback control loop on port `9001`.*

#### Step 2: Initialize the UAV Edge Node Interface

In a separate shell terminal, establish the client state controller:

```bash
python3 src/edge_node.py

```

*The client initializes the PCA reference sweep hardware maps and waits for packet-dispatch directives.*

#### Step 3: Execute the ns-3 Network Simulation Architecture

Move your `anasta_simulation.cc` script into the standard ns-3 workspace `scratch/` directory, compile the binary object, and execute it:

```bash
cd /path/to/ns3-directory/
./ns3 build
./ns3 run scratch/anasta_simulation

```

### 4.3 Expected Behavior Verification

* **Spatial Crossings:** As the simulated ns-3 drone moves inside the configured 30-meter radial proximity of the RF jammer coordinate, the terminal window will show real-time inverse-square drop calculations.
* **Dynamic Convergence:** In the `digital_twin.py` logging console, look for the channel state updates. As packets drop, you will observe the structural transition log:
```text
[Twin Engine] Seq: 42 | Tier: 1 | Channel PDR: 0.95 | Bounds: Normal
... [RF Interference Event Triggered] ...
[Twin Engine] Seq: 55 | Tier: 2 | Channel PDR: 0.62 | Bounds: INFLATED (Rt=4.10)

```


* **Graceful Resumption:** As the mobility model extracts the drone past the interference coordinates, the sliding-window queue will eject the lost packets, returning the PDR metric to $>0.80$ and stabilizing tracking back to Tier 1 baseline bounds.

```

```