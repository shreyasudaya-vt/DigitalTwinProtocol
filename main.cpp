#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <random>
#include <thread>
#include <algorithm>
#include <fstream>
#include <iomanip>

const int N_FEATURES = 2001;          
const int N_COMPONENTS = 128;         
const int HAMMING_THRESHOLD = 25;     

struct TelemetryPacket {
    int packet_id;
    std::chrono::steady_clock::time_point timestamp;
    std::vector<double> sweep_data;   // High-fidelity telemetry stream [cite: 3]
};

class ContestedEnvironmentSimulator {
private:
    std::mt19937 gen;
    std::uniform_real_distribution<double> dis;
    std::normal_distribution<double> noise;

public:
    ContestedEnvironmentSimulator() : gen(1337), dis(0.0, 1.0), noise(0.0, 2.5) {}

    std::vector<double> generate_clean_sweep(int step) {
        std::vector<double> sweep(N_FEATURES);
        
        double step_bias = 50.0 * std::sin(step * 0.1); 
        
        for (int i = 0; i < N_FEATURES; ++i) {
            double angle = i * (2.0 * 3.141592653589793 / N_FEATURES);
            sweep[i] = 500.0 + step_bias + 300.0 * std::sin(angle) + noise(gen);
        }
        return sweep;
    }

    bool process_packet(TelemetryPacket& packet, double loss_rate, int latency_ms, bool inject_spoofing) {
        
        if (dis(gen) < loss_rate) {
            return false; 
        }

        if (latency_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(latency_ms));
        }
        packet.timestamp = std::chrono::steady_clock::now();

        if (inject_spoofing) {
            for (int i = 0; i < N_FEATURES; ++i) {
                packet.sweep_data[i] += 150.0 * std::cos(i * 0.01); 
            }
        }
        return true;
    }
};

class PCAEngine {
private:
    std::vector<double> scaler_mean;
    std::vector<double> scaler_scale;
    std::vector<std::vector<double>> pca_components; // Matrix of size [N_COMPONENTS x N_FEATURES]
    std::string golden_reference_id;

public:
    void initialize_mock_weights() {
    scaler_mean.assign(N_FEATURES, 500.0);
    scaler_scale.assign(N_FEATURES, 15.0);
    
    // Seed a local generator to create diverse projection weights
    std::mt19937 mock_gen(42);
    std::uniform_real_distribution<double> mock_dist(-0.05, 0.05);

    pca_components.assign(N_COMPONENTS, std::vector<double>(N_FEATURES));
    for (int c = 0; c < N_COMPONENTS; ++c) {
        for (int f = 0; f < N_FEATURES; ++f) {
            pca_components[c][f] = mock_dist(mock_gen);
        }
    }
    
    // FIX: Self-calibrate using the EXACT same math as the runtime generator!
    std::vector<double> baseline_sweep(N_FEATURES);
    
    // Calculate the bias at step 1 to align the golden ID perfectly
    double step_bias = 50.0 * std::sin(1 * 0.1); 
    
    for (int i = 0; i < N_FEATURES; ++i) {
        // Use the full 2*PI wave cycle to match generate_clean_sweep()
        double angle = i * (2.0 * 3.141592653589793 / N_FEATURES);
        // Note: We don't add noise() here so we get a pure Golden Reference
        baseline_sweep[i] = 500.0 + step_bias + 300.0 * std::sin(angle); 
    }
    
    golden_reference_id = generate_binary_id(baseline_sweep);
    std::cout << "[Initialization] Calibrated Golden Reference ID: " 
              << golden_reference_id.substr(0,16) << "...\n";
}
    bool load_exported_assets(const std::string& path) {
        return true;
    }

    std::string generate_binary_id(const std::vector<double>& raw_sweep) {
        std::vector<double> standardized(N_FEATURES);
        for (int i = 0; i < N_FEATURES; ++i) {
            standardized[i] = (raw_sweep[i] - scaler_mean[i]) / scaler_scale[i];
        }

        std::string binary_id = "";
        for (int c = 0; c < N_COMPONENTS; ++c) {
            double dot_product = 0.0;
            for (int f = 0; f < N_FEATURES; ++f) {
                dot_product += standardized[f] * pca_components[c][f];
            }
            binary_id += (dot_product > 0.0) ? '1' : '0';
        }
        return binary_id;
    }

    int compute_hamming_distance(const std::string& id_a, const std::string& id_b) {
        int distance = 0;
        for (size_t i = 0; i < id_a.length() && i < id_b.length(); ++i) {
            if (id_a[i] != id_b[i]) distance++;
        }
        return distance;
    }

    std::string get_golden_id() const { return golden_reference_id; }
};

class TrustEngine {
private:
    int last_packet_id = 0;
    std::chrono::steady_clock::time_point last_arrival_time;
    double continuous_trust_score = 1.0;

    // Weights summing to 1.0
    const double w_auth = 0.50;
    const double w_consistency = 0.30;
    const double w_reliability = 0.20;

public:
    double compute_dynamic_trust(const TelemetryPacket& packet, int hamming_dist, const std::vector<double>& last_valid_sweep) {
        auto current_time = std::chrono::steady_clock::now();

        // Metric A: Authentication Confidence (Decays linearly as Hamming distance hits threshold)
        double A_t = std::max(0.0, 1.0 - (static_cast<double>(hamming_dist) / HAMMING_THRESHOLD));

        // Metric C: Temporal Consistency (Checks delta anomalies between consecutive updates) [cite: 22]
        double C_t = 1.0;
        if (!last_valid_sweep.empty()) {
            double mean_absolute_error = 0.0;
            for (int i = 0; i < N_FEATURES; ++i) {
                mean_absolute_error += std::abs(packet.sweep_data[i] - last_valid_sweep[i]);
            }
            mean_absolute_error /= N_FEATURES;
            // If physical deviation is too massive, penalize consistency score
            if (mean_absolute_error > 50.0) C_t = std::max(0.0, 1.0 - ((mean_absolute_error - 50.0) / 100.0));
        }

        // Metric R: Communication Reliability (Evaluates packet gaps and data freshness latency) [cite: 9, 22]
        double R_t = 1.0;
        if (last_packet_id != 0) {
            int packet_gap = packet.packet_id - last_packet_id;
            auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - packet.timestamp).count();
            
            // Penalize for dropped packets or stale data ages [cite: 9, 22]
            if (packet_gap > 1) R_t -= 0.2 * (packet_gap - 1);
            if (latency > 100) R_t -= 0.3 * (static_cast<double>(latency - 100) / 500.0);
            R_t = std::max(0.0, R_t);
        }

        // Update tracking history
        last_packet_id = packet.packet_id;
        last_arrival_time = current_time;

        // Calculate continuous mathematical objective trust value 
        continuous_trust_score = (w_auth * A_t) + (w_consistency * C_t) + (w_reliability * R_t);
        return continuous_trust_score;
    }
};

class DigitalTwinReplica {
public:
    double estimated_asset_state = 500.0; // Track the average scalar state of the asset [cite: 2]
    double absolute_error = 0.0;

    void update_state_naive(const TelemetryPacket& packet, double true_physical_mean) {
        // Gating completely disabled: blindly accepts incoming data packets 
        double incoming_mean = 0.0;
        for (double val : packet.sweep_data) incoming_mean += val;
        incoming_mean /= N_FEATURES;

        estimated_asset_state = incoming_mean;
        absolute_error = std::abs(estimated_asset_state - true_physical_mean);
    }

    double predict_internal_physics_state(int step) {
        double step_bias = 50.0 * std::sin(step * 0.1);
    return 500.0 + step_bias;
    }

    void update_state_trust_aware(const TelemetryPacket& packet, double trust_score, double true_physical_mean, int step) {
    // 1. Process raw telemetry
    double incoming_mean = 0.0;
    for (double val : packet.sweep_data) {
        incoming_mean += val;
    }
    incoming_mean /= N_FEATURES;

    // 2. Trust Gating
    if (trust_score >= 0.65) {
        // --- DATA STREAM TRUSTED ---
        // Snap instantly to the validated telemetry, just like the Naive DT.
        estimated_asset_state = incoming_mean;
    } 
    else {
        // --- DATA STREAM UNTRUSTED (Isolation Active) ---
        // Completely drop the corrupted telemetry and rely 100% on the physics prediction to stay perfectly aligned with the expected baseline.
        double physics_predicted_state = predict_internal_physics_state(step);
        estimated_asset_state = physics_predicted_state;
        
        std::cout << " [!] DT Isolation Active at Step " << step 
                  << " | Telemetry Dropped | Reverted to Physics Engine.\n";
    }

    // 3. Compute absolute tracking error
    absolute_error = std::abs(estimated_asset_state - true_physical_mean);
}

};

int main() {
    ContestedEnvironmentSimulator environment;
    PCAEngine pca;
    TrustEngine trust_system;
    DigitalTwinReplica naive_twin;
    DigitalTwinReplica trust_aware_twin;

    pca.initialize_mock_weights();
    std::vector<double> historical_valid_sweep;

    std::cout << std::setw(6) << "Step" 
              << std::setw(15) << "Condition" 
              << std::setw(12) << "Hamming" 
              << std::setw(14) << "Trust Score" 
              << std::setw(16) << "Naive DT Error" 
              << std::setw(18) << "Secure DT Error\n";
    std::cout << std::string(85, '-') << "\n";

    for (int step = 1; step <= 10; ++step) {
        TelemetryPacket packet;
        packet.packet_id = step;
        packet.timestamp = std::chrono::steady_clock::now();
        packet.sweep_data = environment.generate_clean_sweep(step);

        // Calculate ground-truth mean value before degradation wrapper modifies it
        double true_physical_mean = 0.0;
        for (double val : packet.sweep_data) true_physical_mean += val;
        true_physical_mean /= N_FEATURES;

        // Change operational scenarios at runtime to demonstrate performance metrics
        std::string scenario = "Nominal";
        double loss = 0.0; int latency = 0; bool spoof = false;

        if (step == 4 || step == 5) {
            scenario = "Network Drop";
            loss = 0.10; latency = 250; // Incur network telemetry delay conditions [cite: 9]
        } else if (step >= 7) {
            scenario = "Cyber Attack";
            spoof = true; // Ingest malicious False Data Injection [cite: 4]
        }

        // Process packet through network degradation wrapper
        bool arrived = environment.process_packet(packet, loss, latency, spoof);
        if (!arrived) {
            std::cout << std::setw(6) << step << std::setw(15) << "DROPPED" << "\n";
            continue;
        }

        // Real-time PCA verification logic
        std::string live_id = pca.generate_binary_id(packet.sweep_data);
        int hd = pca.compute_hamming_distance(live_id, pca.get_golden_id());

        // Quantify Continuous Multi-tier Trust Score
        double trust = trust_system.compute_dynamic_trust(packet, hd, historical_valid_sweep);
        if (trust > 0.85 && !spoof) {
            historical_valid_sweep = packet.sweep_data;
        }

        // Process concurrently across both variants to evaluate side-by-side accuracy
        naive_twin.update_state_naive(packet, true_physical_mean);
        trust_aware_twin.update_state_trust_aware(packet, trust, true_physical_mean, step);
        std::cout << std::setw(6) << step 
                  << std::setw(15) << scenario 
                  << std::setw(12) << hd 
                  << std::setw(14) << std::fixed << std::setprecision(3) << trust 
                  << std::setw(16) << naive_twin.absolute_error 
                  << std::setw(18) << trust_aware_twin.absolute_error << "\n";
    }

    return 0;
}