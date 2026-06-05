#!/bin/bash

# Safe cleanup if the user aborts the bash script early (Ctrl+C)
cleanup() {
    echo "🛑 Script interrupted manually. Cleaning up background processes..."
    kill -INT $TWIN_PID $EDGE_PID 2>/dev/null
    exit 1
}
trap cleanup SIGINT

# Define scenarios to evaluate
SCENARIOS=("Scenario_A" "Scenario_B" "Scenario_C")

for SCENARIO in "${SCENARIOS[@]}"
do
    echo "=========================================================="
    echo "🚀 Starting Automated Evaluation for: $SCENARIO"
    echo "=========================================================="

    # 1. Purge any stale logs from aborted runs to prevent data contamination
    rm -f thesis_telemetry_log.csv

    # 2. Start the Digital Twin Server in the background
    python3 digital_twin.py &
    TWIN_PID=$!
    
    # 3. Start the Edge Node in the background
    python3 edge_node.py &
    EDGE_PID=$!

    # Give sockets and hardware initialization a brief moment to bind
    sleep 2

    # 4. Launch the NS-3 Network Simulation
    # This acts as a blocking call and runs for exactly 300 seconds
    echo "🌐 Launching ns-3 Network Simulation Model..."
    ./ns3 run "scratch/anasta_sim --scenario=$SCENARIO"

    # 5. Gracefully terminate background Python threads
    echo "🛑 Stopping background Python applications..."
    # Sending SIGINT (-INT) forces digital_twin.py into its KeyboardInterrupt block,
    # cleanly executing self.log_file.close() to flush all remaining buffers.
    kill -INT $TWIN_PID 2>/dev/null
    kill -INT $EDGE_PID 2>/dev/null
    
    # Allow the OS filesystem buffer time to settle
    sleep 3

    # 6. Data Processing & Isolation
    if [ -f "thesis_telemetry_log.csv" ]; then
        echo "📊 Instantly rendering publication plots for $SCENARIO..."
        # Runs plot.py while the active log matches the completed scenario
        python3 plot.py
        
        # Rename the log file to preserve raw telemetry data for historical reference
        mv thesis_telemetry_log.csv "telemetry_${SCENARIO}.csv"
        echo "💾 Raw telemetry log successfully saved as: telemetry_${SCENARIO}.csv"
    else
        echo "⚠️ Error: thesis_telemetry_log.csv was not detected for $SCENARIO!"
    fi
    
    echo "=========================================================="
    echo "Done with $SCENARIO. Transitioning to next test cycle..."
    echo ""
done

echo "=========================================================="
echo "✅ All Core Scenarios Executed and Plotted Successfully!"
echo "📂 Check your workspace for updated assets:"
echo "   - Scenario_A_Health.png (Now showing dynamic Kalman tracking!)"
echo "   - Scenario_B_Jamming.png"
echo "   - Scenario_C_Injection.png"
echo "=========================================================="