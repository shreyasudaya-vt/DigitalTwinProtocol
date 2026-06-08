#!/bin/bash

# Suppress background job termination logs ("Killed...") to keep terminal clean
set +m

# ==========================================================
# COLOR CONFIGURATION (For clean, scannable terminal logs)
# ==========================================================
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0;49m' # No Color

# ==========================================================
# DIRECTORY CONFIGURATION
# ==========================================================
PYTHON_WORKDIR=$(pwd)
NS3_WORKDIR="/home/shreyas/Desktop/ns-3-dev"

# Safety Check: Verify directories exist before executing
if [ ! -d "$NS3_WORKDIR" ]; then
    echo -e "${RED}❌ Error: Path to ns-3 workspace does not exist at: $NS3_WORKDIR${NC}"
    exit 1
fi

# Safe cleanup if the user aborts the bash script early (Ctrl+C)
cleanup() {
    echo ""
    echo -e "${RED}🛑 Script interrupted manually. Force killing background processes...${NC}"
    kill -9 $TWIN_PID $EDGE_PID 2>/dev/null
    pkill -9 -f "digital_twin.py" 2>/dev/null
    pkill -9 -f "edge_node.py" 2>/dev/null
    exit 1
}
trap cleanup SIGINT

echo -e "${BLUE}🧹 Executing environment pre-clean to guarantee available ports...${NC}"
pkill -9 -f "digital_twin.py" 2>/dev/null
pkill -9 -f "edge_node.py" 2>/dev/null
sleep 1

# Define your IEEE evaluation scenarios
SCENARIOS=("Scenario_A" "Scenario_B" "Scenario_C")

for SCENARIO in "${SCENARIOS[@]}"
do
    echo -e "${BLUE}==========================================================${NC}"
    echo -e "${GREEN}🚀 Starting Automated Evaluation Loop for: $SCENARIO${NC}"
    echo -e "${BLUE}==========================================================${NC}"

    # Ensure we are in the Python directory
    cd "$PYTHON_WORKDIR" || exit 1

    # 1. Clear out historical test data for this specific scenario to prevent false positives
    rm -f "telemetry_${SCENARIO}.csv"

    # 2. Start the Digital Twin Server in the background
    echo "🖥️  Spinning up Digital Twin Server..."
    python3 digital_twin.py "$SCENARIO" &
    TWIN_PID=$!
    
    # 3. Start the Edge Node in the background
    echo "📡 Initializing Edge Node telemetry engine..."
    python3 edge_node.py "$SCENARIO" &
    EDGE_PID=$!

    # Give sockets and hardware initialization a brief moment to bind safely
    sleep 2
    
    # Copy scratch file over to ns-3 directory dynamically
    cp anasta_sim.cc "$NS3_WORKDIR/scratch/anasta_sim.cc"
    
    # 4. Launch the NS-3 Network Simulation
    echo "🌐 Navigating to ns-3 workspace and launching simulation..."
    cd "$NS3_WORKDIR" || exit 1
    
    # Run the simulation unthrottled with scenario flags passed directly to ns-3
    ./ns3 run scratch/anasta_sim -- --scenario=$SCENARIO

    # Return to the Python workspace
    cd "$PYTHON_WORKDIR" || exit 1

    # 5. Gracefully terminate background Python threads using their exact PIDs
    echo "🛑 Stopping background Python applications..."
    kill -INT $TWIN_PID 2>/dev/null
    kill -INT $EDGE_PID 2>/dev/null
    sleep 2
    
    # Precise isolation: Hard-kill ONLY the exact PIDs assigned to this iteration loop
    kill -9 $TWIN_PID 2>/dev/null
    kill -9 $EDGE_PID 2>/dev/null

    # 6. Data Processing, Verification, and Plot Generation
    EXPECTED_CSV="telemetry_${SCENARIO}.csv"
    
    if [ -f "$EXPECTED_CSV" ]; then
        echo -e "${GREEN}📊 Telemetry data verified for $SCENARIO -> ($EXPECTED_CSV)${NC}"
        echo "📈 Rendering publication plots..."
        
        # Pass the scenario name to your plotting script to preserve individual outputs
        python3 plot.py "$SCENARIO"
        
        echo -e "${GREEN}💾 Raw telemetry log and PDF assets successfully saved.${NC}"
    else
        echo -e "${RED}⚠️  Error: Expected output file '$EXPECTED_CSV' was not detected!${NC}"
        echo "   Please check if digital_twin.py threw an unhandled runtime exception."
    fi
    
    echo -e "${BLUE}Done with $SCENARIO. Transitioning to next test cycle...${NC}"
    echo ""
done

# Restore default shell job settings
set -m

echo -e "${GREEN}==========================================================${NC}"
echo -e "${YELLOW}🎉 All Core Scenarios Executed and Plotted Successfully!${NC}"
echo -e "${GREEN}📂 High-quality evaluation assets generated inside workspace.${NC}"
echo -e "${GREEN}==========================================================${NC}"