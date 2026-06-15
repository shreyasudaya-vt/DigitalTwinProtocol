#!/bin/bash

# Suppress background job termination logs
set +m

# ==========================================================
# COLOR CONFIGURATION
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

if [ ! -d "$NS3_WORKDIR" ]; then
    echo -e "${RED}❌ Error: Path to ns-3 workspace does not exist at: $NS3_WORKDIR${NC}"
    exit 1
fi

cleanup() {
    echo ""
    echo -e "${RED}🛑 Script interrupted manually. Force killing background processes...${NC}"
    pkill -9 -f "digital_twin.py" 2>/dev/null
    pkill -9 -f "edge_node.py" 2>/dev/null
    exit 1
}
trap cleanup SIGINT

echo -e "${BLUE}🧹 Executing environment pre-clean...${NC}"
pkill -9 -f "digital_twin.py" 2>/dev/null
pkill -9 -f "edge_node.py" 2>/dev/null
fuser -k 9000/udp 2>/dev/null
fuser -k 5000/udp 2>/dev/null
sleep 1

SCENARIOS=("Scenario_A" "Scenario_B" "Scenario_C" "Scenario_D")

for SCENARIO in "${SCENARIOS[@]}"
do
    echo -e "${BLUE}==========================================================${NC}"
    echo -e "${GREEN}🚀 Starting Automated Evaluation Loop for: $SCENARIO${NC}"
    echo -e "${BLUE}==========================================================${NC}"

    cd "$PYTHON_WORKDIR" || exit 1
    rm -f "telemetry_${SCENARIO}.csv"

    echo "🖥️  Spinning up Digital Twin Server..."
    python3 -u digital_twin.py "$SCENARIO" &
    TWIN_PID=$!
    disown $TWIN_PID # Detach from shell to silence "Killed" messages

    echo "📡 Initializing Edge Node telemetry engine..."
    python3 -u edge_node.py "$SCENARIO" &
    EDGE_PID=$!
    disown $EDGE_PID # Detach from shell to silence "Killed" messages

    sleep 2
    
    if ! kill -0 $TWIN_PID 2>/dev/null || ! kill -0 $EDGE_PID 2>/dev/null; then
        echo -e "${RED}❌ Error: Python telemetry engines crashed during startup!${NC}"
        kill -9 $TWIN_PID $EDGE_PID 2>/dev/null
        continue
    fi
    
    cp anasta_sim.cc "$NS3_WORKDIR/scratch/anasta_sim.cc"
    
    echo "🌐 Navigating to ns-3 workspace and launching simulation..."
    cd "$NS3_WORKDIR" || exit 1
    
    ./ns3 run scratch/anasta_sim -- --scenario=$SCENARIO
    NS3_STATUS=$?

    cd "$PYTHON_WORKDIR" || exit 1

    # Safe guard: If NS-3 failed to build or crashed, alert the user.
    if [ $NS3_STATUS -ne 0 ]; then
        echo -e "${RED}⚠️  NS-3 returned a non-zero exit code! Simulation may have failed.${NC}"
    fi

    echo "🛑 Stopping background Python applications..."
    kill -SIGINT $TWIN_PID $EDGE_PID 2>/dev/null
    sleep 3
    kill -9 $TWIN_PID $EDGE_PID 2>/dev/null

    EXPECTED_CSV="telemetry_${SCENARIO}.csv"
    
    if [ -f "$EXPECTED_CSV" ] && [ -s "$EXPECTED_CSV" ]; then
        echo -e "${GREEN}💾 Telemetry data generated and verified for $SCENARIO -> ($EXPECTED_CSV)${NC}"
    else
        echo -e "${RED}⚠️  Error: Expected output file '$EXPECTED_CSV' is missing or completely EMPTY!${NC}"
    fi
    
    echo ""
done

# Run the plotting script ONCE at the end of the loop
echo -e "${BLUE}==========================================================${NC}"
echo -e "${YELLOW}📊 Generating Final Journal Figures and Metrics...${NC}"
echo -e "${BLUE}==========================================================${NC}"
python3 plot.py

set -m

echo ""
echo -e "${GREEN}🎉 All Core Scenarios Executed and Plotted Successfully!${NC}"
