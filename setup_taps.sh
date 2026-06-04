#!/bin/bash

# Ensure the script is run with sudo privileges
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root. Try: sudo ./setup_tap.sh" 
   exit 1
fi

echo "--- Setting up WSL tap0 interface for ns-3 ---"

# 1. Check for ethtool and install it if missing
if ! command -v ethtool &> /dev/null; then
    echo "[*] ethtool not found. Installing..."
    apt-get update -qq && apt-get install -y ethtool -qq
fi

# 2. Delete the interface if it already exists (allows you to run this script repeatedly)
if ip link show tap0 &> /dev/null; then
    echo "[*] Existing tap0 interface found. Recreating..."
    ip link delete tap0
fi

# 3. Create the virtual TAP interface
echo "[*] Creating tap0 interface..."
ip tuntap add mode tap tap0

# 4. Assign the Host OS IP address (10.0.0.1)
echo "[*] Assigning IP 10.0.0.1/24 to tap0..."
ip addr add 10.0.0.1/24 dev tap0

# 5. Bring the interface online
echo "[*] Bringing tap0 up..."
ip link set dev tap0 up

# 6. Disable TX checksum offloading (The WSL quirk fix)
echo "[*] Disabling TX checksum offloading via ethtool..."
ethtool -K tap0 tx off

echo "--- Setup Complete! ---"
echo "You can verify it by running: ip addr show tap0"
