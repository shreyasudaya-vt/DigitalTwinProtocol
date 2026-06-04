#!/bin/bash
sudo ip tuntap add dev tap0 mode tap 2>/dev/null || echo "tap0 already exists"
sudo ip addr add 10.0.0.1/24 dev tap0 2>/dev/null || echo "address already set"
sudo ip link set tap0 up

# Ensure route exists explicitly (WSL2 doesn't always auto-add it)
sudo ip route add 10.0.0.0/24 dev tap0 2>/dev/null || echo "route already set"

# Add static ARP for ns-3 node so host doesn't drop packets waiting for ARP reply
TAP_MAC=$(cat /sys/class/net/tap0/address)
sudo arp -s 10.0.0.2 $TAP_MAC

echo "=== tap0 status ==="
ip addr show tap0
echo "=== route ==="
ip route show | grep 10.0.0
echo "=== arp ==="
arp -n | grep 10.0.0
