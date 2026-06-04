import socket, threading, time

# Listener: receives what ns-3 forwards back to the host
def listener():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("10.0.0.1", 5000))          # host tap0 IP, ns-3 forwards here
    print("Listening on 10.0.0.1:5000 ...")
    while True:
        data, addr = s.recvfrom(4096)
        print(f"[host] received from ns-3: {data.decode()!r}")

threading.Thread(target=listener, daemon=True).start()
time.sleep(0.5)   # let listener bind first

# Sender: packets go into tap0 → ns-3 node (10.0.0.2) port 9000
sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
for i in range(5):
    msg = f"Hello ns-3, packet {i}"
    sender.sendto(msg.encode(), ("10.0.0.2", 9000))  # ns-3 node's IP
    print(f"[host] sent: {msg!r}")
    time.sleep(1)

input("Press Enter to quit...")