import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", 5000))

print("[TWIN] listening...")

while True:
    data, _ = sock.recvfrom(1024)
    print("[TWIN] received:", data.decode())