import socket
import time

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

while True:
    sock.sendto(b"Hello", ("127.0.0.1", 9000))
    print("[EDGE] sent Hello")
    time.sleep(1)