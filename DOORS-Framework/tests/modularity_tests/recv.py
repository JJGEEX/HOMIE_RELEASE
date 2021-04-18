#!/usr/bin/env python3
import socket

def main():
    HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
    PORT = 5555        # Port to listen on (non-privileged ports are > 1023)
    f = open("temp.wav", "wb")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                f.write(data)
    f.close()

if __name__ == "__main__":
    main()
