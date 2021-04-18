#!/usr/bin/env python3

import socket
import wave
from os import system

def conn_stuff():
    HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
    PORT = 5555        # Port to listen on (non-privileged ports are > 1023)
    CHUNK = int(65536/2)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        
        while True:
            try:
                print(f"listening for a connection IP: {HOST} PORT: {PORT}\n")
                s.listen()
                conn, addr = s.accept()
                print(f"got a connection from {addr}\n")
                with conn:
                    msg = conn.recv(CHUNK)
                    if len(msg) > 5:
                        code = msg[:6]
                        msg = msg[6:]
                    else:
                        print(f"message is too short {msg}")
                        continue
                    print(code)
                    if b"CNRDY\0" in code:
                        print("got a CNRDY, will send YEETO")
                        conn.sendall(b"YEETO\0")
                        data = conn.recv(CHUNK)
                        while b"FLUSH\0" not in data:
                            data = conn.recv(CHUNK)
                        print("received flush!!!")
                        send_stuff(conn)
                    elif b"CNERR\0" in code:
                        print("got a CNERR, need to clear out buffer")
                    elif b"GDATA\0" in code:
                        print("got a GDATA, the back-end got good data!")
                    elif b"APCKT\0" in code:
                        print("got a APCKT, assuming the rest of the data is audio")
                        receive_stuff(conn, msg, len(msg))
                        conn.sendall(b"ADONE");
                    else:
                        print("code didn't match anything")

                    conn.close()

            except KeyboardInterrupt:
                print("keyboard stop. Closing connection and file")
                break
            except Exception as e:
                print("something went wrong")
                print(e)
                break
        s.close()

def send_stuff(conn):
    files = [
    "offLight.wav",
    "onLight.wav",
    "plugOff.wav",
    "plugOn.wav",
    "silence.wav",
    "playSong.wav",
    ]
    CHUNK = int(65536/2)


    for i in range(len(files)):
        print("{}. {}".format(i, files[i]))
    name = int(input("enter file number to send : "))
    f = open("./voice_files/"+files[name], "rb")

    print("FILE: {}".format(files[name]))
    header = f.read(44)
    size = 1
    while size > 0:
        read = f.read(CHUNK)
        size = len(read)
        print(size)
        try:
            conn.sendall(read)
        except BrokenPipeError:
            print("connection to front end died")
            break
    f.close()

def receive_stuff(conn, recv, size):
    CHUNK = int(65536/2)
    temp = wave.open("temp.wav", 'wb')
    temp.setnchannels(1) #mono
    temp.setsampwidth(2)
    temp.setframerate(16000)
    
    while size > 0:
        try:
            print(f"for data : {size}")
            temp.writeframesraw(recv)
            recv = conn.recv(CHUNK)
            size = len(recv)

        except KeyboardInterrupt:
            break
    temp.close()
    print("going to play the audio real quick")
    system("play temp.wav")



def main():
    conn_stuff()

if __name__ == "__main__":
    main()
