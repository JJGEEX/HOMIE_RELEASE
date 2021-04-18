#! /usr/bin/env python3
import serial
import vosk_rec

def rec_data():
    decoder = vosk_rec.Decoder()
    try:
        port = serial.Serial("/dev/ttyACM0", baudrate=921600, timeout=10.0)
    except Exception as exp:
        print(type(exp))
        print("serial port wasn't found")
    start = "are ya ready kids"
    confirm = "aye aye captain"
    stop = "done!!!"

    while True:
        print("waiting for input")
        rcv = port.read(1024).decode(errors="ignore")
        print(rcv)
        if(start in rcv): #start message was detected
            print("msg was found from Justin!!!")
            port.write(confirm.encode("utf-8"))
            rcv = rcv[start+len(start):] #remove message
            decoder.decode_stream(port, rcv)

def main():
    rec_data()

if __name__ == "__main__":
    main()
