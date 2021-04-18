#! /usr/bin/env python3
import serial
import modules.vosk_rec as vosk_rec

def rec_data():
    decoder = vosk_rec.Decoder()
    try:
        port = serial.Serial("/dev/ttyACM0", baudrate=230400, timeout=3.0)
    except Exception as exp:
        print(type(exp))
        print("serial port wasn't found")
    start = "are ya ready kids?"
    confirm = "ay ay captain"
    stop = "done!!!"

    while True:
        rcv = port.read(1024)
        start = rcv.find(msg)
        print(start)
        if(start != -1): #start message was detected
            print("msg was found from Justin!!!")
            ser.write(confirm)
            rcv = rcv[start+len(start):] #remove message
            decoder.decode_stream(port, rcv)

def main():
    rec_data()

if __name__ == "__main__":
    main()
