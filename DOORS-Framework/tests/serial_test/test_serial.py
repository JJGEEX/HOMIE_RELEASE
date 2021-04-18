#!/usr/bin/env python3
import serial
import numpy as np
import wavio
import time
import vosk_rec as vr
import copy

# ser = serial.Serial(
#     port='COM5',\
#     baudrate=9600,\
#     parity=serial.PARITY_NONE,\
#     stopbits=serial.STOPBITS_ONE,\
#     bytesize=serial.EIGHTBITS,\
#         timeout=0)

# Protocol is  Homie Front End: "\n\rAre Ya Ready Kids?\n\r"
#  Response "Aye Aye Captain!" -> if Failed it will say "I cant hear you"

SAMPLE_RATE = 8000
RECEIVE_FILE = "audio.wav"
COMPLETE_FILE = "complete.wav"

port = serial.Serial("/dev/ttyACM0", baudrate = 921600, timeout = 10.0)
decoder = vr.Decoder()
allData = b''

restart = "r"
while(restart == "r"):
    print("Press Push Button When Ready")
    while port.inWaiting() < 17:
        pass

    message = port.read(17).decode(errors='ignore')
    print(message)
    if message == "are ya ready kids": #front-end ready
        print("received message, sending confirmation")
        my_message = "aye aye captain".encode('ascii', errors='ignore')
        port.write(my_message) #back-end ready
        print(my_message)
        while True: #voice recognition
            rcv = port.read(80000)
            if(len(allData) == 0):
                print("making deep copy")
                allData = copy.deepcopy(rcv)
            my_audio = np.frombuffer(rcv, np.int16)
            wavio.write(RECEIVE_FILE, my_audio, SAMPLE_RATE) #temporary file
            result = decoder.decode_file(RECEIVE_FILE)
            if(len(result) == 0): #nothing was detected, stopping
                my_audio = np.frombuffer(allData, np.int16)
                wavio.write(COMPLETE_FILE, my_audio, SAMPLE_RATE)
                result = decoder.decode_file(COMPLETE_FILE)
                break
            else: #something was detected, adding it to a buffer
                if(len(allData) == 0):
                    print("allData empty")
                    exit()
                if(len(rcv) == 0):
                    print("rcv empty")
                    exit()
                allData = np.concatenate([allData, rcv])
    write = input("enter w to write messages")
    if(write == "w"):
        my_message = "i cant hear you".encode('ascii', errors='ignore')
        port.write(my_message) #front-end stop listening
        print(my_message)
        time.sleep(1)
        my_message = "ooohhh".encode('ascii', errors='ignore')
        port.write(my_message)
        print("ooohhh")
    print("DONE")
    restart = input("enter r to restart")
