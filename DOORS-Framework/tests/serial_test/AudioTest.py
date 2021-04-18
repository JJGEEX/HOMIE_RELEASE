import serial
import numpy as np
import matplotlib.pyplot as plt
import wavio
from scipy.fftpack import fft
from scipy.io import  wavfile as wav
import time

# ser = serial.Serial(
#     port='COM5',\
#     baudrate=9600,\
#     parity=serial.PARITY_NONE,\
#     stopbits=serial.STOPBITS_ONE,\
#     bytesize=serial.EIGHTBITS,\
#         timeout=0)

# Protocol is  Homie Front End: "\n\rAre Ya Ready Kids?\n\r"
#  Response "Aye Aye Captain!" -> if Failed it will say "I cant hear you"

port = serial.Serial("com4", baudrate = 921600, timeout = 10.0)
# port = serial.Serial("com4", baudrate = 460800, timeout = 150.0)
print("Press Push Button When Ready")
while port.inWaiting() < 17:

    pass

message = port.read(17).decode(errors='ignore')
print(message)
if message == "are ya ready kids":
    my_message = "aye aye captain".encode('ascii', errors='ignore')
    port.write(my_message)
    print("aye aye captain")
    rcv = port.read(80000)
    my_message = "i cant hear you".encode('ascii', errors='ignore')
    port.write(my_message)
    print("i cant hear you")



    # print(rcv)
    my_audio = np.frombuffer(rcv, np.int16)
    time.sleep(1)
    my_message = "ooohhh".encode('ascii', errors='ignore')
    port.write(my_message)
    print("ooohhh")

    # print(my_audio)
    # print(len(my_audio))
    # sample_rate = 16000
    sample_rate = 8000
    wavio.write("audio.wav", my_audio, sample_rate)
