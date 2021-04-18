#!/usr/bin/env python3 
import sys
sys.path.append('tools')
sys.path.append('modules')

import vosk_rec as vosk_rec
import sklearn_sims as sklearn_sims
#import modules.serial_comm as serial_comm
import voice_synth as vs
import front_info as fi
import module_loader as ml
from tests.main_tests.main_test import run_tests
import os #for recording, temporary usage
import time #for testing
from pygame import mixer
from parse import *
import socket
import time
import pathlib

def main():
    info = fi.get_fe_info()
    info["path"] = pathlib.Path().absolute()
    voice = vs.VoiceSynth(info)
    decoder = vosk_rec.Decoder(info)
    classes = ml.class_builder()
    filename = "{}/temp/downSamp.wav".format(info["path"])
    os.system("clear") #clearing out text from vosk intialization
    menu = ("enter \"reuse\" to use previous recording\n"
            "enter \"r\" to record for 10 seconds\n"
            "enter \"wifi\" to test the wifi functionality\n"
            "enter \"test\" to enter the testing menu\n"
            "enter \"exit\" to exit the program: ")

    while True:
        print()
        #record = input(menu)
        record = "wifi"
        print()
        record = record.strip().lower()
        msg = ""
        func = None
        
        if(record == "exit"):
            exit()
        elif(record == "r" or record == "wifi" or record == "reuse"):
            if record == "r": #do a local recording
                local()
                sentence = decoder.decode_file(filename)
            elif record == "wifi": #test using wifi capability
                while True:
                    sentence = decoder.listen_stream()
                    if sentence == "":
                        send_error(info)
                        continue
                    elif sentence == "stop":
                        send_stop(info)
                        continue
                    msg, func, mod = sklearn_sims.compare_command(sentence, classes, info)
                    if "no match for" in msg:
                        send_error(info)
                        continue
                    run_results(msg, func, mod, classes, voice, info)
            elif record == "reuse": #reuse previous recording
                sentence = decoder.decode_file(filename)
            else:
                print("that shouldn't have happened: "+record)
                exit()
            print("vosk sentence: "+sentence)
            run_results(msg, func, mod, classes, voice, info)
        elif(record == "test"):
            run_tests(decoder, voice, classes)
        else:
            print(record,"is not an option \n")

def run_results(msg, func, mod, classes, voice):
    print(msg)
    voice.sendToFront(msg)
    if func: #we got a func back
        if mod in classes.keys(): #classes funcs should manipulate themselves
            func(classes[mod])
        else:
            time.sleep(2)
            func()

def local(): #function for recording and testing locally
    rec_com = [ #commands for recording audio
        "echo \"recording for 10 seconds\"",
        "arecord -t wav -D \"hw:2,0\" -d 10 -f S16_LE -r 48000 {}/temp/temp.wav".format(info["path"]),
        "ffmpeg -i {}/temp/temp.wav -isr 48000 -ar 8000 {}/temp/downSamp.wav".format(info["path"],info["path"]),
        "rm {}/temp/temp.wav".format(info["path"]),
        "clear",
        "echo \"done recording\"",
        ]

    try: #removing original recording file if it exists
        os.system("rm downSamp.wav")
    except:
        print(end = "")
    for i in rec_com:
        os.system(i)

def send_error(info): #send error for
    CHUNK = int(65536/2)
    IP, PORT = info["front"]
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (IP, PORT)
    print ('sending error to {} port {}\n'.format(IP, PORT))
    time.sleep(1)
    try:
        sock.connect(server_address)
    except:
        print("connection to {} on port {} died. Couldn't send error packet".format(IP, PORT))
        return
    print("sending error")
    sock.sendall(b"VRERR\0")
    sock.close()

def send_stop(info):
    CHUNK = int(65536/2)
    IP, PORT = info["front"]
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (IP, PORT)
    time.sleep(1)
    print ('connecting to {} port {}\n'.format(IP, PORT))
    try:
        sock.connect(server_address)
    except:
        print("connection to {} on port {} died. Couldn't send stop packet".format(IP, PORT))
        return
    print("sending cancel")
    sock.sendall(b"CANCL\0")
    sock.close()
                
if __name__ == "__main__":
        main()
