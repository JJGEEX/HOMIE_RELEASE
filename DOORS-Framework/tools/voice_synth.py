#!/usr/bin/env python
import sys
sys.path.append("temp")

from subprocess import call
from os import system
import socket
import sys
import time

class VoiceSynth:
        
        def __init__(self, info):
                self.ip, self.port = info["front"]
                self.enabled = True

        def speak(self, sentence):
                if self.enabled:
                        comb = ""
                        for i in sentence:
                                comb += i
                        comb = comb.replace("\n", " ")
                        command = 'espeak \"{}\" --stdout |aplay 2>/dev/null'.format(comb)
                        system(command)
        def sendToFront(self, sentence): #still working on this
                fName = "./temp/voice.wav"
                temp = "./temp/vTemp.wav"
                cleanup = [
                        "rm "+fName,
                    ]
                comms = [
                        "espeak -w "+temp+" -s 130 \""+sentence+"\"",  #converting msg to voice synth .wav file
                        "ffmpeg -i "+temp+" -ar 16000 "+fName, #downsampling .wav file for front-end
                        "rm "+temp, #removing the temp file
                        "clear"
                    ]

                if self.enabled == False:
                        print("voice synth not on. Will not send audio to front end")
                        return
                for i in cleanup:
                    try:
                        system(i)
                    except:
                        continue
                for i in comms:
                    try:
                        system(i)
                    except:
                        continue
                SIZE = int(65536/2)
                #open file for sending
                f = open(fName, "rb")
                binaryHeader = f.read(44) #remove .wav header info for raw format
                # Create a TCP/IP socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

                # Connect the socket to the port where the server is listening
                server_address = (self.ip, self.port)
                print("msg to send {}".format(sentence))
                print('voice synth connecting to {} port {}'.format(self.ip, self.port))
                time.sleep(2)
                try:
                    sock.connect(server_address)
                except:
                    print("connection to {} port {} refused".format(self.ip, self.port))
                    return
                time.sleep(2)
                size = 1
                while size > 0:
                        read = f.read(SIZE)
                        if size == 1: #first loop we append audio 
                            read = b"APCKT\0" + read
                        size = len(read)
                        try:
                                sock.sendall(read)
                        except BrokenPipeError:
                                print("connection to IP {} PORT {} died".format(self.ip, self.port))
                                break
                                f.close()
                                return
                while True:
                        data = sock.recv(SIZE)
                        if b"ADONE" in data:
                            break
                print("ADONE RECEIVED FOR VOICE SYNTH") 
                f.close()
                sock.close()

        def enable(self):
                self.enabled = True
        
        def disable(self):
                self.enabled = False
