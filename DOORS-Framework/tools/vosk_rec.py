#!/usr/bin/env python3

from vosk import Model, KaldiRecognizer, SetLogLevel
import sys
import socket
import os
import wave
import json
import struct
import math
import random
import wave
import time
from pydub import AudioSegment, silence #for detecting silence in audio file

class Decoder:
        def __init__(self, info):
                model = Model(os.getcwd()+"/modules/model")
                self.rec = KaldiRecognizer(model, 8000)
                self.ip, self.port = info["front"]

        def decode_file(self, aud_file):
                SetLogLevel(0)
                sentence = ""
                results = ""
                confidence = 0
                tot = 0

                wf = wave.open(aud_file, "rb")
                if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE": #checking certain file characteristics
                        print ("Audio aud_file must be WAV format mono PCM.")
                        exit (1)

                
                while True: #loop for doing voice recognition
                        data = wf.readframes(4000)
                        if len(data) == 0: #done reading audio file
                                break
                        if self.rec.AcceptWaveform(data): #finished recognition on segment of audio file
                                items = self.rec.Result()
                                results = json.loads(items)
                                if len(results.items()) > 1: #false recognition, sometimes nothing is detected
                                        for i in results["result"]: 
                                            confidence += i["conf"]
                                            tot += 1
                                        sentence = sentence + " " + results["text"]
                                else:
                                        print(self.rec.PartialResult())
                f_res = json.loads(self.rec.FinalResult())
                if len(f_res.items()) > 1:
                    return f_res["text"]
                wf.close()                        
                if tot > 0 and confidence/tot > .8: #checking confidence of recognition
                        return sentence.lower().strip()
                elif tot > 0:
                        print("confidence too low: "+str(confidence/tot))
                return ""

        def listen_stream(self):
                HOST = self.ip
                PORT = self.port
                CHUNK = 32768
                TIMEOUT = 10

                while True:
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                                totData = 0
                                connDied = False

                                ret = self.try_connection(HOST, PORT, s, "send CNRDY")
                                if ret == False:
                                        s.close()
                                        continue
                                print("connected")
                                s.sendall(b"CNRDY\0") #sending connection ready 
                                data = b""
                                s.settimeout(2)
                                while b"YEETO" not in data: #getting rid of bad data
                                    try:
                                            data = s.recv(CHUNK)
                                            print("bad data : {}".format(len(data)))
                                            if len(data) == 0:
                                                    print("conn died during handshake")
                                                    time.sleep(2)
                                                    connDied = True
                                                    break

                                    except:
                                            print("timed out from connection and didn't get YEETO")
                                            print("exception: {}".format(sys.exc_info[0]))
                                            connDied = True
                                            break
                                if connDied:
                                        continue
                                s.settimeout(None)
                                s.sendall(b"FLUSH\0") #letting front know bad data has been flushed
                                FTOT, FTEMP = self.init_temp_tot_wave() #init FTOT and FTEMP files
                                s.settimeout(5)
                                while True:
                                        temp = self.open_temp_wave(FTEMP) #get temorary wave file
                                        try:
                                                data = s.recv(CHUNK)
                                        except:
                                                print("connection with {} {} died".format(HOST, PORT))
                                                print("exception: {}".format(sys.exc_info[0]))
                                                connDied = True
                                                break
                                        size = len(data)
                                        totData += size
                                        if data == None or size == 0:#check for when we 
                                                #receive packets of zero size
                                                print("connection from front-end closed")
                                                print(f"FRONT CLOSE tot data received : {totData}")
                                                break
                                        print(f"got data: {len(data)}")
                                        temp.writeframesraw(data)
                                        temp.close()
                                        self.combine_files([FTOT, FTEMP]) 
                                        #combining wave file data
                                        if(self.detect_silence(FTOT)): 
                                                #2 seconds of silence detected
                                                break
                                if connDied:
                                        break   
                        try:
                                s.close()
                                print(f"BACK CLOSE tot data received : {totData}")
                                if totData != 0: #we got zero data from the connection
                                        self.send_gdata()
                                        break
                        except BrokenPipeError:
                                print(f"connection died with {HOST} port {PORT}")

                results = self.decode_file(FTOT) #get results from file
                print("FINAL RESULT from stream: "+results)
                return results

        def clear_socket(self): #prototype for clearing socket data
                HOST = self.ip
                PORT = self.port

                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    self.try_connection(HOST, PORT, sock, "CLEAR SOCKET")
                    sock.settimeout(TIMEOUT) # 10 second timeout
                    size = 1
                    while size > 0:
                        sock.recv(1024) #just receive data and throw it away
                    sock.close()

        def send_cnerr(self):
                HOST = self.ip
                PORT = self.port

                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                        print("sending connection error")
                        self.try_connection(HOST, PORT, sock, "SEND CNERR")
                        sock.sendall(b"CNERR\0")
                        sock.close()

        def send_gdata(self):
                HOST = self.ip
                PORT = self.port

                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                        print("sending good data")
                        self.try_connection(HOST, PORT, sock, "SEND GDATA")
                        sock.sendall(b"GDATA\0")
                        sock.close()

        def init_temp_tot_wave(self):
                FTOT = "./temp/recv.wav"
                FTEMP = "./temp/temp.wav"

                tot = wave.open(FTOT, 'wb')
                tot.setnchannels(1) #mono
                tot.setsampwidth(2)
                tot.setframerate(8000)
                tot.close()

                temp = wave.open(FTEMP, 'wb')
                temp.setnchannels(1) #mono
                temp.setsampwidth(2)
                temp.setframerate(8000)
                temp.close()
                return FTOT, FTEMP

        def open_temp_wave(self, FTEMP):
                temp = wave.open(FTEMP, 'wb')
                temp.setnchannels(1) #mono
                temp.setsampwidth(2)
                temp.setframerate(8000)
                return temp


        def try_connection(self, HOST, PORT, s, funcName):
                print("trying to connect "+HOST+ " " +str(PORT))
                print(f"{funcName} connecting to front-end")
                time.sleep(2)
                s.settimeout(5)
                try:
                        s.connect((HOST, PORT))
                        s.settimeout(None)
                        return True
                except ConnectionRefusedError:
                        print("connection to {} on port {} refused.".format(HOST, PORT))
                        print("will try again in 5 seconds\n")
                        time.sleep(5)
                        return False
                except OSError:
                        print("couldn't find {} on port {}".format(HOST, PORT))
                        print("wil try again in 5 seconds")
                        time.sleep(5)
                        return False
                except TimeoutError:
                        print("connection timed out for {} port {}".format(HOST, PORT))
                        print("will try again in 5 seconds\n")
                        time.sleep(5)
                        return False


        def send_mstop(self):
                HOST = self.ip
                PORT = self.port

                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                        print("sending MSTOP")
                        while True:
                                try:
                                        sock.connect((HOST, PORT))
                                        break
                                except ConnectionRefusedError:
                                        print("connection to {} on port {} refused.".format(HOST, PORT))
                                        print("will try again in 5 seconds\n")
                                        time.sleep(5)
                                except OSError:
                                        print("couldn't find {} on port {}".format(HOST, PORT))
                                        print("wil try again in 5 seconds")
                                        time.sleep(5)
                        sock.sendall(b"MSTOP\0")
                        sock.close()

        def combine_files(self, files):
                data = []

                for infile in files:
                        w = wave.open(infile, "rb")
                        data.append( [w.readframes(w.getnframes())] )
                        w.close()

                output = wave.open(files[0], "wb")
                output.setnchannels(1) #mono
                output.setsampwidth(2)
                output.setframerate(8000)
                output.writeframes(data[0][0])
                output.writeframes(data[1][0])
                output.close()
                                                
        def detect_silence(self, fileName):
                myaudio = intro = AudioSegment.from_wav(fileName)
                dBFS = myaudio.dBFS
                print(dBFS)
                pieces = silence.detect_silence(myaudio, 1000, dBFS-0)
                pieces = [((start/1000),(stop/1000)) for start,stop in pieces] #convert to sec

                for i in pieces:
                        if i[1] - i[0] > 3:
                            print("big silence: "+str(i[0]) + " " + str(i[1]))
                            return True
                return False
