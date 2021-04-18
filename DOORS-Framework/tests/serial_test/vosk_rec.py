#!/usr/bin/env python3

from vosk import Model, KaldiRecognizer, SetLogLevel
import sys
import os
import wave
import json
import struct
import math
import random

class Decoder:
    def __init__(self):
        model = Model("/home/pi/Documents/DOORS/modules/model")
        self.rec = KaldiRecognizer(model, 8000)

    def decode_file(self, aud_file):
        SetLogLevel(0)

        wf = wave.open(aud_file, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            print ("Audio aud_file must be WAV format mono PCM.")
            exit (1)

        results = []
        
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if self.rec.AcceptWaveform(data):
                results.append(self.rec.Result())
 
        for i in results:
            y = json.loads(i)
            print("---VOSK TEXT---",y["text"])
        print("results:", results)
        return results

    def decode_stream(self, socket, initData):
        fname = 'temp.wav'
        cur = 1
        obj = wave.open(fname, 'wb')
        obj.setchannels(1) #mono
        obj.setsampwidth(2)
        obj.setframerate(8000)
        obj.writeframesraw(initData)
        obj.close
        results = self.decode_file(fname)
        print("results "+cur+":"+results)


        while true:
            obj = wave.open(fname, 'wb')
            obj.setchannels(1) #mono
            obj.setsampwidth()
            obj.setframerate(8000)
            obj.writeframesraw(socket.read(1024))
            obj.close
            results = self.decode_file(fname)
            print("results "+cur+":"+results)
            cur += 1
