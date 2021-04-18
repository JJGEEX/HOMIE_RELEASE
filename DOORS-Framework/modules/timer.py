import sys
sys.path.append("../tools")

from word2number import w2n
import signal
from parse import *
import socket
import front_info as fe
import voice_synth as vs

def command_handler(sentence, info):
    msg = sentence+" is not a known command"
    function = None
    comms, classify = commands()
    for i in comms: #pulling parsing formats
        for j in i:
            res = parse(j, sentence) #try and parse using parse formats
            if res:
                msg, function = setTimer(res[0])
    return msg, function

def commands():
    comm = [
            [
                "set a timer for {}",
                "start a timer for {}",
                "setup a timer for {}",
            ]
        ]
    classify = [
            "parse"
            ]
    return comm, classify

def handler(signal, frame): #handler for timer
        info = fe.get_fe_info()
        voice = vs.VoiceSynth(info)
        msg = "\n\nTime is up for timer!\n"
        print(msg)
        voice.sendToFront(msg)
        

def setTimer(timeStr): #only going to focus on time for now
        temp = ""
        arr = timeStr.split()
        num = 0
        strNum = ""
        msg = ""
        timeSwitch = { #dictionary for scaling the time
            "second": 1,
            "minute": 60,
            "hour": 3600,
            }

        try:
                timeFormat = arr[-1].strip() #get time format
                arr = arr[:-1] #remove time format
        except: #failed to pull timeformat from string
                msg = "no time format was detected for setting a timer"
                return msg, None

        if(timeFormat[-1] is "s"): #removing trailing s. EX: seconds, minutes
                timeFormat = timeFormat[:-1]

        for f in range(len(arr),0,-1): #pulling time amount out of string
                try:
                        strtemp = " ".join(arr[f-1:]) #pull substring and see if it's a number          
                        numtemp = w2n.word_to_num(strtemp)
                        num = int(numtemp)
                except ValueError:
                        break

        if(timeFormat not in timeSwitch.keys()): #error 1: no time format
                msg = timeFormat+" is not a valid time format"
                return msg, None
        elif(num == 0): #error 2: time requested is 0 for timer
                msg = "can't set a timer for 0 "+timeFormat
                return msg, None

        msg = "\nsetting timer for "+str(num)+" "+timeFormat
        if num > 1:
                msg += "s"
        def setSignal():
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(num * timeSwitch[timeFormat])
        return msg, setSignal
