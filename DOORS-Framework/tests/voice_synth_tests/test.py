#!/usr/bin/env python3
import modules.sklearn_sims as sklearn_sims
import modules.local_commands as lc
import modules.voice_synth as vs
import os #for recording, temporary usage

def main():
        voice = vs.VoiceSynth()
        voice.disable()
        stopwatch = lc.Stopwatch()

        #testSetTimer(voice)
        testPlaySong(voice)
        #testStopMusic(voice)
        #testWeather(voice)
        #testStartWatch(stopwatch, voice)
        #testStopWatch(stopwatch, voice)
                        
def testSetTimer(voice):
        sent1 = "set a timer for 3 seconds" #works correctly
        sent2 = "set a timer for 0 seconds" #works correctly
        sent3 = "set a timer for seconds" #test defaults to 0 second error. Might change to no time detected error
        sent4 = "set a timer for 5" #works correctly
        sent5 = "set a timer for 33 seconds" #works correctly
        sent6 = "set a timer for"
        arr = [sent1, sent2, sent3, sent4, sent5]
        match = "set a timer for"
        title = "TESTING SET TIMER FUNCTIONALITY"
        testNum = 1

        print(title)
        for i in range(0, len(arr)):
                testInfo(arr[i], i)
                lc.check_command(match, arr[i], None, voice)
                input("press enter for the next test\n\n")
        print("END*******************************************")
        
def testPlaySong(voice): #music keeps playing for speaker, maybe pause it?
        sent1 = "play the song country roads" #works correctly
        sent2 = "play the song" #works correctly
        sent3 = "play the song random" #works correctly
        match = "play the song" #works correctly
        arr = [sent1, sent2, sent3, sent1]
        
        for i in range(0, len(arr)):
                testInfo(arr[i], i)
                lc.check_command(match, arr[i], None, voice)
                input("press enter for the next test\n\n")
        print("END*******************************************")

def testWeather(voice): #need to change error message and code for this function
        sent1 = "what's the weather in denver" #works correctly
        sent2 = "what's the weather in america" #works correctly
        sent3 = "what's the weather in random" #works correctly
        sent4 = "what's the weather in" #works correctly
        sent5 = "what's the weather" #works correctly
        match = "what's the weather"
        arr = [sent1, sent2, sent3, sent4, sent5]

        for i in range(0, len(arr)):
                testInfo(arr[i], i)
                lc.check_command(match, arr[i], None, voice)
                input("press enter for the next test\n\n")
        print("END*******************************************")
        

def testStartWatch(watch, voice): #need to check for a stopwatch within the function
        sent1 = "start a stopwatch"
        match = "start a stopwatch"
        arr = [sent1]

        for i in range(0, len(arr)):
                testInfo(arr[i], i)
                lc.check_command(match, arr[i], watch, voice)
                input("press enter for the next test\n\n")
        print("END*******************************************")

def testStopWatch(watch, voice): #need to check for a stopwatch within the function
        sent1 = "stop the stopwatch"
        match = "stop the stopwatch"
        arr = [sent1, sent1]

        for i in range(0, len(arr)):
                testInfo(arr[i], i)
                lc.check_command(match, arr[i], watch, voice)
                input("press enter for the next test\n\n")
        print("END*******************************************")

def testStopMusic(voice): #maybe check if music is playing?
        sent1 = "stop playing music"
        match = "stop playing music"
        arr = [sent1]

        for i in range(0, len(arr)):
                testInfo(arr[i], i)
                lc.check_command(match, arr[i], None, voice)
                input("press enter for the next test\n\n")
        print("END*******************************************")

def testInfo(command, testNum):
        print("TEST",testNum+1)
        print("command to be tested:",command)

if __name__ == "__main__":
    main()
