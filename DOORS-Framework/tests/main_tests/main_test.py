import os
import sys
sys.path.append('tools')

import sklearn_sims as sklearn_sims
from pygame import mixer
from parse import parse

def run_tests(decoder, voice, classes):
        t_range = ["1", "2", "3", "4", "5", "6"]
        t_menu = (
                "TEST 1: \"set a timer for 3 seconds\"\n"
                "TEST 2: \"play the song country roads\"\n"
                "TEST 3: \"stop playing music\"\n"
                "TEST 4: \"what's the weather in denver\"\n"
                "TEST 5: \"start a stopwatch\"\n"
                "TEST 6: \"stop the stopwatch\"\n"
                "enter \"7\" to exit this menu\n"
                "Enter a test number for the test you would like to run: "
                )
        num = None

        os.system("clear")

        while True:
            num = input(t_menu).strip()
            if num in t_range:
                    f_name = os.getcwd()+"/tests/voice_files/file_"+num+".wav"
                    if num == "3":
                        mixer.init()
                        mixer.music.pause()
                    os.system("aplay "+f_name)
                    if num == "3":
                        mixer.music.unpause()
                    os.system("clear")
                    sentence = decoder.decode_file(f_name)
                    if sentence == "":
                        print("nothing detected within vosk")
                        continue
                    msg, func, bMod = sklearn_sims.compare_command(sentence, classes)
                    check_test(num, msg)
            elif num != "7":
                print(str(num)+" isn't a valid option!")
            else:
                break

def check_test(num, sentence):
        sentence = sentence.strip()

        if num == "1":
            print("EXPECTED: setting timer for 3 seconds")
            print("RETURNED: "+sentence)
            temp = "setting timer for 3 seconds" == sentence
            print("EQUAL: "+str(temp))
        if num == "2":
            print("EXPECTED: Song Country Roads will be played")
            print("RETURNED: "+sentence)
            temp = "Song Country Roads will be played" == sentence
            print("EQUAL: "+str(temp))
        if num == "3":
            print("EXPECTED: music is stopped")
            print("RETURNED: "+sentence)
            temp = "music is stopped" == sentence
            print("EQUAL: "+str(temp))
        if num == "4":
            print("EXPECTED: using city {}\n"
                "looks like {} today\n"
                "humidity is at {} percent\n"
                "temperature is {} degrees fahrenheit\n")
            print("RETURNED: "+sentence)
            check = ("using city {}\n"
                "looks like {} today\n"
                "humidity is at {} percent\n"
                "temperature is {} degrees fahrenheit")
            ret = parse(check, sentence)
            temp = ret is not None
            print("EQUAL: "+str(temp))
        if num == "5":
            print("EXPECTED: started a stopwatch")
            print("RETURNED: "+sentence)
            temp = "started a stopwatch" == sentence
            print("EQUAL: "+str(temp))
        if num == "6":
            print("EXPECTED: stopwatch ran for x seconds")
            print("RETURNED: "+sentence)
            ret = parse("stopwatch ran for {} seconds", sentence)
            temp = ret is not None
            print("EQUAL: "+str(temp))
        print()
