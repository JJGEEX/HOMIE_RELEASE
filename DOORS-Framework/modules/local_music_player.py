from pygame import mixer #for audio control
from fuzzywuzzy import fuzz #for music searching
from fuzzywuzzy import fuzz #for music searching
import os #for parsing files in directory
import pwd #for finding current directory
import eyed3 #for mp3 metadata pulling
from parse import *
#import mutagen #for .wav files metadata pulling
from tinytag import TinyTag
import socket
import sys

def command_handler(sentence, info):
    msg = ""
    function = None
    comms, classify = commands()

    for i in comms[0]:
        res = parse(i, sentence)
        if res:
            msg, function = playSong(res[0], info)
            if len(msg) > 0:
                break

    if len(msg) > 0:
        return msg, function
    elif sentence in comms[1]:
        msg, function = stopSong()
    elif sentence in comms[2]:
        msg, function = continueSong()
    else:
        msg = sentence+" is not a known command"
        function = None
    return msg, function

def commands():
    coms = [
                [
                    "play the song {} by {}", 
                    "play the song {}"
                ],
                [
                    "stop playing music", "you must stop playing music",
                    "stop whatever music is playing", "stop the music",
                    "the music must stop"
                ],
                [
                    "continue playing music", "unpause the music",
                    "continue playing the song", "unpause the song"
                ]
            ]
    comp_types = [
        "parse",
        "cosine",
        "cosine"
        ]
    return coms, comp_types

def playSong(songName, info):
        songName = songName.strip()
        songs = []
        highDis = 0
        highDitle = ""
        highPath = None
        dis = 0
        path = "/home/"+pwd.getpwuid(os.getuid()).pw_name+"/Music/" #dir for music for current user
        onlyfiles = [f for f in os.listdir(path)] #only mp3 and wav files pulled

        if(len(songName) == 0):
                msg = "No song name was given"
                return msg, None

        for i in onlyfiles:
                fpath = path+i.strip()
                try:
                        tag = TinyTag.get(fpath)
                except:
                        continue
                if tag.title is None:
                        continue
                dis = fuzz.ratio(songName.lower(), tag.title.lower())
                if dis > 79: # 80% similarity or more is saved
                        songs.append([dis, tag.title])
                        if dis > highDis: #getting the most similar
                                highDis = dis
                                highTitle = tag.title
                                highPath = i
        if(highPath):
                totPath = path+highPath
                tmp = "{}/temp/tmpSend.wav".format(info["path"])
                song = "{}/temp/songSend.wav".format(info["path"])
                msg = "Song "+highTitle+" will be played"
                convert = "ffmpeg -i "+ totPath + " -ar 16k -ac 1 "+ song
                rmFile = "rm "+song

                try:
                    os.system(rmFile)
                except:
                    print(end = "")

                try:
                    os.system(convert)
                except:
                    msg = "couldn't convert file for song "+songName
                    func = None
                    return msg, func
                while True:
#                    option = input("send to front-end? ")
                    option = "yes" 

                    if option == "yes" or option == "y":
                        def send():
                            sendToFront(song, info)
                        return msg, send
                    elif option == "no" or option == "n":
                        def play():
                            mixer.init(16000, -16, 1)
                            mixer.music.load(song)
                            mixer.music.play()
                        return msg, play
                
        else:
                msg = "No songs in the local library matched "+songName
                return msg, None
        return 0

def sendToFront(songName, info):
    ip, port = info["front"]
    SIZE = int(65536/2)
    #open file for sending
    f = open(songName, "rb")
    binaryHeader = f.read(44) #remove .wav header info for raw format
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect the socket to the port where the server is listening
    server_address = (ip, port)
    while True:
        try:
            sock.connect(server_address)
            break
        except:
            print("connection to {} port {} refused. Can't send song".format(ip,port))
    print ('connecting to %s port %s' % server_address)
    size = 1
    while size > 0:
            read = f.read(SIZE)
            if size == 1:
                read = b"APCKT\0" + read
            size = len(read)
            try:
                sock.send(read)
            except KeyboardInterrupt:
                print("got keyboard interrupt for local music player")
                break
            except socket.error as ex:
                print("something went wrong with connection to {} port {}".format(ip,port))
                print("ERROR: {}".format(ex))
                while True:
                    try:
                        sock.connect(server_address)
                        break
                    except:
                        print("connection to {} port {} refused. Can't send song".format(ip,port))
                print ('connecting to %s port %s' % server_address)
                continue
    sock.settimeout(5)
    while True:
        try:
            data = sock.recv(SIZE)
            if b"ADONE" in data:
                break
        except:
            print("connection timed out in local music connection receive")
            f.close()
            return
    print("RECEIVED ADONE FOR LOCAL MUSIC PLAYER") 
    sock.close()
    f.close()

def stopSong():
        if mixer.get_init():
            def stopMusicFunc():
                mixer.music.pause()
            msg = "music is stopped"
            return msg, stopMusicFunc()
        else:
            return "no music playing", None
        
def continueSong():
    if mixer.get_init():
        def contMusicFunc():
                mixer.music.unpause()
        msg = "music will be unpaused"
        return msg, contMusicFunc()
    else:
        return "no music playing", None
