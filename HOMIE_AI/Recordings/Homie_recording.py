import os
import time
import random
from datetime import datetime
import subprocess

import numpy as np
from scipy.io.wavfile import write
import sounddevice as sd
import PySimpleGUI as sg
       
        
  

    
random.seed(datetime.now())


sample_rate = 16000
filename = "Homie_Recordings"

#------------------------------------------------
layout = [
          [sg.Button('Record Audio', button_color=('white', 'black'), key='start')]
          ]
#------------------------------------------------



current_dir = os.path.abspath(os.getcwd())
newpath = current_dir +"/"+ filename

if not os.path.exists(newpath):
    os.makedirs(newpath)





#records the audio
def record_audio(filename):
    duration = 1
    print("Recording..........")
    myrecording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2)
    time.sleep(duration)
    sd.wait()
    write(filename, sample_rate, myrecording.astype(np.float32))




if __name__ == "__main__":
    
    window = sg.Window("Homie", layout)
    while True:
        event, values = window.read(timeout=100)
        if event == sg.WINDOW_CLOSED:
            break
        elif event== "start":
            window["start"].Update("Recording",  button_color=('white', "white"))
            record_audio(newpath+"/"+ "homie_"+ str(random.randint(0,9999999))+ ".wav")
            window["start"].Update('Record Audio',  button_color=('white', 'black'))
            
        
    window.close()
    
    

        