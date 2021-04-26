from os import listdir
from os.path import isdir, join
import librosa
import random
import numpy as np
from scipy.io.wavfile import write
# import matplotlib.pyplot as plt
# import python_speech_features

# Settings

feature_sets_file = 'all_targets_mfcc_sets.npz'
perc_keep_samples = 1.0 # 1.0 is keep all samples
val_ratio = 0.1
test_ratio = 0.1
sample_rate = 8000
num_mfcc = 8 #change these ------------------------------------------------------
len_mfcc = 8


# Dataset path and view possible targets
dataset_path = '(path to voice commands folder)/voice-commands/'

# Create an all targets list
all_targets = [name for name in listdir(dataset_path) if isdir(join(dataset_path, name))]
#all_targets.remove('_background_noise_')






for i in all_targets:
    # if i in ["homie", "bird", "happy", "five", "go"]:  
        # filenames = [dataset_path + i+"/"+ fname for fname in listdir(join(dataset_path, i))]#get a list of all the filenames in a given directory
        # filenames = filenames[500:1000]
        
        # #get the average of a file type
        # avgs = [np.average(np.absolute(librosa.load(path, sr=sample_rate)[0])) for path in filenames]
        # average = sum(avgs)/len(avgs)
        # print(i)
        # print(average)
        # print("")
        
    if i in ["homie"]:
        filenames = [dataset_path + i+"/"+ fname for fname in listdir(join(dataset_path, i))]
        #this sees if a recording is too quiet and makes it louder
        for path in filenames:
            file = librosa.load(path, sr=sample_rate)[0]
            
            if np.average(np.absolute(file)) < 0.03:
                file = file*2
                print(path.split("/")[-1])
                
            write(dataset_path + "h2/s" +path.split("/")[-1], sample_rate, file.astype(np.float32))
            
        # if i in ["homie"]:
        # filenames = [dataset_path + i+"/"+ fname for fname in listdir(join(dataset_path, i))]
        
        # for path in filenames:
            ##This is used to potentially double the samples
            # file = librosa.load(path, sr=sample_rate)[0]
            

                
            # write(dataset_path + "homie/d" +path.split("/")[-1], sample_rate, (2*file).astype(np.float32))
            # write(dataset_path + "homie/h" +path.split("/")[-1], sample_rate, (file/2).astype(np.float32))
            
        
        
        

