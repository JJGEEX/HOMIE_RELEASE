from os import listdir
from os.path import isdir, join
import librosa
import random
import numpy as np
import matplotlib.pyplot as plt
import python_speech_features

# Settings

feature_sets_file = 'all_targets_mfcc_sets.npz'
perc_keep_samples = 1.0 # 1.0 is keep all samples
val_ratio = 0.1
test_ratio = 0.1
sample_rate = 8000
num_mfcc = 8 #change these ------------------------------------------------------
len_mfcc = 8


# Dataset path and view possible targets
dataset_path = 'C:/Users/thwai/Desktop/HOMIE_AI/Generating the NN/voice-commands'

# Create an all targets list
all_targets = [name for name in listdir(dataset_path) if isdir(join(dataset_path, name))]
all_targets.remove('_background_noise_')


target_list = all_targets

# Create list of filenames along with ground truth vector (y)
filenames = []
y = []
for index, target in enumerate(target_list):
    # print(join(dataset_path, target))
    filenames.append(listdir(join(dataset_path, target)))
    y.append(np.ones(len(filenames[index])) * index)
    if 'homie' in target:
        Homie_index = index
        print(index)
    
    
    


# Flatten filename and y vectors
filenames = [item for j,sublist in enumerate(filenames) for i, item in enumerate(sublist) if (i < 1500 or j == Homie_index)]
y = [item for j,sublist in enumerate(y) for i, item in enumerate(sublist) if (i < 1500 or j == Homie_index)]

# Associate filenames with true output and shuffle
filenames_y = list(zip(filenames, y))
random.shuffle(filenames_y)
filenames, y = zip(*filenames_y)

# Only keep the specified number of samples (shorter extraction/training)
print(len(filenames))
filenames = filenames[:int(len(filenames) * perc_keep_samples)]
print(len(filenames))

# Calculate validation and test set sizes
val_set_size = int(len(filenames) * val_ratio)
test_set_size = int(len(filenames) * test_ratio)

# Break dataset apart into train, validation, and test sets
filenames_val = filenames[:val_set_size]
filenames_test = filenames[val_set_size:(val_set_size + test_set_size)]
filenames_train = filenames[(val_set_size + test_set_size):]

# Break y apart into train, validation, and test sets
y_orig_val = y[:val_set_size]
y_orig_test = y[val_set_size:(val_set_size + test_set_size)]
y_orig_train = y[(val_set_size + test_set_size):]


# Function: Create MFCC from given path
def calc_mfcc(path):
    # Load wavefile
    signal, fs = librosa.load(path, sr=sample_rate)

    # print(len(signal)/fs)
    # Create MFCCs from sound clip
    mfccs = python_speech_features.base.mfcc(signal,
                                             samplerate=fs,
                                             winlen=0.128,
                                             winstep=0.1246,
                                             # winlen=0.256,
                                             # winstep=0.050,
                                             numcep=num_mfcc,
                                             # nfilt=26,
                                             nfilt=8,
                                             nfft=1024,
                                             preemph=0.0,
                                             ceplifter=0,
                                             appendEnergy=False)
    # print(mfccs)
    return mfccs.transpose()


# Function: Create MFCCs, keeping only ones of desired length
def extract_features(in_files, in_y):
    prob_cnt = 0
    out_x = []
    out_y = []

    for index, filename in enumerate(in_files):

        # Create path from given filename and target item
        path = join(dataset_path, target_list[int(in_y[index])],
                    filename)

        # Check to make sure we're reading a .wav file
        if not path.endswith('.wav'):
            continue

        # Create MFCCs
        mfccs = calc_mfcc(path)

        # Only keep MFCCs with given length
        if mfccs.shape[1] == len_mfcc:
            out_x.append(mfccs)
            out_y.append(in_y[index])
        else:
            print('Dropped:', index, mfccs.shape)
            prob_cnt += 1

    return out_x, out_y, prob_cnt

x_train, y_train, prob = extract_features(filenames_train,
                                          y_orig_train)
print('Removed percentage:', prob / len(y_orig_train))
x_val, y_val, prob = extract_features(filenames_val, y_orig_val)
print('Removed percentage:', prob / len(y_orig_val))
x_test, y_test, prob = extract_features(filenames_test, y_orig_test)
print('Removed percentage:', prob / len(y_orig_test))


# Save features and truth vector (y) sets to disk
np.savez(feature_sets_file,
         x_train=x_train,
         y_train=y_train,
         x_val=x_val,
         y_val=y_val,
         x_test=x_test,
         y_test=y_test)