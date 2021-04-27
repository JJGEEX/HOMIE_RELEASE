import numpy as np
import sounddevice as sd
import scipy.signal
import timeit
import python_speech_features
import wavio
import simpleaudio as sa
import tensorflow as tf

# Parameters
window_time = 1 # in seconds
num_mfcc = 8
num_samples_per_word = 8
model_path = "wake_word_stop_model.tflite"
debug_time = 1
debug_acc = 0
word_threshold = 0.8
AI_sample_rate = 8000
Num_Callbacks_per_window = 2 #current limit due to too much time taken


#things modified by parameters
window_stride = window_time/Num_Callbacks_per_window #window step size before AI checks for a match
sample_rate = 48000
resample_rate = AI_sample_rate
num_channels = 1


window = np.zeros(int(window_stride * resample_rate) * Num_Callbacks_per_window)

mfcc_mem = np.zeros((num_mfcc, num_samples_per_word))

# Load model (interpreter)
interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)

# Decimate (filter and downsample)
def decimate(signal, old_fs, new_fs):
    # Check to make sure we're downsampling
    if new_fs > old_fs:
        print("Error: target sample rate higher than original")
        return signal, old_fs

    # We can only downsample by an integer factor
    dec_factor = old_fs / new_fs
    if not dec_factor.is_integer():
        print("Error: can only decimate by integer factor")
        return signal, old_fs

    # Do decimation
    resampled_signal = scipy.signal.decimate(signal, int(dec_factor))

    return resampled_signal, new_fs

# This gets called every window_stride seconds
def sd_callback(rec, frames, time, status):
    # GPIO.output(led_pin, GPIO.LOW)

    # Start timing for testing
    start = timeit.default_timer()

    # Notify if errors
    if status:
        print('Error:', status)

    # Remove 2nd dimension from recording sample
    rec = np.squeeze(rec)

    # Resample
    rec, new_fs = decimate(rec, sample_rate, resample_rate)
    
    
    # Save recording onto sliding window
    window_size = new_fs //Num_Callbacks_per_window
    
    # extra
    tmp = window[window_size-192: window_size]


    for i in range(Num_Callbacks_per_window - 1):
        window[i*window_size:(i+1)*window_size] = window[(i+1)*window_size:(i+2)*window_size]
    # window[:len(window) // 2] = window[len(window) // 2:]
    # window[len(window) // 2:] = rec
    window[len(window) - window_size:] = rec
    

    # Compute features
    new_mfccs = python_speech_features.base.mfcc(window,
                                             samplerate=new_fs,
                                             winlen=0.125,
                                             winstep=0.125,
                                             # winlen=0.256,
                                             # winstep=0.050,
                                             numcep=num_mfcc,
                                             # nfilt=26,
                                             nfilt=8,
                                             nfft=2048,
                                             preemph=0.0,
                                             ceplifter=0,
                                             appendEnergy=False,
                                             winfunc=np.hanning)
    # print(len(mfccs[0])) # This one should be the ceps
    # new_mfccs = new_mfccs.transpose()
    # mfcc_mem[6:8] = mfcc_mem[4:6]
    # mfcc_mem[4:6] = mfcc_mem[2:4]
    # mfcc_mem[4:8] = mfcc_mem[0:4]
    # mfcc_mem[0:4] = new_mfccs
    # print(new_mfccs)

    mfccs = new_mfccs.transpose()
    # for i in range(Num_Callbacks_per_window - 1):
    #     mfccs[]

    # print(len(mfccs[0])) #this one has time component
    # print(mfccs.shape)
    # Make prediction from model
    in_tensor = np.float32(mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1))
    interpreter.set_tensor(input_details[0]['index'], in_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    val = output_data[0][0]

    if val > word_threshold:
        print('homie')
        print(val)
        recording = np.append(tmp , window)
        mfcc_audio = (recording*(2**14)).astype(int)
        print(mfccs)
        # wavio.write("audio.wav", mfcc_audio, 8000)
        # np.savetxt('array.csv', mfcc_audio, delimiter=',', fmt='%i')
        # GPIO.output(led_pin, GPIO.HIGH)
        # with open('valid.npy', 'wb') as f:
        #     np.save(f, in_tensor)
        
    # if val < 0.01:
        # print('def not stop')
        # print(val)
        # recording = np.append(tmp , window)
        # mfcc_audio = (recording*(2**14)).astype(int)
        # # print(mfccs)
        # wavio.write("audio.wav", mfcc_audio, 8000)
        # np.savetxt('array.csv', mfcc_audio, delimiter=',', fmt='%i')

    if debug_acc:
        print(val)


    if debug_time:
        print(timeit.default_timer() - start)



# Start streaming from microphone
with sd.InputStream(channels=num_channels,
                    samplerate=sample_rate,
                    blocksize=int(sample_rate * window_stride),
                    callback=sd_callback):
    while True:
        pass
        x = 1
print("done")