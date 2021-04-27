from os import listdir
from os.path import isdir, join
from tensorflow.keras import layers, models, metrics
import numpy as np

feature_sets_filename = 'all_targets_mfcc_sets.npz'
model_filename = 'wake_word_stop_model.h5'
wake_word = 'homie'

# Load feature sets
feature_sets = np.load(feature_sets_filename)

# Assign feature sets
x_train = feature_sets['x_train']
y_train = feature_sets['y_train']
x_val = feature_sets['x_val']
y_val = feature_sets['y_val']
x_test = feature_sets['x_test']
y_test = feature_sets['y_test']

print("percent 'Stop'")
print(sum(y_val) / len(y_val))

#get the wake_word
dataset_path = 'C:/Users/thwai/Desktop/HOMIE_AI/Generating the NN/voice-commands'
all_targets = all_targets = [name for name in listdir(dataset_path) if isdir(join(dataset_path, name))]
all_targets.remove('_background_noise_')
wake_word_index = all_targets.index(wake_word)

#convert truth arrays to 1 for wake word and 0 for other
y_train = np.equal(y_train, wake_word_index).astype('float64')
y_val = np.equal(y_val, wake_word_index).astype('float64')
y_test = np.equal(y_test, wake_word_index).astype('float64')




# CNN for TF expects (batch, height, width, channels)
# So we reshape the input tensors with a "color" channel of 1
x_train = x_train.reshape(x_train.shape[0],
                          x_train.shape[1],
                          x_train.shape[2],
                          1)
x_val = x_val.reshape(x_val.shape[0],
                      x_val.shape[1],
                      x_val.shape[2],
                      1)
x_test = x_test.reshape(x_test.shape[0],
                        x_test.shape[1],
                        x_test.shape[2],
                        1)


# Input shape for CNN is size of MFCC of 1 sample
sample_shape = x_test.shape[1:]
# print(sample_shape)
# sample_shape = (12,12,1)

# -----------------------------------------------------------------------------------
# Build model
# Based on: https://www.geeksforgeeks.org/python-image-classification-using-keras/
model = models.Sequential()
model.add(layers.Conv2D(32,
                        (2, 2),
                        activation='relu',
                        input_shape=sample_shape))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(8, (2, 2), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(layers.Conv2D(16, (2, 2), activation='relu'))
model.add(layers.BatchNormalization())
# model.add(layers.Conv2D(64, (2, 2), activation='relu'))
# model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Classifier
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
# model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.15))
model.add(layers.Dense(1, activation='sigmoid'))

# -----------------------------------------------------------------------------------
# # Build model
# # Based on: https://www.geeksforgeeks.org/python-image-classification-using-keras/
# model = models.Sequential()
# model.add(layers.Conv2D(32,
#                         (2, 2),
#                         activation='relu',
#                         input_shape=sample_shape))
# model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#
# model.add(layers.Conv2D(32, (2, 2), activation='relu'))
# model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#
# model.add(layers.Conv2D(64, (2, 2), activation='relu'))
# model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#
# # Classifier
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(1, activation='sigmoid'))

# -------------------------------------------------------------------------------------

# Display model
model.summary()

# Add training parameters to model
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=[metrics.BinaryAccuracy()])
              # metrics=['acc'])

# Train
history = model.fit(x_train,
                    y_train,


                    epochs=15,
                    batch_size=50,


                    # epochs=30,
                    # batch_size=100,
                    validation_data=(x_val, y_val))


# Plot results
import matplotlib.pyplot as plt

acc = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# Save the model as a file
models.save_model(model, model_filename)

# See which are 'stop'
# for idx, y in enumerate(y_test):
#     if y == 1:
#         print(idx)

# # TEST Load model and run it against test set
tfmodel = models.load_model(model_filename)
for i in range(100, 110):
    print('Answer:', y_test[i], ' Prediction:', model.predict(np.expand_dims(x_test[i], 0)))
tfmodel.evaluate(x=x_test, y=y_test)