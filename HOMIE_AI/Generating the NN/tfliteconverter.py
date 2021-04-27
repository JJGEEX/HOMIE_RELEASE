import tensorflow as tf

model = tf.keras.models.load_model('wake_word_stop_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("wake_word_stop_model.tflite", "wb").write(tflite_model)