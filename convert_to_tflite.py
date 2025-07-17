import tensorflow as tf

# 1) Load your Keras model from the H5 file
model = tf.keras.models.load_model('models/disease_cnn.h5', compile=False)

# 2) Create a TFLite converter from the Keras model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Optional: optimize for size
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]

# 3) Perform the conversion
tflite_model = converter.convert()

# 4) Save the TFLite model to disk
with open('models/disease_cnn.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Converted to TFLite: models/disease_cnn.tflite")
