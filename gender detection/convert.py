
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

def convertToTlite():
    converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file( 'age_recognition.h5' )
    converter.post_training_quantize = True
    tflite_buffer = converter.convert()
    open( 'age_recognition.tflite' , 'wb' ).write( tflite_buffer )
convertToTlite()


def loadModel():
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="age_recognition.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    input_shape = input_details[0]['shape']
    print(np.random.random_sample(input_shape))
    img = image.load_img('testimages/female.jpg',target_size=(257,257))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    input_data = np.array(img, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)
