import numpy as np
import tensorflow as tf
import keras2onnx as k2o
import os
input = tf.keras.Input(shape = (32, 32), dtype = 'float32', name = 'Input')

x = input
x = tf.keras.layers.Reshape((32, 32, 1))(x)
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(1, (3, 3), activation='relu')(x)
x = tf.keras.layers.Reshape((4, 4),  name = 'Output')(x)

model = tf.keras.Model(inputs = input, outputs = x)

print(model.summary())

model.compile()

test = np.e * np.ones([1, 32, 32], dtype = 'float32')

result = model(test)

print(result.numpy())


#save keras model

model.save(os.getcwd()+"\\myKerasModel")

#save model

onnxModel = k2o.convert_keras(model, 'myOnnxModel', target_opset=7)
k2o.save_model(onnxModel, os.getcwd()+"\\myModel.onnx")
