#!/usr/bin/env python

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images.shape
len(train_labels)
train_labels

test_images.shape

len(test_labels)

train_images = train_images / 255.0

test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

scores = model.evaluate(test_images,  test_labels, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
model.save("New_model.h5")

file1 = open("results.txt","w")
file1.write(str(scores[1]*100))


if (scores[1]*100) < 88:
    print("Not great accuracy")
    
else:
    print("Best model created!")
    print("Accuracy is :",scores[1]*100)

