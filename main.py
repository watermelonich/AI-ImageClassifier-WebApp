import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import tensorflow as tf
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Flatten, Dense

# Load CIFAR-10 dataset
(X_train, y_train), (X_val, y_val) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
X_train = X_train / 255.0
X_val = X_val / 255.0

# One-hot encode the target labels
y_train = to_categorical(y_train, 10)
y_val = to_categorical(y_val, 10)

# Build the model
model = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(1000, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_val, y_val))

# Save the trained model
model.save('cifar10_model.h5')
