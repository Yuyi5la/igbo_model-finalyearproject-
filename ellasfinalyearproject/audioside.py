import os
import numpy as np
import tensorflow as tf
import keras
from tensorflow.python.keras import Sequential
from keras.layers import LSTM, Dense ,Flatten, Conv2D, MaxPooling2D
from keras.layers import TimeDistributed, Bidirectional
from keras.preprocessing.sequence import pad_sequences
import scipy
from scipy.io import wavfile
from python_speech_features import mfcc
import librosa

# Install python_speech_features library if you don't have it already
# pip install python_speech_features
# pip install scipy

# Define the dataset path
dataset_path =  'C:\\Users\\HP\\Documents\\seset'

# Function to extract MFCC features from audio files
def extract_features(audio_path):
    rate, audio = wavfile.read(audio_path)
    mfcc_features = mfcc(audio, rate)
    return mfcc_features

# Read the dataset and split it into audio paths and responses

audio_data_list = []
sampling_rate_list = []

with open('C:\\Users\\HP\\Documents\\seset', 'r', encoding='utf-8') as file:
    lines = file.read().split('\n')
    for line in lines:
        audio_file_path = line.strip()
        if audio_file_path:  # Check if the line is not empty
            audio_data, sampling_rate = librosa.load(audio_file_path, sr=None)
            audio_data_list.append(audio_data)
            sampling_rate_list.append(sampling_rate)

# Extract features from audio files
audio_features = [extract_features(audio_path) for audio_path in audio_data_list]

# Pad sequences to ensure they have the same length
padded_audio_features = pad_sequences(audio_features, padding='post', truncating='post', dtype='float32')

# Define the LSTM model for audio
model = keras.Sequential()
model.add(Bidirectional(LSTM(units=64, return_sequences=True), input_shape=(None, padded_audio_features.shape[2])))
model.add(TimeDistributed(Dense(units=64, activation='relu')))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=len(audio_features) + 1, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(padded_audio_features,  epochs=10, batch_size=32)

# Save the model for later use
model.save('audio_model.h5')
