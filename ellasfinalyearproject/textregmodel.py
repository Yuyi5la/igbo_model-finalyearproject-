import os
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
import tensorflow as tf
import keras
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

dataset = 'c:\\Users\\HP\\Documents\\DS_FILES\\datasets.txt'

# Read the dataset and split it into Igbo text and responses
igbo_texts = []
responses = []

with open(dataset, 'r', encoding='utf-8') as file:
    lines = file.readlines()

for line in lines:
    parts = line.strip().split('|')
    if len(parts) == 2:
        igbo_texts.append(parts[0].strip())
        responses.append(parts[1].strip())

# Preprocess text data
def preprocess_text(text):
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Convert text to lowercase
    text = text.lower()
    return text

# Tokenize text using Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(igbo_texts + responses)
igbo_sequences = tokenizer.texts_to_sequences(igbo_texts)
response_sequences = tokenizer.texts_to_sequences(responses)

# Pad sequences to ensure they have the same length
max_seq_length = max(len(seq) for seq in igbo_sequences + response_sequences)
padded_igbo_sequences = pad_sequences(igbo_sequences, maxlen=max_seq_length, padding='post')
padded_response_sequences = pad_sequences(response_sequences, maxlen=max_seq_length, padding='post')

# Define vocabulary size
vocab_size = len(tokenizer.word_index) + 1

# Define the LSTM model
model = keras.Sequential([
    keras.layers.Embedding(input_dim=vocab_size, output_dim=64, input_length=max_seq_length),
    keras.layers.LSTM(units=64, return_sequences=True),
    keras.layers.Dense(units=vocab_size, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(padded_igbo_sequences, padded_response_sequences, epochs=20, batch_size=32)

# Save the model for later use
model.save('igbo_model.h5')


#model.load_weights('igbo_model.h5')
#model= tf.keras.models.load_model('igbo_model.h5')

