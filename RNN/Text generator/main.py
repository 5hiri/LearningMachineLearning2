import numpy as np
import tensorflow as tf
import os

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, GRU

os.chdir(os.path.dirname(__file__))

#Load text data
text = open('training_data_1.txt', 'r').read()

#Character level tokenization
chars = sorted(set(text))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = np.array(chars)

#convert text ot numerical indices
text_as_int = np.array([char_to_idx[c] for c in text])

#Create sequences
seq_length = 100
examples_per_epoch = len(text) // seq_length
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

#spliit input and target
def split_input_target(chunk):
    input_text = chunk[:-1]
    input_text = tf.expand_dims(input_text, 0)
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

#Build the RNN Model
#Define RNN Model
vocab_size = len(chars)
embedding_dim = 256
rnn_units = 1024

model = Sequential([
    Embedding(vocab_size, embedding_dim, batch_input_shape=[None, None]),
    LSTM(rnn_units, return_sequences=True, stateful=False),
    Dense(vocab_size)
])

model.summary()

#Train model
model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True))
#training
epochs = 20
history = model.fit(dataset, epochs=epochs)

#Generate text
def generate_text(model, start_string, num_generate=1000):
    input_eval = [char_to_idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    model.reset_state()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx_to_char[predicted_id])

    return (start_string + ''.join(text_generated))

print(generate_text(model, start_string="ALIENS: "))