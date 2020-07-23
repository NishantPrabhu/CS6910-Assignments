
# ================================================ #
# MACHINE TRANSLATION WITH LSTM AND ATTENTION      #
#                                                  #
# Assignment 4 | Task 4                            #
# Submitted by Group 19                            #
# ================================================ #

# Dependencies
import os
import re
import string
import pickle
import argparse
import numpy as np
from tqdm import tqdm

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding


def preprocess_line(line, english=True):
    """ Cleans a line of punctuation, numbers, special
    characters and converts the line to lowercase """

    line = line.translate(str.maketrans("", "", puncts))
    line = re.sub("\u200b", " ", line)
    line = re.sub("\u200d", " ", line)
    line = re.sub("\d+", " ", line)
    line = line.lower()
    if not english:
        line = "startseq " + line + " endseq"
    line = " ".join(line.split())
    return line


def tokenize(lines):
    """ Fits a keras Tokenizer on lines """

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    vocab_size = len(tokenizer.word_index) + 1
    return tokenizer, vocab_size


def data_generator(input_seq, output_seq, batch_size):
    """ Generator object to feed batch_size sizes of data
    at every iteration. Consumes much less memory. """

    enc_in, dec_trg = [], []
    count = 0
    while True:
        for in_seq, out_seq in zip(input_seq, output_seq):
            enc_in.append(np.flip(in_seq))
            dec_trg.append(out_seq[1:])
            count += 1
            if count == batch_size:
                yield (np.array(enc_in), np.array(dec_trg))
                enc_in, dec_trg = [], []
                count = 0


class Encoder(tf.keras.Model):

    def __init__(self):
        super(Encoder, self).__init__()

        # Initialize layers
        self.embedding_layer = Embedding(
            en_vocab_size, embedding_dim, name="encoder_embedding",
            embeddings_initializer=embed_init
        )
        self.lstm_layer = LSTM(
            units=hidden_dim,
            return_state=True,
            return_sequences=True,
            recurrent_initializer='glorot_uniform',
            name="encoder_lstm"
        )

    def call(self, x):
        x = self.embedding_layer(x) # -> (batch_size, seq length, embedding_dim)
        enc_out, enc_h, enc_c = self.lstm_layer(x)
        # enc_out -> (batch_size, seq length, hidden_dim)
        # enc_h -> (batch_size, hidden_dim)
        # enc_c -> (batch_size, hidden_dim)
        enc_states = [enc_h, enc_c]

        return enc_out, enc_states


class Attention(tf.keras.layers.Layer):
    """ Scaled dot product attention """

    def __init__(self):
        super(Attention, self).__init__()
        pass

    def call(self, states, enc_out):

        dec_h = tf.expand_dims(states[0], 1)
        # dec_h -> query (batch_size, 1, 256), enc_out -> values (batch_size, maxlen_en, 256)

        align_score = tf.matmul(enc_out, dec_h, transpose_b=True)
        align_score /= hidden_dim
        # align_score => (batch_size, maxlen_en, 1)

        weights = tf.nn.softmax(align_score, 1)
        # weights => (batch_size, maxlen_en, 1)

        context_vector = tf.reduce_sum(enc_out * weights, 1)
        # context_vector => (batch_size, 256)

        context_vector = tf.expand_dims(context_vector, 1)
        # context_vector => (batch_size, 1, 256)

        return context_vector, weights


class Decoder(tf.keras.Model):

    def __init__(self):
        super(Decoder, self).__init__()

        # Intialize layers
        self.embedding_layer = Embedding(
            hi_vocab_size, embedding_dim, name="decoder_embedding"
        )
        self.lstm_layer = LSTM(
            units=hidden_dim,
            return_sequences=True,
            return_state=True,
            name="decoder_lstm"
        )
        self.fc_layer = Dense(
            units=hi_vocab_size,
            activation="softmax",
            name="decoder_output"
        )
        self.attention_layer = Attention()

    def call(self, decoder_inputs):

        x, enc_out, init_states = decoder_inputs

        x = self.embedding_layer(x) # -> (batch_size, seq length, embedding dim)
        dec_out, dec_h, dec_c = self.lstm_layer(
            x, initial_state=init_states
        )
        # dec_out -> (batch_size, seq length, hidden_dim)
        # dec_h -> (batch_size, hidden_dim)
        # dec_c -> (batch_size, hidden_dim)
        init_states = [dec_h, dec_c]

        context_vector, weights = self.attention_layer(
            init_states, enc_out
        )
        # context_vector -> (batch_size, 1, hidden_dim
        # weights -> (batch_size, out_seq_length, in_seq_length)

        dec_out = tf.concat([dec_out, context_vector], axis=-1)
        # dec_out -> (batch_size, out_vocab_size)

        out = tf.reshape(dec_out, (-1, dec_out.shape[2]))
        out = self.fc_layer(out)

        return out, init_states, weights


def get_embedding_map(file_path):
    """ Loads glove embeddings from a file and
    generates embedding matrix for embedding layer"""

    print("\n [INFO] Processing GloVe embeddings... \n")

    with open(file_path, "r") as f:
        lines = f.read().split("\n")

    embedding_map = {}
    # Extract word and vector from every line and ...
    # ... store it in a dictionary
    for line in tqdm(lines):
        try:
            word, vec = line.split()[0], \
                np.array(line.split()[1:]).astype("float")
        except:
            continue

        embedding_map[word] = vec

    return embedding_map


def embed_init(shape, dtype=None):
    """ GloVe embeddings initializer for english embeddings """

    # This function definition is specific to tensorflow

    # Generate embedding weights in matrix form
    # Initialize a uniformly distributed tensor of size (vocab_size, embed_dim)
    # For words available in GloVe vocabulary, replace the vectors
    # Others will remain random to start with and tune with the model
    embedding_matrix = np.random.uniform(0, 1, shape)
    tokenizer_vocab = np.array(list(en_tokenizer.word_index.keys()))

    for word in tokenizer_vocab:
        if word in embedding_map.keys():
            idx = np.where(word == tokenizer_vocab)[0]
            embedding_matrix[idx] = embedding_map[word]

    return embedding_matrix


def compute_loss(real, pred):
    """ Loss function to update the model """

    loss = tf.keras.losses.CategoricalCrossentropy()
    return loss(real, pred)


def learning_step(enc_in, dec_trg):
    """
        Given a training batch of samples, this function performs
        predictions, computes loss and updates the model
    """

    loss = 0
    with tf.GradientTape() as tape:

        # Get representation of input sequence
        enc_out, enc_states = encoder_model(enc_in)

        # Initialize decoder states and input
        dec_in = tf.expand_dims(
            [hi_tokenizer.word_index["startseq"]] * batch_size, 1
        )
        dec_states = enc_states

        for t in range(1, maxlen_hi-1):
            # Generate predictions
            preds, dec_states, _ = decoder_model(
                [dec_in, enc_out, dec_states]
            )
            # Compute loss
            loss += compute_loss(
                to_categorical(dec_trg[:, t], num_classes=hi_vocab_size),
                preds
            )
            # Teacher forcing
            dec_in = tf.expand_dims(dec_trg[:, t], 1)

        batch_loss = loss.numpy() / int(dec_trg.shape[1])

        # Collect trainable variables
        variables = (
            encoder_model.trainable_variables + decoder_model.trainable_variables
        )

        # Compute gradients
        grads = tape.gradient(loss, variables)

        # Update the model
        optimizer.apply_gradients(zip(grads, variables))
        return batch_loss


# MAIN SCRIPT

if __name__ == "__main__":

    # Accept arguments from argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--FileRoot", type=str, required=True, help='root directory')
    parser.add_argument("-s", "--SaveRoot", type=str, required=True, help='root directory for saving models')
    parser.add_argument("-g", "--GlovePath", type=str, required=True, help='path to glove embeddings text file')
    parser.add_argument("-m", "--EmbedDim", type=int, required=True, default=200, help='embedding dim')
    parser.add_argument("-d", "--HiddenDim", type=int, required=True, default=256, help='lstm hidden dim')
    parser.add_argument("-n", "--NumSamples", type=int, required=True, default=30000, help='number of samples to train on')
    parser.add_argument("-e", "--Epochs", type=int, required=True, default=30, help='number of training epochs')
    parser.add_argument("-b", "--BatchSize", type=int, required=True, default=64, help='training batch size')
    args = vars(parser.parse_args())

    # Constants
    embedding_dim = args["EmbedDim"]
    hidden_dim = args["HiddenDim"]

    # Load files
    root = args["FileRoot"]
    save_root = args["SaveRoot"]

    with open(root + "/train.en", "r") as f:
        train_en_raw = f.read().split("\n")

    with open(root + "/train.hi", "r") as f:
        train_hi_raw = f.read().split("\n")

    with open(root + "/dev.en", "r") as f:
        dev_en_raw = f.read().split("\n")

    with open(root + "/dev.hi", "r") as f:
        dev_hi_raw = f.read().split("\n")

    # Clean the text
    puncts = string.punctuation + train_hi_raw[3][-1]

    # Obtain cleaned corpus
    train_en_ = [preprocess_line(line, True) for line in train_en_raw]
    train_hi_ = [preprocess_line(line, False) for line in train_hi_raw]
    dev_en = [preprocess_line(line, True) for line in dev_en_raw]
    dev_hi = [preprocess_line(line, False) for line in dev_hi_raw]

    # Select only those lines for training which have less than 10 words
    train_en, train_hi = [], []

    for i in range(len(train_en_)):
        if (len(train_en_[i].split()) <= 30) & (len(train_hi_[i].split()) <= 30):
            train_en.append(train_en_[i])
            train_hi.append(train_hi_[i])

    # Choose the first NumSamples lines to reduce training time
    SIZE = min(args["NumSamples"], len(train_en))
    train_en = train_en[:SIZE]
    train_hi = train_hi[:SIZE]

    # Tokenize lines
    en_tokenizer, en_vocab_size = tokenize(train_en + dev_en)
    hi_tokenizer, hi_vocab_size = tokenize(train_hi + dev_hi)

    # Save tokenizers
    with open(save_root + "/en_tokenizer.pkl", "wb") as f:
        pickle.dump(en_tokenizer, f)

    with open(save_root + "/hi_tokenizer.pkl", "wb") as f:
        pickle.dump(hi_tokenizer, f)

    # Convert texts to sequences and pad, find max lengths
    train_en_seq = en_tokenizer.texts_to_sequences(train_en)
    train_hi_seq = hi_tokenizer.texts_to_sequences(train_hi)

    train_en_seq = pad_sequences(train_en_seq, padding="post")
    train_hi_seq = pad_sequences(train_hi_seq, padding="post")

    maxlen_en = train_en_seq.shape[1]
    maxlen_hi = train_hi_seq.shape[1]

    # GloVe Embeddings
    embedding_map = get_embedding_map(args["GlovePath"])

    # Create models and optimizer
    encoder_model = Encoder()
    decoder_model = Decoder()
    optimizer = tf.keras.optimizers.Adam()

    # Training
    epochs = args["Epochs"]
    batch_size = args["BatchSize"]
    save_root = args["SaveRoot"]
    steps_per_epoch = len(train_en_seq) // batch_size
    training_loss = []

    for epoch in range(epochs):

        print("\nEpoch {}".format(epoch+1))
        print("-------------------------------------------------")

        # Initialize data generator and loss counter
        data_gen = data_generator(train_en_seq, train_hi_seq, batch_size)
        total_loss = 0

        for step in range(steps_per_epoch):
            # Generate batch of training examples
            enc_input, dec_target = next(data_gen)
            # Perform learning step and record loss
            mean_loss = learning_step(
                enc_input, dec_target
            )
            total_loss += mean_loss

            if step % 100 == 0:
                print("Epoch {} - Batch {} - Loss: {:.4f}".format(
                    epoch+1, step+1, mean_loss
                ))

        print("\nEpoch {} - Average loss: {:.4f}\n".format(
            epoch+1, total_loss/steps_per_epoch
        ))

        training_loss.append(total_loss/steps_per_epoch)

        # Save model
        encoder_model.save(save_root + "/models/encoder_{}".format(epoch+1))
        decoder_model.save(save_root + "/models/decoder_{}".format(epoch+1))

        print("\n=====================================================")


    # Save training loss
    with open(save_root + "/training_loss.pkl", "wb") as f:
        pickle.dump(training_loss, f)
