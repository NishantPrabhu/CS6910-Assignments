
# Dependencies

import string
import pickle
import argparse
import numpy as np
from math import sqrt
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Text processing

class PreprocessText(object):

    def __init__(self, punctuation, word_limit):
        self.puncts = punctuation
        self.w_limit = word_limit


    def _word_limit_filter(self, in_sents, out_sents):
        """ Preserves sentences shorter in length than the word limit """

        train_x, train_y = [], []
        for in_sent, out_sent in zip(in_sents, out_sents):
            if len(in_sent.split()) <= 30 & len(out_sent.split()) <= 30:
                train_x.append(in_sent)
                train_y.append(out_sent)
        return train_x, train_y


    def _preprocess_line(self, line, english=True):
        """ Cleans a line of punctuation, numbers, special
        characters and converts the line to lowercase """

        line = line.translate(str.maketrans("", "", self.puncts))
        words = line.split()
        words = [w.lower() for w in words]
        words = [w for w in words if w not in ['\u200b', '\u200d']]
        words = [w for w in words if not w.isnumeric()]
        words = ['startseq'] + words + ['endseq']

        return " ".join(words)


    def _tokenize_sentences(self, sentences):
        """ Fits a tokenizer to the corpus and computes vocabulary size """

        tok = Tokenizer(filters='')
        tok.fit_on_texts(sentences)
        vocab_size = len(tok.word_index) + 1
        return tok, vocab_size


    def _generate_sequences(self, tokenizer, sentences):
        """ Generate sequences of integer tokens for each sentence """

        tokens = tokenizer.texts_to_sequences(sentences)
        tokens = pad_sequences(tokens, padding='post')
        return tokens


    def flow(self, train_input, train_output, dev_input, dev_output):
        """ Calls the above functions in sequence and returns the sequence
        of integer tokens, tokenizers, vocabulary sizes and cleaned lines
        for input and output language """

        train_input = [self._preprocess_line(line) for line in train_input]
        train_output = [self._preprocess_line(line) for line in train_output]
        dev_input = [self._preprocess_line(line) for line in dev_input]
        dev_output = [self._preprocess_line(line) for line in dev_output]

        if self.w_limit is not None:
            train_input, train_output = self._word_limit_filter(train_input, train_output)

        in_tokenizer, in_vocab_size = self._tokenize_sentences(train_input + dev_input)
        out_tokenizer, out_vocab_size = self._tokenize_sentences(train_output + dev_output)

        in_tokens = self._generate_sequences(in_tokenizer, train_input)
        out_tokens = self._generate_sequences(out_tokenizer, train_output)

        tokens = (in_tokens, out_tokens)
        tokenizers = (in_tokenizer, out_tokenizer)
        vocab_sizes = (in_vocab_size, out_vocab_size)
        lines = (train_input, train_output)

        return tokens, tokenizers, vocab_sizes, lines


# Positional encoding

class PositionalEncoder(object):

    def __init__(self, d_model):
        self.d = d_model


    def _compute_for_position(self, pos):
        """ Computes the positional encoding vector for given
            position in a sequence
        """
        exp = np.array([2*i if i % 2 == 0 else 2*(i-1) for i in range(self.d)])/float(self.d)
        values = []

        for i in range(self.d):
            # For even positions
            if i % 2 == 0:
                values.append(np.sin((pos+1) / (10000 ** exp[i])))
            # For odd positions
            else:
                values.append(np.cos((pos+1) / (10000 ** exp[i])))

        return values


    def compute(self, size):
        """ Generates a tensor of positional encoding for all positions
        of a sequence using the function above """

        encodings = []
        for t in range(size):
            values = self._compute_for_position(t)
            encodings.append(values)

        return torch.FloatTensor(encodings).unsqueeze(0).to(device)


# Masking

def create_padding_mask(x):
    """ Returns 1 for positions where a padding zero is present, 0
    for all other locations """

    mask = x.eq(0).float().unsqueeze(1).unsqueeze(1)
    return mask.to(device)


def create_future_mask(size):
    """ Returns an upper triangular matrix of ones whose diagonal
    elements are all 0 """

    mask = 1. - torch.tril(torch.ones((size, size)))
    return mask.to(device)


def create_masks(enc_in, dec_trg):
    """ Given input and outputs for the model, returns all three masks
    needed for the encoder and deocder """

    # Encoder padding mask
    enc_pad_mask = create_padding_mask(enc_in)

    # Decoder padding mask for unmasked mha
    dec_pad_mask = create_padding_mask(enc_in)

    # Decoder future mask for masked mha
    dec_lookahead_mask = create_future_mask(dec_trg.shape[1])
    dec_trg_pad_mask = create_padding_mask(dec_trg)
    dec_lookahead_mask = torch.max(dec_lookahead_mask, dec_trg_pad_mask)
    # This is done to hide the padding and the future tokens together

    return enc_pad_mask, dec_lookahead_mask, dec_pad_mask


# Core layers

class SelfAttention(torch.nn.Module):

    def __init__(self, depth):
        super(SelfAttention, self).__init__()
        self.depth = depth

    def forward(self, q, k, v, mask):
        """ Performs scaled dot product attention """

        align_score = torch.matmul(q, torch.transpose(k, -1, -2))

        if mask is not None:
            # All locations where 1 is returned by the mask
            # become -infinity (close to it)
            # so no information is drawn from them
            align_score += mask * -1e09

        attention_weights = F.softmax(align_score / sqrt(d_model), dim=-1)
        z_score = torch.matmul(attention_weights, v)

        return z_score, attention_weights


class MultiheadAttention(torch.nn.Module):

    def __init__(self, n_heads, depth):
        super(MultiheadAttention, self).__init__()

        self.n_heads = n_heads
        self.depth = depth
        self.dim = d_model // n_heads
        self.W_q = torch.nn.Linear(depth, d_model)
        self.W_k = torch.nn.Linear(depth, d_model)
        self.W_v = torch.nn.Linear(depth, d_model)
        self.W_o = torch.nn.Linear(d_model, depth)
        self.attention_layer = SelfAttention(d_model)


    def split_heads(self, x, batch_size):
        """ Adds an attention head dimension to the input tensor;
        New dimension is added at position 1 """

        x = torch.reshape(x, (batch_size, -1, self.n_heads, self.dim))
        return torch.transpose(x, 1, 2)


    def forward(self, q, k, v, mask):

        batch_size = q.shape[0]

        # Take in the Q, K, V tensors with d_model features
        q = self.W_q(q)  # -> (batch_size, q_len, d_model)
        k = self.W_k(k)  # -> (batch_size, k_len, d_model)
        v = self.W_v(v)  # -> (batch_size, v_len, d_model)

        # Split them across n attention heads
        q = self.split_heads(q, batch_size) # -> (batch_size, n_heads, q_len, dim)
        k = self.split_heads(k, batch_size) # -> (batch_size, n_heads, k_len, dim)
        v = self.split_heads(v, batch_size) # -> (batch_size, n_heads, v_len, dim)

        # Compute z score and self attention weights
        z_score, attention_weights = self.attention_layer(q, k, v, mask)
        # z_score -> (batch_size, n_heads, q_len, dim)
        # attention_weights -> (batch_size, n_heads, q_len, k_len)

        z_score = z_score.permute([0, 2, 1, 3]) # -> (batch_size, q_len, n_heads, dim)
        z_score = torch.reshape(z_score, (batch_size, -1, d_model))

        # Bring the tensor back to d_model dimensions
        out = self.W_o(z_score)

        return out, attention_weights


class Feedforward(torch.nn.Module):
    """ Feedforward layer for encoder and decoder layers """

    def __init__(self, depth, hidden_depth):
        super(Feedforward, self).__init__()
        self.fc_1 = torch.nn.Linear(depth, hidden_depth)
        self.fc_2 = torch.nn.Linear(hidden_depth, depth)

    def forward(self, x):
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        return x


class AddNormalize(torch.nn.Module):
    """ Addition and normalization layer for encoder and
        decoder layers
    """

    def __init__(self):
        super(AddNormalize, self).__init__()
        pass

    def _normalize(self, x):
        """ Computes mean and std of input tensor along
        feature dimension """

        mean = x.mean(dim=-1).unsqueeze(2)
        std = x.std(dim=-1).unsqueeze(2)
        return (x - mean) / std


    def forward(self, x, y):
        """ Add inputs and normalize """

        return self._normalize(x + y)


# Encoder and Decoder layers

class Encoder(torch.nn.Module):

    def __init__(self, n_heads, depth):
        super(Encoder, self).__init__()

        self.add_norm_1 = AddNormalize()
        self.add_norm_2 = AddNormalize()
        self.mha = MultiheadAttention(n_heads, depth)
        self.feedforward = Feedforward(depth, int(0.5*depth))

    def forward(self, x, pad_mask):
        """ Forward pass through encoder layer as defined in the paper """

        mha_out, mha_weights = self.mha(x, x, x, pad_mask)
        add_norm_out = self.add_norm_1(x, mha_out)
        ff_out = self.feedforward(add_norm_out)
        enc_out = self.add_norm_2(ff_out, add_norm_out)

        return enc_out, mha_weights


class Decoder(torch.nn.Module):

    def __init__(self, n_heads, depth):
        super(Decoder, self).__init__()

        self.add_norm_1 = AddNormalize()
        self.add_norm_2 = AddNormalize()
        self.add_norm_3 = AddNormalize()
        self.masked_mha = MultiheadAttention(n_heads, depth)
        self.mha = MultiheadAttention(n_heads, depth)
        self.feedforward = Feedforward(depth, int(0.5*depth))

    def forward(self, x, enc_out, pad_mask, future_mask):
        """ Forward pass through the decoder layer as defined in the paper """

        masked_mha_out, masked_mha_weights = self.masked_mha(x, x, x, future_mask)
        add_norm1_out = self.add_norm_1(x, masked_mha_out)

        mha_out, mha_weights = self.mha(add_norm1_out, enc_out, enc_out, pad_mask)
        add_norm2_out = self.add_norm_2(mha_out, add_norm1_out)
        ff_out = self.feedforward(add_norm2_out)
        dec_out = self.add_norm_3(ff_out, add_norm2_out)

        return dec_out, masked_mha_weights, mha_weights


# Transformer model

class Transformer(torch.nn.Module):
    """ Combining all the layers defined above into the Transformer """

    def __init__(self, n_encoders, n_decoders, n_heads, glove_weights):
        super(Transformer, self).__init__()
        self.n_encoders = n_encoders
        self.n_decoders = n_decoders

        self.enc_embedding = torch.nn.Embedding(en_vocab_size, embedding_dim)
        self.dec_embedding = torch.nn.Embedding(hi_vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoder(embedding_dim)
        self.encoders = nn.ModuleList([Encoder(n_heads, embedding_dim) for _ in range(n_encoders)])
        self.decoders = nn.ModuleList([Decoder(n_heads, embedding_dim) for _ in range(n_decoders)])
        self.output_fc = torch.nn.Linear(embedding_dim, hi_vocab_size)

        # Add glove embedding to encoder
        self.enc_embedding.weight = torch.nn.Parameter(glove_weights)


    def forward(self, enc_in, dec_in):

        # Generate embeddings and add positional encoding
        # Create masks
        enc_pad_mask, dec_lookahead_mask, dec_pad_mask = create_masks(enc_in, dec_in)
        enc_embed = self.enc_embedding(enc_in) + self.pos_encoder.compute(enc_in.shape[1])
        dec_embed = self.dec_embedding(dec_in) + self.pos_encoder.compute(dec_in.shape[1])

        # Encoder stack pass
        x_enc = enc_embed
        enc_weights = {}

        # Pass the tensor sequentially through each encoder layer
        for i in range(self.n_encoders):
            x_enc, mha_weights = self.encoders[i](x_enc, enc_pad_mask)
            enc_weights.update({f"{i+1}": mha_weights})

        enc_out = x_enc

        # Decoder stack pass
        x_dec = dec_embed
        dec_masked_weights = {}
        dec_weights = {}

        # Pass the tensor sequentially through each decoder layer
        for i in range(self.n_decoders):
            x_dec, masked_mha_weights, mha_weights = \
                self.decoders[i](x_dec, enc_out, dec_pad_mask, dec_lookahead_mask)
            dec_masked_weights.update({f"{i+1}": masked_mha_weights})
            dec_weights.update({f"{i+1}": mha_weights})

        dec_out = x_dec

        # Generate predictions with log softmax activation
        # This is so that we can use nonlinear logloss as loss function
        preds = self.output_fc(dec_out)
        preds = F.log_softmax(preds, dim=-1)

        return preds, enc_weights, dec_masked_weights, dec_weights


# Helper functions

def get_embedding_weights(file_path):
    """ Loads glove embeddings from a file and
    generates embedding matrix for embedding layer"""

    print("\n[INFO] Processing GloVe embeddings... \n")

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

    # Generate embedding weights in matrix form
    # Initialize a uniformly distributed tensor of size (vocab_size, embed_dim)
    # For words available in GloVe vocabulary, replace the vectors
    # Others will remain random to start with and tune with the model
    embedding_matrix = np.random.uniform(0, 1, (en_vocab_size, args['EmbedDim']))
    tokenizer_vocab = np.array(list(en_tokenizer.word_index.keys()))

    for word in tokenizer_vocab:
        if word in embedding_map.keys():
            idx = np.where(word == tokenizer_vocab)[0]
            embedding_matrix[idx] = embedding_map[word]

    return torch.FloatTensor(embedding_matrix)


def data_generator(inputs, outputs, batch_size):
    """ Generator object to feed batch_size sizes of data
    at every iteration. Consumes much less memory. """

    enc_in, dec_in, dec_trg = [], [], []
    count = 0

    while True:
        for in_seq, out_seq in zip(inputs, outputs):
            enc_in.append(in_seq)
            dec_in.append(out_seq[:-1])
            dec_trg.append(out_seq[1:])
            count += 1

            if count == batch_size:
                yield (
                    torch.LongTensor(enc_in),
                    torch.LongTensor(dec_in),
                    torch.LongTensor(dec_trg)
                )
                enc_in, dec_in, dec_trg = [], [], []
                count = 0


def compute_loss(pred, real):
    """ Computes loss for each example in batch and returns
    the aggregate """

    total_loss = 0
    for y_pred, y_true in zip(pred, real):
        loss = F.nll_loss(y_pred, y_true, reduction='mean')
        total_loss += loss
    return total_loss


def learning_step(enc_in, dec_in, dec_trg):
    """ Updates model with given batch of examples """

    # Zero out optimizer gradients
    optimizer.zero_grad()
    # Generate predictions, ignore the weights
    preds, _, _, _ = model(enc_in, dec_in)
    # Compute loss
    loss = compute_loss(preds, dec_trg)
    # Backpropagate loss and compute gradients
    loss.backward()
    # Update the model
    optimizer.step()

    return loss.item()/len(enc_in)


# Main script

if __name__ == "__main__":

    device = torch.device("cuda")

    # Accept arguments from argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--FileRoot", type=str, required=True, help='root directory')
    parser.add_argument("-s", "--SaveRoot", type=str, required=True, help='root directory for saving models')
    parser.add_argument("-g", "--GlovePath", type=str, required=True, help='path to glove embeddings text file')
    parser.add_argument("-m", "--EmbedDim", type=int, required=True, default=200, help='embedding dim')
    parser.add_argument("-d", "--DModel", type=int, required=True, default=256, help='lstm hidden dim')
    parser.add_argument("-e", "--Epochs", type=int, required=True, default=20, help='number of training epochs')
    parser.add_argument("-b", "--BatchSize", type=int, required=True, default=32, help='training batch size')
    args = vars(parser.parse_args())

    file_root = args["FileRoot"]
    save_root = args["SaveRoot"]

    # Load files
    with open(file_root + "/train.en", 'r') as f:
        train_en = f.read().split("\n")

    with open(file_root + "/train.hi", 'r') as f:
        train_hi = f.read().split("\n")

    with open(file_root + "/dev.en", "r") as f:
        dev_en = f.read().split("\n")

    with open(file_root + "/dev.hi", "r") as f:
        dev_hi = f.read().split("\n")


    # Preprocess texts
    puncts = string.punctuation + '।'
    pre = PreprocessText(puncts, None)

    # Generate data for input / output
    tokens, tokenizers, vocab_sizes, lines = pre.flow(train_en, train_hi, dev_en, dev_hi)
    train_en_seq, train_hi_seq = tokens
    en_tokenizer, hi_tokenizer = tokenizers
    en_vocab_size, hi_vocab_size = vocab_sizes
    en_lines, hi_lines = lines


    with open(save_root + "/en_tokenizer.pkl", "wb") as f:
        pickle.dump(en_tokenizer, f)

    with open(save_root + "/hi_tokenzier.pkl", "wb") as f:
        pickle.dump(hi_tokenizer, f)

    # Glove embeddings
    embedding_weights = get_embedding_weights(args["GlovePath"])

    # Training loop
    embedding_dim = args["EmbedDim"]
    d_model = args["DModel"]

    epochs = args["Epochs"]
    batch_size = args["BatchSize"]
    steps_per_epoch = len(train_en_seq) // batch_size

    # Define the model and optimizer
    model = Transformer(n_encoders=2, n_decoders=2, n_heads=4,
                        glove_weights=embedding_weights).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    model.train()

    training_loss = []

    print("\n[INFO] Beginning training...\n")

    for epoch in range(epochs):

        print("Epoch {}".format(epoch+1))
        print("--------------------------------------------")

        # Initialize data generator and loss counter
        data_gen = data_generator(train_en_seq, train_hi_seq, batch_size)
        epoch_loss = 0

        for batch in range(steps_per_epoch):
            # Generate batch of training samples
            enc_in, dec_in, dec_trg = next(data_gen)

            enc_in = enc_in.to(device)
            dec_in = dec_in.to(device)
            dec_trg = dec_trg.to(device)

            # Perform learning step and record loss
            batch_loss = learning_step(enc_in, dec_in, dec_trg)
            epoch_loss += batch_loss

            if batch % 500 == 0:
                print("Epoch {} - Batch {} - Loss {:.4f}".format(
                    epoch+1, batch, batch_loss
                ))

        # Epoch statistics
        average_loss = epoch_loss / steps_per_epoch
        print("Epoch {} - Average loss {:.4f}".format(
            epoch+1, average_loss
        ))

        training_loss.append(average_loss)

        # Save model
        torch.save(model.state_dict(), save_root + f"/models/model_{epoch+1}")

        print("\n==========================================\n")

    # Save training loss
    with open(save_root + "/training_loss.pkl", "wb") as f:
        pickle.dump(training_loss, f)
