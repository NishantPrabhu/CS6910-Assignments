
# ================================================ #
# IMAGE CAPTIONING WITH RNN                        #
#                                                  #
# Assignment 4 | Task 1                            #
# Submitted by Group 19                            #
# ================================================ #

# Dependencies
import os
import pickle
import string
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def extract_features(img_dir, load_path=None):
    """ Extracts VGG features for all images in a dir """

    if load_path is None:

        # Transforms
        img_transform = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Create a VGG model and remove the top layer
        # Returns a feature vector of size (4096,)
        vgg = models.vgg16(pretrained=True)
        layers = list(vgg.children())[:-1]
        model = torch.nn.Sequential(*layers).to(device)
        features = {}

        for name in tqdm(os.listdir(img_dir), leave=False):

            # Load image from path
            path = img_dir + "/" + name
            image = Image.open(path)
            image = img_transform(image)

            # Reshape to size (batch_size, height, width)
            image = image.unsqueeze(0).to(device)

            # Extract features
            fs = model(image)
            features[name] = fs

        return features

    else:

        with open(load_path, "rb") as f:
            features = pickle.load(f)

        return features


def get_caption_map(captions):
    """ Maps all captions of an image to path of image """

    mapping = {}
    for line in captions:
        if len(line) < 2:
            continue
        image_id, image_desc = line.split("#")[0], line.split()[1:]
        image_desc = " ".join(image_desc)

        # If this is a new image, add its path to the dict
        # And initialize list of captions
        if image_id not in mapping.keys():
            mapping[image_id] = []

        # If key is present, append description to its list
        mapping[image_id].append(image_desc)

    return mapping


def clean_descriptions(descriptions):
    """ Performs text cleaning, like removing punctuation,
    numbers and special characters which do not carry meaning """

    table = str.maketrans("", "", string.punctuation)
    for _, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            # Split into individual words
            desc = desc_list[i].split()
            # Lowercase all words, numbers are kept as is
            desc = [w.lower() for w in desc]
            # Remove punctuation with translation table
            desc = [w.translate(table) for w in desc]
            # Removing hanging words like "a", "s", etc.
            desc = [w for w in desc if len(w) > 1]
            # Remove characters that are not alphabets
            desc = [w for w in desc if w.isalpha()]
            # Append starting and ending tokens
            desc_list[i] = "startseq " + " ".join(desc) + " endseq"


def get_vocabulary(descriptions):
    """ Generates a vocabulary given all
    dictionary map of file descriptions """

    # Using a set because it auto-handles duplicates
    all_words = set()

    for key in descriptions.keys():
        # Get all words from each sentence and update the set
        [all_words.update(d.split()) for d in descriptions[key]]

    return list(all_words)


def get_all_captions(descriptions):
    """ Generate a list of all captions """

    all_captions = []
    for l in descriptions.values():
        all_captions.extend(l)
    return all_captions


def get_max_length(descriptions):
    """ Length of longest description """

    all_captions = get_all_captions(descriptions)
    return max([len(c.split()) for c in all_captions])


def create_tokenizer(descriptions):
    """ Tokenizer for sentences """

    lines = get_all_captions(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    vocab_size = len(tokenizer.word_index) + 1
    return tokenizer, vocab_size


def data_generator(descriptions, image_features, batch_size):
    """ Generator object to feed batches of image
    features and description features. Memory efficient. """

    in_1, in_2, out = [], [], []
    count = 0

    while True:
        # For each image ...
        for key, desc_list in descriptions.items():
            image = image_features[key][0]

            # For each desc in list of descriptions ...
            for desc in desc_list:

                # Obtain integer sequence of desc
                seq = tokenizer.texts_to_sequences([desc])[0]

                # Starting from first word ...
                # Input is sequence until that word
                # Target is next word
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]

                    in_1.append(image)
                    in_2.append(in_seq)
                    out.append(out_seq)
                    count += 1

                    if count == batch_size:
                        yield (
                            torch.FloatTensor(in_1),
                            torch.LongTensor(in_2),
                            torch.LongTensor(out)
                        )
                        in_1, in_2, out = [], [], []
                        count = 0


class Network(torch.nn.Module):

    def __init__(self, depth, embedding_dim, glove_weights):
        super(Network, self).__init__()

        self.fc_1 = torch.nn.Linear(4096, depth)
        self.fc_2 = torch.nn.Linear(depth, depth)
        self.fc_3 = torch.nn.Linear(depth, vocab_size)
        self.dropout_1 = torch.nn.Dropout(0.5)
        self.dropout_2 = torch.nn.Dropout(0.5)
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.rnn = torch.nn.RNN(embedding_dim, depth, batch_first=True)

        # Initialize embedding weights to GloVe
        self.embedding.weight = torch.nn.Parameter(glove_weights)


    def forward(self, img_in, caption_in):

        x1 = self.dropout_1(img_in)         # (batch_size, 4096)
        x1 = self.fc_1(img_in)              # (batch_size, depth)
        x1 = F.relu(x1)

        x2 = self.embedding(caption_in)     # (batch_size, max_length, embedding_dim)
        x2 = self.dropout_2(x2)             # (batch_size, max_length, embedding_dim)
        x2, _ = self.rnn(x2)                # (batch_size, depth)
        x2 = x2[:, -1, :]                   # Pick only the last sequence output

        x3 = self.fc_2(x1 + x2)             # (batch_size, depth)
        x3 = F.relu(x3)
        out = self.fc_3(x3)                 # (batch_size, vocab_size)
        out = F.log_softmax(out, dim=-1)

        return out


def get_embedding_weights(file_path):
    """ Loads glove embeddings from a file and
    generates embedding matrix for embedding layer"""

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
    embedding_matrix = np.random.uniform(0, 1, (vocab_size, embedding_dim))
    tokenizer_vocab = np.array(list(tokenizer.word_index.keys()))

    for word in tokenizer_vocab:
        if word in embedding_map.keys():
            idx = np.where(word == tokenizer_vocab)[0]
            embedding_matrix[idx] = embedding_map[word]

    return torch.FloatTensor(embedding_matrix)


def loss_function(preds, real):
    loss = F.nll_loss(preds, real)
    return loss


def learning_step(img_in, cpt_in, cpt_out):
    # Zero out optimizers gradients
    optimizer.zero_grad()
    # Generate predictions
    preds = model(img_in, cpt_in)
    # Compute loss
    loss = loss_function(preds, cpt_out)
    # Compute gradients
    loss.backward()
    # Update model weights
    optimizer.step()

    return loss


# SCRIPT START

if __name__ == "__main__":

    # Constants and device specifications
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth = 256
    embedding_dim = 200

    # Extract features for images from file and save
    file_root = "../data/task1/flickr/"
    save_root = "../saved_data/image_captioning/rnn/"
    glove_path = "../glove/glove.6B/glove.6B.200d.txt"

    # Extract image features and store them in a dictionary
    features = extract_features(file_root + "flickr_images", load_path=save_root + "/image_features.pkl")

    with open(save_root + "image_features.pkl", "wb") as f:
        pickle.dump(features, f)

    # Obtain all captions
    with open(file_root + "flickr_captions.txt", "r") as f:
        captions = f.read().split("\n")

    # Generate a map of filename -> caption list ...
    # ... and generate its vocabulary
    caption_map = get_caption_map(captions)
    clean_descriptions(caption_map)
    vocab = get_vocabulary(caption_map)

    with open(save_root + "caption_map.pkl", "wb") as f:
        pickle.dump(caption_map, f)

    with open(save_root + "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    # Split data into training (0.7) and test (0.3) sets
    filenames = list(features.keys())
    split_point = int(0.7 * len(filenames))
    train_names, test_names = filenames[:split_point], filenames[split_point:]

    # Store training features/captions and testing features/captions ...
    # ... separate files
    train_features, train_captions = {}, {}
    test_features, test_captions = {}, {}

    for name in filenames:
        if name in train_names:
            train_features.update({name: features[name]})
            train_captions.update({name: caption_map[name]})
        else:
            test_features.update({name: features[name]})
            test_captions.update({name: caption_map[name]})

    # Save these for quick loading later
    with open(save_root + "train_features.pkl", "wb") as f:
        pickle.dump(train_features, f)

    with open(save_root + "train_captions.pkl", "wb") as f:
        pickle.dump(train_captions, f)

    with open(save_root + "test_features.pkl", "wb") as f:
        pickle.dump(test_features, f)

    with open(save_root + "test_captions.pkl", "wb") as f:
        pickle.dump(test_captions, f)

    # Tokenize captions
    tokenizer, vocab_size = create_tokenizer(train_captions)
    max_length = get_max_length(caption_map)

    with open(save_root + "tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    # Generate embedding map so the embedding layer can be initialized
    embedding_weights = get_embedding_weights(glove_path)

    # Training
    epochs = 100
    batch_size = 16

    # Training hyperparameters
    steps = len(train_captions) // batch_size
    model = Network(depth, embedding_dim, embedding_weights).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    training_loss = []

    print("\n[INFO] Beginning training... \n")

    for epoch in range(epochs):

        print("Epoch {}".format(epoch+1))
        print("--------------------------------------------")

        # Initialize the data generator and loss counter
        data_gen = data_generator(train_captions, train_features, batch_size)
        total_loss = 0

        for batch in range(steps):
            # Generate batch of examples
            img_in, cpt_in, cpt_out = next(data_gen)

            img_in = img_in.to(device)
            cpt_in = cpt_in.to(device)
            cpt_out = cpt_out.to(device)

            # Perform learning step and record loss
            loss = learning_step(img_in, cpt_in, cpt_out)

            total_loss += loss

            if batch % 100 == 0:
                print("Epoch {} - Batch {} - Loss {:.4f}".format(
                    epoch+1, batch, loss
                ))

        print("Epoch {} - Average loss {:.4f}".format(
            epoch+1, total_loss/steps
        ))

        training_loss.append(total_loss/steps)

        # Save model
        torch.save(model.state_dict(), save_root + "models/model_{}".format(epoch+1))

        print("\n==============================================\n")


    # Save training loss
    with open(save_root + "/training_loss.pkl", "wb") as f:
        pickle.dump(training_loss, f)
