
# Dependencies
import numpy as np
import random
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
plt.style.use('ggplot')


# Data loader class

class DataLoader():

    def __init__(self):
        self.data = None
        self.dataset = None

    def create_dataset(self, data, batch_size, shuffle=True):

        data = np.asarray(data)

        if shuffle:
            idx = np.arange(len(data))
            random.shuffle(idx)
            self.data = data[idx]
        else:
            self.data = data

        # Split into batches
        assert len(data) % batch_size == 0
        n_splits = len(data) // batch_size
        idx_splits = np.split(idx, n_splits)

        # Data accumulator
        chunks = []

        for group in idx_splits:
            chunks.append(data[group])

        self.dataset = np.array(chunks)
        return self.dataset


# RBM Class

class RBM():

    def __init__(self, v_nodes, h_nodes, visible_binary, cd_steps, batch_size, learning_rate, weights=None, v_bias=None, h_bias=None):
        self.v_nodes = v_nodes
        self.h_nodes = h_nodes
        self.lr = learning_rate
        self.batch_size = batch_size
        self.binary = visible_binary
        self.k = cd_steps

        # Weights initializer
        if weights is not None:
            assert weights.shape == (self.v_nodes, self.h_nodes)
            self.W = weights
        else:
            self.W = np.random.normal(0, 1, (self.v_nodes, self.h_nodes))

        # Hidden biases initializer
        if h_bias is not None:
            assert len(h_bias) == self.h_nodes
            self.h_bias = h_bias
        else:
            self.h_bias = np.zeros((self.h_nodes, 1))

        # Visible biases initializer
        if v_bias is not None:
            assert len(v_bias) == self.v_nodes
            self.v_bias = v_bias
        else:
            self.v_bias = np.zeros((self.v_nodes, 1))


    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))


    def convert_to_binary(self, activations):
        rand = np.random.random(activations.shape)
        return (rand < activations).astype('float')


    def calculate_h_from_v(self, v_states):
        try:
            v_states = v_states.reshape((self.v_nodes, self.batch_size))
        except:
            raise ValueError('Expected {} dims for visbile state, got {}'.format(self.v_nodes, v_states.shape[0]))

        activations = self.sigmoid(np.dot(self.W.T, v_states) + self.h_bias)
        assert activations.shape == (self.h_nodes, self.batch_size)

        # Convert activations stochastically to binary
        return self.convert_to_binary(activations)


    def calculate_v_from_h(self, h_states):
        try:
            h_states = h_states.reshape((self.h_nodes, self.batch_size))
        except:
            raise ValueError('Expected {} dims for hidden state, got {}'.format(self.h_nodes, h_states.shape[0]))

        activations = self.sigmoid(np.dot(self.W, h_states) + self.v_bias)
        assert activations.shape == (self.v_nodes, self.batch_size)

        if self.binary:
            return self.convert_to_binary(activations)
        else:
            return activations


    def get_negative_sample(self, v_states):
        """K-step contrastive divergence"""

        try:
            v_states = v_states.reshape((self.v_nodes, self.batch_size))
        except:
            raise ValueError('Expected {} dims for visible input, got {}'.format(self.v_nodes, v_states.shape[0]))
        assert self.k > 0

        v_, h_ = v_states, None

        for _ in range(self.k):
            h_ = self.calculate_h_from_v(v_)
            v_ = self.calculate_v_from_h(h_)

        return v_


    def calculate_grads(self, pos_samples, neg_samples):
        """Batch gradients of weights and biases"""

        try:
            pos_samples = pos_samples.reshape((self.v_nodes, self.batch_size))
        except:
            raise ValueError('Expected {} dims for visible input, got {}'.format(self.v_nodes, pos_samples.shape[0]))

        try:
            neg_samples = neg_samples.reshape((self.v_nodes, self.batch_size))
        except:
            raise ValueError('Expected {} dims for visible input, got {}'.format(self.v_nodes, neg_samples.shape[0]))

        w_grad = np.dot(pos_samples, self.sigmoid(np.dot(self.W.T, pos_samples) + self.h_bias).T) - \
                 np.dot(neg_samples, self.sigmoid(np.dot(self.W.T, neg_samples) + self.h_bias).T)

        v_bias_grad = np.mean(pos_samples - neg_samples, axis=1)

        h_bias_grad = np.mean(self.sigmoid(np.dot(self.W.T, pos_samples) + self.h_bias) - \
                        self.sigmoid(np.dot(self.W.T, neg_samples) + self.h_bias), axis=1)

        # Clip gradients so they don't cause overflows
        w_grad = np.clip(w_grad, a_min=-5.0, a_max=5.0)
        v_bias_grad = np.clip(v_bias_grad, a_min=-5.0, a_max=5.0)
        h_bias_grad = np.clip(h_bias_grad, a_min=-5.0, a_max=5.0)

        return w_grad, v_bias_grad, h_bias_grad


    def update_params(self, grads):
        w_grad, v_bias_grad, h_bias_grad = grads

        try:
            w_grad = w_grad.reshape(self.W.shape)
        except:
            raise ValueError('Bad weight gradient shape: expected {}, got {}'.format(self.W.shape, w_grad.shape))

        try:
            v_bias_grad = v_bias_grad.reshape(self.v_bias.shape)
        except:
            raise ValueError('Bad visible bias gradient shape: expected {}, got {}'.format(self.v_bias.shape, v_bias_grad.shape))

        try:
            h_bias_grad = h_bias_grad.reshape(self.h_bias.shape)
        except:
            raise ValueError('Bad hidden bias gradient shape: expected {}, got {}'.format(self.h_bias.shape, h_bias_grad.shape))

        self.W += self.lr * w_grad
        self.v_bias += self.lr * v_bias_grad
        self.h_bias += self.lr * h_bias_grad


    def get_state_energy(self, pos_samples):

        try:
            pos_samples = pos_samples.reshape((self.v_nodes, self.batch_size))
        except:
            raise ValueError('Expected {} dims for visible state, got {}'.format(self.v_nodes, pos_samples.shape[0]))

        h_ = self.calculate_h_from_v(pos_samples)
        v_ = self.calculate_v_from_h(h_)

        if self.binary:
            v_ = self.convert_to_binary(v_)

        energy = np.mean(np.dot(self.W.T, v_) * h_) + np.mean(h_ * self.h_bias) + np.mean(v_ * self.v_bias)
        return energy


    def get_reconstruction_error(self, pos_samples):

        try:
            pos_samples = pos_samples.reshape((self.v_nodes, self.batch_size))
        except:
            raise ValueError('Expected {} dims for visible state, got {}'.format(self.v_nodes, pos_samples.shape[0]))

        h_ = self.calculate_h_from_v(pos_samples)
        v_ = self.calculate_v_from_h(h_)

        if self.binary:
            v_ = self.convert_to_binary(v_)

        # RMSE error
        error = np.mean(np.power(np.mean(np.power(pos_samples - v_, 2), axis=1), 0.5))
        return error


    def train(self, dataset, epochs=10, verbosity='shallow'):
        """
        Verbosity options: 'silent', 'shallow', 'deep_(logging_freq)'
        """

        assert (verbosity == 'shallow') | (verbosity == 'silent') | ('deep' in verbosity)

        avg_energy_hist = []
        avg_rec_error_hist = []

        if 'deep' in verbosity:

            try:
                log_freq = verbosity.split('_')[1]
            except:
                raise ValueError('Bad deep verbosity parameter, expected format: "deep_(integer)"')

            for epoch in range(epochs):

                print("\nEpoch {}".format(epoch))
                batch_energy = []
                batch_rec_error = []

                for pos_samples in tqdm(dataset):

                    neg_samples = self.get_negative_sample(pos_samples)

                    # Calculate gradients
                    w_grad, v_bias_grad, h_bias_grad = self.calculate_grads(pos_samples, neg_samples)
                    grads = [w_grad, v_bias_grad, h_bias_grad]

                    # Get system energy
                    energy = self.get_state_energy(pos_samples)
                    batch_energy.append(energy)

                    # Get reconstruction error
                    rec_error = self.get_reconstruction_error(pos_samples)
                    batch_rec_error.append(rec_error)

                    # Update params
                    self.update_params(grads)

                # Log
                if epoch % log_freq == 0:
                    print("Epoch {} \t Average reconstruction error: {:.5f}".format(epoch, np.mean(batch_rec_error)))

                # Calculate average energy and reconstruction error over all samples
                avg_energy_hist.append(np.mean(batch_energy))
                avg_rec_error_hist.append(np.mean(batch_rec_error))

        elif 'shallow' in verbosity:

            print('')
            outer = tqdm(total=epochs, desc='Epoch', position=0)
            rec_error_log = tqdm(total=0, position=1, bar_format='{desc}')
            save_log = tqdm(total=0, position=2, bar_format='{desc}')

            for epoch in range(epochs):

                batch_energy = []
                batch_rec_error = []

                for pos_samples in dataset:

                    neg_samples = self.get_negative_sample(pos_samples)

                    # Calculate gradients
                    w_grad, v_bias_grad, h_bias_grad = self.calculate_grads(pos_samples, neg_samples)
                    grads = [w_grad, v_bias_grad, h_bias_grad]

                    # Get system energy
                    energy = self.get_state_energy(pos_samples)
                    batch_energy.append(energy)

                    # Get reconstruction error
                    rec_error = self.get_reconstruction_error(pos_samples)
                    batch_rec_error.append(rec_error)

                    # Update params
                    self.update_params(grads)

                # Log
                rec_error_log.set_description_str("Average reconstruction error: {:.5f}".format(np.mean(batch_rec_error)))
                outer.update(1)

                # Calculate average energy and reconstruction error over all samples
                avg_energy_hist.append(np.mean(batch_energy))
                avg_rec_error_hist.append(np.mean(batch_rec_error))

        elif 'silent' in verbosity:

            for epoch in range(epochs):

                print("\nEpoch {}".format(epoch))
                batch_energy = []
                batch_rec_error = []

                for pos_samples in dataset:

                    neg_samples = self.get_negative_sample(pos_samples)

                    # Calculate gradients
                    w_grad, v_bias_grad, h_bias_grad = self.calculate_grads(pos_samples, neg_samples)
                    grads = [w_grad, v_bias_grad, h_bias_grad]

                    # Get system energy
                    energy = self.get_state_energy(pos_samples)
                    batch_energy.append(energy)

                    # Get reconstruction error
                    rec_error = self.get_reconstruction_error(pos_samples)
                    batch_rec_error.append(rec_error)

                    # Update params
                    self.update_params(grads)

                # Calculate average energy and reconstruction error over all samples
                avg_energy_hist.append(np.mean(batch_energy))
                avg_rec_error_hist.append(np.mean(batch_rec_error))

        return avg_energy_hist, avg_rec_error_hist


    def save_params(self, epoch, root_path):
        """Add the trailing slash in root path"""

        np.save(root_path + str(epoch) + '_weights.npy', self.W)
        np.save(root_path + str(epoch) + '_v_bias.npy', self.v_bias)
        np.save(root_path + str(epoch) + '_h_bias.npy', self.h_bias)


    def get_hidden(self, X):
        """Returns hidden representation of data samples"""

        # Shape check for input
        try:
            X = X.reshape((self.v_nodes, len(X)))
        except:
            raise ValueError('Expected {} dims for visible state, got {}'.format(self.v_nodes, X.shape[0]))

        h_ = self.sigmoid(np.dot(self.W.T, X) + self.h_bias)
        h_ = self.convert_to_binary(h_)
        assert h_.shape == (self.h_nodes, X.shape[1])

        return h_.T


    def plot_trend(self, series, name):
        """Useful for plotting reconstruction error and state energy"""

        plt.figure(figsize=(6, 6))
        plt.plot(series, color='blue')
        plt.title(name, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        plt.show()
