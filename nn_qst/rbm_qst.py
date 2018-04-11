import numpy as np
from utils import *


class RBM:

    def __init__(self, num_visible, num_hidden, params=None):
        self.num_hidden = num_hidden
        self.num_visible = num_visible

        if params is None:

            np_rng = np.random.RandomState(1234)

            params = {
                "lambda": {
                    "W": np.asarray(np_rng.uniform(low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                                                   high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                                                   size=(num_visible, num_hidden))),
                    "b": np.random.random(self.num_visible),
                    "c": np.random.random(self.num_hidden)
                },

                "mu": {
                    "W": np.asarray(np_rng.uniform(low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                                                   high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                                                   size=(num_visible, num_hidden))),
                    "b": np.random.random(self.num_visible),
                    "c": np.random.random(self.num_hidden)
                }
            }

            # params = {
            #     "lambda": {
            #         "W": np.random.random((self.num_hidden, self.num_visible)),
            #         "b": np.random.random(self.num_visible),
            #         "c": np.random.random(self.num_hidden)
            #     },
            #
            #     "mu": {
            #         "W": np.random.random((self.num_hidden, self.num_visible)),
            #         "b": np.random.random(self.num_visible),
            #         "c": np.random.random(self.num_hidden)
            #     }
            # }

        self.params = params

    def train(self, data, sparsed_states, max_epochs=1000, overlap_each=100, onum_samples=1000, onum_steps=100, learning_rate=0.1, overlap=False, debug=False):
        """Train the machine.

        Args:
            data (np.array): A matrix where each row is a training example consisting
                of the states of visible units.

        """
        num_examples = data.shape[0]

        for epoch in range(max_epochs):
            pos_hidden_activations = np.dot(data, self.params['lambda']['W'].T) + self.params['lambda']['c']

            pos_hidden_probs = self._logistic(pos_hidden_activations)
            pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden)

            neg_visible_activations = np.dot(pos_hidden_states, self.params['lambda']['W']) + self.params['lambda']['b']
            neg_visible_probs = self._logistic(neg_visible_activations)

            error = np.sum((data - neg_visible_probs) ** 2)
            if debug:
                print("Epoch %s: error is %s" % (epoch, error))

            if overlap and epoch % overlap_each == 0:
                sampled_from_RBM = np.array([self.daydream(onum_steps)[-1] for _ in range(onum_samples)])
                sampled_from_RBM = into_dict(sampled_from_RBM)

                ideal_state = into_dict(sparsed_states)

                print('Fidelity is {}'.format(fidelity_dicts(ideal_state, sampled_from_RBM)))

            gradients = grad_lambda_ksi(data, self.params)

            self.params['lambda']['c'] -= learning_rate * gradients['c']
            self.params['lambda']['b'] -= learning_rate * gradients['b']
            self.params['lambda']['W'] -= learning_rate * gradients['W']

    def run_visible(self, data, probs=False):
        """Assuming the RBM has been trained (so that weights for the network have been learned),
        run the network on a set of visible units, to get a sample of the hidden units.

        Args:
            data (np.array): A matrix where each row consists of the states of the visible units.
            probs (bool): Whether return probabilities or activations of hidden layer. Defaults to False.

        Returns:
            hidden_states (np.array): A matrix where each row consists of the hidden units activated from the visible
                units in the data matrix passed in.

        """
        num_examples = data.shape[0]

        # Create a matrix, where each row is to be the hidden units (plus a bias unit)
        # sampled from a training example.
        hidden_states = np.ones((num_examples, self.num_hidden))

        # Calculate the activations of the hidden units.
        hidden_activations = np.dot(data, self.params['lambda']['W']) + self.params['lambda']['c']
        # Calculate the probabilities of turning the hidden units on.
        hidden_probs = self._logistic(hidden_activations)

        if probs:
            return hidden_probs

        # Turn the hidden units on with their specified probabilities.
        hidden_states = hidden_probs > np.random.rand(num_examples, self.num_hidden)
        hidden_states = hidden_states.astype('float')

        return hidden_states

    def run_hidden(self, data):
        """Assuming the RBM has been trained (so that weights for the network have been learned),
        run the network on a set of hidden units, to get a sample of the visible units.

        Args:
            data (np.array): A matrix where each row consists of the states of the hidden units.

        Returns:
            visible_states (np.array): A matrix where each row consists of the visible units activated from the hidden
                units in the data matrix passed in.

        """
        num_examples = data.shape[0]

        visible_states = np.ones((num_examples, self.num_visible))

        # Calculate the activations of the visible units.
        visible_activations = np.dot(data, self.params['lambda']['W'].T) + self.params['lambda']['b']
        # Calculate the probabilities of turning the visible units on.
        visible_probs = self._logistic(visible_activations)
        # Turn the visible units on with their specified probabilities.
        visible_states = visible_probs > np.random.rand(num_examples, self.num_visible)
        visible_states = visible_states.astype('float')

        return visible_states

    def daydream(self, num_samples, debug=False):
        """Randomly initialize the visible units once, and start running alternating Gibbs sampling steps
        (where each step consists of updating all the hidden units, and then updating all of the visible units),
        taking a sample of the visible units at each step.

        Note that we only initialize the network *once*, so these samples are correlated.

        Returns:
            samples: A matrix, where each row is a sample of the visible units produced while the network was
                daydreaming.

        """
        samples = np.ones((num_samples, self.num_visible))

        # Take the first sample from a uniform distribution.
        samples[0] = np.random.rand(self.num_visible)

        if debug:
            print(samples[0])

        # Start the alternating Gibbs sampling.
        for i in range(1, num_samples):
            visible = samples[i - 1]

            # Calculate the activations of the hidden units.
            hidden_activations = self.params['lambda']['W'].dot(visible) + self.params['lambda']['c']

            # Calculate the probabilities of turning the hidden units on.
            hidden_probs = self._logistic(hidden_activations)

            if debug:
                print('-------------')
                print('Hidden activations:')
                print(hidden_activations)

                print('-------------')
                print('Hidden probs:')
                print(hidden_probs)

            # Turn the hidden units on with their specified probabilities.
            hidden_states = hidden_probs > np.random.rand(self.num_hidden)

            # Recalculate the probabilities that the visible units are on.
            visible_activations = self.params['lambda']['W'].T.dot(hidden_states) + self.params['lambda']['b']
            visible_probs = self._logistic(visible_activations)
            visible_states = visible_probs > np.random.rand(self.num_visible)
            samples[i] = visible_states

        return samples

    def _logistic(self, x):
        return 1.0 / (1 + np.exp(-x))


if __name__ == '__main__':
    raise RuntimeError('not a main file')
