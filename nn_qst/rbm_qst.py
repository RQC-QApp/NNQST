import numpy as np
import utils
import paper_functions


class RBM_QST:

    def __init__(self, num_visible, num_hidden):
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.objectives = list()

        np_rng = np.random.RandomState(42)

        self.weights_lambda = np.asarray(np_rng.uniform(
                                         low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                                         high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                                         size=(num_visible, num_hidden)))

        # Insert weights for the bias units into the first row and first column.
        self.weights_lambda = np.insert(self.weights_lambda, 0, 0, axis=0)
        self.weights_lambda = np.insert(self.weights_lambda, 0, 0, axis=1)

        self.weights_mu = np.asarray(np_rng.uniform(
                                     low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                                     high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                                     size=(num_visible, num_hidden)))

        # Insert weights for the bias units into the first row and first column.
        self.weights_mu = np.insert(self.weights_mu, 0, 0, axis=0)
        self.weights_mu = np.insert(self.weights_mu, 0, 0, axis=1)

    def train(self, data_raw, ideal_state, max_epochs=1000, overlap_each=100, onum_samples=1000, onum_steps=100, learning_rate=0.1, overlap=False, debug=False, precise=False):
        """Train the machine.

        Args:
            data (np.array): A matrix where each row is a training example consisting
                of the states of visible units.

        """
        num_examples = data_raw.shape[0]

        #############
        # Converting dataset to a histogram representation
        
        occurs, data_hist = utils.dataset_to_hist(data_raw)        
        # selecting only states with nonzero occurencies
        data_hist = data_hist[occurs!=0]        
        occurs = occurs[occurs!=0]        
        ################
        
        # Insert bias units of 1 into the first column.
        data_hist = np.insert(data_hist, 0, 1, axis=1)        
        data_raw = np.insert(data_raw, 0, 1, axis=1)        
        
        
        for epoch in range(max_epochs):
            if debug and epoch % 100 == 0:
                self.objectives.append(paper_functions.objective_func(self.weights_lambda, self.weights_mu, data_raw))
                print("Epoch %s: objective is %s" % (epoch, self.objectives[-1]))

            if overlap and epoch % overlap_each == 0:
                sampled_from_RBM = np.array([self.daydream(onum_steps)[-1] for _ in range(onum_samples)])
                sampled_from_RBM = into_dict(sampled_from_RBM)

                ideal_state = into_dict(ideal_state)
                print('Fidelity is {}'.format(utils.fidelity_dicts(ideal_state, sampled_from_RBM)))

            # gradients = paper_functions.grad_lambda_ksi(data, self.weights_lambda, self.weights_mu, precise)
            gradients = paper_functions.grad_lambda_ksi_MANUAL(occurs, data_hist, self.weights_lambda, self.weights_mu)
            self.weights_lambda -= learning_rate * gradients

    def daydream(self, num_samples, debug=False):
        """Randomly initialize the visible units once, and start running alternating Gibbs sampling steps
        (where each step consists of updating all the hidden units, and then updating all of the visible units),
        taking a sample of the visible units at each step.

        Note that we only initialize the network *once*, so these samples are correlated.

        Returns:
            samples: A matrix, where each row is a sample of the visible units produced while the network was
                daydreaming.

        """
        # Create a matrix, where each row is to be a sample of of the visible units
        # (with an extra bias unit), initialized to all ones.
        samples = np.ones((num_samples, self.num_visible + 1))

        samples[0, 1:] = np.random.rand(self.num_visible)

        for i in range(1, num_samples):
            visible = samples[i - 1, :]

            # Calculate the activations of the hidden units.
            hidden_activations = np.dot(visible, self.weights_lambda)
            # Calculate the probabilities of turning the hidden units on.
            hidden_probs = self._logistic(hidden_activations)
            # Turn the hidden units on with their specified probabilities.
            hidden_states = hidden_probs > np.random.rand(self.num_hidden + 1)
            # Always fix the bias unit to 1.
            hidden_states[0] = 1

            # Recalculate the probabilities that the visible units are on.
            visible_activations = np.dot(hidden_states, self.weights_lambda.T)
            visible_probs = self._logistic(visible_activations)
            visible_states = visible_probs > np.random.rand(self.num_visible + 1)
            samples[i, :] = visible_states

        # Ignore the bias units (the first column), since they're always set to 1.
        return samples[:, 1:]

    def _logistic(self, x):
        return 1.0 / (1 + np.exp(-x))


if __name__ == '__main__':
    raise RuntimeError('not a main file')
