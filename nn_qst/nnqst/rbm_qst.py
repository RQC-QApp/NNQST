import numpy as np

from . import paper_functions


# NOTE: Calculation of Objective function is disabled
class RBM_QST:
    """Class for RBM to perform Quantum State Tomography.

    Methods:
        * __init__(self, quantum_system, num_visible, num_hidden)
        * train_amplitudes(self, dataset, max_epochs=1000, learning_rate=0.1, debug=False, precise=True)
        * train_phases(self, dataset, basis_set, max_epochs=1000, learning_rate=0.1, debug=False, precise=False)
        * daydream(self, num_samples, debug=False)
        * run_visible(self, visible, states=False)
        * _logistic(self, x)

    """
    def __init__(self, quantum_system, num_visible, num_hidden):
        """Initialisation.

        Args:
            quantum_system (list):
            num_visible (int):
            num_hidden (int):

        """
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.objectives = list()

        np_rng = np.random.RandomState(42)

        self.quantum_system = quantum_system
        # Amplitudes `lambda`.
        self.weights_lambda = np.asarray(np_rng.uniform(
                                         low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                                         high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                                         size=(num_visible, num_hidden)))
        # Insert weights for the bias units into the first row and first column.
        self.weights_lambda = np.insert(self.weights_lambda, 0, 0, axis=0)
        self.weights_lambda = np.insert(self.weights_lambda, 0, 0, axis=1)

        # Phases `mu`.
        self.weights_mu = np.asarray(np_rng.uniform(
                                     low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                                     high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                                     size=(num_visible, num_hidden)))
        # Insert weights for the bias units into the first row and first column.
        self.weights_mu = np.insert(self.weights_mu, 0, 0, axis=0)
        self.weights_mu = np.insert(self.weights_mu, 0, 0, axis=1)

    def train_amplitudes(self, dataset, max_epochs=1000, learning_rate=0.1, debug=False, precise=True):
        """Train the machine to learn amplitudes.

        Args:
            dataset (np.array): A matrix where each row is a training example consisting
                of the states of visible units.
            max_epochs (int, optional): Defaults to 1000.
            learning_rate (float, optional): Defaults to 0.1.
            debug (bool, optional): Defaults to False.
            precise (bool, optional): Defaults to True.

        """
        # Converting dataset to a histogram representation.
        Nqub = self.num_visible
        dataset_Z = dataset['I' * Nqub]

        sigmas = np.array(list(dataset_Z.keys()))
        sigmas = np.insert(sigmas, 0, 1, axis=1)  # Insert bias units of 1 into the first column.

        basis_set_Z = ['I' * Nqub]
        for epoch in range(max_epochs):
            if debug and epoch % 500 == 0:
                # Saving objective function.
                self.objectives.append(paper_functions.objective_func(self.quantum_system,
                                                                      self.weights_lambda,
                                                                      self.weights_mu,
                                                                      dataset,
                                                                      basis_set_Z))
                print("Epoch %s: objective is %s" % (epoch, self.objectives[-1]))

            gradients = paper_functions.grad_lambda_ksi(dataset,
                                                        self.weights_lambda,
                                                        self.weights_mu,
                                                        precise)

            self.weights_lambda -= learning_rate * gradients

    def train_phases(self, dataset, basis_set, max_epochs=1000, learning_rate=0.1, debug=False, precise=False):
        """Train the machine to learn phases.

        Args:
            dataset (dict): Dict of {basis: list of states}.
            basis_set (list): List of bases (strings).
            max_epochs (int, optional): Defaults to 1000.
            learning_rate (float, optional): Defaults to 0.1.
            debug (bool, optional): Defaults to False.
            precise (bool, optional): Defaults to False.

        """
        for epoch in range(max_epochs):
            # Converting dataset to a histogram representation.
            #
            # It seems all we have to do is to convert `training_set` using `basis_operations`
            # into different basis and then pass such `training_set` to `utils.dataset_to_hist()`.
            # Also, we need an array of coefficients (both amplitudes and phases) of every state.
            #
            # Calculating of gradients.
            # All the operations of rotations were carried out here
            # and just `occurs` and `data_hist` are passed to (15) of Nature paper.
            #
            # Old update rule without S_ij Fisher Information matrix
            # gradients = paper_functions.grad_mu_ksi(dataset, basis_set, self.weights_lambda, self.weights_mu)
            # self.weights_mu -= learning_rate * gradients
            #
            # New update rule with S_ij.
            self.weights_mu = paper_functions.update_weights_mu_Fisher(dataset,
                                                                       self.weights_lambda,
                                                                       self.weights_mu,
                                                                       learning_rate)

            if debug and epoch % 500 == 0:
                self.objectives.append(paper_functions.objective_func(self.quantum_system,
                                                                      self.weights_lambda,
                                                                      self.weights_mu,
                                                                      dataset,
                                                                      basis_set))
                print("Epoch %s: objective is %s" % (epoch, self.objectives[-1]))

    def daydream(self, num_samples, debug=False):
        """Randomly initialize the visible units once, and start running alternating Gibbs sampling steps
        (where each step consists of updating all the hidden units, and then updating all of the visible units),
        taking a sample of the visible units at each step.

        Note that we only initialize the network *once*, so these samples are correlated.

        Args:
            num_samples (int):
            debug (bool, optional): Defaults to False.

        Returns:
            np.array: A matrix, where each row is a sample of the visible units produced while the network was
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

    def run_visible(self, visible, states=False):
        """Assuming the RBM has been trained (so that weights for the network have been learned),
        run the network on a set of visible units, to get a sample of the hidden units.

        Args:
            visible (np.array): Vector of zeros and ones.
            states (bool, optional): Whether returned array consists of 1s and 0s or probabilities (floats from 0 to 1). Defaults to False.

        Returns:
            np.array:

        """
        visible = np.insert(visible, 0, 1, axis=0)
        # Calculate the activations of the hidden units.
        hidden_activations = np.dot(visible, self.weights_lambda)
        # Calculate the probabilities of turning the hidden units on.
        hidden_probs = self._logistic(hidden_activations)

        if states:
            # Turn the hidden units on with their specified probabilities.
            hidden_states = hidden_probs > np.random.rand(self.num_hidden + 1)
            # Always fix the bias unit to 1.
            hidden_states[0] = 1
            return hidden_states
        else:
            return hidden_probs

    def _logistic(self, x):
        """Calculate sigmoid.

        Args:
            x (float):

        Returns:
            float:

        """
        return 1.0 / (1 + np.exp(-x))


if __name__ == '__main__':
    raise RuntimeError('Not a main file')
