import numpy as np

import utils
import paper_functions


class RBM_QST:

    def __init__(self, num_visible, num_hidden, basis_states):
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.basis_states = np.insert(basis_states, 0, 1, axis=1)
        self.objectives = list()

        np_rng = np.random.RandomState(42)

        # Amplitudes -- `lambda`.
        self.weights_lambda = np.asarray(np_rng.uniform(
                                         low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                                         high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                                         size=(num_visible, num_hidden)))
        # Insert weights for the bias units into the first row and first column.
        self.weights_lambda = np.insert(self.weights_lambda, 0, 0, axis=0)
        self.weights_lambda = np.insert(self.weights_lambda, 0, 0, axis=1)

        # Phases -- `mu`.
        self.weights_mu = np.asarray(np_rng.uniform(
                                     low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                                     high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                                     size=(num_visible, num_hidden)))
        # Insert weights for the bias units into the first row and first column.
        self.weights_mu = np.insert(self.weights_mu, 0, 0, axis=0)
        self.weights_mu = np.insert(self.weights_mu, 0, 0, axis=1)

    def train_amplitudes(self, data_raw, max_epochs=1000, learning_rate=0.1, debug=False, precise=False):
        """Train the machine to learn amplitudes.

        Args:
            data_raw (np.array): A matrix where each row is a training example consisting
                of the states of visible units.

        """
        # Converting dataset to a histogram representation.
        occurs, data_hist = utils.dataset_to_hist(data_raw)

        # Insert bias units of 1 into the first column.
        data_hist = np.insert(data_hist, 0, 1, axis=1)
        data_raw = np.insert(data_raw, 0, 1, axis=1)

        for epoch in range(max_epochs):
            if debug and epoch % 100 == 0:
                self.objectives.append(paper_functions.objective_func(self.weights_lambda, self.weights_mu, data_raw))
                print("Epoch %s: objective is %s" % (epoch, self.objectives[-1]))

            # Calculating of `gradient`.
            N_b = 1  # TODO: In this version we have only one basis, but in general: N_b = len(datasets).
            tmp1 = N_b * self._averaged_D_lambda_p_lambda(precise=precise)
            # Due to we have only one basis we calculate just one component.
            tmp2 = self._averaged_D_k_Q_b('lambda', occurs, data_hist)
            tmp2 = tmp2.real
            gradients = tmp1 - tmp2

            self.weights_lambda -= learning_rate * gradients

    # TODO: Implement a case with precise=False
    def _averaged_D_lambda_p_lambda(self, precise=False, num_samples=10, num_steps=10):
        """(A18) of arxive paper. (17) of Nature paper.

        """
        res = None
        if precise:
            num_units = self.weights_lambda.shape[0] - 1
            all_states = utils.get_all_states(num_units)
            all_states = np.insert(all_states, 0, 1, axis=1)

            # Sum of gradients.
            res = np.array([paper_functions.p_k(x, self.weights_lambda) *
                            paper_functions.D_k(x, self.weights_lambda)
                            for x in all_states])
            res = np.sum(res, axis=0)
            res /= paper_functions.Z_lambda(self.weights_lambda)

        else:
            raise ValueError('This case is not implemented yet')
            # # sampled = np.array([self.daydream(num_steps)[-1] for _ in range(num_samples)]).copy()
            # # sampled = np.insert(sampled, 0, 1, axis=1)
            #
            # probs = list()
            # for i in range(len(self.basis_states)):
            #     tmp_data = self.basis_states[i]
            #     tmp_prob = paper_functions.p_k(tmp_data, self.weights_lambda)
            #     probs.append(tmp_prob)
            # probs = np.array(probs)
            # probs = probs / probs.sum()
            #
            # sampled = self.basis_states[np.random.choice(self.basis_states.shape[0], size=num_samples, p=probs)]
            #
            # res = np.array([paper_functions.D_k(x, self.weights_lambda) for x in sampled])
            # res = np.sum(res, axis=0)  # Sum of gradients.
            # res /= num_samples

        return res

    def train_phases(self, data_raw, operations, max_epochs=1000, learning_rate=0.1, debug=False, precise=False):
        """Train the machine to learn phases.

        Args:
            data_raw (dict): Dict of {basis: list of states}.
            operations (str): List of strings of `operations`.

        """
        for epoch in range(max_epochs):
            gradients_total = 0.0
            for basis_operations in operations:
                training_set = data_raw[basis_operations]

                # Converting dataset to a histogram representation.
                #
                # It seems all we have to do is to convert `training_set` using `basis_operations`
                # into different basis and then pass such `training_set` to `utils.dataset_to_hist()`.
                # Also, we need an array of coefficients (both amplitudes and phases) of every state.
                #
                amplitudes = [1] * len(training_set)
                phases = [0] * len(training_set)
                after_evolution = utils.system_evolution(training_set, basis_operations, amplitudes, phases)
                # Now we have a list of states and corresponding coefficients.
                after_evolution = list(after_evolution.items())

                # occurs, data_hist = utils.dataset_to_hist(training_set)
                data_hist = list(map(lambda x: list(map(float, x[0])), after_evolution))
                occurs = list(map(lambda x: x[1], after_evolution))

                # Insert bias units of 1 into the first column.
                data_hist = np.insert(data_hist, 0, 1, axis=1)
                training_set = np.insert(training_set, 0, 1, axis=1)

                # Calculating of gradients.
                # All the operations of rotations were carried out here
                # and just `occurs` and `data_hist` are passed to (15) of Nature paper.
                gradients_tmp = self._averaged_D_k_Q_b('mu', occurs, data_hist)
                gradients_total += gradients_tmp.imag / len(training_set)

            self.weights_mu -= learning_rate * gradients_total

            if debug and epoch % 2 == 0:
                # TODO: Don't know exactly what dataset should be passed here                   ----------------v
                self.objectives.append(paper_functions.objective_func(self.weights_lambda, self.weights_mu, training_set))
                print("Epoch %s: objective is %s" % (epoch, self.objectives[-1]))

    def _averaged_D_k_Q_b(self, k, occurs, dataset_hist):
        """(A16) of arxive paper. (15) of Nature paper.

        Args:
            k (str): Possible values: 'mu' for phases and 'lambda' for amplitudes.

        """
        if k == 'mu':
            weights = self.weights_mu
        elif k == 'lambda':
            weights = self.weights_lambda
        else:
            raise ValueError('Wrong k.')

        quasi_probs = np.sum([occurs[i] * paper_functions.Q_b(dataset_hist[i, :], self.weights_lambda, self.weights_mu)
                              for i in range(len(dataset_hist))])

        # Sum of quasi_prob * gradients.
        res = np.array([occurs[i] * paper_functions.D_k(dataset_hist[i, :], weights) *
                        paper_functions.Q_b(dataset_hist[i, :], self.weights_lambda, self.weights_mu)
                        for i in range(len(dataset_hist))])
        res = np.sum(res, axis=0)
        res /= quasi_probs

        return res

    # TODO: It seems that we need another function to sample phases from this RBM.
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

    def run_visible(self, visible, states=False):
        """Assuming the RBM has been trained (so that weights for the network have been learned),
        run the network on a set of visible units, to get a sample of the hidden units.

        Args:
            visible (np.array): Vector of zeros and ones.

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
        return 1.0 / (1 + np.exp(-x))


if __name__ == '__main__':
    raise RuntimeError('not a main file')
