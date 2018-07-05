import numpy as np
import itertools
import matplotlib.pyplot as plt

from . import state_operations, state_representations


def random_phases(size):
    """Generate a list of random phases.

    Args:
        size (int): Length of list.

    Returns:
        List of random phases.

    """
    np.random.seed(1)
    return 2 * np.pi * np.random.random(size)


def sample_from_hist(histogram, size=100):
    """Sample dataset using `histogram`-data.

    Args:
        histogram (np.array): Histogram of states - `states` and corresponding `probabilities`.
        size (int, optional): Number of samples. Defaults to 100.

    Returns:
        np.array: Array of sampled states.

    """
    states = histogram[:, 0]
    probs = histogram[:, 1].astype(float)
    # Sample tuples from `states` with corresponding probabilities `probs`.
    sampled = np.random.choice(states, size=size, p=probs)
    # Make `sampled` a list of lists with 0.0 and 1.0 values.
    sampled = np.array(list(map(list, sampled))).astype(float)
    return sampled


def get_all_states(n):
    """Produces array of all possible binary strings of length `n`.

    Args:
        n (int): Number of qubits (length of bit string).

    Returns:
        np.array: All possibla binary strings.

    """
    all_states = np.array(list(map(np.array, itertools.product([0, 1], repeat=n))))
    return all_states


def generate_Isinglike_basis_set(n_qub):
    basis_set = []

    for symb in ['H', 'K']:
        for k in range(n_qub):
            basis = 'I' * k + symb + 'I' * (n_qub - k - 1)
            basis_set.append(basis)

    return basis_set


def dataset_w(n_vis, n_samples, hist=False):
    sparsed_states = np.eye(n_vis)
    random_indices = np.random.randint(0, n_vis, n_samples)
    dataset = []
    for i in random_indices:
        dataset.append(sparsed_states[i])

    dataset = np.array(dataset)
    if hist:
        plt.hist(random_indices, bins=n_vis)
        plt.show()

    return dataset


def ideal_w(n_vis):
    sparsed_states = np.eye(n_vis)
    return sparsed_states


def generate_dataset(quantum_system, basis_set, amplitudes, phases, num_units, num_samples):
    Isinglike_dataset = dict()
    # Measurements in ZZZ... basis.
    # Resulting states and coefficients.

    for basis in basis_set:
        # Resulting states and coefficients.
        res = state_operations.system_evolution(quantum_system, basis, amplitudes, phases)
        hist = state_representations.dict_to_hist(res)  # States and corresponding probabilities.
        Isinglike_dataset[basis] = sample_from_hist(hist, num_samples)

    # Converting dataset to histogram representation
    for op in Isinglike_dataset.keys():
        sigmas = Isinglike_dataset[op]
        occurs, sigmas = state_representations.dataset_to_hist(sigmas)
        Isinglike_dataset[op] = {}
        for i in range(len(sigmas)):
            sigma = sigmas[i, :]
            Isinglike_dataset[op][tuple(sigma)] = occurs[i]

    return Isinglike_dataset


if __name__ == '__main__':
    raise RuntimeError('Not a main file')
