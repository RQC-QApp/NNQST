# File name: generators.py
# Authors: Yaroslav Kharkov <y.kharkov@unsw.edu.au>, Anton Karazeev <a.karazeev@rqc.ru>
# Based on https://www.nature.com/articles/s41567-018-0048-5 paper
#
# This file is part of NNQST project (https://github.com/RQC-QApp/NNQST)
#
# Description: Samplers for datasets of measurements and generators of states.

import numpy as np
import itertools
import matplotlib.pyplot as plt

from . import state_operations, state_representations


def random_phases(size):
    """Generate a list of random phases.

    Args:
        size (int): Length of list.

    Returns:
        np.array: List of random phases.

    """
    return 2 * np.pi * np.random.random(size)


def sample_from_probabilities(histogram, size=100):
    """Sample dataset using `histogram`-data.

    Args:
        histogram (list): Histogram of states containing states and corresponding probabilities.
        size (int, optional): Number of samples. Defaults to 100.

    Returns:
        np.array: Array of sampled states.

    """
    histogram = np.array(histogram)
    states = histogram[:, 0]
    probs = histogram[:, 1].astype(float)
    # Sample tuples from `states` with corresponding probabilities `probs`.
    sampled = np.random.choice(states, size=size, p=probs)
    return sampled


def get_all_states(n, state_type="tuple"):
    """Produces array of all possible binary strings of length `n`.

    Args:
        n (int): Number of qubits (length of bit string).
        state_type (str, optional): Representation of every state: "tuple" or "list". Defaults to "tuple".

    Returns:
        list: All possible binary strings.

    """
    if state_type == "tuple":
        all_states = list(map(tuple, itertools.product([0, 1], repeat=n)))
    elif state_type == "list":
        all_states = list(map(list, itertools.product([0, 1], repeat=n)))
    else:
        raise ValueError("`state_type` must be 'tuple' or 'list'")
    return all_states


def generate_Isinglike_basis_set(n_qub):
    """Generate basis set for Ising model.

    Args:
        n_qub (int): Number of qubits.

    Returns:
        list: List of strings.

    """
    basis_set = []

    for symb in ['H', 'K']:
        for k in range(n_qub):
            basis = 'I' * k + symb + 'I' * (n_qub - k - 1)
            basis_set.append(basis)

    return basis_set


def ideal_w(n_vis):
    """Basis states for easy state W.

    Args:
        n_vis (int):

    Returns:
        list: List of states (tuples).

    """
    sparsed_states = np.eye(n_vis)
    sparsed_states = [tuple(map(int, x)) for x in sparsed_states]
    return sparsed_states


def dataset_w(n_vis, n_samples, hist=False):
    """Generate measurement for easy state called W.

    Args:
        n_vis (int):
        n_samples (int):
        hist (bool, optional): Wheteher to plot histogram or not. Defaults to False.

    Returns:
        list: List of states (tuples).

    """
    sparsed_states = ideal_w(n_vis)
    random_indices = np.random.randint(0, n_vis, n_samples)
    # Fill dataset.
    dataset = list()
    for i in random_indices:
        dataset.append(sparsed_states[i])

    if hist:
        plt.hist(random_indices, bins=n_vis)
        plt.show()

    return dataset


def generate_dataset(states, basis_set, amplitudes, phases, num_samples):
    """Generate measurements for the `states` in given bases `basis_set`.

    Args:
        states (list): List of states (tuples).
        basis_set (list): List of bases (strings).
        amplitudes (list): List of amplitudes (floats).
        phases (list): List of phases (floats).
        num_samples (int):

    Returns:
        dict: dict of dicts {basis: {state: occurrences}}.

    """
    dataset_tmp = dict()

    # Sampling states.
    for basis in basis_set:
        # Resulting states and coefficients.
        evolved_system = state_operations.system_evolution(states, basis, amplitudes, phases)
        # States and corresponding probabilities.
        probabilities = state_representations.get_probabilities(evolved_system)
        dataset_tmp[basis] = sample_from_probabilities(probabilities, num_samples)

    dataset = dict()
    # Converting dataset to histogram representation.
    for basis in basis_set:
        sigmas = dataset_tmp[basis]
        occurrences, sigmas = state_representations.get_occurrences(sigmas)
        dataset[basis] = dict()
        for i in range(len(sigmas)):
            dataset[basis][sigmas[i]] = occurrences[i]

    return dataset


def generate_Isinglike_dataset(num_qbits, states, amplitudes, phases, num_samples):
    """Generate measurements for Ising model.

    Args:
        num_qbits (int):
        states (list): List of states (tuples).
        amplitudes (list): List of amplitudes (floats).
        phases (list): List of phases (floats).
        num_samples (int):

    Returns:
        dict: dict of dicts {basis: {state: occurrences}}.

    """
    Isinglike_basis_set = generate_Isinglike_basis_set(num_qbits)
    Isinglike_dataset = generate_dataset(states, Isinglike_basis_set, amplitudes, phases, num_samples)

    return Isinglike_dataset


if __name__ == '__main__':
    raise RuntimeError('Not a main file')
