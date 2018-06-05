import matplotlib.pyplot as plt
from collections import Counter
import paper_functions
import numpy as np
import itertools
import math


def into_dict(dataset):
    """Convert list of np.array into dict.

    Args:
        dataset (np.array): Sampled or ideal states.

    Returns:
        dict: {state: frequency}.

    """
    dataset = dataset.copy()
    dataset = list(map(lambda x: x.astype('int'), dataset))
    dataset = list(map(lambda x: ''.join(map(str, x)), dataset))
    dataset = dict(Counter(dataset))

    num_samples = np.sum(list(dataset.values()))

    for state in dataset:
        dataset[state] = np.sqrt(dataset[state] / num_samples)

    return dataset


def evolution(state, operations, coefficient=1, verbose=False):
    """Applies the sequence of `operations` to given `state`.

    Args:
        state (tuple): State.
        operations (str): String consisting of 'I', 'H' and 'K'.
        coefficient (complex, optional): Coefficient by given state. Defaults to 1.
        verbose (bool, optional): Printing log info. Defaults to False.

    Returns:
        Dict of states and corresponding coefficients.

    """
    def apply_k(value: int):
        assert value in [0, 1]

        res = dict()
        if value == 0:
            res[0] = 1 / np.sqrt(2)
            res[1] = 1 / np.sqrt(2)
        elif value == 1:
            res[0] = -1j / np.sqrt(2)
            res[1] = 1j / np.sqrt(2)
        return res

    def apply_h(value: int):
        assert value in [0, 1]

        res = dict()
        if value == 0:
            res[0] = 1 / np.sqrt(2)
            res[1] = 1 / np.sqrt(2)
        elif value == 1:
            res[0] = 1 / np.sqrt(2)
            res[1] = -1 / np.sqrt(2)
        return res

    all_states = dict()
    all_states[state] = coefficient

    h_indices = [x[0] for x in enumerate(operations) if x[1] == 'H']
    k_indices = [x[0] for x in enumerate(operations) if x[1] == 'K']

    for indices, apply, letter in [(h_indices, apply_h, 'H'), (k_indices, apply_k, 'K')]:
        for i in indices:
            # It's necessary to save keys before iterating over dict.
            all_states_copy = all_states.copy()
            for s in all_states_copy:
                # Apply operation to i'th qubit.
                applied = apply(s[i])
                for v in applied:
                    tmp = s[:i] + (v,) + s[i + 1:]
                    if tmp in all_states:
                        all_states[tmp] *= applied[v]
                    else:
                        all_states[tmp] = all_states_copy[s] * applied[v]
            if verbose:
                print(letter, i)
                print(all_states)

    if verbose:
        print("=======")
        print("Result:")
        print(all_states)

    return all_states


def merge_dicts(dict1, dict2):
    """Sums up two dicts into the new one.

    """
    tmp = {k: dict1.get(k, 0) + dict2.get(k, 0) for k in set(dict1) | set(dict2)}
    res = {k: tmp[k] for k in tmp if abs(tmp[k]) > 0.0}
    return res


def dict_to_quantum_system(quantum_dict):
    """Converts dict of states and corresponding coefficients into three
    lists -- of states, of amplitudes and of phases.

    Args:
        quantum_dict (dict): States and corresponding probabilities.

    Returns:
        Tuple of three lists -- `quantum_system`, `amplitudes`, `phases`.

    """
    phases = list()
    amplitudes = list()
    quantum_system = list()

    for key in quantum_dict:
        r_theta = polar(quantum_dict[key])
        quantum_system.append(key)
        amplitudes.append(r_theta[0])
        phases.append(r_theta[1])

    return quantum_system, amplitudes, phases


def dict_to_hist(quantum_dict):
    """Converts dict of states into histogram -- list of tuples of states and corresponding probabilities.

    Args:
        quantum_dict (dict): Dict of states and corresponding coefficients.

    Returns:
        np.array of tuples (state, probability).

    """
    res = list()
    for state in quantum_dict:
        tmp = (state, abs(quantum_dict[state]) ** 2)  # State and corresponding probability.
        res.append(tmp)
    return np.array(res)


def system_evolution(quantum_system, operations, amplitudes, phases):
    """Performs an evolution/rotation for `quantum_system` using the
    sequence of `operations`.

    Args:
        quantum_system (list): List of states.
        operations (str): Sequence of operations.
        amplitudes (list): List of floats.
        phases (list): List of floats.

    Returns:
        Dict of {states: coefficients}.

    """
    assert len(operations) == len(quantum_system[0]), "Lengths must be the same."

    total = dict()
    for i in range(len(quantum_system)):
        # Pass state, operations and coefficient == a * exp(i * b).
        state = tuple(quantum_system[i])
        tmp = evolution(state, operations,
                        amplitudes[i] * np.exp(1j * phases[i]))
        total = merge_dicts(total, tmp)

    return total


def random_phases(size):
    """Generate a list of random phases.

    Args:
        size (int): Length of list.

    Returns:
        List of random phases.

    """
    return 2 * np.pi * np.random.random(size)


def polar(z):
    """Convert complex number to polar representation.

    Args:
        z (complex): Number to convert.

    Returns:
        Tuple of float for `r` and float for angle `theta`.

    """
    a = z.real
    b = z.imag
    r = math.hypot(a, b)
    theta = math.atan2(b, a)
    if theta < 0:
        theta += 2 * np.pi
    return r, theta


def U_XX(j, n):
    """Makes a sequence of operations for XX basis.

    Args:
        j (int): Position of 'HH'.
        n (int): Length of the sequence.

    Returns:
        str: Sequence of operations.

    """
    assert j < n - 1
    assert n > 2
    return 'I' * j + "HH" + 'I' * (n - j - 2)


def U_XY(j, n):
    """Makes a sequence of operations for XY basis.

    Args:
        j (int): Position of 'HK'.
        n (int): Length of the sequence.

    Returns:
        str: Sequence of operations.

    """
    assert j < n - 1
    assert n > 2
    return 'I' * j + "HK" + 'I' * (n - j - 2)


def U_ZZ(n):
    """Makes a sequence of operations for ZZ basis.

    Args:
        n (int): Length of the sequence.

    Returns:
        str: Sequence of operations.

    """
    return 'I' * n


def sample_from_hist(histogram, size=100):
    """Sample dataset using `histogram`-data.

    Args:
        histogram (np.array): Histogram of states -- `states` and corresponding `probabilities`.
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


def dataset_to_hist(dataset):
    """Turns `dataset` into statistics: elements and theirs occurrences.

    Args:
        dataset (np.array): Dataset of states.

    Returns:
        tuple: np.array of occurrences and np.array of states.

    """
    cntr = Counter(map(tuple, dataset))

    tmp = list(cntr.items())
    data_hist = list(map(lambda x: x[0], tmp))
    data_hist = np.array(list(map(list, data_hist)), dtype=int)
    occurs = np.array(list(map(lambda x: x[1], tmp)))

    return occurs, data_hist


def fidelity_dicts(dict1, dict2):
    res = 0
    for state in dict1:
        if state in dict2:
            res += dict1[state] * dict2[state]
    res = res ** 2
    return res


def fidelity(state1, state2):
    return state1.dot(state2) ** 2


def normalize(vector):
    return np.sqrt(vector / vector.sum())


def plot_histogram(data, number_to_keep=False):
    """Plot a histogram of data.

    Args:
        data (dict): A dictionary of  {'000': 5, '010': 113, ...}.
        number_to_keep (int, optional): The number of terms to plot and rest is made into a
            single bar called other values. Defaults to False.

    """
    if number_to_keep is not False:
        data_temp = dict(Counter(data).most_common(number_to_keep))
        data_temp["rest"] = sum(data.values()) - sum(data_temp.values())
        data = data_temp

    labels = sorted(data)
    values = np.array([data[key] for key in labels], dtype=float)
    pvalues = values / sum(values)
    numelem = len(values)
    ind = np.arange(numelem)  # the x locations for the groups
    width = 0.35  # the width of the bars
    _, ax = plt.subplots(figsize=(20, 10))
    rects = ax.bar(ind, pvalues, width, color='seagreen')
    # add some text for labels, title, and axes ticks
    ax.set_ylabel('Probabilities', fontsize=12)
    ax.set_xticks(ind)
    ax.set_xticklabels(labels, fontsize=12, rotation=70)
    ax.set_ylim([0., min([1.2, max([1.2 * val for val in pvalues])])])
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                '%f' % float(height),
                ha='center', va='bottom')
    plt.show()


def fidelity_RBM(trained_RBM, ideal_state, num_samples=1000, num_steps=10):
    sampled_from_RBM = np.array([trained_RBM.daydream(num_steps)[-1] for _ in range(num_samples)])
    sampled_from_RBM = into_dict(sampled_from_RBM)

    return fidelity_dicts(ideal_state, sampled_from_RBM), sampled_from_RBM


def generate_phases_dataset(quantum_system, amplitudes, phases, num_units, num_samples):
    phases_dataset = dict()

    # Measurements in ZZZ... basis.
    operations = U_ZZ(num_units)
    # Resulting states and coefficients.
    res = system_evolution(quantum_system, operations, amplitudes, phases)
    hist = dict_to_hist(res)  # States and corresponding probabilities.
    phases_dataset[operations] = sample_from_hist(hist, num_samples)

    for j in range(num_units - 1):
        operations = U_XX(j, num_units)
        # Resulting states and coefficients.
        res = system_evolution(quantum_system, operations, amplitudes, phases)
        hist = dict_to_hist(res)  # States and corresponding probabilities.
        phases_dataset[operations] = sample_from_hist(hist, num_samples)

    for j in range(num_units - 1):
        operations = U_XY(j, num_units)
        # Resulting states and coefficients.
        res = system_evolution(quantum_system, operations, amplitudes, phases)
        hist = dict_to_hist(res)  # States and corresponding probabilities.
        phases_dataset[operations] = sample_from_hist(hist, num_samples)

    return phases_dataset


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


def get_all_states(n):
    all_states = np.array(list(map(np.array, itertools.product([0, 1], repeat=n))))
    return all_states
