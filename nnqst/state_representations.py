import numpy as np
import collections
import math


def polar(z):
    """Convert complex number to polar representation.

    Args:
        z (complex): Number to convert.

    Returns:
        tuple: Tuple of float for `r` and float for angle `theta`.

    """
    a = z.real
    b = z.imag
    r = math.hypot(a, b)
    theta = math.atan2(b, a)
    if theta < 0:
        theta += 2 * np.pi
    return r, theta


def dict_to_quantum_system(quantum_dict):
    """Convert dict of states and corresponding coefficients into three
    lists - of states, of amplitudes and of phases.

    Args:
        quantum_dict (dict): States and corresponding probabilities.

    Returns:
        tuple: Tuple of three lists -- `quantum_system`, `amplitudes`, `phases`.

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


def into_dict(dataset):
    """Convert list of np.array into dict.

    Args:
        dataset (np.array): Sampled or ideal states.

    Returns:
        dict: {state: frequency}.

    """
    dataset = dataset.copy()
    dataset = dict(collections.Counter(dataset))

    num_samples = np.sum(list(dataset.values()))
    for state in dataset:
        dataset[state] = np.sqrt(dataset[state] / num_samples)

    return dataset


def get_probabilities(quantum_system):
    """Convert dict of states into histogram - list of tuples of states and corresponding probabilities.

    Args:
        quantum_system (dict): Dict of states and corresponding coefficients.

    Returns:
        list: list of tuples (state, probability).

    """
    res = [(state, abs(quantum_system[state]) ** 2) for state in quantum_system]  # State and corresponding probability.
    return res


def get_occurrences(dataset):
    """Turn `dataset` into statistics: elements and theirs occurrences.

    Args:
        dataset (np.array): Dataset of states.

    Returns:
        tuple: np.array of occurrences and np.array of states.

    """
    cntr = collections.Counter(dataset)

    tmp = cntr.items()
    data_hist = list(map(lambda x: x[0], tmp))
    occurs = list(map(lambda x: x[1], tmp))

    return occurs, data_hist


if __name__ == '__main__':
    raise RuntimeError('Not a main file')
