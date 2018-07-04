import numpy as np
import collections
import math


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


def dict_to_quantum_system(quantum_dict):
    """Converts dict of states and corresponding coefficients into three
    lists - of states, of amplitudes and of phases.

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
    dataset = dict(collections.Counter(dataset))

    num_samples = np.sum(list(dataset.values()))

    for state in dataset:
        dataset[state] = np.sqrt(dataset[state] / num_samples)

    return dataset


def dict_to_hist(quantum_dict):
    """Converts dict of states into histogram - list of tuples of states and corresponding probabilities.

    Args:
        quantum_dict (dict): Dict of states and corresponding coefficients.

    Returns:
        np.array of tuples (state, probability).

    """
    res = list()
    print('qdict=', quantum_dict)
    for state in quantum_dict:
        tmp = (state, abs(quantum_dict[state]) ** 2)  # State and corresponding probability.
        res.append(tmp)
    return np.array(res)


def dataset_to_hist(dataset):
    """Turns `dataset` into statistics: elements and theirs occurrences.

    Args:
        dataset (np.array): Dataset of states.

    Returns:
        tuple: np.array of occurrences and np.array of states.

    """
    cntr = collections.Counter(map(tuple, dataset))

    tmp = list(cntr.items())
    data_hist = list(map(lambda x: x[0], tmp))
    data_hist = np.array(list(map(list, data_hist)), dtype=int)
    occurs = np.array(list(map(lambda x: x[1], tmp)))

    return occurs, data_hist


if __name__ == '__main__':
    raise RuntimeError('Not a main file')
