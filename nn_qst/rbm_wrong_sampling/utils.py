import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import itertools

import paper_functions


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

    data is a dictionary of  {'000': 5, '010': 113, ...}
    number_to_keep is the number of terms to plot and rest is made into a
    single bar called other values
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
