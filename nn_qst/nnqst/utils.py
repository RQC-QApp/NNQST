import matplotlib.pyplot as plt
import numpy as np
import collections

from . import paper_functions, generators


def normalize(vector):
    """Normalize vector.

    Args:
        vector (np.array):

    Returns:
        np.array:

    """
    return np.sqrt(vector / vector.sum())


def psi_RBM(trained_RBM):
    """Wave function of learned state by `trained_RBM`.

    Args:
        trained_RBM (RBM_QST):

    Returns:
        dict:

    """
    psi_RBM = dict()

    Nqub = trained_RBM.num_visible
    all_states = generators.get_all_states(Nqub)
    all_states = np.insert(all_states, 0, 1, axis=1)

    stat_sum = paper_functions.Z_lambda(trained_RBM.weights_lambda)
    for sigma in all_states:
        prob_k = paper_functions.p_k(sigma, trained_RBM.weights_lambda)
        phase_k = paper_functions.phi_k(sigma, trained_RBM.weights_mu)

        sigma_tup = tuple(sigma[1:])
        psi_RBM[sigma_tup] = np.sqrt(prob_k / stat_sum) * np.exp(1j * phase_k / 2.)
    return psi_RBM


def plot_histogram(data, number_to_keep=False):
    """Plot a histogram of data.

    Args:
        data (dict): A dictionary of  {'000': 5, '010': 113, ...}.
        number_to_keep (int, optional): The number of terms to plot and rest is made into a
            single bar called other values. Defaults to False.

    """
    if number_to_keep is not False:
        data_temp = dict(collections.Counter(data).most_common(number_to_keep))
        data_temp["rest"] = sum(data.values()) - sum(data_temp.values())
        data = data_temp

    labels = sorted(data)
    values = np.array([data[key] for key in labels], dtype=float)
    pvalues = values / sum(values)
    numelem = len(values)
    ind = np.arange(numelem)  # The x locations for the groups.
    width = 0.35  # The width of the bars.
    _, ax = plt.subplots(figsize=(20, 10))
    rects = ax.bar(ind, pvalues, width, color='seagreen')
    # Add some text for labels, title, and axes ticks.
    ax.set_ylabel('Probabilities', fontsize=12)
    ax.set_xticks(ind)
    ax.set_xticklabels(labels, fontsize=12, rotation=70)
    ax.set_ylim([0., min([1.2, max([1.2 * val for val in pvalues])])])
    # Attach some text labels.
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                '%f' % float(height),
                ha='center', va='bottom')
    plt.show()


if __name__ == '__main__':
    raise RuntimeError('Not a main file')
