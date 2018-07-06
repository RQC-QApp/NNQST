import numpy as np

from . import utils, state_representations


def fidelity_dicts(dict1, dict2):
    """Calculate fidelity between states corresponding to `dict1` and `dict2`.

    Args:
        dict1 (dict): {state: <amplitude * exp(phase)>}.
        dict2 (dict): {state: <amplitude * exp(phase)>}.

    Returns:
        float:

    """
    res = 0.
    for state in dict1:
        if state in dict2:
            res += np.conj(dict1[state]) * dict2[state]
    res = np.abs(res) ** 2
    return res


def fidelity_RBM_PRECISE(trained_RBM, ideal_state):
    """Calculate fidelity between learned state by RBM and `ideal_state` PRECISELY.

    Args:
        trained_RBM (RBM_QST):
        ideal_state (dict):

    Returns:
        float:

    """
    psi_rbm = utils.psi_RBM(trained_RBM)
    return fidelity_dicts(ideal_state, psi_rbm)


def fidelity_RBM(trained_RBM, ideal_state, num_samples=1000, num_steps=10):
    """Short summary.

    Args:
        trained_RBM (RBM_QST):
        ideal_state (dict):
        num_samples (int, optional): Defaults to 1000.
        num_steps (int, optional): Defaults to 10.

    Returns:
        float:

    """
    sampled_from_RBM = np.array([trained_RBM.daydream(num_steps)[-1] for _ in range(num_samples)])
    sampled_from_RBM = list(map(lambda x: tuple(map(int, x)), sampled_from_RBM))
    sampled_from_RBM = state_representations.into_dict(sampled_from_RBM)

    return fidelity_dicts(ideal_state, sampled_from_RBM), sampled_from_RBM


if __name__ == '__main__':
    raise RuntimeError('Not a main file')
