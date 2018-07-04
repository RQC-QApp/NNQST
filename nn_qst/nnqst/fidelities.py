import numpy as np

from . import utils, state_representations


def fidelity_dicts(dict1, dict2):
    res = 0
    for state in dict1:
        if state in dict2:
            res += np.conj(dict1[state]) * dict2[state]
    res = np.abs(res) ** 2
    return res


def fidelity(state1, state2):
    return abs(state1.dot(np.conj(state2))) ** 2


def fidelity_RBM_PRECISE(trained_RBM, ideal_state):
    psi_rbm = utils.psi_RBM(trained_RBM)
    return fidelity_dicts(ideal_state, psi_rbm)


def fidelity_RBM(trained_RBM, ideal_state, num_samples=1000, num_steps=10):
    sampled_from_RBM = np.array([trained_RBM.daydream(num_steps)[-1] for _ in range(num_samples)])
    sampled_from_RBM = state_representations.into_dict(sampled_from_RBM)

    return fidelity_dicts(ideal_state, sampled_from_RBM), sampled_from_RBM


if __name__ == '__main__':
    raise RuntimeError('Not a main file')
