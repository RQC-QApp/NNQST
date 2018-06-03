import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import itertools
import math


def evolution(state, operations, coefficient=1, verbose=False):
    """
    Args:
        state (tuple): State.
        operations (str): String consisting of 'I', 'H' and 'K'.
        coefficient (complex): Coefficient by given state.

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
    tmp = {k: dict1.get(k, 0) + dict2.get(k, 0) for k in set(dict1) | set(dict2)}
    res = {k: tmp[k] for k in tmp if abs(tmp[k]) > 0.0}
    return res


def dict_to_quantum_system(quantum_dict):
    """
    Returns:
        quantum_system (list):
        amplitudes (list):
        phases (list):

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
    res = list()
    for state in quantum_dict:
        tmp = (state, abs(quantum_dict[state]) ** 2)  # State and corresponding probability.
        res.append(tmp)
    return np.array(res)


def system_evolution(quantum_system, operations, amplitudes, phases):
    assert len(operations) == len(quantum_system), "Lengths must be the same."

    total = dict()
    for i in range(len(quantum_system)):
        # Pass state, operations and coefficient == a * exp(i * b).
        tmp = evolution(quantum_system[i], operations,
                        amplitudes[i] * np.exp(1j * phases[i]))
        total = merge_dicts(total, tmp)

    return total


def random_phases(size):
    return 2 * np.pi * np.random.random(size)


def polar(z):
    a = z.real
    b = z.imag
    r = math.hypot(a, b)
    theta = math.atan2(b, a)
    if theta < 0:
        theta += 2 * np.pi
    return r, theta


def U_XX(j, n):
    assert j < n - 1
    assert n > 2
    return 'I' * j + "HH" + 'I' * (n - j - 2)


def U_XY(j, n):
    assert j < n - 1
    assert n > 2
    return 'I' * j + "HK" + 'I' * (n - j - 2)


def U_ZZ(n):
    return 'I' * n


def sample_from_hist(histogram, size=100):
    states = histogram[:, 0]
    probs = histogram[:, 1].astype(float)
    sampled = np.random.choice(states, size=size, p=probs)
    sampled = np.array(list(map(list, sampled))).astype(float)
    return sampled
