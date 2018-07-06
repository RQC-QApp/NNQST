# File name: state_operations.py
# Authors: Yaroslav Kharkov <y.kharkov@unsw.edu.au>, Anton Karazeev <a.karazeev@rqc.ru>
# Based on https://www.nature.com/articles/s41567-018-0048-5 paper
#
# This file is part of NNQST project (https://github.com/RQC-QApp/NNQST)
#
# Description: Operations over state and quantum systems (rotations, etc.).

import numpy as np


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


def evolution(state, operations, coefficient=1, verbose=False):
    """Applies the sequence of `operations` to given `state`.

    Args:
        state (tuple): State.
        operations (str): String consisting of 'I', 'H' and 'K'.
        coefficient (complex, optional): Coefficient by given state. Defaults to 1.
        verbose (bool, optional): Printing log info. Defaults to False.

    Returns:
        dict: Dict of states and corresponding coefficients.

    """
    def apply_k(value):
        """Apply K gate.

        Args:
            value (int): 0 or 1.

        Returns:
            complex: Coefficient to the state (depending on `value`) after applying H gate.

        """
        assert value in [0, 1]

        res = dict()
        if value == 0:
            res[0] = 1 / np.sqrt(2)
            res[1] = 1 / np.sqrt(2)
        elif value == 1:
            res[0] = -1j / np.sqrt(2)
            res[1] = 1j / np.sqrt(2)
        return res

    def apply_h(value):
        """Apply H gate.

        Args:
            value (int): 0 or 1.

        Returns:
            float: Coefficient to the state (depending on `value`) after applying H gate.

        """
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


def system_evolution(states, operations, amplitudes, phases):
    """Performs an evolution/rotation for `states` using the
    sequence of `operations`.

    Args:
        states (list): List of tuples.
        operations (str): Sequence of operations.
        amplitudes (list): List of floats.
        phases (list): List of floats.

    Returns:
        dict: Dict of {states: coefficients}.

    """
    total = {}

    for i in range(len(states)):
        # Pass state, operations and coefficient == a * exp(i * b).
        tmp = evolution(states[i], operations,
                        amplitudes[states[i]] * np.exp(1j * phases[states[i]]))
        total = merge_dicts(total, tmp)
    return total


def merge_dicts(dict1, dict2):
    """Sums up two dicts into the new one.

    Args:
        dict1 (dict):
        dict2 (dict):

    Returns:
        dict:

    """
    tmp = {k: dict1.get(k, 0) + dict2.get(k, 0) for k in set(dict1) | set(dict2)}
    res = {k: tmp[k] for k in tmp if abs(tmp[k]) > 0.0}
    return res


if __name__ == '__main__':
    raise RuntimeError('Not a main file')
