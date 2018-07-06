# File name: collect_tomo_data.py
# Authors: Yaroslav Kharkov <y.kharkov@unsw.edu.au>, Anton Karazeev <a.karazeev@rqc.ru>
# Based on https://www.nature.com/articles/s41567-018-0048-5 paper
#
# This file is part of NNQST project (https://github.com/RQC-QApp/NNQST)
#
# Description: Quantum Tomography Data Collection module.

import numpy as np
from itertools import product


###############################################################
# Tomography circuit generation
###############################################################

def build_state_tomography_circuits(Q_program, name, qubits, qreg, creg,
                                    silent=False):
    """Add state tomography measurement circuits to a QuantumProgram.

    The quantum program must contain a circuit 'name', which is treated as a
    state preparation circuit. This function then appends the circuit with a
    tomographically overcomplete set of measurements in the Pauli basis for
    each qubit to be measured. For n-qubit tomography this result in 3 ** n
    measurement circuits being added to the quantum program.

    Returns:
        A list of names of the added quantum state tomography circuits.
        Example: ['circ_meas_X', 'circ_measY', 'circ_measZ']

    """
    labels = __add_meas_circuits(Q_program, name, qubits, qreg, creg)
    if not silent:
        print('>> created state tomography circuits for "%s"' % name)
    return labels


def __tomo_dicts(qubits, basis=None, states=False):
    """Helper function.

    Build a dictionary assigning a basis element to a qubit.

    Args:
        qubit (int): the qubit to add
        tomos (list[dict]): list of tomo_dicts to add to
        basis (list[str], optional): basis to use. If not specified
            the default is ['X', 'Y', 'Z']

    Returns:
        A new list of tomo_dict

    """
    if isinstance(qubits, int):
        qubits = [qubits]

    if basis is None:
        basis = __DEFAULT_BASIS

    if states:
        ns = len(list(basis.values())[0])
        lst = [(b, s) for b in basis.keys() for s in range(ns)]
    else:
        lst = basis.keys()

    return [dict(zip(qubits, b)) for b in product(lst, repeat=len(qubits))]


def __add_meas_circuits(Q_program, name, qubits, qreg, creg):
    """Add measurement circuits to a quantum program.

    See: build_state_tomography_circuits.
         build_process_tomography_circuits.

    """
    orig = Q_program.get_circuit(name)

    labels = []

    for dic in __tomo_dicts(qubits):

        # Construct meas circuit name.
        label = '_meas_'
        for qubit, op in dic.items():
            label += op  # + str(qubit)
        circuit = Q_program.create_circuit(label, [qreg], [creg])

        # Add gates to circuit.
        for qubit, op in dic.items():
            circuit.barrier(qreg[qubit])
            if op == "X":
                circuit.u2(0., np.pi, qreg[qubit])  # H.
            elif op == "Y":
                circuit.u2(0., 0.5 * np.pi, qreg[qubit])  # H.S^*.
            circuit.measure(qreg[qubit], creg[qubit])
        # Add circuit to QuantumProgram.
        Q_program.add_circuit(name + label, orig + circuit)
        # Add label to output.
        labels.append(name + label)
        # delete temp circuit.
        del Q_program._QuantumProgram__quantum_program[label]

    return labels


###############################################################
# Tomography circuit labels
###############################################################

def __tomo_labels(name, qubits, basis=None, states=False):
    """Helper function.

    """
    labels = []
    state = {0: 'p', 1: 'm'}
    for dic in __tomo_dicts(qubits, states=states):
        label = ''
        if states:
            for qubit, op in dic.items():
                label += op[0] + state[op[1]] + str(qubit)
        else:
            for qubit, op in dic.items():
                label += op[0] + str(qubit)
        labels.append(name + label)
    return labels


def state_tomography_circuit_names(name, qubits):
    """Return a list of state tomography circuit names.

    This list is the same as that returned by the
    build_state_tomography_circuits function.

    Args:
        name (string): the name of the original state preparation
                       circuit.
        qubits: (list[int]): the qubits being measured.

    Returns:
        A list of circuit names.

    """
    return __tomo_labels(name + '_meas', qubits)


###############################################################
# Tomography preparation and measurement bases
###############################################################

# Default Pauli basis
# This corresponds to measurements in the X, Y, Z basis where
# Outcomes 0,1 are the +1,-1 eigenstates respectively.
# State preparation is also done in the +1 and -1 eigenstates.
__DEFAULT_BASIS = {'X': [np.array([[0.5, 0.5],
                                   [0.5, 0.5]]),
                         np.array([[0.5, -0.5],
                                   [-0.5, 0.5]])],
                   'Y': [np.array([[0.5, -0.5j],
                                   [0.5j, 0.5]]),
                         np.array([[0.5, 0.5j],
                                   [-0.5j, 0.5]])],
                   'Z': [np.array([[1, 0],
                                   [0, 0]]),
                         np.array([[0, 0],
                                   [0, 1]])]}


def build_tomo_curcuit_core(Q_program, qreg, creg):
    """Prepare quantum state for tomography reconstruction.
    The state could be arbitrary.

    """
    # Example state with 3 qubits.
    circuit_name = 'tomo_c'
    my_circuit = Q_program.create_circuit(circuit_name, [qreg], [creg])
    my_circuit.h(qreg[0])
    my_circuit.cx(qreg[0], qreg[1])
    my_circuit.cx(qreg[0], qreg[2])

    my_circuit.cx(qreg[1], qreg[0])
    my_circuit.y(qreg[0])

    my_circuit.h(qreg[2])
    my_circuit.cx(qreg[2], qreg[0])

    return my_circuit


def collect_tomo_data(Q_program, backend, shots):
    """Executing data collection from IBM backend.
    The measurements bases consists of the complete set of 3**n bases states.

    """
    qubits = [0, 1, 2]
    n_qubits = len(qubits)
    qreg = Q_program.create_quantum_register("q", n_qubits)
    creg = Q_program.create_classical_register("c", n_qubits)

    _tomo_circuits = build_state_tomography_circuits(Q_program, 'tomo_c', qubits, qreg, creg)
    result = Q_program.execute(_tomo_circuits, backend=backend, shots=shots)

    tomo_data = {}
    for i_circ in range(len(_tomo_circuits)):
        data = result.get_data(_tomo_circuits[i_circ])
        basis = _tomo_circuits[i_circ][-n_qubits:]
        tomo_data[basis] = data['counts']
    return tomo_data
