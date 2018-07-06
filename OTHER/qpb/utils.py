from qiskit.tools.visualization import circuit_drawer, plot_histogram, plot_state
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, QuantumProgram
from qiskit import available_backends, execute
import qiskit

import matplotlib.pyplot as plt
import numpy as np
import itertools


def n_cx(circuit, controls, anticontrols, targets, aux):
    controls = controls + anticontrols

    def n_cx_one_target(target):
        def n_cx_12(circuit, controls, target):
            if (len(controls) == 1):
                circuit.cx(controls[0], target)
            elif (len(controls) == 2):
                circuit.ccx(controls[0], controls[1], target)
            else:
                raise ValueError("The controlled X with more than 2 controls is implemented as different function")

        if len(controls) <= 2:
            n_cx_12(circuit, controls, target)
        else:
            # # Left Toffoli gates.
            # circuit.ccx(controls[1], controls[2], aux[0])
            # for k, control_qubit in enumerate(controls[3:], 1):
            #     circuit.ccx(control_qubit, aux[k - 1], aux[k])

            # Center Toffoli gate.
            n_cx_12(circuit, [controls[0], aux[len(controls[3:])]], target)

            # # Right Toffoli gates.
            # for k, control_qubit in enumerate(reversed(controls[3:]), 1):
            #     circuit.ccx(control_qubit, aux[len(controls[3:]) - k], aux[len(controls[3:]) - k + 1])
            # circuit.ccx(controls[1], controls[2], aux[0])

    # Left X gates.
    for anticontrol in anticontrols:
        circuit.x(anticontrol)

    # Left Toffoli gates.
    circuit.ccx(controls[1], controls[2], aux[0])
    for k, control_qubit in enumerate(controls[3:], 1):
        circuit.ccx(control_qubit, aux[k - 1], aux[k])

    for target in targets:
        n_cx_one_target(target)

    # Right Toffoli gates.
    for k, control_qubit in enumerate(reversed(controls[3:]), 1):
        circuit.ccx(control_qubit, aux[len(controls[3:]) - k], aux[len(controls[3:]) - k + 1])
    circuit.ccx(controls[1], controls[2], aux[0])

    # Right X gates.
    for anticontrol in anticontrols:
        circuit.x(anticontrol)


def n_cz(circuit, controls, anticontrols, targets, aux):
    controls = controls + anticontrols

    def n_cz_one_target(target):
        def n_cz_12(circuit, controls, target):
            if (len(controls) == 1):
                circuit.h(target)
                circuit.cx(controls[0], target)
                circuit.h(target)
            elif (len(controls) == 2):
                circuit.h(target)
                circuit.ccx(controls[0], controls[1], target)
                circuit.h(target)
            else:
                raise ValueError("The controlled X with more than 2 controls is implemented as different function")

        if len(controls) <= 2:
            n_cz_12(circuit, controls, target)
        else:
            # # Left Toffoli gates.
            # circuit.ccx(controls[1], controls[2], aux[0])
            # for k, control_qubit in enumerate(controls[3:], 1):
            #     circuit.ccx(control_qubit, aux[k - 1], aux[k])

            # Center Toffoli gate.
            n_cz_12(circuit, [controls[0], aux[len(controls[3:])]], target)

            # # Right Toffoli gates.
            # for k, control_qubit in enumerate(reversed(controls[3:]), 1):
            #     circuit.ccx(control_qubit, aux[len(controls[3:]) - k], aux[len(controls[3:]) - k + 1])
            # circuit.ccx(controls[1], controls[2], aux[0])

    # Left X gates.
    for anticontrol in anticontrols:
        circuit.x(anticontrol)

    # Left Toffoli gates.
    circuit.ccx(controls[1], controls[2], aux[0])
    for k, control_qubit in enumerate(controls[3:], 1):
        circuit.ccx(control_qubit, aux[k - 1], aux[k])

    for target in targets:
        n_cz_one_target(target)

    # Right Toffoli gates.
    for k, control_qubit in enumerate(reversed(controls[3:]), 1):
        circuit.ccx(control_qubit, aux[len(controls[3:]) - k], aux[len(controls[3:]) - k + 1])
    circuit.ccx(controls[1], controls[2], aux[0])

    # Right X gates.
    for anticontrol in anticontrols:
        circuit.x(anticontrol)


def plot_statevector(statevector):
    plt.figure(figsize=(6, 4))

    x = np.arange(1, 1 + len(statevector))
    states = list(map(lambda x: ''.join(map(str, x)), itertools.product([0, 1], repeat=int(np.log2(len(statevector))))))

    plt.stem(x, statevector.real, 'dodgerblue', basefmt='C5-')
    plt.xticks(x, states)
    # plt.title('Amplitude (real)')
    plt.title('Amplitude amplification', fontsize=15)
    plt.xlabel('States', fontsize=15)
    plt.ylabel('Amplitude (real)', fontsize=15)
    plt.show()
