import matplotlib.pyplot as plt
from qiskit.tools.visualization import circuit_drawer, plot_histogram, plot_state
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, QuantumProgram
from qiskit import available_backends, execute
import qiskit
import numpy as np
import itertools


def n_controlled_Z_12(circuit, controls, target):
    if (len(controls) > 2):
        raise ValueError('The controlled Z with more than 2 controls is implemented as different function')
    elif (len(controls) == 1):
        circuit.h(target)
        circuit.cx(controls[0], target)
        circuit.h(target)
    elif (len(controls) == 2):
        circuit.h(target)
        circuit.ccx(controls[0], controls[1], target)
        circuit.h(target)


def n_controlled_Z(circuit, controls, target, aux):
    if len(controls) <= 2:
        n_controlled_Z_12(circuit, controls, target)
    else:
        # Left.
        circuit.ccx(controls[1], controls[2], aux[0])
        for k, control_qubit in enumerate(controls[3:], 1):
            circuit.ccx(control_qubit, aux[k - 1], aux[k])
        # Center.
        n_controlled_Z_12(circuit, [controls[0], aux[len(controls[3:])]], target)
        # Right.
        for k, control_qubit in enumerate(reversed(controls[3:]), 1):
            circuit.ccx(control_qubit, aux[len(controls[3:]) - k], aux[len(controls[3:]) - k + 1])
        circuit.ccx(controls[1], controls[2], aux[0])


def input_state(circuit, f_in, f_out):
    """Multiqubit input state for Grover search."""
    circuit.h(f_in)

    circuit.x(f_out)
    circuit.h(f_out)


def black_box_u_f(circuit, f_in, f_out, aux, n, exactly_1_k_sat_formula):
    """Circuit that computes the black-box function from f_in to f_out.
    Create a circuit that verifies whether a given exactly-1 k-SAT
    formula is satisfied by the input. The exactly-1 version
    requires exactly one literal out of every clause to be satisfied.
    """
    num_clauses = len(exactly_1_k_sat_formula)
    for (k, clause) in enumerate(exactly_1_k_sat_formula):
        # This loop ensures aux[k] is 1 if an odd number of literals
        # are true
        for literal in clause:
            if literal > 0:
                circuit.cx(f_in[literal - 1], aux[k])
            else:
                circuit.x(f_in[-literal - 1])
                circuit.cx(f_in[-literal - 1], aux[k])
        # Flip aux[k] if all literals are true, using auxiliary qubit
        # (ancilla) aux[num_clauses]
        circuit.ccx(f_in[0], f_in[1], aux[num_clauses])
        circuit.ccx(f_in[2], aux[num_clauses], aux[k])
        # Flip back to reverse state of negative literals and ancilla
        circuit.ccx(f_in[0], f_in[1], aux[num_clauses])
        for literal in clause:
            if literal < 0:
                circuit.x(f_in[-literal - 1])
    # The formula is satisfied if and only if all auxiliary qubits
    # except aux[num_clauses] are 1
    if (num_clauses == 1):
        circuit.cx(aux[0], f_out[0])
    elif (num_clauses == 2):
        circuit.ccx(aux[0], aux[1], f_out[0])
    elif (num_clauses == 3):
        circuit.ccx(aux[0], aux[1], aux[num_clauses])
        circuit.ccx(aux[2], aux[num_clauses], f_out[0])
        circuit.ccx(aux[0], aux[1], aux[num_clauses])
    else:
        raise ValueError('We only allow at most 3 clauses')
    # Flip back any auxiliary qubits to make sure state is consistent
    # for future executions of this routine; same loop as above.
    for (k, clause) in enumerate(exactly_1_k_sat_formula):
        for literal in clause:
            if literal > 0:
                circuit.cx(f_in[literal - 1], aux[k])
            else:
                circuit.x(f_in[-literal - 1])
                circuit.cx(f_in[-literal - 1], aux[k])
        circuit.ccx(f_in[0], f_in[1], aux[num_clauses])
        circuit.ccx(f_in[2], aux[num_clauses], aux[k])
        circuit.ccx(f_in[0], f_in[1], aux[num_clauses])
        for literal in clause:
            if literal < 0:
                circuit.x(f_in[-literal - 1])


def inversion_about_average(circuit, f_in, aux, n):
    circuit.h(f_in)
    circuit.x(f_in)

    controls = [f_in[j] for j in range(n - 1)]
    n_controlled_Z(circuit, controls, f_in[n - 1], aux)

    circuit.x(f_in)
    circuit.h(f_in)


def build_oracle(qc, f_in, aux, query):
    """Building oracle which corresponds to `query`.

    """
    query = query[::-1]
    for i in range(len(query)):
        if query[i] == '0':
            qc.x(f_in[i])

    target = f_in[0]
    controls = [f_in[i] for i in range(1, len(f_in))]

    n_controlled_Z(qc, controls, target, aux)

    for i in range(len(query)):
        if query[i] == '0':
            qc.x(f_in[i])

    return qc


def build_diffusion_operator(qc, f_in, aux):
    """Building Grover's diffusion operator.

    """
    target = f_in[0]
    controls = [f_in[i] for i in range(1, len(f_in))]

    qc.h(f_in)
    qc.x(f_in)
    n_controlled_Z(qc, controls, target, aux)
    qc.x(f_in)
    qc.h(f_in)
    return qc


def build_grover_search_qc(n, n_iter, query, measure=False):
    assert(len(query) == n)

    f_in = QuantumRegister(n, name='name')
    c = ClassicalRegister(n, name='nameans')

    aux = None
    qc = None
    # len(controls) = n - 1
    if n > 3:
        aux = QuantumRegister(n - 3, name='aux')
        qc = QuantumCircuit(f_in, aux, c, name='grover')
    else:
        qc = QuantumCircuit(f_in, c, name='grover')

    # Preparing uniform superposition.
    qc.h(f_in)

    if n_iter == 0:
        qc = build_oracle(qc, f_in, aux, query)

    for i in range(n_iter):
        qc = build_oracle(qc, f_in, aux, query)
        # if i < 1:
        qc = build_diffusion_operator(qc, f_in, aux)

    ans = ClassicalRegister(n, name='nameans')
    if measure:
        qc.measure(f_in, ans)

    return qc


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
