import numpy as np

from . import generators, state_operations


def Z_lambda(weights):
    """Calculate statisticsl sum Z_lambda.

    Args:
        weights (np.array):

    Returns:
        float:

    """
    num_units = weights.shape[0] - 1
    all_states = generators.get_all_states(num_units)
    all_states = np.insert(all_states, 0, 1, axis=1)
    res = np.sum(list(map(lambda x: p_k(x, weights), all_states)))
    return res


def objective_func(quantum_system, weights_lambda, weights_mu, dataset, basis_set):
    """Calculate objective function.

    Args:
        quantum_system (list):
        weights_lambda (np.array):
        weights_mu (np.array):
        dataset (dict): Measurements
        basis_set (list): List of bases (strings).

    Returns:
        float:

    """
    res = 0
    Nqub = weights_lambda.shape[0] - 1
    Nb = len(basis_set)
    stat_sum = Z_lambda(weights_lambda)

    amplitudes, phases = {}, {}
    all_states = generators.get_all_states(Nqub)
    all_states = np.insert(all_states, 0, 1, axis=1)

    for sigma in all_states:
        amplitudes[tuple(sigma[1:])] = np.sqrt(p_k(sigma, weights_lambda) / stat_sum)
        phases[tuple(sigma[1:])] = phi_k(sigma, weights_mu) / 2

    for basis in basis_set:
        dataset_b = dataset[basis]

        tmp = 0.
        rot_state = state_operations.system_evolution(quantum_system, basis, amplitudes, phases)
        occurs = list(dataset_b.values())  # vector of occurencies

        for state in rot_state.keys():
            if state in dataset_b:
                psi_i = rot_state[state]
                n_occurs = dataset_b[state]
                tmp += n_occurs * np.log(abs(psi_i) ** 2)

        tmp /= np.sum(occurs)
        res += tmp
    res *= -1. / Nb

    return res


def p_k_sigma_h(sigma, h, weights):
    """(A2) of arxive paper. (6) of Nature paper.

    Args:
        sigma (np.array):
        h (np.array):
        weights (np.array):

    Returns:
        np.array:

    """
    #             v-- W                                   v-- b_bias              v-- c_bias.
    tot = h.dot(weights[1:, 1:]).dot(sigma) + sigma.dot(weights[1:, 0]) + h.dot(weights[0, 1:])
    return np.exp(tot)


def boltzmann_margin_distribution(sigma, weights, verbose=False):
    """(A3) of arxive paper. (7) of Nature paper.

    Args:
        sigma (np.array):
        weights (np.array):
        verbose (bool, optional): Defaults to False.

    Returns:
        np.array:

    """
    sigma[0] = 1
    tmp = sigma.dot(weights)

    tmp = tmp[1:]  # Ignore bias unit.
    if verbose:
        print(tmp)
    tmp = 1 + np.exp(tmp)

    tmp = np.log(tmp)
    tmp = np.sum(tmp)

    tmp += np.dot(sigma[1:], (weights[1:, 0]))  # b.

    return np.exp(tmp)


def p_k(sigma, weights, verbose=False):
    """(A3) of arxive paper.

    Args:
        sigma (np.array):
        weights (np.array):
        verbose (bool, optional): Defaults to False.

    Returns:
        np.array:

    """
    return boltzmann_margin_distribution(sigma, weights, verbose)


def phi_k(sigma, weights):
    """(A4)~ of arxive paper.

    Args:
        sigma (np.array):
        weights (np.array):

    Returns:
        np.array:

    """
    # print('> p_k:', p_k(sigma, weights))
    # print('> sigma:', sigma)
    # print('> weights:', weights)
    # print(' ')
    return np.log(p_k(sigma, weights))


def psi_lambda_mu(sigma, Z_lambda, weights_lambda, weights_mu):
    """(A4) of arxive paper.

    Args:
        sigma (np.array):
        Z_lambda (float):
        weights_lambda (np.array):
        weights_mu (np.array):

    Returns:
        complex:

    """
    tmp = 1j * phi_k(sigma, weights_mu) / 2
    tmp = np.exp(tmp)
    tmp *= np.sqrt(p_k(sigma, weights_lambda) / Z_lambda)
    return tmp


def D_k(sigma, weights):
    """(A8) and (A9) of arxive paper.

    Args:
        sigma (np.array): Input.
        weights (np.array):

    Returns:
        np.array:

    """
    sigma[0] = 1
    tmp = np.dot(sigma, weights)
    tmp = tmp[1:]  # Ignore bias unit.
    tmp = np.exp(tmp)
    grad_c = tmp / (1 + tmp)

    grad_b = sigma[1:].copy()
    grad_W = np.outer(grad_b, grad_c)

    res = np.insert(grad_W, 0, 0, axis=0)
    res = np.insert(res, 0, 0, axis=1)

    res[0, 1:] = grad_c
    res[1:, 0] = grad_b

    return res


def averaged_D_lambda_Q_b(dataset, weights_lambda, weights_mu):
    """(A16) of arxive paper. (15) of Nature paper.
    Measurements only in Z basis !!!

    Args:
    dataset (dict):
    weights_lambda (np.array):
    weights_mu (np.array):

    Returns:
        np.array:

    """
    Nqub = weights_lambda.shape[0] - 1

    dataset_Z = dataset['I' * Nqub]  # Selecting only Z - basis measurements.
    # Array of sigmas.
    sigmas = np.array(list(dataset_Z.keys()))
    sigmas = np.insert(sigmas, 0, 1, axis=1)
    occurs = list(dataset_Z.values())  # List of occurrences for each sigma.

    tmp2 = np.zeros((len(sigmas), weights_lambda.shape[0], weights_lambda.shape[1]))
    for i in range(len(sigmas)):
        sigma = sigmas[i, :]
        sigma_state = tuple(sigma[1:])
        n_occur = dataset_Z[sigma_state]
        tmp2[i, :, :] = n_occur * D_k(sigma, weights_lambda)

    tmp2 = np.sum(tmp2, axis=0)
    tmp2 /= np.sum(occurs)
    return tmp2


def averaged_D_lambda_p_lambda_PRECISE(dataset, weights_lambda):
    """(A18) of arxive paper. (17) of Nature paper.

    Args:
        dataset (dict):
        weights_lambda (np.array):

    Returns:
        np.array:

    """
    Nqub = weights_lambda.shape[0] - 1
    all_states = generators.get_all_states(Nqub)
    all_states = np.insert(all_states, 0, 1, axis=1)

    stat_sum = Z_lambda(weights_lambda)

    # Sum of gradients.
    tmp1 = np.sum(list(map(lambda x: p_k(x, weights_lambda) * D_k(x, weights_lambda), all_states)), axis=0)
    tmp1 /= stat_sum

    return tmp1


def grad_lambda_ksi(dataset, weights_lambda, weights_mu, precise=True):
    """(A14) of arxive paper. (13) of Nature paper.
    Gradient for amplitudes reconstruction. Note that in this version we perform measurements only in Z-basis.

    Args:
    dataset (dict):
    weights_lambda (np.array):
    weights_mu (np.array):
    precise (bool, optional): Defaults to True.

    Returns:
        np.array:

    """
    N_b = 1

    # PRECISE method with exact evaluation of partition function Z.
    if precise:
        tmp1 = N_b * averaged_D_lambda_p_lambda_PRECISE(dataset, weights_lambda)
    else:
        raise ValueError('This case is not implemented yet')

    # Since we have only one basis we calculate just one component.
    tmp2 = averaged_D_lambda_Q_b(dataset, weights_lambda, weights_mu)
    tmp2 = tmp2.real

    return tmp1 - tmp2


def averaged_D_mu_Q_b(sigma_b, weights_lambda, weights_mu, basis):
    """(A16) of arxive paper. (15) of Nature paper.

    Args:
        sigma_b (np.array): Spin configuration, an array of {1,0}, len=N+1.
        weights_lambda (np.array): Weights for amplitudes RBM, shape=(N+1, M+1).
        weights_mu (np.array): Weights for phases RBM, shape=(N+1, M+1)
        basis (string): String of basis transforms for each qubit, len=N
            (e.g. 'IIHKII', where I - identity, H - rotation in X basis, K - rotation in Y basis).

    Returns:
        float:

    """
    Nqub = weights_lambda.shape[0] - 1

    all_states = generators.get_all_states(Nqub)
    all_states = np.insert(all_states, 0, 1, axis=1)

    # Realizing formula B17 from Appendix, arxiv version
    sigma1, sigma0 = sigma_b.copy(), sigma_b.copy()

    if basis.find('H') != -1:
        symb = 'H'
        delta_symb = 1  # Coefficient due to H matrix.
    elif basis.find('K') != -1:
        symb = 'K'
        delta_symb = -1j  # Coefficient due to K matrix.

    indx_b = basis.find(symb) + 1  # Shift by 1 because of the fictious 0th sigma unit.

    sigma1[indx_b] = 1
    sigma0[indx_b] = 0

    prob1 = p_k(sigma1, weights_lambda)
    prob0 = p_k(sigma0, weights_lambda)
    phi1 = phi_k(sigma1, weights_mu)
    phi0 = phi_k(sigma0, weights_mu)

    xi_lambda_mu = np.sqrt(prob1 / prob0) * np.exp(1j * (phi1 - phi0) / 2.)  # Eq. B18.

    numerator = D_k(sigma0, weights_lambda) + D_k(sigma1, weights_lambda) * xi_lambda_mu * (1. - 2. * sigma_b[indx_b]) * delta_symb
    denominator = 1. + xi_lambda_mu * (1. - 2. * sigma_b[indx_b]) * delta_symb

    aver_D_mu_Qb = numerator / denominator

    return aver_D_mu_Qb


def grad_mu_ksi(dataset, basis_set, weights_lambda, weights_mu):
    """(A14) of arxive paper. (13) of Nature paper.
    Gradient for phases reconstruction.

    Args:
        dataset (dict):
        basis_set (list): List of bases (strings).
        weights_lambda (np.array):
        weights_mu (np.array):

    Returns:
        float:

    """
    Nb = len(dataset.keys())

    tmp, res = 0., 0.
    for basis in basis_set:
        sigmas = list(dataset[basis].keys())
        sigmas = np.array(sigmas)
        sigmas = np.insert(sigmas, 0, 1, axis=1)
        occurs = list(dataset[basis].values())

        for sigma in sigmas:
            n_occurs = dataset[basis][sigma[1:]]
            tmp += n_occurs * averaged_D_mu_Q_b(sigma, weights_lambda, weights_mu, basis)
        res += tmp.imag / (np.sum(occurs) * Nb)

    return res


def update_weights_mu_Fisher(batch, weights_lambda, weights_mu, learning_rate, use_denom=False):
    """Fisher Information Matrix.

    Args:
        batch (dict):
        weights_lambda (np.array):
        weights_mu (np.array):
        learning_rate (float):
        use_denom (bool, optional): Defaults to False.

    Returns:
        np.array: Matrix of gradients for `weigths_mu`.

    """
    weights_flatten = weights_mu.flatten()

    dimW = len(weights_flatten)
    fim = np.zeros((dimW, dimW))
    av_flatten_grad_mu = np.zeros(dimW)
    basis_set = list(batch.keys())

    n_tot = 0
    for basis in basis_set:
        sigmas = list(batch[basis].keys())
        sigmas = np.array(sigmas)
        sigmas = np.insert(sigmas, 0, 1, axis=1)
        for sigma in sigmas:
            sigma_tup = tuple(sigma[1:])
            # Calculating S_ij.
            n_occurs = batch[basis][sigma_tup]
            tmp = averaged_D_mu_Q_b(sigma, weights_lambda, weights_mu, basis).imag
            flatten_grad_mu = tmp.flatten()

            av_flatten_grad_mu += n_occurs * flatten_grad_mu
            fim += n_occurs * np.outer(flatten_grad_mu, flatten_grad_mu)
            # Calculating <g_j>_B.
            n_tot += n_occurs

    fim /= n_tot
    av_flatten_grad_mu /= n_tot

    fim += 1.e-5 * np.eye(dimW)

    tmp = np.dot(fim, av_flatten_grad_mu)
    if use_denom:
        denom = np.dot(av_flatten_grad_mu, np.matmul(fim, av_flatten_grad_mu))
        eta = learning_rate / np.sqrt(denom)
    else:
        eta = learning_rate

    upd_weights_mu = weights_mu.flatten() - eta * tmp
    upd_weights_mu = upd_weights_mu.reshape(weights_mu.shape)

    return upd_weights_mu


if __name__ == '__main__':
    raise RuntimeError('Not a main file')
