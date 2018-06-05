import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import itertools
import utils


def Z_lambda(weights):
    num_units = weights.shape[0] - 1
    all_states = utils.get_all_states(num_units)
    all_states = np.insert(all_states, 0, 1, axis=1)
    res = np.sum(list(map(lambda x: p_k(x, weights), all_states)))
    return res


def objective_func(weights_lambda, weights_mu, dataset):
    res = 0

    stat_sum = Z_lambda(weights_lambda)
    res = np.sum(list(map(lambda x: np.log(psi_lambda_mu(x, stat_sum, weights_lambda, weights_mu) ** 2), dataset)))

    res /= len(dataset)
    res *= -1

    return res


def p_k_sigma_h(sigma, h, weights):
    """(A2) of arxive paper. (6) of Nature paper.

    """
    #             v-- W                                   v-- b_bias              v-- c_bias.
    tot = h.dot(weights[1:, 1:]).dot(sigma) + sigma.dot(weights[1:, 0]) + h.dot(weights[0, 1:])
    return np.exp(tot)


def boltzmann_margin_distribution(sigma, weights):
    """(A3) of arxive paper. (7) of Nature paper.

    """
    sigma[0] = 1
    tmp = sigma.dot(weights)

    tmp = tmp[1:]  # Ignore bias unit.
    tmp = 1 + np.exp(tmp)

    tmp = np.log(tmp)
    tmp = np.sum(tmp)

    tmp += np.dot(sigma[1:], (weights[1:, 0]))  # b.

    return np.exp(tmp)


def p_k(sigma, weights):
    """(A3) of arxive paper.

    """
    return boltzmann_margin_distribution(sigma, weights)


def phi_k(sigma, weights):
    """(A4)~ of arxive paper.

    """
    return np.log(p_k(sigma, weights))


def psi_lambda_mu(sigma, Z_lambda, weights_lambda, weights_mu):
    """(A4) of arxive paper.

    """
    tmp = 1j * phi_k(sigma, weights_mu) / 2
    tmp = np.exp(tmp)
    tmp = np.sqrt(p_k(sigma, weights_lambda) / Z_lambda)
    return tmp


def D_k(sigma, weights):
    """(A8) and (A9) of arxive paper.

    Args:
        sigma (np.array): input.

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


# TODO: Probably it's better to do everything with phases before calling this function.
# So we don't need a `case` variable at all.
def Q_b(sigma, weights_lambda, weights_mu, case="ampl"):
    """(A13) of arxive paper. (12) of Nature paper.

    Args:
        case (str): "ampl" or "phase"

    """
    if case == "ampl":
        tmp = np.exp(1j * phi_k(sigma, weights_mu) / 2)
        tmp *= np.sqrt(p_k(sigma, weights_lambda))

    elif case == "phase":
        raise ValueError("Not implemeted!")

        # coeff, sigma = u(sigma)
        # tmp = np.exp(1j * phi_k(sigma, weights_mu) / 2)
        # tmp *= np.sqrt(p_k(sigma, weights_lambda))
        # tmp *= coeff

    else:
        raise ValueError("Wrong case")

    return tmp


def grad_lambda_ksi_MANUAL(occurs, dataset_hist, weights_lambda, weights_mu):
    """(A14) of arxive paper. (13) of Nature paper.

    """
    num_units = weights_lambda.shape[0] - 1
    all_states = utils.get_all_states(num_units)
    all_states = np.insert(all_states, 0, 1, axis=1)

    tmp1 = np.sum(list(map(lambda x: p_k(x, weights_lambda) * D_k(x, weights_lambda), all_states)), axis=0)
    tmp1 /= np.sum(list(map(lambda x: p_k(x, weights_lambda), all_states)))

    tmp2 = np.zeros((len(dataset_hist), weights_lambda.shape[0], weights_lambda.shape[1]))
    for i in range(len(dataset_hist)):
        tmp2[i, :, :] = occurs[i] * D_k(dataset_hist[i, :], weights_lambda)
    tmp2 = np.sum(tmp2, axis=0)

    res = tmp1 - tmp2 / np.sum(occurs)
    return res


# def grad_lambda_ksi(dataset, weights_lambda, weights_mu, precise=False):
#     """(A14) of arxive paper. (13) of Nature paper.
#
#     """
#     N_b = 1  # TODO: In this version we have only one basis, but in general: N_b = len(datasets).
#
#     tmp1 = None
#     if precise:
#         tmp1 = N_b * averaged_D_lambda_p_lambda_PRECISE(dataset, weights_lambda)
#     else:
#         tmp1 = N_b * averaged_D_lambda_p_lambda(dataset, weights_lambda)
#
#     # Due to we have only one basis we calculate just one component.
#     tmp2 = averaged_D_lambda_Q_b(dataset, weights_lambda, weights_mu)
#     tmp2 = tmp2.real / len(dataset)
#
#     return tmp1 - tmp2


# def averaged_D_lambda_p_lambda_PRECISE(batch, weights):
#     """(A18) of arxive paper. (17) of Nature paper.
#
#     """
#     stat_sum = Z_lambda(weights)
#
#     # Sum of gradients.
#     res = np.array(list(map(lambda x: p_k(x, weights) * D_k(x, weights), batch)))
#     res = np.sum(res, axis=0)
#     res /= stat_sum
#
#     return res


# def averaged_D_lambda_Q_b(batch, weights_lambda, weights_mu):
#     """(A16) of arxive paper. (15) of Nature paper.
#
#     """
#     # TODO: How to pass basis transformation matrices here?
#     quasi_probs = np.sum(list(map(lambda x: Q_b(x, weights_lambda, weights_mu), batch)))  # Sum of quasi probs (complex numbers).
#
#     res = np.sum(list(map(lambda x: D_k(x, weights_lambda) * Q_b(x, weights_lambda, weights_mu), batch)), axis=0)  # quasi_prob * gradients.
#     res /= quasi_probs
#
#     return res


# def averaged_D_lambda_p_lambda(batch, weights):
#     """(A18) of arxive paper. (17) of Nature paper.
#
#     """
#     res = np.sum(list(map(lambda x: D_k(x, weights), batch)), axis=0)  # Sum of gradients.
#     res /= len(batch)
#     return res

################################################
################################################
################################################

# def update_params(dataset, params, learning_rate=0.0000000000000000001):
#     res = averaged_D_lambda_p_lambda(dataset, params)  # ???.
#
#     N_b = 1  # TODO: In this version we have only one basis, but in general: N_b = len(datasets).
#     res['W'] *= N_b
#     res['b'] *= N_b
#     res['c'] *= N_b
#
#     # Due to we have only one basis we calculate just one component.
#     tmp = averaged_D_lambda_Q_b(dataset, params)
#     tmp['W'] = tmp['W'].real
#     tmp['b'] = tmp['b'].real
#     tmp['c'] = tmp['c'].real
#
#     avg_grad = {}
#     avg_grad['W'] = res['W'].copy() - (tmp['W'] / len(dataset))
#     avg_grad['b'] = res['b'].copy() - (tmp['b'] / len(dataset))
#     avg_grad['c'] = res['c'].copy() - (tmp['c'] / len(dataset))
#
#     res['W'] -= tmp['W']
#     res['b'] -= tmp['b']
#     res['c'] -= tmp['c']
#
#     W_flatten = res['W'].flatten()
#     flatten_gradients = np.zeros(len(W_flatten) + len(res['c']) + len(res['b']))
#     flatten_gradients[:len(W_flatten)] = W_flatten
#     flatten_gradients[len(W_flatten):len(W_flatten) + len(res['c'])] = res['c']
#     flatten_gradients[len(W_flatten) + len(res['c']):len(W_flatten) + len(res['c']) + len(res['b'])] = res['b']
#
#     # Fisher Information Matrix.
#     fim = np.outer(flatten_gradients, flatten_gradients)
#     fim /= len(dataset)
#     fim_inv = np.linalg.inv(fim)
#
#     grad_W_flatten = avg_grad['W'].flatten()
#     flatten_avg_gradients = np.zeros(len(grad_W_flatten) + len(avg_grad['c']) + len(avg_grad['b']))
#     flatten_avg_gradients[:len(grad_W_flatten)] = grad_W_flatten
#     flatten_avg_gradients[len(grad_W_flatten):len(grad_W_flatten) + len(avg_grad['c'])] = avg_grad['c']
#     flatten_avg_gradients[len(grad_W_flatten) + len(avg_grad['c']):len(grad_W_flatten) + len(avg_grad['c']) + len(avg_grad['b'])] = avg_grad['b']
#
#     denom = 0
#     for i in range(len(flatten_avg_gradients)):
#         for j in range(len(flatten_avg_gradients)):
#             denom += fim[i, j] * flatten_avg_gradients[i] * flatten_avg_gradients[j]
#     denom = np.sqrt(denom.copy())
#
#     W_flatten = params['lambda']['W'].flatten()
#     flatten_params = np.zeros(len(W_flatten) + len(params['lambda']['c']) + len(params['lambda']['b']))
#     flatten_params[:len(W_flatten)] = W_flatten
#     flatten_params[len(W_flatten):len(W_flatten) + len(params['lambda']['c'])] = params['lambda']['c']
#     flatten_params[len(W_flatten) + len(params['lambda']['c']):len(W_flatten) + len(params['lambda']['c']) + len(params['lambda']['b'])] = params['lambda']['b']
#     for j in range(len(flatten_params)):
#         flatten_params[j] -= flatten_avg_gradients[j] * np.sum(fim_inv[:, j]) * (learning_rate / denom)
#
#     params['lambda']['W'] = flatten_params[:len(W_flatten)].reshape((len(params['lambda']['b']), len(params['lambda']['c'])))
#     params['lambda']['c'] = flatten_params[len(W_flatten):len(W_flatten) + len(params['lambda']['c'])]
#     params['lambda']['b'] = flatten_params[len(W_flatten) + len(params['lambda']['c']):len(W_flatten) + len(params['lambda']['c']) + len(params['lambda']['b'])]
#
#     return params
