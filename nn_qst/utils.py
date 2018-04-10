import numpy as np


def fidelity(state1, state2):
    return state1.dot(state2) ** 2


def normalize(vector):
    return np.sqrt(vector / vector.sum())


def p_k_sigma_h(sigma, h, weights, b_bias, c_bias):
    """(A2) of arxive paper.

    """
    tot = h.dot(weights).dot(sigma)
    tot += sigma.dot(b_bias)
    tot += h.dot(c_bias)
    return np.exp(tot)


def boltzmann_margin_distribution(sigma, weights, b_bias, c_bias):
    """(A3) of arxive paper. (7) of Nature paper.

    """
    tmp = weights.dot(sigma)
    tmp += c_bias
    tmp = np.exp(tmp)
    tmp += 1
    tmp = np.log(tmp)
    tmp = tmp.sum()
    tmp += sigma.dot(b_bias)
    return np.exp(tmp)


def p_k(k, sigma, params):
    """(A3) of arxive paper.

    Args:
        k (str): either "lambda" or "mu".

    """
    return boltzmann_margin_distribution(sigma, params[k]['W'], params[k]['b'], params[k]['c'])


def p_lambda(sigma, params):
    """(A3)~

    """
    return p_k('lambda', sigma, params)


def p_mu(sigma, params):
    """(A3)~

    """
    return p_k('mu', sigma, params)


def phi_mu(sigma, params):
    """(A4)~

    """
    return np.log(p_mu(sigma, params))


def psi_lambda_mu(sigma, Z_lambda, params):
    """(A4) of arxive paper.

    """
    tmp = 1j * phi_mu(sigma, params) / 2
    tmp = np.exp(tmp)
    tmp *= np.sqrt(p_lambda(sigma, params) / Z_lambda)
    return tmp


def D_k(k, sigma, params):
    """(A8) and (A9) of arxive paper.

    Args:
        sigma (np.array): input.

    """
    weights = params[k]['W']
    c_bias = params[k]['c']

    grad_b = sigma.copy()  # (A11).

    tmp = weights.dot(sigma)
    tmp += c_bias
    tmp = np.exp(tmp)
    grad_c = tmp / (1 + tmp)  # (A12).

    grad_W = np.outer(grad_c, sigma)  # (A10).

    res = {
        'W': grad_W,
        'c': grad_c,
        'b': grad_b
    }

    return res


def Q_b(sigma, params, u=None):
    """(A13) of arxive paper. (12) of Nature paper.

    Args:
        u (?type?): basis transformation matrix.

    """
    tmp = 1j * phi_mu(sigma, params) / 2
    tmp = np.exp(tmp)
    tmp *= np.sqrt(p_lambda(sigma, params))

    if u:
        return u * tmp
    else:
        return tmp


def averaged_D_lambda_Q_b(batch, params):
    """(A16) of arxive paper. (15) of Nature paper.

    """
    quasi_probs = None
    for x in batch:
        # TODO: How to pass basis transformation matrices here?
        if quasi_probs is None:
            quasi_probs = Q_b(x, params)
        else:
            quasi_probs += Q_b(x, params)

    res_W = None
    res_b = None
    res_c = None
    for x in batch:
        gradients = D_k('lambda', x, params)
        quasi_prob = Q_b(x, params)

        if res_W is None:
            res_W = quasi_prob * gradients["W"]
            res_b = quasi_prob * gradients["b"]
            res_c = quasi_prob * gradients["c"]
        else:
            res_W += quasi_prob * gradients["W"]
            res_b += quasi_prob * gradients["b"]
            res_c += quasi_prob * gradients["c"]

    res_W /= quasi_probs
    res_b /= quasi_probs
    res_c /= quasi_probs

    res = {
        'W': res_W,
        'c': res_c,
        'b': res_b
    }

    return res


def averaged_D_lambda_p_lambda(batch, params):
    """(A18) of arxive paper. (17) of Nature paper.

    """
    n = len(batch)

    res_W = None
    res_b = None
    res_c = None

    for x in batch:
        gradients = D_k('lambda', x, params)

        if res_W is None:
            res_W = gradients["W"]
            res_b = gradients["b"]
            res_c = gradients["c"]
        else:
            res_W += gradients["W"]
            res_b += gradients["b"]
            res_c += gradients["c"]

    res_W /= n
    res_b /= n
    res_c /= n

    res = {
        'W': res_W,
        'c': res_c,
        'b': res_b
    }

    return res
