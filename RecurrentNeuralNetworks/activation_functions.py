import numpy as np


def sigmoid(Z):
    """
    Implements the sigmoid function.

    Parameters
    ----------
    Z: (ndarray of any shape) or (scalar) - input to the sigmoid function

    Returns
    -------
    A: (ndarray of same shape as Z) or (scalar) - output from the sigmoid function
    """

    A = 1 / (1 + np.exp(-Z))
    return A


def tanh(Z):
    """
    Implements the tanh function.

    Parameters
    ----------
    Z: (ndarray of any shape) or (scalar) - input to the tanh function

    Returns
    -------
    A: (ndarray of same shape as Z) or (scalar) - output from the tanh function
    """

    A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
    return A


def softmax(Z):
    """
    Implements the softmax activation.

    Parameters
    ----------
    Z: (ndarray of any shape) - input to the activation function

    Returns
    -------
    A: (ndarray of same shape as Z) - output of the activation function
    """

    # Subtracting the maximum value in each column for numerical stability to avoid overflow
    Z_stable = Z - np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(Z_stable)
    A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    return A
