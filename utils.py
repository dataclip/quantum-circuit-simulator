import numpy as np


def n_kron(*inputs):
    """Return Kronecker product of a variable number of input.
        Args:
            Variable number of input matrices and vectors
        Returns:
            Kronecker product
    """
    kp = np.array([[1.0]])
    for op in inputs:
        kp = np.kron(kp, op)
    return kp
