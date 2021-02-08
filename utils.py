import numpy as np


def n_kron(*inputs):
    """Create a new circuit.
        Args:
            Variable number of input matrices and vectors
        Returns:
            Kronecker product
    """
    kp = np.array([[1.0]])
    for op in inputs:
        kp = np.kron(kp, op)
    return kp
