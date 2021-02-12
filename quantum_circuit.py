import numpy as np
from numpy.linalg import multi_dot
from utils import *


class QuantumCircuit:
    """Quantum Circuit Simulator Class."""

    def __init__(self, num_qubits):
        """Create a new circuit.
            Args:
                num_qubits: The number of qubits in the circuit.
        """
        self.num_qubits = num_qubits

    def get_ground_state(self):
        """Return a Numpy.array corresponding to the ground state (all qubits in state |0>)
            of a  n qubit system .
            Returns:
                np.ndarray of size 2**num_qubits with first element 1.0 and rest 0.0
        """
        state = np.zeros(2 ** self.num_qubits)
        state[0] = 1.0
        return state

    def get_state_index(self, state):
        """Return indices of the elements of a 1d Numpy array (state vector).
            Args:
                state: 1d np.array
            Returns:
                binary indices of the array. For example, for a 2 qubit system,'00', '01',
                '10', '11'.
        """

        state_index = np.arange(0, len(state))
        state_binary_index = []

        for num in state_index:
            binary = "{0:b}".format(num)
            state_binary_index.append(binary)

        for i in range(len(state_binary_index)):
            state_binary_index[i] = (len(state_binary_index[-1]) -
                                     len(state_binary_index[i])) * '0' + state_binary_index[i]
        return state_binary_index

    def op_single_q_gates(self, gate_unitary, target_qubits, **params):
        """Return matrix operator for a single qubit gate.
            Args:
                gate_unitary: unitary matrix corresponding to a single qubit gate,
                for example, a Hadamard gate. If gate is U3, additional parameters theta, lambda and phi should be
                passed.
                target_qubits: list of target qubits
            Returns:
                Unitary operator of size 2**n x 2**n for given gate and target qubits. For example, for a 2 qubit
                system, O = U X I X I
                where O = operator, U = Unitary gate and I=Identity matrix of size 2x2. The
                position of U is varied based on the target qubit.
            Raises:
                ValueError if the gate size is not equal to 2^(len(target)*2)
        """

        # if gate_unitary.size != 2 ** (len(target_qubits) * 2):
        #     raise ValueError('You must set gate.size==2^(len(target_qubits)*2)')

        I = np.identity(2)
        list_of_identity_elements = [I] * self.num_qubits
        keys = []
        values = []
        for key, value in params.items():
            keys.append(key)
            values.append(value)

        if gate_unitary == 'U3':

            U3 = [[np.cos(values[0] / 2.0), -np.exp((0 + 1j) * values[1]) * np.sin(values[0] / 2.0)],
                  [np.exp((0 + 1j) * values[1]) * np.sin(values[0] / 2.0),
                   np.exp((0 + 1j) * values[2] + (0 + 1j) * values[1]) * np.cos(values[0] / 2.0)]]
            list_of_identity_elements[target_qubits[0]] = U3
            op = n_kron(*list_of_identity_elements)

        else:
            list_of_identity_elements[target_qubits[0]] = gate_unitary
            op = n_kron(*list_of_identity_elements)

        return op

    def op_cnot(self, target_qubits):
        """Return matrix operator for a CNOT (Controlled-X) gate.
            Args:
                :param target_qubits: list of target qubits
            Returns:
                Unitary operator for a controlled-X gate. For example, for a 3 qubit system:
                CNOT_op = n_kron(P0, I, I) + n_kron(P1, I, X) where
                controlled qubit is 0 and target qubit is 1. P0 and P1 are projection operators.
                n_kron calculates the Kronecker product of variable numbers of inputs. The
                location of X is varied based on the target qubit.
            Raises:
                ValueError if there are more than one controlled or target qubits
        """

        if len(target_qubits) > 2:
            raise ValueError('You must set one control and one target qubit')
        elif target_qubits[1] == self.num_qubits:
            raise ValueError('Index starts from zero, please set target qubit index = total qubits -1')

        zero = np.array([[1.0],
                         [0.0]])
        one = np.array([[0.0],
                        [1.0]])
        P0 = np.dot(zero, zero.T)
        P1 = np.dot(one, one.T)
        X = np.array([[0, 1],
                      [1, 0]])

        I = np.identity(2)
        i_list_p0 = [I] * self.num_qubits
        i_list_p0[0] = P0

        i_list_p1 = [I] * self.num_qubits
        i_list_p1[target_qubits[0]] = P1
        i_list_p1[target_qubits[1]] = X
        op = n_kron(*i_list_p0) + n_kron(*i_list_p1)
        return op

    def measure(self, initial_state, ops):
        """Return dot products of initial state and Unitary operators
            Args:
                :param initial_state: state vector of ann qubit system
                :param ops: list of unitary operators
            Returns:
                Final state of the system after non-unitary operations on initial state
        """
        list_ops = [op for op in ops]
        list_ops.append(initial_state)
        final_state = multi_dot(list_ops)
        return final_state

    def get_counts(self, final_state, num_shots=1000):
        """Return counts of occurrences for each state after a given number of measurements
            Args:
                :param num_shots: number of measurements
                :param final_state: Final state of the circuit
            Returns:
               a dictionary containing number of occurrences for each state with non-zero probability amplitude
               a numpy array of the count of each state

        """
        prob = np.real(final_state * np.conjugate(final_state))
        final_state_binary_index = self.get_state_index(final_state)
        weighted_random_array = np.array([np.random.choice(final_state_binary_index, p=prob)
                                          for i in range(num_shots)])
        state_index, state_count = np.unique(weighted_random_array, return_counts=True)
        counts = {}
        for key, value in zip(state_index, state_count):
            counts[key] = value
        return counts, state_count


