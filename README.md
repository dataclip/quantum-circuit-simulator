{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Quantum Circuit Simulator\n",
    "\n",
    "<p>This is a solution for task 3 of the qosf mentorship program."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "from quantum_circuit import QuantumCircuit\n",
    "from utils import *\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "For a 3 qubit system initialize the circuit:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "qc = QuantumCircuit(3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Calculate initial state where all qubits are in state <0|:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1., 0., 0., 0., 0., 0., 0., 0.])"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "initial_state = qc.get_ground_state()\n",
    "initial_state"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Calculate operators for Hadamard gate applied to qubit 0 and a CNOT gate applied to 1 where controlled qubit is at 0:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "H = 1./np.sqrt(2) * np.array([[1, 1],[1, -1]])\n",
    "op1 = qc.op_single_q_gates(H, [0])\n",
    "op2 = qc.op_cnot([0, 1])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "calculate final state vector using the measure method of QuantumCricuit class:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "final_state = qc.measure(initial_state, [op2, op1])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Finally, for a 1000 shots calculate probability of occurences of each state:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'000': 483, '110': 517}"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "qc.get_counts(final_state)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Another example, use H gate on qubit 0 will produce equal probability of finding + and - states."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'0': 505, '1': 495}"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "qc = QuantumCircuit(1)\n",
    "initial_state = qc.get_ground_state()\n",
    "op1 = qc.op_single_q_gates(H, [0])\n",
    "final_state = qc.measure(initial_state, [op1])\n",
    "qc.get_counts(final_state)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Use H gate on qubit 0 followed by another H gate on the same qubit for a 1 qubit system: This is will return t state 0 because H is unitary."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'0': 1000}"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "qc = QuantumCircuit(1)\n",
    "initial_state = qc.get_ground_state()\n",
    "op1 = qc.op_single_q_gates(H, [0])\n",
    "op2 = qc.op_single_q_gates(H, [0])\n",
    "final_state = qc.measure(initial_state, [op2, op1])\n",
    "qc.get_counts(final_state)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Use X gate:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = np.array([[0,1],\n",
    "            [1,0]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0., 1.])"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "qc = QuantumCircuit(1)\n",
    "initial_state = qc.get_ground_state()\n",
    "op1 = qc.op_single_q_gates(X, [0])\n",
    "final_state = qc.measure(initial_state, [op1])\n",
    "final_state"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
