from qutip import *
from qutip import gates
from qutip import qeye, sigmax, basis, tensor, expect
from qutip import Qobj
from qutip import ket2dm, basis, ptrace, tensor

import numpy as np
from collections import defaultdict
from numpy.random import multivariate_normal, choice
import scipy.sparse as sp

import matplotlib.pyplot as plt
from itertools import product
from itertools import product
from functools import partial, reduce
from operator import mul
from itertools import product

from qutip import Qobj, qeye, sigmax, basis, tensor
from qutip.measurement import measure

from collections import Counter

# my noise file
import depolarizing_noise
from depolarizing_noise import *
from depolarizing_noise import ibm_sherbrooke_params


def print_state_nicely(state):
    state = state.full().flatten()  # Convert to a flat NumPy array
    n_qubits = int(np.log2(len(state)))
    basis_labels = [''.join(map(str, bits)) for bits in product([0, 1], repeat=n_qubits)]

    for amplitude, label in zip(state, basis_labels):
        if np.abs(amplitude) > 1e-10:  # Only show significant components
            print(f"{amplitude:.3g} |{label}>")
            

def single_qubit_gate(G, target, N=4):
    ops = [qeye(2)] * N
    ops[target] = G
    return Qobj(tensor(ops).full())


def manual_cnot(control, target, N=4, flatten=True):
    """
    Constructs a CNOT gate on an N-qubit system where `control` and `target` are the qubit indices.
    If `flatten=True`, returns the result as a 2^N x 2^N operator with dims=[[2^N], [2^N]].
    """
    if control == target:
        raise ValueError("Control and target must be different.")
    if not (0 <= control < N) or not (0 <= target < N):
        raise ValueError(f"Qubit indices must be between 0 and {N-1}")

    I = qeye(2)
    X = sigmax()
    P0 = basis(2, 0) * basis(2, 0).dag()
    P1 = basis(2, 1) * basis(2, 1).dag()

    def build_term(ctrl_proj, target_op):
        ops = []
        for i in range(N):
            if i == control:
                ops.append(ctrl_proj)
            elif i == target:
                ops.append(target_op)
            else:
                ops.append(I)
        return tensor(ops)

    U = build_term(P0, I) + build_term(P1, X)

    if flatten:
        return Qobj(U.full(), dims=[[2**N], [2**N]])
    else:
        return U  # return with proper qubit-structured dims



def custom_measure(psi_original, i, N):
    """
    Manually perform a projective measurement on qubit `i` of an N-qubit pure state,
    returning the measurement result and collapsed (pure) state.
    """
    if i < 0 or i >= N:
        raise ValueError(f"Invalid qubit index i={i} for N={N}")

    # Ensure correct structure of the state
    psi = psi_original.copy()
    dim_list = [[2]*N, [1]*N]
    psi = Qobj(psi.full().reshape((2**N, 1)), dims=dim_list)

    # Build projectors P0 and P1 for qubit i
    P0_list = [qeye(2)] * N
    P1_list = [qeye(2)] * N
    P0_list[i] = basis(2, 0) * basis(2, 0).dag()
    P1_list[i] = basis(2, 1) * basis(2, 1).dag()
    P0 = tensor(P0_list)
    P1 = tensor(P1_list)
    
    # Compute probabilities (unnormalized)
    p0 = float((psi.dag() * P0 * psi).real)
    p1 = float((psi.dag() * P1 * psi).real)

    total = p0 + p1
    p0_norm = p0 / total if total > 0 else 0
    p1_norm = p1 / total if total > 0 else 0

    # Sample measurement outcome
    outcome = np.random.choice([0, 1], p=[p0_norm, p1_norm])

    # Collapse the state based on outcome
    if outcome == 0:
        collapsed = (P0 * psi).unit()
    else:
        collapsed = (P1 * psi).unit()

    return outcome, collapsed



def noisy_reset_with_gate(psi_og, i, N, p, T1, T2, tg, additional_noise = False):
    """
    Reset a qubit to |0⟩ via:
    - projective measurement
    - conditional application of a noisy X gate if outcome == 1
    """
    theta=np.pi
    phi=0
    psi=np.pi/2
    
    outcome, collapsed_state = custom_measure(psi_og, i, N)  # Measure qubit 0 and get collapsed state

    if outcome == 0:
        return collapsed_state
    else:
        # Apply noisy X gate (you already have it defined with correct params)
        noisy_X = Qobj(X_gate_construction(p, T1, T2, tg, additional_noise))
        U = single_qubit_gate(noisy_X, i, N)
        collapsed_state_full = Qobj(collapsed_state.full())  # Fix dims
        new_state = U * collapsed_state_full
        return new_state


def measure_all(psi_original, N):
    """
    Sequentially measure all qubits in a pure N-qubit state using custom_measure.
    Returns the measurement result as a bitstring and the final collapsed state.
    """
    psi = psi_original.copy()
    result_bits = []

    for i in range(N):
        outcome, psi = custom_measure(psi, i, N)
        result_bits.append(str(outcome))

    bitstring = ''.join(result_bits)
    return bitstring, psi


def simulate_circuit(psi, shots=1000):
    """
    Simulate projective measurement of all qubits using measure_all
    for a specified number of shots. Works on 4-qubit pure states.
    """
    N = 4
    outcomes = []

    for _ in range(shots):
        bitstring, psi_out = measure_all(psi, N)
        outcomes.append(bitstring)

    # Count frequencies of all 16 possible bitstrings
    labels = [''.join(bits) for bits in product('01', repeat=N)]
    counts = {label: outcomes.count(label) for label in labels}

    # Plot
    plt.bar(counts.keys(), counts.values(), color='skyblue')
    plt.title(f"Measurement outcomes over {shots} shots")
    plt.xlabel("Basis state")
    plt.ylabel("Counts")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
def simulate_noisy_4_qubit_surface(psi0_flat, p, T1, T2, tg, shots=100, plot_bool=False, additional_noise=False):
    """
    Simulate a 4-qubit noisy circuit with fresh noise per shot.
    - psi0_flat: initial 4-qubit state
    - H_gate_construction: function to generate noisy H gates
    - manual_cnot: function to construct full-system CNOT gate
    - p, T1, T2, tg: noise parameters
    - shots: number of circuit repetitions
    """
    outcomes = []
    psi_outcomes = []

    for _ in range(shots):
        # Step 0: reset qubits 0, 3 at the start 
        psi_in = noisy_reset_with_gate(psi0_flat,0,4, p, T1, T2, tg, additional_noise)
        psi_in = noisy_reset_with_gate(psi_in,3,4, p, T1, T2, tg, additional_noise)
        
        # Step 1: Rebuild circuit with fresh noisy H gates
        U = single_qubit_gate(Qobj(H_gate_construction(p, T1, T2, tg, additional_noise)), 0, 4)
        U *= manual_cnot(0, 1, 4)
        U *= manual_cnot(0, 2, 4)
        U *= manual_cnot(1, 3, 4)
        U *= manual_cnot(2, 3, 4)
        U *= single_qubit_gate(Qobj(H_gate_construction(p, T1, T2, tg, additional_noise)), 0, 4)
        
        # Step 2: Apply noisy circuit to fresh initial state
        U = Qobj(U.full(), dims=[[2]*4, [2]*4])
        psi = U * psi_in

        
        psi_outcomes.append(psi)
        
        # Step 3: Measure qubits 0 and 3 sequentially
        result0, psi = custom_measure(psi, i=0, N=4)
        result3, psi = custom_measure(psi, i=3, N=4)

        # Step 4: Measure remaining qubits (1 and 2) to get full bitstring
        result1, psi = custom_measure(psi, i=1, N=4)
        result2, psi = custom_measure(psi, i=2, N=4)

        # Step 5: Assemble outcome bitstring in correct qubit order (0 to 3)
        bitstring = f"{result0}{result1}{result2}{result3}"
        outcomes.append(bitstring)

    # Tally and plot
    counts = Counter(outcomes)
    labels = [''.join(bits) for bits in product('01', repeat=4)]
    counts = {label: counts.get(label, 0) for label in labels}

    # Plot
    if plot_bool:
        plt.bar(counts.keys(), counts.values(), color='skyblue')
        plt.title(f"Noisy 4-qubit circuit outcomes over {shots} shots"+ 
            f'\n $p = {p:.3e}$',)
        plt.xlabel("Basis state")
        plt.ylabel("Counts")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return counts, psi_outcomes



def run_parameter_sweep_with_closeness(param_name, param_range, shots, psi0_flat, p, T1, T2, tg, simulate_func, ideal_state, additional_noise = False):
    """
    Run a sweep over a specified noise parameter for a 4-qubit noisy surface code simulation,
    returning trace closeness to unity instead of raw trace norm.

    Parameters:
    - param_name: str, one of 'p', 'T1', 'T2', or 'tg'
    - param_range: iterable of values to sweep over for the parameter
    - shots: int, number of shots per parameter setting
    - psi0_flat: initial state
    - p, T1, T2, tg: default values for noise parameters
    - simulate_func: function handle to the simulation function
    - ideal_state: Qobj, ideal pure state for fidelity comparisons

    Returns:
    - results_dict: dictionary {param_val: [psi_outcomes]}
    - closeness_dict: dictionary {param_val: [1 - abs(1 - trace norm)]}
    - fidelity_dict: dictionary {param_val: [fidelity values]}
    - avg_closeness: dictionary {param_val: mean closeness}
    - avg_fidelities: dictionary {param_val: mean fidelity}
    """
    results_dict = {}
    closeness_dict = {}
    fidelity_dict = {}
    fidelity_close_dict = {}
    avg_closeness = {}
    avg_fidelities = {}
    avg_fidelities_close = {}

    for val in param_range:
        kwargs = {'p': p, 'T1': T1, 'T2': T2, 'tg': tg}
        kwargs[param_name] = val

        counts, psi_outcomes = simulate_func(psi0_flat, **kwargs, shots=shots, additional_noise=additional_noise)
        results_dict[val] = psi_outcomes

        trace_vals = [psi.norm()**2 for psi in psi_outcomes]
        closeness_vals = [1 - abs(1 - tr) for tr in trace_vals]
        
        fidelity_vals = [fidelity(psi, ideal_state) for psi in psi_outcomes]
        fid_close_val = [1 - np.abs(1 - fid) for fid in fidelity_vals]
        

        closeness_dict[val] = closeness_vals
        fidelity_dict[val] = fidelity_vals
        fidelity_close_dict[val] = fid_close_val
        avg_closeness[val] = np.mean(closeness_vals)
        avg_fidelities[val] = np.mean(fidelity_vals)
        avg_fidelities_close[val] = np.mean(fid_close_val)
        
    return results_dict, closeness_dict, fidelity_dict, avg_closeness, avg_fidelities, fidelity_close_dict, avg_fidelities_close


def hellinger_distance_diag(rho, sigma):
    """
    Compute the classical Hellinger distance between the diagonals of two density matrices.
    
    Parameters:
    - rho: Qobj, quantum state (pure or mixed)
    - sigma: Qobj, quantum state (pure or mixed)
    
    Returns:
    - Classical Hellinger distance between the diagonals (float)
    """
    # Ensure density matrices
    rho_diag = np.real(rho.diag())
    sigma_diag = np.real(sigma.diag())
    
    # Normalize just in case
    rho_diag /= np.sum(rho_diag)
    sigma_diag /= np.sum(sigma_diag)
    
    # Classical Hellinger distance formula
    diff = np.sqrt(rho_diag) - np.sqrt(sigma_diag)
    H = (1 / np.sqrt(2)) * np.linalg.norm(diff)
    return H
