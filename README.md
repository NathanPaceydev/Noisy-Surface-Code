# Noisy Surface Code Simulations

This repository investigates the effect of physically motivated stochastic gate noise on quantum error correction using surface codes. It includes implementations in both Qiskit and QuTiP, and extends a noisy gates framework to support realistic quantum circuit behavior. The work is based on simulations of 4-qubit and 9-qubit planar surface codes, with a focus on benchmarking fidelity and understanding decoherence impacts.

## 📁 Repository Structure

### `Calculations/`
Jupyter notebooks for computing symbolic and numerical properties of the noise model:
- `2D_covariance_calcs.ipynb`: Numerical integration for 2D noise covariance matrices.
- `3D_covariance_calcs.ipynb`: Similar calculations in 3D, supporting arbitrary rotation axes.
- `symbolic_calcs 2D.ipynb` & `symbolic_calcs 3D.ipynb`: Symbolic derivations of Itô variances and covariances used in the noise model.

### `Noisy Quantum Gates Library/`
Development of noisy gate models and their implementation:
- `Noisy Gate Breakdown.ipynb`: Detailed breakdown of stochastic noise applied to single-qubit gates.
- `Noisy 2-qubit Gate Breakdown.ipynb`: (In development) Intended extension toward noisy 2-qubit gate modeling.

### `Noisy Surface code/`
Experiments and simulations involving the 4-qubit surface code:
- `noisy4qubitSurface.ipynb`: Implements a 4-qubit surface code simulation using both Qiskit and QuTiP.
- `Qiskit Qutip Trajectory Comparison.ipynb`: Compares results between Qiskit and QuTiP trajectories.
- `Qiskit Qutip noise comparison.ipynb`: Plots and compares noise statistics across simulators.
- `depolarizing_noise.py`: Defines the noise models used in simulations.
- `qutip_functions.py`: Utility functions for simulating measurement trajectories and noise in QuTiP.

### `Qiskit Surface code/`
Implements larger-scale simulations in Qiskit:
- `9-qubit Surface Model.ipynb`: A 9-qubit planar code model following Fowler et al.'s layout.
- `Simple Model Circuit.ipynb`: A simpler prototype for testing noise integration in Qiskit.

### `Report and Presentation/`
Contains documentation and presentation material:
- `Surface codes Report.pdf`: Full report describing the motivation, methods, and results.
- `Surface Codes Presentation.pptx/pdf`: Slides summarizing the key points.
- `Report/Surface codes Report/`: Includes LaTeX sources (`Covariance Derivations/`) and figures for the PDF report.

## 📘 Summary

This project:
- Simulates surface code layouts in Qiskit and QuTiP.
- Introduces a generalized noise model with support for arbitrary-axis rotations and mid-circuit operations.
- Validates results via fidelity metrics and statistical comparisons.
- Lays the groundwork for benchmarking error mitigation schemes.


## 🤝 Contributors

* Cherilyn Lee
* Gabriela Christen
* Nathaniel James Pacey

Supervised at EPFL by Prof. Nicolas Macris.

