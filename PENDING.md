# Pending Algorithms and Atoms for Ingestion

This file tracks clever, novel, or surprising algorithms and functions identified across the source repositories that are prime candidates for future ingestion.

## 1. BioSPPy
- **SSF Segmenter (`ssf_segmenter`)**: Uses a Slope Sum Function to enhance the QRS complex before thresholding, making it highly robust to certain types of noise during ECG R-peak detection.
- **Christov Segmenter (`christov_segmenter`)**: Implements a complex adaptive thresholding scheme combining several criteria (steepness, integration) to accurately identify R-peaks.
- **Hamilton Segmenter (`hamilton_segmenter`)**: A classic robust R-peak detector using bandpass filtering and moving averages to identify peaks.

## 2. E2E-PPG
- **Kazemi Peak Detection (`kazemi_peak_detection`)**: A specialized peak detection algorithm tailored for PPG signals, effectively handling their characteristic baseline wander and motion artifacts.
- **PPG Reconstruction (`ppg_reconstruction`)**: Employs advanced signal processing to 'fill in' corrupted signal segments, a novel necessity for noisy wearable data processing.
- **Signal Quality Assessment (`ppg_sqa`)**: A dedicated module that quantifies the reliability of the PPG signal before allowing further downstream analysis.

## 3. Institutional-Quant-Engine
- **Avellaneda-Stoikov Market Making (`23_market_making_avellaneda`)**: A sophisticated stochastic control model that dynamically adjusts bid-ask spreads based on inventory risk and price volatility.
- **Almgren-Chriss Optimal Execution**: Calculates the optimal trajectory for liquidating a large position while optimally balancing market impact against timing risk.
- **Probability of Informed Trading (`41_pin_informed_trading`)**: Uses order flow data to estimate the likelihood that trades are driven by asymmetric information (the PIN model).
- **Limit Order Queue Estimator**: Estimates the precise position of an order within a limit order book's queue, which is a critical piece for HFT simulations.

## 4. quantfin
- **Functional Monte Carlo (`MonteCarlo.hs`)**: Utilizes functional programming (Haskell's type system) to enforce strict correctness in stochastic path generation and contingent claim evaluation.
- **Volatility Surface Modeling**: Implements robust interpolation and calibration techniques for implied volatility surfaces.

## 5. Pulsar_Folding
- **Dedispersion Brute-Force (`DM_can`)**: Performs a brute-force search over a range of Dispersion Measures (DM) to find the optimal shift that maximizes the Signal-to-Noise Ratio (SNR) of the folded radio pulse profile.
- **Spline Band-Pass Correction**: Uses interpolative splines to model and subtract instrument-induced band-pass artifacts across frequency channels, 'flattening' the spectrogram for better signal detection.

## 6. Tempo.jl
- **Graph-Based Time Scale Management (`scales.jl`)**: Instead of fixed conversion factors, uses a `MappedDiGraph` to represent astronomical time scales as nodes. It dynamically computes transformation paths for complex conversions (e.g., UTC to TDB).
- **High-Precision Duration (`duration.jl`)**: A `Duration` struct that splits time into an integer part (seconds) and a fractional part, preventing precision loss over geological and astronomical time scales.
- **Metaprogramming for Scales**: Uses Julia macros (`@timescale`) to generate type-safe scale aliases safely at compile-time.

## 7. pronto
- **Modular State Estimation (RBIS)**: The Recursive Bayesian Incremental State (RBIS) engine. It provides a robust framework for sensor fusion that famously decouples measurement modules (LIDAR, Visual Odometry, GPS) into plugin-like atoms that feed the core estimator.

## 8. rust_robotics
- **High-Performance Numerical Solvers and Control Laws**: Implements custom kinematics/dynamics solvers for N-joint arm control and core path planning algorithms (like Dijkstra's) safely and efficiently in Rust.

## 9. Molecular-Docking
- **Quantum MWIS Solver (`quantum_solver_molecular.py`)**: Reduces the molecular docking problem to a Maximum Weight Independent Set (MWIS) problem and solves it using the physical properties of neutral atoms (Rydberg blockade). Uses Detuning Map Managers (DMM) to encode optimization weights directly into the quantum Hamiltonian.
- **Greedy Lattice Mapping (`greedy_lattice_mapping.py`)**: Implements a heuristic to map abstract interaction graphs onto physical 2D atom lattices under strict hardware constraints.

## 10. mint
- **Axial Attention (`axial_attention.py`)**: Implements factorized attention over Multiple Sequence Alignments (MSAs), allowing for highly memory-efficient processing of extremely large protein datasets.
- **Rotary Positional Embeddings (`rotary_embedding.py`)**: Implements RoPE positional embeddings, a novel approach to relative position encoding in Transformers that generalizes well across different sequence lengths.
