# Interesting Ingestion Targets

Algorithms from referenced repositories that represent novel, clever, or surprising implementations — not standard library wrappers, textbook formulas, or simple filter chains.

---

## BioSPPy (`PIA-Group/BioSPPy`)

| Entrypoint | Why interesting |
|---|---|
| `biosppy.signals.ecg.christov_segmenter` | Adaptive threshold state machine for QRS detection with dynamic moving-average buffers that distinguish R-peaks from T-waves |
| `biosppy.signals.ecg.engzee_segmenter` | Threshold-intersection peak detection using consecutive-sample validation — unusual numerical method vs standard peak-finding |
| `biosppy.signals.ecg.gamboa_segmenter` | Histogram CDF-based QRS detection with second-derivative analysis on normalized signals |
| `biosppy.signals.ecg.hamilton_segmenter` | Multi-stage rule-based detector with buffered statistics, dual slope analysis, and RR-interval validation |
| `biosppy.signals.ecg.ASI_segmenter` | Finite state machine on double-derivative-squared signals with exponential decay thresholding |
| `biosppy.signals.ecg.ZZ2018` | Fuzzy-logic signal quality assessment fusing beat detection agreement, spectral content, kurtosis, and power distribution |
| `biosppy.signals.emg.solnik_onset_detector` | Teager-Kaiser energy operator (`x[k]² - x[k+1]*x[k-1]`) — non-linear signal conditioning for EMG onset |
| `biosppy.signals.emg.bonato_onset_detector` | Dual-threshold state machine for EMG onset with temporal validation and failure recovery |
| `biosppy.signals.emg.abbink_onset_detector` | Transition-index method counting threshold crossings in fixed windows for onset/offset boundary detection |
| `biosppy.signals.pcg.homomorphic_filter` | Log-domain filtering (`exp(filter(log(\|signal\|)))`) for heart sound envelope extraction — spectral flattening technique |
| `biosppy.signals.ppg.find_onsets_kavsaoglu2016` | Adaptive segmentation with dynamic window sizing based on current BPM estimate and low-pass filtered BPM changes |
| `biosppy.signals.ppg.find_onsets_elgendi2013` | Squared-signal processing with truncation, multi-scale moving averages, and prominence-based peak selection |
| `biosppy.signals.abp.find_onsets_zong2003` | Multi-threshold validation combining positive/negative slope analysis with dynamic threshold adjustment |

---

## E2E-PPG (`HealthSciTech/E2E-PPG`)

| Entrypoint | Why interesting |
|---|---|
| `ppg_reconstruction.gan_rec` | Iterative peak-aligned GAN noise reconstruction — stitches signals at peak boundaries with resampling for continuity |
| `ppg_reconstruction.reconstruction` | Dynamic peak-boundary signal splicing: upsamples, finds peaks in clean and reconstructed, splices at peak boundaries, downsamples |
| `ppg_sqa.heart_cycle_detection` | Adaptive beat segmentation using systolic peak-defined windows (half mean inter-peak interval) |
| `ppg_sqa.template_matching_features` | Dual-domain beat coherence: Euclidean distance AND Pearson correlation against averaged template |
| `kazemi_peak_detection.Wrapper_function` | Windowed refinement of NN predictions — nested peak detection with 15-sample adaptive windows and 35-sample minimum spacing |

---

## Institutional-Quant-Engine (`Rakeshks7/Institutional-Quant-Engine`)

| Entrypoint | Why interesting |
|---|---|
| `execution_hft.17_optimal_execution_almgren.AlmgrenChriss` | Hyperbolic-sine execution trajectories balancing market impact vs urgency via risk aversion parameter |
| `execution_hft.23_market_making_avellaneda.AvellanedaStoikov` | Dynamic bid-ask spread adjustment with non-linear inventory-dependent reservation price |
| `execution_hft.28_order_flow_imbalance.OrderFlowImbalance` | Directional pressure detection from bid/ask price-quantity changes — predicts short-term moves without price levels |
| `execution_hft.49_limit_order_queue_estimator.QueueEstimator` | FIFO queue decay model estimating fill probability by tracking ahead-order attrition |
| `execution_hft.41_pin_informed_trading.PINModel` | MLE decomposition of order flow into informed vs noise components to detect insider activity |
| `risk_portfolio.16_hierarchical_risk_parity.HRP` | Hierarchical clustering for capital allocation to risk-balanced clusters rather than individual securities |
| `risk_portfolio.27_copula_dependence.CopulaDependence` | T-copula tail risk capturing non-linear crash co-dependence independent of linear correlation |
| `risk_portfolio.24_tail_risk_evt.EVTModel` | Generalized Pareto Distribution for tail CVaR — fat-tail risk that normal distributions miss |
| `research_math.19_triangular_arbitrage_graph.TriangularArbitrage` | Log-weighted currency graph with negative-cycle detection for risk-free arbitrage discovery |
| `research_math.34_fractional_diff_stationarity.FractionalDiff` | Fractional-order differencing preserving memory/trend while achieving stationarity — superior to integer differencing for ML |
| `research_math.35_hawkes_process_arrival.HawkesProcess` | Self-exciting point process where trades trigger cascading follow-on trades — captures microstructure clustering |
| `strategies.29_dynamic_hedge_kalman.DynamicHedge` | Kalman-filtered time-varying hedge ratios that adapt to regime shifts in paired positions |
| `strategies.47_supply_chain_propagation.SupplyChainShock` | Directed-graph shock propagation through weighted supply chain edges — predicts downstream impacts |
| `ops_compliance.50_trade_surveillance_wash_graph.WashTradeDetector` | Graph cycle detection on trade networks to flag collusion rings |
| `derivatives_pricing.30_stochastic_volatility_heston.HestonModel` | Correlated stochastic vol paths capturing leverage effect (vol spikes when price drops) |

---

## quantfin (`boundedvariation/quantfin`)

| Entrypoint | Why interesting |
|---|---|
| `Quant.VolSurf.localVol` | Dupire local volatility extraction from variance surface via finite differences with automatic strike adjustment |
| `Quant.Models.charFuncOption` | Characteristic function inversion with damped exponential transforms for option pricing (Carr-Madan method) |
| `Quant.Models.Heston.evolve'` | Non-standard Heston discretization compensating for squared-volatility in variance process |
| `Quant.Models.Merton.evolve'` | Jump-diffusion with dynamic drift adjustment for jump intensity correction |
| `Quant.Math.Utilities.tdmaSolver` | Tridiagonal matrix solver using Haskell's ST monad for in-place mutable vector operations |
| `Quant.RNG.MWC64X.skip` | Skip-ahead RNG via modular exponentiation — generates non-sequential random streams without intermediate states |
| `Quant.ContingentClaim.specify` | Monadic DSL for path-dependent exotic options with automatic cash flow sorting by observation time |
| `Quant.MonteCarlo.runSimulationAnti` | Antithetic variate variance reduction via complementary simulation averaging |

---

## Pulsar_Folding (`PetchMa/Pulsar_Folding`)

| Entrypoint | Why interesting |
|---|---|
| `de_disperse` + `DM_can` | Brute-force DM grid search: per-channel cyclic time-shifts via `delay = DM / (0.000241 * freq²)` with SNR maximization |
| `top` (FFT peak extraction) | Greedy iterative argmax-and-zero on FFT magnitude spectrum for candidate period discovery |
| Spline bandpass removal | 2nd-order spline fit on coarse channels (width=8) to remove bandpass ripple artifacts without overfitting weak pulsar signals |

---

## Tempo.jl (`JuliaSpaceMissionDesign/Tempo.jl`)

| Entrypoint | Why interesting |
|---|---|
| `Tempo.offset_tt2tdb` | 3-pass Newton-like iteration solving nonlinear transcendental equation for TDB via nested sine functions |
| `Tempo.offset_tt2tdbh` | 6-term Fourier series (Harada & Fukushima 2003) for high-precision TDB at 10μs accuracy |
| `Tempo.tai2utc` | Fixed-point iteration for leap-second ambiguity — predicts and corrects TAI-UTC jumps across boundaries |
| `Tempo.utc2tai` | Leap-second spread detection comparing Δt across day boundaries, spreading leap into preceding day |
| `Tempo.jd2cal` | Generalized Kahan-Babuska compensated summation (Klein 2006) for precision-preserving Julian date arithmetic |
| `Tempo.find_year` | O(1) Gregorian year from Julian day via `(400 * j2d + 292194288) ÷ 146097` exploiting 400-year cycle |
| `Tempo.find_month` | Branchless month formula `(10 * dayinyear + offset) ÷ 306` with leap-year-dependent offset |
| `Tempo.apply_offsets` | DAG pathfinding across timescale graph, chaining offset functions along discovered path |

---

## pronto (`PickNikRobotics/pronto`)

| Entrypoint | Why interesting |
|---|---|
| `pronto::DynamicStanceEstimator::getStance` | Exhaustive search over all 15 contact configurations minimizing Newton-Euler dynamics violation to infer stance |
| `pronto::FlexEstimator::update` | System-ID leg flexibility compensation modeling HAA joint as 2nd-order transfer function (Kp, Tw, Zeta, Tz) |
| `pronto::LegOdometer::estimateVelocity` | Multi-mode covariance estimation weighting stance variance with ground reaction force impact analysis |
| `pronto::BacklashFilter::processSample` | Kalman-filtered velocity zero-crossing detection with exponential innovation suppression for joint backlash |
| `pronto::BlipFilter::getValue` | Median-based impulse detector with predictive outlier rejection — replaces central value only if both predicted and actual deviate |
| `pronto::StateEstimator::EKFSmoothBackwardsPass` | Rauch-Tung-Striebel smoother iterating backward through measurement history with INS update association |
| `pronto::InverseSchmittTrigger::getStance` | Inverted hysteresis: triggers contact on signal falling BELOW threshold, with separate time constants for contact/release |
| `pronto::FootContactClassifier::update` | Multi-modal classifier combining foot force-z and torques with separate weak/strong Schmitt triggers and blackout periods |
| `pronto::YawLock::getCorrection` | Kinematic foot-lock constraint: compares FK-computed foot positions to locked config, with slip detection threshold |
| `pronto::TorqueAdjustment::processSample` | IHMC-inspired compliance correction treating transmission as spring element with per-joint tunable stiffness |

---

## rust_robotics (`shassen14/rust_robotics`)

| Entrypoint | Why interesting |
|---|---|
| `models::ground_vehicles::bicycle_kinematic::Model::calculate_b` | Jacobian linearization with complex 2nd-order coupling term `l_t³ * u0 / (cos(u[1])² * denom^1.5)` |
| `models::humanoid::n_joint_arm2_d::ModelD::inverse_kinematics` | Geometric Jacobian accumulation over serial chain with pseudo-inverse joint angle correction |
| `models::ground_vehicles::longitudinal_dynamics::Model::get_derivatives` | Combined aerodynamic drag + rolling resistance + grade resistance with static stiction handling at zero velocity |

---

## Molecular-Docking (`pasqal-io/Molecular-Docking`)

| Entrypoint | Why interesting |
|---|---|
| `src.solver.quantum_solver_molecular.q_solver` | Adiabatic quantum evolution for MWIS using Rydberg blockade with dynamic detuning modulation on neutral-atom hardware |
| `src.solver.optimiser.minimize_bandwidth` | Iterative reverse Cuthill-McKee with threshold-based matrix truncation to minimize qubit arrangement bandwidth |
| `src.solver.greedy_subgraph_vv.solve_breadthfirst_2_sol` | Fork-based divide-and-conquer: spawns two branches running quantum solvers on subgraphs while pruning inferior states |
| `src.solver.greedy_lattice_mapping.GreedyMapping` | Multi-factor heuristic scoring (degree matching + non-adjacency penalty + history pruning) for graph-to-Rydberg-lattice embedding |
| `src.graph.mapping.add_quantum_link` | Inserts 2n ancillary link nodes to maintain entanglement between distant qubits — chain construction from quantum annealing |
| `src.graph.mapping.map_to_UDG` | COBYLA-optimized node coordinates matching `C₆/r⁶` interaction strengths to adjacency structure for Unit Disk Graph realization |
| `src.graph.interaction_graph.build_weighted_binding_interaction_graph` | Molecular docking as graph: ligand-receptor feature pairs as nodes weighted by pharmacophoric potentials with distance compatibility edges |
| `src.graph.mapping.build_complementary_graph` | Maximum Clique → MWIS reduction via complement graph, enabling quantum solver for clique enumeration |
| `src.solver.mwis_SA` | Simulated annealing with problem-aware temperature scaling calibrated from graph size and weight range |

---

## mint (`VarunUllanat/mint`)

| Entrypoint | Why interesting |
|---|---|
| `mint.rotary_embedding.RotaryEmbedding.forward` | RoPE: complex rotation matrices via half-rotation trick on Q/K tensors for relative position encoding |
| `mint.multihead_attention.with_incremental_state` | Decorator dynamically modifying class inheritance for UUID-keyed KV cache management during autoregressive decoding |
| `mint.axial_attention.RowSelfAttention` | Factorized 2D MSA attention along rows with dynamic batching based on token budget constraints |
| `mint.axial_attention.RowSelfAttention.align_scaling` | Scales attention weights by inverse sqrt of row count to normalize contribution in high-dimensional alignments |
| `mint.data.FastaBatchedDataset.get_batch_indices` | Greedy first-fit-decreasing bin packing for GPU-optimal sequence batching under token budget |
| `mint.modules.apc` | Average Product Correction for contact prediction: removes background via `(a₁*a₂)/a₁₂` denoising of attention maps |
| `downstream.TCR-Epitope.tcr-interface.teim_utils.encoding_dist_mat` | Dynamic center-alignment encoding of variable-length distance matrices with hardcoded padding offsets by epitope length class |
