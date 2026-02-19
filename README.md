# ageoa

High-level verified atoms for AGEO-Matcher. This package provides safe, icontract-decorated wrappers for common scientific libraries, plus a **Ghost Witness** system for lightweight abstract graph verification.

## What are atoms?

An *atom* is a thin wrapper around a library function (numpy, scipy, etc.) annotated with formal pre/postconditions via [icontract](https://github.com/Paricigor/icontract). Every atom carries:

- **Preconditions** (`@require`) -- validated before execution (e.g., "input must not be empty", "filter order must be positive")
- **Postconditions** (`@ensure`) -- validated after execution (e.g., "output shape matches input shape", "FFT output must be complex-valued")
- **Slow postconditions** -- expensive checks (round-trip reconstruction, stability via pole analysis, PSD eigenvalue checks) gated by `AGEOA_SLOW_CHECKS=1`

## Ghost Witness system

Each atom is paired with a **ghost witness** -- a lightweight executable shadow that propagates metadata (`shape`, `dtype`, `sampling_rate`, `domain`) instead of computing actual values. Running witnesses through a computation graph catches structural mismatches before any expensive computation runs.

```python
from ageoa.ghost.registry import REGISTRY, list_registered
from ageoa.ghost.simulator import simulate_graph, SimNode

# After importing atom modules, all 16 DSP atoms are auto-registered
import ageoa.numpy.fft
import ageoa.scipy.signal

print(list_registered())
# ['butter', 'cheby1', 'cheby2', 'dct', 'fft', 'firwin', 'freqz',
#  'graph_fourier_transform', 'graph_laplacian', 'heat_kernel_diffusion',
#  'idct', 'ifft', 'irfft', 'lfilter', 'rfft', 'sosfilt']
```

The `@register_atom(witness)` decorator on each heavy atom binds it to its witness in the global `REGISTRY`. The AGEO-Matcher synthesizer uses this to run a ghost simulation pass before assembly.

## Installation

```bash
pip install -e .
```

Requires Python >= 3.10, numpy >= 1.24, scipy >= 1.10, icontract >= 2.6.

## Modules

```
ageoa/
  numpy/
    arrays.py          Array manipulation (reshape, concatenate, ...)
    emath.py           Extended math functions
    fft.py             FFT/IFFT/RFFT/IRFFT with round-trip contracts
    linalg.py          Linear algebra (det, inv, eig, svd, ...)
    polynomial.py      Polynomial operations
    random.py          Random number generation
  scipy/
    fft.py             DCT/IDCT with round-trip contracts
    integrate.py       Numerical integration
    linalg.py          Sparse/dense linear algebra
    optimize.py        Optimization routines
    signal.py          Filter design (butter, cheby1/2, firwin) and application (lfilter, sosfilt, freqz)
    sparse_graph.py    Graph signal processing (Laplacian, GFT, heat diffusion)
    stats.py           Statistical tests
  biosppy/
    ecg.py             ECG processing (bandpass_filter, r_peak_detection, peak_correction, template_extraction, heart_rate_computation, ssf_segmenter, christov_segmenter)
    pcg.py             PCG processing (shannon_energy, pcg_segmentation)
    eda.py             EDA processing (gamboa_segmenter, eda_feature_extraction)
  pasqal/
    docking.py         Molecular docking graph decomposition (sub_graph_embedder, graph_transformer, quantum_mwis_solver)
  pulsar/
    pipeline.py        Pulsar folding and dedispersion (delay_from_DM, de_disperse, fold_signal, SNR)
  ghost/
    abstract.py        AbstractSignal, AbstractBeatPool metadata types
    registry.py        REGISTRY dict, @register_atom decorator, get_witness()
    witnesses.py       18 concrete ghost witnesses + AbstractFilterCoefficients, AbstractGraphMeta
    simulator.py       simulate_graph(), SimNode, SimResult, PlanError
```

## DSP contract patterns

See [INGESTION.md](INGESTION.md) section 13 for the three DSP-specific contract patterns:

1. **Epsilon-metric round-trip** -- invertible transform pairs (FFT/IFFT, DCT/IDCT)
2. **Stability** -- filter design via `_poles_inside_unit_circle`
3. **Total variation reduction** -- graph signal smoothing

## Testing

```bash
# Standard tests
pytest

# With expensive postcondition validation
AGEOA_SLOW_CHECKS=1 pytest
```

## License

MIT
