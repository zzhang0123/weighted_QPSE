# Weighted Quadratic Power Spectrum Estimator (weighted_QPSE)

A Python package for cosmological power spectrum estimation from interferometric visibility data using quadratic estimator methods with non-uniform frequency weighting.

## Overview

This package implements quadratic estimator methodology for extracting cosmological 21cm power spectra from radio interferometer data, with particular emphasis on handling non-uniform frequency weighting due to radio frequency interference (RFI) flagging and instrumental effects.

## Getting Started

**Highly Suggested**: See [test_nonuniform_weighting.ipynb](test_nonuniform_weighting.ipynb) notebook, which demonstrates:
- Detailed window function analysis with different weighting schemes.
- Basic power spectrum estimation workflow
- Monte Carlo validation with simulated data
- Covariance matrix calculation
- Four-panel comparison plots (mixing )



## Key Features

- **Quadratic Estimator Framework**: Implements optimal quadratic estimators for power spectrum extraction
- **Non-uniform Weighting**: Handles flagged frequency channels with sophisticated weighting schemes
- **Apodization Methods**: Advanced tapering around RFI flags to minimize spectral leakage
- **Window Function Analysis**: Comprehensive tools for analyzing estimator response functions
- **Bias Correction**: Automatic noise bias evaluation and subtraction
- **Covariance Estimation**: Both theoretical and Monte Carlo covariance matrix calculation

## Mathematical Framework

### Quadratic Estimator

The power spectrum estimate for delay mode α is given by:

```
p̂_α = (1/N_α) * x† Q_α x - b_α
```

where:
- `x` is the weighted visibility vector
- `Q_α` is the quadratic form matrix for delay mode α
- `N_α` is the normalization factor
- `b_α` is the noise bias correction

### Window Functions

The window function describes how the estimator responds to different delay modes:

```
W_αβ = Tr(Q_α R_β) / N_α
```

where `R_β` are the response matrices encoding instrument effects.

### Covariance Matrix

For Gaussian fields, the covariance between estimates is:

```
Cov[p̂_α, p̂_β] = 2 * Re{Tr[C E_α C E_β]}
```

where `C` is the total data covariance and `E_α = R_α / N_α` are normalized response matrices.


## Core Components

### Classes

#### `QE_window`
Window function analysis for quadratic estimators.

**Parameters:**
- `freqs`: Frequency array [Hz]
- `normalised_beam`: Beam response normalized by Ω_p
- `d_Omega`: Solid angle element [steradians]

**Methods:**
- `generate_window_coeffs(weight1, weight2, quad_form="Q")`: Generate window functions

#### `nonuniform_pspec`  
Full power spectrum estimation with non-uniform weighting.

**Parameters:**
- `freqs`: Frequency array [Hz]
- `normalised_beam`: Normalized beam pattern
- `d_Omega`: Solid angle element
- `taper`: Global tapering window type ('blackman-harris', 'none')
- `test_with_isotropic_signal`: Use isotropic signal assumption for testing

**Methods:**
- `p_hat(vis1, vis2, flag1, flag2, ...)`: Estimate power spectrum from visibilities

### Key Functions

#### Data Processing
- `trace_AB(A, B)`: Efficient matrix trace computation
- `example_beam()`: Load example beam pattern data
- `generate_freq_weights(flag1, flag2, ...)`: Create frequency-domain weights

#### Quadratic Forms
- `get_quad_Fourier_list(Ndlys, d_eta)`: Generate Fourier transform quadratic forms
- `generate_quad_Beam(beam, d_Omega, ...)`: Create beam quadratic form matrix
- `generate_response_matrix(...)`: Combine all response components

#### Estimator Components  
- `generate_window_func(quad_form_mat, resp_mat_list)`: Compute window functions
- `evaluate_bias(quad_form_mat, quad_Weights, norm, noise_cov)`: Calculate noise bias

#### Flagging and Weighting
- `apodize_around_flags(flag_arr, ramp_width)`: Apply half-cosine ramps around flags
- `taper_apodised_flags(flag_arr, ramp_width, taper_coeffs)`: Combine flag apodization with global tapering

#### Analysis Tools
- `calculate_p_hat_covariance(bps, resp_mat_arr, norm_li, ...)`: Theoretical covariance matrix

## Usage Examples

📓 **Start here**: Work through the [test_nonuniform_weighting.ipynb](test_nonuniform_weighting.ipynb) notebook for a complete understanding.

### Basic Power Spectrum Estimation

```python
import numpy as np
from nonuniform_pspec import nonuniform_pspec

# Load your data
freqs = np.linspace(100e6, 200e6, 1024)  # 1024 frequency channels
vis1 = your_visibility_data_1
vis2 = your_visibility_data_2  
flag1 = your_flag_array_1  # Boolean: True = flagged
flag2 = your_flag_array_2

# Initialize estimator
estimator = nonuniform_pspec(freqs, taper='blackman-harris')

# Estimate power spectrum
window_coeffs, bias, norm, p_hat = estimator.p_hat(
    vis1, vis2, flag1, flag2, 
    ramp_width=20,  # Width of apodization around flags
    quad_form="Q"   # Use full response matrix
)
```

### Window Function Analysis

```python
from nonuniform_pspec import QE_window
import uvtools.dspec as dspec

# Initialize window analyzer
window_obj = QE_window(freqs)

# Generate different weighting schemes
weights_uniform = np.ones(len(freqs))
weights_tapered = dspec.gen_window('blackman-harris', len(freqs))

# Analyze window functions
windows_uniform = window_obj.generate_window_coeffs(
    weights_uniform, weights_uniform, quad_form="Q"
)
windows_tapered = window_obj.generate_window_coeffs(  
    weights_tapered, weights_tapered, quad_form="Q"
)
```

### Advanced Flagging and Apodization

```python
from nonuniform_pspec import taper_apodised_flags

# Create sophisticated weights around RFI flags
flag_array = your_boolean_flags  # True = flagged channel
weights = taper_apodised_flags(
    flag_array,
    ramp_width=25,  # Wider ramps = better spectral leakage suppression
    taper_coeffs=None  # Uses Blackman-Harris by default
)
```

## Key Parameters

### Apodization Parameters
- **`ramp_width`**: Width of half-cosine ramps around flags
  - Larger values → better spectral leakage suppression
  - Typical range: 10-50 channels
  - Trade-off: wider ramps reduce effective bandwidth

### Quadratic Form Types
- **`"Q"`**: Full response matrix (includes weights, beam, and Fourier components)
  - Recommended for realistic analysis
  - Accounts for all instrumental effects
- **`"Q_alt"`**: Fourier-only quadratic forms
  - Useful for testing and comparison
  - Ignores weighting and beam effects
  - Don't use it when flagging occurs — the mixing effect is significant.

### Tapering Options
- **`'blackman-harris'`**: Excellent sidelobe suppression (recommended)


## Implementation Notes


### Data Requirements
- **Visibility data**: Complex-valued, shape `(N_freqs,)`
- **Flag arrays**: Boolean, shape `(N_freqs,)`, True = flagged
- **Frequency array**: Real-valued, shape `(N_freqs,)`, units in Hz
- **Beam data**: Complex-valued, shape `(N_freqs, N_directions)`

## Details

### Half-Cosine Apodization
Left ramp: `w(t) = 0.5 * (1 + cos(π*t))` for t ∈ [0,1]
Right ramp: `w(t) = 0.5 * (1 - cos(π*t))` for t ∈ [0,1]

### Fourier Transform Conventions
- Uses `numpy.fft` conventions throughout
- Delay coordinates: `η = fftshift(fftfreq(N_freqs, d=df))`
- DFT normalization includes delay bin spacing `d_eta`

### Beam Integration
- Assumes normalized beam: `beam[ν,Ω] / Ω_p(ν)`  
- Solid angle integration: `∫ dΩ → Σ_k * d_Omega`
- Supports both anisotropic and isotropic signal models

## Performance Tips

1. **Apodization Width**: Default value `ramp_width=20`, increase if spectral leakage is still problematic
(or you can consider use other apodisations rather than half cosine..)
2. **Beam Handling**: Set (`test_with_isotropic_signal=Flase`) for real data analysis
3. **Covariance**: Maybe use Monte Carlo covariance for final error bars, theoretical calculations (Gaussian data approximations) for quick estimates.

## Dependencies

- `numpy`: Core numerical operations
- `scipy`: Special functions and integration
- `uvtools.dspec`: Window function generation  
- `matplotlib`: Plotting (for notebooks)
- `tqdm`: Progress bars (for Monte Carlo)




