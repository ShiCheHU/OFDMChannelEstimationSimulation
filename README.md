# SISO OFDM Channel Estimation Simulation

A Python simulation project for **SISO OFDM channel estimation and interpolation** under time-varying multipath fading channels.

This project implements a complete baseband simulation chain, including:

- random bit generation
- digital modulation / demodulation
- doubly-selective channel dataset generation
- pilot-aided channel estimation
- frequency / time-frequency interpolation
- equalization and BER evaluation
- NMSE evaluation in both pilot domain and full time-frequency grid
- comparison between **analytic** and **Monte Carlo estimated** channel covariance matrices

The code is organized into four files:

- `main.py` — end-to-end simulation, evaluation, plotting, and result saving
- `modulation.py` — bit generation, modulation, and demodulation
- `channelestimation.py` — pilot-domain channel estimation and interpolation
- `channel_model.py` — channel generation and covariance / autocorrelation construction

---

## 1. Project Scope

This project is intended for studying and comparing classical pilot-aided channel estimation methods for SISO OFDM systems, especially under:

- multipath fading channel, three kinds of fading are considered:
    - Rayleigh fading
    - Rician fading
    - TDL
- two kinds of slow time-frequency selective fading:
    - Jakes 
    - AR(1) models

The simulation assumes the standard per-subcarrier OFDM model

\[
Y[m,k] = H[m,k]X[m,k] + W[m,k]
\]

under the assumption that the cyclic prefix is sufficient and inter-carrier interference is negligible.

---

## 2. Implemented Methods

### 2.1 Channel estimation of pilots

The following pilot estimators are implemented:

- **LS** (Least Squares)
- **MMSE**
- **LMMSE**

For MMSE / LMMSE, the required covariance matrices can be obtained from two different sources:

1. **Analytic covariance** derived from the assumed channel model, where the covariance matrix is computed based on the asumption of random process
2. **Monte Carlo covariance** estimated from a large number of simulated channel realizations

---

### 2.2 Interpolation methods

The following interpolation methods are implemented:

- **Linear interpolation**
- **Quadratic interpolation**
- **Wiener / LMMSE interpolation**
- **DFT-based interpolation**

Both 1D and 2D interpolation structures are supported depending on pilot pattern and selected method.

---

### 2.3 Pilot patterns

The following pilot patterns are supported:

- **Block-type pilots**
- **Comb-type pilots**
- **Scattered-type pilots**

---

## 3. File Structure

```text
.
├── main.py
├── modulation.py
├── channelestimation.py
├── channel_model.py
└── corr_stats/                  # auto-generated covariance cache directory
```

### `main.py`
Responsible for the full simulation workflow:

- simulation configuration
- pilot-grid construction
- transmit-grid generation
- channel realization loading / generation
- received-grid generation
- pilot extraction
- channel estimation and interpolation
- equalization and demodulation
- BER / NMSE computation
- plotting and saving figures

### `modulation.py`
Contains:

- constellation generation
- bit generation
- modulation
- demodulation
- unit test for supported modulations

### `channelestimation.py`
Contains:

- LS / MMSE / LMMSE pilot estimation
- linear / quadratic / Wiener / DFT interpolation
- support for covariance input by file path or in-memory dictionary

### `channel_model.py`
Contains:

- fading channel generation
- Jakes / AR(1) time-selective process modeling
- Rayleigh / Rician / TDL profile generation
- covariance / autocorrelation generation
- covariance file save / load utilities

---

## 4. Channel and Covariance Modeling

The channel is modeled in the delay-time domain as

\[
h[m,n] = \sum_{\ell=0}^{L-1} \sqrt{P_\ell} \, g_{\ell,m} \, \delta[n-d_\ell]
\]

where:

- \(d_\ell\): path delay index
- \(P_\ell\): average path power from the power delay profile (PDP)
- \(g_{\ell,m}\): time-selective fading coefficient at OFDM symbol index \(m\)

The corresponding CFR is

\[
H[m,k] = \sum_{\ell=0}^{L-1} \sqrt{P_\ell} \, g_{\ell,m} \, e^{-j\frac{2\pi}{N}kd_\ell}
\]

Covariance matrices used by MMSE / LMMSE estimation and Wiener interpolation can be supplied through:

- **theory-based covariance**
- **Monte Carlo covariance**

Monte Carlo covariance files are cached automatically in `./corr_stats`.

---

## 5. Simulation Outputs

The simulation produces performance metrics such as:

- **BER**
- **full-grid NMSE**
- **pilot-domain NMSE**
- **theoretical pilot-domain NMSE for LS and MMSE**

Saved figures typically include:

- BER versus SNR
- full-grid NMSE versus SNR
- pilot-domain NMSE: simulation versus theory

---

## 6. How to Run

Place all four Python files in the **same directory** and run:

```bash
python main.py
```

> `channel_model.py` should be in the same directory as `main.py`, unless you modify the Python import path.

---

## 7. Main Configuration Parameters

`main.py` defines a `SimulationConfig` dataclass. Typical parameters include:

- `n_frame` — number of independent channel realizations
- `n_sym` — number of OFDM symbols per frame
- `n_fft` — number of subcarriers
- `cp_len` — cyclic prefix length
- `modulation` — `QPSK`, `16QAM`, `64QAM`, etc.
- `channel_type` — `Rayleigh`, `Rician`, `TDL`
- `fading_model` — `jakes` or `ar1`
- `tau_max_samples` — maximum delay spread in samples
- `num_path` — number of channel paths
- `f_max` — maximum Doppler frequency in Hz
- `pilot_pattern` — `block`, `comb`, or `scattered`
- `est_method` — `LS`, `MMSE`, or `LMMSE`
- `interp_method` — `linear`, `quadratic`, `wiener`, or `dft`
- `corr_source` — `theory` or `mc`
- `corr_mc_frames` — number of Monte Carlo frames used for covariance estimation
- `corr_dir` — covariance cache directory
- `corr_path` — explicit covariance file path (optional)

---

## 8. Correlation Matrix Input Path

For MMSE / LMMSE estimation and Wiener interpolation, the correlation matrix source can be selected explicitly.

Two common modes are supported:

- **Theory-based** covariance derived from channel assumptions
- **Monte Carlo-based** covariance estimated from `x` simulated channel frames

The saved file path is used to record the covariance source clearly, for example:

- `corr_theory_...npz`
- `corr_mc1000_...npz`

This makes it easy to compare performance under different covariance sources in `main.py`.

---

## 9. Typical Research Use Cases

This codebase is suitable for:

- pilot-aided SISO OFDM channel estimation study
- comparing LS / MMSE / LMMSE estimators
- comparing linear / quadratic / Wiener / DFT interpolation
- studying analytic versus Monte Carlo covariance matrices
- evaluating BER and NMSE under different pilot structures
- building a baseline for more realistic 4G/5G systems

---

## 10. Notes and Limitations

- The current simulation uses the standard diagonal-per-subcarrier OFDM model and does **not** explicitly model ICI.
- Theoretical NMSE curves are provided for **pilot-domain channel estimation**, not for the interpolated full-grid CFR.
- MMSE / LMMSE performance strongly depends on the consistency between the assumed covariance matrix and the actual simulated channel statistics.

---

## 11. Suggested Future Extensions

Possible extensions include:

- channel tracking and prediction
- Kalman filtering
- BEM-based doubly-selective channel estimation
- decision-directed channel refinement
- MIMO OFDM extension
- explicit ICI modeling for fast fading scenarios

---

## 12. License

- MIT License

---

## 13. Author

If you use this repository as part of your research or technical blog, you may consider adding:

- Shicheng Hu
- University of Chinese Academy of Science
- https://github.com/ShiCheHU
- hushch2018@163.com

for easier academic or professional reference.

