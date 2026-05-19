
"""
Channel estimation and interpolation for SISO OFDM.
Support functions:
- pilot-domain channel estimation
- 1D interpolation in frequency domain
- 2D interpolation in time-frequency domain
- test functions for estimation and interpolation

Support estimation:
- LS
- MMSE / LMMSE pilot denoising

Support interpolation:
- linear interpolation
- quadratic interpolation
- Wiener (LMMSE) interpolation
- DFT interpolation
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np

from channel_model import load_correlation_stats

EPS = 1e-12


def _extract_corr_matrices(corr_stats: Dict | None = None,
                           corr_path: str | None = None) -> Dict:
    """
    Load or validate correlation statistics.

    Input:
    - corr_stats: dict or None, already loaded correlation statistics.
    - corr_path: str or None, path of saved correlation statistics.

    Output:
    - corr_stats: dictionary containing at least R_freq_matrix and/or R_time_matrix.
    """
    if corr_stats is None and corr_path is None:
        raise ValueError("Either corr_stats or corr_path must be provided.")
    if corr_stats is None:
        if not os.path.exists(corr_path):
            raise FileNotFoundError(f"Correlation file not found: {corr_path}")
        corr_stats = load_correlation_stats(corr_path)
    return corr_stats


def ls_estimate(y_p: np.ndarray, x_p: np.ndarray) -> np.ndarray:
    """
    Least-squares channel estimation at pilot positions.

    Input:
    - y_p: received pilot symbols, shape (..., Np)
    - x_p: transmitted pilot symbols, shape (..., Np) or (Np,)

    Output:
    - h_ls: LS channel estimates at pilot positions, shape (..., Np)
    """
    y_p = np.asarray(y_p, dtype=complex)
    x_p = np.asarray(x_p, dtype=complex)
    return y_p / (x_p + EPS)


def mmse_estimate(y_p: np.ndarray,
                  x_p: np.ndarray,
                  r_pp: np.ndarray,
                  noise_var: float) -> np.ndarray:
    """
    MMSE/LMMSE pilot-domain channel estimation.

    Input:
    - y_p: received pilot symbols, shape (..., Np)
    - x_p: transmitted pilot symbols, shape (..., Np) or (Np,)
    - r_pp: pilot-position channel correlation matrix, shape (Np, Np)
    - noise_var: AWGN variance in frequency domain.

    Output:
    - h_mmse: MMSE channel estimates at pilot positions, shape (..., Np)
    """
    h_ls = ls_estimate(y_p, x_p)
    x_abs2 = np.mean(np.abs(np.asarray(x_p, dtype=complex)) ** 2)
    eta = noise_var / (x_abs2 + EPS)
    filt = r_pp @ np.linalg.inv(r_pp + eta * np.eye(r_pp.shape[0], dtype=complex))
    return np.asarray(h_ls) @ filt.T


def lmmse_estimate(y_p: np.ndarray,
                   x_p: np.ndarray,
                   r_pp: np.ndarray,
                   noise_var: float) -> np.ndarray:
    """
    Linear MMSE pilot-domain channel estimation.

    Input:
    - y_p: received pilot symbols, shape (..., Np)
    - x_p: transmitted pilot symbols, shape (..., Np) or (Np,)
    - r_pp: pilot-position channel correlation matrix, shape (Np, Np)
    - noise_var: AWGN variance in frequency domain.

    Output:
    - h_lmmse: LMMSE channel estimates at pilot positions, shape (..., Np)
    """
    return mmse_estimate(y_p, x_p, r_pp, noise_var)


def interpolate_linear_1d(h_p: np.ndarray, pilot_idx: np.ndarray, n_fft: int) -> np.ndarray:
    """
    One-dimensional linear interpolation in frequency.

    Input:
    - h_p: channel estimates at pilot subcarriers, shape (Np,)
    - pilot_idx: pilot subcarrier indices, shape (Np,)
    - n_fft: number of subcarriers.

    Output:
    - h_est: interpolated CFR on all subcarriers, shape (n_fft,)
    """
    pilot_idx = np.asarray(pilot_idx, dtype=int)
    h_p = np.asarray(h_p, dtype=complex)
    all_k = np.arange(n_fft)
    real_part = np.interp(all_k, pilot_idx, np.real(h_p))
    imag_part = np.interp(all_k, pilot_idx, np.imag(h_p))
    return real_part + 1j * imag_part


def _quadratic_window_interp(x: float, x_pts: np.ndarray, y_pts: np.ndarray) -> complex:
    """
    Quadratic Lagrange interpolation from three support points.

    Input:
    - x: target coordinate.
    - x_pts: support coordinates, shape (3,)
    - y_pts: support values, shape (3,)

    Output:
    - y: interpolated complex value.
    """
    x0, x1, x2 = x_pts
    y0, y1, y2 = y_pts
    L0 = (x - x1) * (x - x2) / ((x0 - x1) * (x0 - x2) + EPS)
    L1 = (x - x0) * (x - x2) / ((x1 - x0) * (x1 - x2) + EPS)
    L2 = (x - x0) * (x - x1) / ((x2 - x0) * (x2 - x1) + EPS)
    return L0 * y0 + L1 * y1 + L2 * y2


def interpolate_quadratic_1d(h_p: np.ndarray, pilot_idx: np.ndarray, n_fft: int) -> np.ndarray:
    """
    One-dimensional quadratic interpolation in frequency.

    Input:
    - h_p: channel estimates at pilot subcarriers, shape (Np,)
    - pilot_idx: pilot subcarrier indices, shape (Np,)
    - n_fft: number of subcarriers.

    Output:
    - h_est: interpolated CFR on all subcarriers, shape (n_fft,)
    """
    pilot_idx = np.asarray(pilot_idx, dtype=int)
    h_p = np.asarray(h_p, dtype=complex)
    Np = len(pilot_idx)
    if Np < 3:
        return interpolate_linear_1d(h_p, pilot_idx, n_fft)

    h_est = np.zeros(n_fft, dtype=complex)
    for k in range(n_fft):
        if k <= pilot_idx[1]:
            win = np.array([0, 1, 2])
        elif k >= pilot_idx[-2]:
            win = np.array([Np - 3, Np - 2, Np - 1])
        else:
            g = np.searchsorted(pilot_idx, k, side='right') - 1
            g = int(np.clip(g, 1, Np - 2))
            win = np.array([g - 1, g, g + 1])
        x_pts = pilot_idx[win]
        y_pts = h_p[win]
        h_est[k] = _quadratic_window_interp(k, x_pts.astype(float), y_pts)
    return h_est


def interpolate_wiener_1d(h_p: np.ndarray,
                          pilot_idx: np.ndarray,
                          n_fft: int,
                          r_freq_full: np.ndarray,
                          noise_var: float = 0.0,
                          pilot_power: float = 1.0) -> np.ndarray:
    """
    Wiener/LMMSE interpolation in frequency domain using a provided correlation matrix.

    Input:
    - h_p: channel estimates at pilot subcarriers, shape (Np,)
    - pilot_idx: pilot subcarrier indices, shape (Np,)
    - n_fft: number of subcarriers.
    - r_freq_full: full frequency covariance matrix, shape (n_fft, n_fft)
    - noise_var: pilot estimation noise variance.
    - pilot_power: pilot symbol power.

    Output:
    - h_est: interpolated CFR on all subcarriers, shape (n_fft,)
    """
    pilot_idx = np.asarray(pilot_idx, dtype=int)
    all_idx = np.arange(n_fft, dtype=int)
    r_dp = r_freq_full[np.ix_(all_idx, pilot_idx)]
    r_pp = r_freq_full[np.ix_(pilot_idx, pilot_idx)]
    r_pp = r_pp + (noise_var / (pilot_power + EPS)) * np.eye(len(pilot_idx), dtype=complex)
    return r_dp @ np.linalg.solve(r_pp, np.asarray(h_p, dtype=complex))


def interpolate_dft_1d(h_p: np.ndarray,
                       pilot_idx: np.ndarray,
                       n_fft: int,
                       cp_len: int) -> np.ndarray:
    """
    DFT-based interpolation for comb-type equally spaced pilots.

    Input:
    - h_p: channel estimates at pilot subcarriers, shape (Np,)
    - pilot_idx: pilot subcarrier indices, shape (Np,)
    - n_fft: total number of subcarriers.
    - cp_len: cyclic prefix length; effective CIR support is assumed <= cp_len.

    Output:
    - h_est: interpolated CFR on all subcarriers, shape (n_fft,)
    """
    pilot_idx = np.asarray(pilot_idx, dtype=int)
    h_p = np.asarray(h_p, dtype=complex)

    if len(pilot_idx) < 2:
        return np.ones(n_fft, dtype=complex) * h_p[0]

    spacing = np.diff(pilot_idx)
    if not np.all(spacing == spacing[0]):
        return interpolate_linear_1d(h_p, pilot_idx, n_fft)

    S = spacing[0]
    Np = len(pilot_idx)

    if pilot_idx[0] != 0 or Np * S > n_fft:
        return interpolate_linear_1d(h_p, pilot_idx, n_fft)

    h_alias = np.fft.ifft(h_p, n=Np)
    support = int(min(cp_len, Np))
    h_trunc = np.zeros(Np, dtype=complex)
    h_trunc[:support] = h_alias[:support]
    h_pad = np.zeros(n_fft, dtype=complex)
    h_pad[:Np] = h_trunc
    h_est = np.fft.fft(h_pad, n=n_fft)
    return h_est


def _interp_time_linear(h_known: np.ndarray, known_idx: np.ndarray, n_sym: int) -> np.ndarray:
    """
    One-dimensional linear interpolation along OFDM symbol index.

    Input:
    - h_known: known channel values along time, shape (Nt_known,)
    - known_idx: known OFDM symbol indices, shape (Nt_known,)
    - n_sym: total number of OFDM symbols.

    Output:
    - h_est: interpolated time sequence, shape (n_sym,)
    """
    known_idx = np.asarray(known_idx, dtype=int)
    t = np.arange(n_sym)
    real_part = np.interp(t, known_idx, np.real(h_known))
    imag_part = np.interp(t, known_idx, np.imag(h_known))
    return real_part + 1j * imag_part


def _interp_time_quadratic(h_known: np.ndarray, known_idx: np.ndarray, n_sym: int) -> np.ndarray:
    """
    One-dimensional quadratic interpolation along OFDM symbol index.

    Input:
    - h_known: known channel values along time, shape (Nt_known,)
    - known_idx: known OFDM symbol indices, shape (Nt_known,)
    - n_sym: total number of OFDM symbols.

    Output:
    - h_est: interpolated time sequence, shape (n_sym,)
    """
    known_idx = np.asarray(known_idx, dtype=int)
    h_known = np.asarray(h_known, dtype=complex)
    Nt = len(known_idx)
    if Nt < 3:
        return _interp_time_linear(h_known, known_idx, n_sym)

    h_est = np.zeros(n_sym, dtype=complex)
    for m in range(n_sym):
        if m <= known_idx[1]:
            win = np.array([0, 1, 2])
        elif m >= known_idx[-2]:
            win = np.array([Nt - 3, Nt - 2, Nt - 1])
        else:
            q = np.searchsorted(known_idx, m, side='right') - 1
            q = int(np.clip(q, 1, Nt - 2))
            win = np.array([q - 1, q, q + 1])
        x_pts = known_idx[win].astype(float)
        y_pts = h_known[win]
        h_est[m] = _quadratic_window_interp(float(m), x_pts, y_pts)
    return h_est


def _interp_time_wiener(h_known: np.ndarray,
                        known_idx: np.ndarray,
                        n_sym: int,
                        r_time_full: np.ndarray,
                        noise_var: float = 0.0) -> np.ndarray:
    """
    Wiener/LMMSE interpolation along OFDM symbol index using a provided time covariance matrix.

    Input:
    - h_known: known channel values along time, shape (Nt_known,)
    - known_idx: known OFDM symbol indices, shape (Nt_known,)
    - n_sym: total number of OFDM symbols.
    - r_time_full: full time covariance matrix, shape (n_sym, n_sym).
    - noise_var: estimation noise variance at known points.

    Output:
    - h_est: interpolated time sequence, shape (n_sym,)
    """
    known_idx = np.asarray(known_idx, dtype=int)
    target_idx = np.arange(n_sym, dtype=int)
    r_tk = r_time_full[np.ix_(target_idx, known_idx)]
    r_kk = r_time_full[np.ix_(known_idx, known_idx)]
    r_kk = r_kk + noise_var * np.eye(len(known_idx), dtype=complex)
    return r_tk @ np.linalg.solve(r_kk, np.asarray(h_known, dtype=complex))


def interpolate_2d(h_p_grid: np.ndarray,
                   pilot_symbol_idx: np.ndarray,
                   pilot_subcarrier_idx: np.ndarray,
                   n_sym: int,
                   n_fft: int,
                   method: str = 'linear',
                   corr_stats: Dict | None = None,
                   corr_path: str | None = None,
                   noise_var: float = 0.0,
                   cp_len: int | None = None) -> np.ndarray:
    """
    Separable 2D interpolation in time-frequency domain.

    Input:
    - h_p_grid: pilot-grid channel estimates, shape (Nt_p, Nf_p)
    - pilot_symbol_idx: OFDM symbol indices carrying pilots, shape (Nt_p,)
    - pilot_subcarrier_idx: pilot subcarrier indices, shape (Nf_p,)
    - n_sym: total number of OFDM symbols.
    - n_fft: number of subcarriers.
    - method: 'linear', 'quadratic', 'wiener', or 'dft'.
    - corr_stats: optional correlation-statistics dictionary.
    - corr_path: optional path of saved correlation statistics.
    - noise_var: noise variance.
    - cp_len: cyclic prefix length for DFT interpolation.

    Output:
    - h_est: interpolated CFR on all time-frequency positions, shape (n_sym, n_fft)
    """
    h_p_grid = np.asarray(h_p_grid, dtype=complex)
    pilot_symbol_idx = np.asarray(pilot_symbol_idx, dtype=int)
    pilot_subcarrier_idx = np.asarray(pilot_subcarrier_idx, dtype=int)

    if method == 'wiener':
        corr_stats = _extract_corr_matrices(corr_stats, corr_path)
        r_freq_full = corr_stats['R_freq_matrix']
        r_time_full = corr_stats['R_time_matrix']
    else:
        r_freq_full = r_time_full = None

    freq_est = np.zeros((len(pilot_symbol_idx), n_fft), dtype=complex)
    for i, _m in enumerate(pilot_symbol_idx):
        if method == 'linear':
            freq_est[i] = interpolate_linear_1d(h_p_grid[i], pilot_subcarrier_idx, n_fft)
        elif method == 'quadratic':
            freq_est[i] = interpolate_quadratic_1d(h_p_grid[i], pilot_subcarrier_idx, n_fft)
        elif method == 'wiener':
            freq_est[i] = interpolate_wiener_1d(h_p_grid[i], pilot_subcarrier_idx, n_fft,
                                                r_freq_full=r_freq_full, noise_var=noise_var)
        elif method == 'dft':
            if cp_len is None:
                raise ValueError('DFT interpolation requires cp_len.')
            freq_est[i] = interpolate_dft_1d(h_p_grid[i], pilot_subcarrier_idx, n_fft, cp_len)
        else:
            raise ValueError('Unsupported interpolation method.')

    h_est = np.zeros((n_sym, n_fft), dtype=complex)
    for k in range(n_fft):
        if len(pilot_symbol_idx) == 1:
            h_est[:, k] = freq_est[0, k]
        else:
            if method in ('linear', 'dft'):
                h_est[:, k] = _interp_time_linear(freq_est[:, k], pilot_symbol_idx, n_sym)
            elif method == 'quadratic':
                h_est[:, k] = _interp_time_quadratic(freq_est[:, k], pilot_symbol_idx, n_sym)
            elif method == 'wiener':
                h_est[:, k] = _interp_time_wiener(freq_est[:, k], pilot_symbol_idx, n_sym,
                                                  r_time_full=r_time_full, noise_var=noise_var)
    return h_est


def estimate_channel_from_pilots(y_p_grid: np.ndarray,
                                 x_p_grid: np.ndarray,
                                 pilot_symbol_idx: np.ndarray,
                                 pilot_subcarrier_idx: np.ndarray,
                                 n_sym: int,
                                 n_fft: int,
                                 est_method: str = 'LS',
                                 interp_method: str = 'linear',
                                 corr_stats: Dict | None = None,
                                 corr_path: str | None = None,
                                 noise_var: float = 0.0,
                                 cp_len: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate full time-frequency channel from pilot observations.

    Input:
    - y_p_grid: received pilot grid, shape (Nt_p, Nf_p)
    - x_p_grid: transmitted pilot grid, shape (Nt_p, Nf_p)
    - pilot_symbol_idx: pilot OFDM symbol indices, shape (Nt_p,)
    - pilot_subcarrier_idx: pilot subcarrier indices, shape (Nf_p,)
    - n_sym: total number of OFDM symbols.
    - n_fft: total number of subcarriers.
    - est_method: 'LS', 'MMSE', or 'LMMSE'
    - interp_method: 'linear', 'quadratic', 'wiener', or 'dft'
    - corr_stats: optional correlation-statistics dictionary.
    - corr_path: optional path of saved correlation statistics.
    - noise_var: AWGN variance.
    - cp_len: cyclic prefix length for DFT interpolation.

    Output:
    - h_p_hat: pilot-domain channel estimates, shape (Nt_p, Nf_p)
    - h_hat: full estimated CFR, shape (n_sym, n_fft)
    """
    y_p_grid = np.asarray(y_p_grid, dtype=complex)
    x_p_grid = np.asarray(x_p_grid, dtype=complex)
    est_method_u = est_method.upper()

    if est_method_u == 'LS':
        h_p_hat = ls_estimate(y_p_grid, x_p_grid)
    else:
        corr_stats = _extract_corr_matrices(corr_stats, corr_path)
        r_freq_full = corr_stats['R_freq_matrix']
        r_pp = r_freq_full[np.ix_(pilot_subcarrier_idx, pilot_subcarrier_idx)]
        if est_method_u == 'MMSE':
            h_p_hat = mmse_estimate(y_p_grid, x_p_grid, r_pp, noise_var)
        elif est_method_u == 'LMMSE':
            h_p_hat = lmmse_estimate(y_p_grid, x_p_grid, r_pp, noise_var)
        else:
            raise ValueError('Unsupported estimation method.')

    h_hat = interpolate_2d(h_p_hat, pilot_symbol_idx, pilot_subcarrier_idx,
                           n_sym, n_fft, method=interp_method,
                           corr_stats=corr_stats, corr_path=corr_path,
                           noise_var=noise_var, cp_len=cp_len)
    return h_p_hat, h_hat


def test_channel_estimation() -> None:
    """
    Simple self-test for estimation and interpolation functions.

    Input:
    - None.

    Output:
    - None.
    """
    rng = np.random.default_rng(0)
    n_sym = 4
    n_fft = 8
    pilot_symbol_idx = np.array([0, 2], dtype=int)
    pilot_subcarrier_idx = np.array([0, 4], dtype=int)

    h_true = np.ones((n_sym, n_fft), dtype=complex) * (1 + 1j)
    x_p = np.ones((len(pilot_symbol_idx), len(pilot_subcarrier_idx)), dtype=complex)
    y_p = h_true[np.ix_(pilot_symbol_idx, pilot_subcarrier_idx)] * x_p

    h_p_hat, h_hat = estimate_channel_from_pilots(
        y_p_grid=y_p,
        x_p_grid=x_p,
        pilot_symbol_idx=pilot_symbol_idx,
        pilot_subcarrier_idx=pilot_subcarrier_idx,
        n_sym=n_sym,
        n_fft=n_fft,
        est_method='LS',
        interp_method='linear'
    )

    assert np.allclose(h_p_hat, 1 + 1j)
    assert h_hat.shape == (n_sym, n_fft)
    print('channelestimation.py test passed.')


if __name__ == '__main__':
    test_channel_estimation()
