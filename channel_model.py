
"""
generation channel dataset
generate channel autocorrelation
Support channel model:
- Rayleigh
- Rician
- TDL

This file implements a sampled time-selective multipath channel model.
The dataset is generated in the form of:
- Hf: frequency-domain CFR, shape (N_sample, N_sy, N_sc)
- Ht: time-domain CIR,      shape (N_sample, N_sy, tau_max_samples)

The default temporal evolution of each path gain uses the Jakes/Clarke
sum-of-sinusoids model. An AR(1) approximation is also provided.

This file also provides:
- analytic correlation matrices derived from the channel assumptions
- Monte Carlo estimated correlation matrices
- save/load helpers for correlation matrices
"""

from __future__ import annotations

import json
import os
from typing import Dict, Tuple

import numpy as np

EPS = 1e-12
try:
    from scipy.special import j0 as bessel_j0
except Exception:  # pragma: no cover
    bessel_j0 = None


def _make_rng(seed=None):
    """
    Create a random number generator.

    Input:
    - seed: int or None, random seed.

    Output:
    - rng: numpy random generator.
    """
    return np.random.default_rng(seed)


def _j0_safe(x: np.ndarray | float) -> np.ndarray | float:
    """
    Evaluate zeroth-order Bessel function J0(x).

    Input:
    - x: scalar or ndarray.

    Output:
    - y: J0(x), or a cosine approximation if scipy is unavailable.
    """
    if bessel_j0 is not None:
        return bessel_j0(x)
    return np.cos(x)


def _build_delay_profile(channel_type, tau_max_samples, num_path,
                         rician_k=0.0, tau_rms=None, rng=None):
    """
    Generate path delays and average path powers.

    Input:
    - channel_type: 'Rayleigh', 'Rician', or 'TDL'.
    - tau_max_samples: int, maximum delay spread in sample indices.
    - num_path: int, number of paths.
    - rician_k: float, Rician K factor in linear scale.
    - tau_rms: float or None, RMS delay spread used by TDL.
    - rng: numpy random generator.

    Output:
    - delays: ndarray, shape (num_path,), integer delay indices.
    - powers: ndarray, shape (num_path,), average power of each path, sum = 1.
    """
    if rng is None:
        rng = _make_rng()

    if tau_max_samples <= 0:
        raise ValueError("tau_max_samples must be positive.")
    if num_path <= 0:
        raise ValueError("num_path must be positive.")
    if num_path > tau_max_samples:
        raise ValueError("num_path should not exceed tau_max_samples if unique delays are used.")

    channel_type = str(channel_type).strip()

    if channel_type == 'Rayleigh':
        delays = np.sort(rng.choice(np.arange(tau_max_samples), size=num_path, replace=False))
        powers = np.ones(num_path, dtype=float) / num_path

    elif channel_type == 'Rician':
        if num_path == 1:
            delays = np.array([0], dtype=int)
            powers = np.array([1.0], dtype=float)
        else:
            nlos_delays = np.sort(rng.choice(np.arange(1, tau_max_samples), size=num_path - 1, replace=False))
            delays = np.concatenate(([0], nlos_delays))
            los_power = rician_k / (rician_k + 1.0)
            nlos_total = 1.0 / (rician_k + 1.0)
            nlos_powers = np.ones(num_path - 1, dtype=float) * (nlos_total / (num_path - 1))
            powers = np.concatenate(([los_power], nlos_powers))

    elif channel_type == 'TDL':
        delays = np.sort(rng.choice(np.arange(tau_max_samples), size=num_path, replace=False))
        if tau_rms is None:
            tau_rms = max(tau_max_samples / 3.0, 1.0)
        powers = np.exp(-delays / max(tau_rms, EPS))
        powers = powers / np.sum(powers)

    else:
        raise ValueError("Unsupported channel_type. Choose 'Rayleigh', 'Rician', or 'TDL'.")

    powers = np.asarray(powers, dtype=float)
    powers = powers / np.sum(powers)
    return delays.astype(int), powers


def get_reference_profile(config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get a deterministic reference delay profile from the configuration.

    Input:
    - config: channel configuration dictionary.

    Output:
    - delays: ndarray, shape (num_path,)
    - powers: ndarray, shape (num_path,)
    """
    profile_seed = config.get('profile_seed', config.get('seed', 0))
    rng = _make_rng(profile_seed)
    delays, powers = _build_delay_profile(
        channel_type=config.get('channel_type', 'TDL'),
        tau_max_samples=int(config.get('tau_max_samples', 8)),
        num_path=int(config.get('num_path', 4)),
        rician_k=float(config.get('rician_k', 0.0)),
        tau_rms=config.get('tau_rms', None),
        rng=rng,
    )
    return delays, powers


def _jakes_fading(num_path, N_sy, f_max, T_sym, n_sin=16, rng=None):
    """
    Generate time-selective complex fading coefficients using the Jakes/Clarke model.

    Input:
    - num_path: int, number of paths.
    - N_sy: int, number of OFDM symbols.
    - f_max: float, maximum Doppler frequency in Hz.
    - T_sym: float, OFDM symbol duration in seconds.
    - n_sin: int, number of sinusoids in the sum-of-sinusoids approximation.
    - rng: numpy random generator.

    Output:
    - g: ndarray, shape (num_path, N_sy), unit-power complex fading process.
    """
    if rng is None:
        rng = _make_rng()

    t = np.arange(N_sy, dtype=float) * T_sym
    g = np.zeros((num_path, N_sy), dtype=complex)

    if abs(f_max) < EPS:
        phases = rng.uniform(0.0, 2.0 * np.pi, size=num_path)
        return np.exp(1j * phases)[:, None] * np.ones((num_path, N_sy), dtype=complex)

    for l in range(num_path):
        theta = rng.uniform(0.0, 2.0 * np.pi)
        phi = rng.uniform(0.0, 2.0 * np.pi, size=n_sin)
        n = np.arange(1, n_sin + 1)
        alpha_n = (2.0 * np.pi * n - np.pi + theta) / (4.0 * n_sin)
        tones = np.exp(1j * (2.0 * np.pi * f_max * np.cos(alpha_n)[:, None] * t[None, :] + phi[:, None]))
        g[l, :] = np.sum(tones, axis=0) / np.sqrt(n_sin)

    power = np.mean(np.abs(g) ** 2, axis=1, keepdims=True)
    g = g / np.sqrt(power + EPS)
    return g


def _ar1_fading(num_path, N_sy, rho, rng=None):
    """
    Generate time-selective complex fading coefficients using an AR(1) process.

    Input:
    - num_path: int, number of paths.
    - N_sy: int, number of OFDM symbols.
    - rho: float, adjacent-symbol correlation coefficient.
    - rng: numpy random generator.

    Output:
    - g: ndarray, shape (num_path, N_sy), unit-power complex fading process.
    """
    if rng is None:
        rng = _make_rng()

    rho = float(np.clip(abs(rho), 0.0, 0.999999))
    g = np.zeros((num_path, N_sy), dtype=complex)
    g[:, 0] = (rng.standard_normal(num_path) + 1j * rng.standard_normal(num_path)) / np.sqrt(2.0)
    for m in range(1, N_sy):
        w = (rng.standard_normal(num_path) + 1j * rng.standard_normal(num_path)) / np.sqrt(2.0)
        g[:, m] = rho * g[:, m - 1] + np.sqrt(1.0 - rho ** 2) * w

    power = np.mean(np.abs(g) ** 2, axis=1, keepdims=True)
    g = g / np.sqrt(power + EPS)
    return g


def _gen_multipath(N_sy, N_sc, tau_max_samples, num_path,
                   rician_k=0.0, f_max=50.0, fs=960e3,
                   channel_type='Rayleigh', fading_model='jakes',
                   tau_rms=None, cp_len=0, n_sin=16, ar_rho=None,
                   normalize=True, seed=None,
                   delays=None, powers=None):
    """
    Generate one multipath channel realization, including CFR and CIR.

    Input:
    - N_sy: int, number of OFDM symbols.
    - N_sc: int, number of subcarriers.
    - tau_max_samples: int, maximum delay spread in samples.
    - num_path: int, number of paths.
    - rician_k: float, Rician K factor in linear scale (0 = Rayleigh-like scattered path only).
    - f_max: float, maximum Doppler frequency in Hz.
    - fs: float, sampling rate in Hz.
    - channel_type: 'Rayleigh', 'Rician', or 'TDL'.
    - fading_model: 'jakes' or 'ar1'.
    - tau_rms: float or None, RMS delay spread used by TDL.
    - cp_len: int, CP length in samples, used to compute OFDM symbol duration.
    - n_sin: int, number of sinusoids used by the Jakes model.
    - ar_rho: float or None, AR(1) coefficient. If None, it is approximated from f_max.
    - normalize: bool, whether to normalize average power to 1.
    - seed: int or None, random seed.
    - delays: ndarray or None, optional pre-defined path delays.
    - powers: ndarray or None, optional pre-defined path powers.

    Output:
    - Hf: ndarray, shape (N_sy, N_sc), frequency-domain CFR.
    - Ht: ndarray, shape (N_sy, tau_max_samples), time-domain CIR.
    - delays: ndarray, shape (num_path,), integer path delays.
    - powers: ndarray, shape (num_path,), average path powers.
    """
    rng = _make_rng(seed)

    if N_sy <= 0 or N_sc <= 0:
        raise ValueError("N_sy and N_sc must be positive.")
    if fs <= 0:
        raise ValueError("fs must be positive.")

    if delays is None or powers is None:
        delays, powers = _build_delay_profile(
            channel_type=channel_type,
            tau_max_samples=tau_max_samples,
            num_path=num_path,
            rician_k=rician_k,
            tau_rms=tau_rms,
            rng=rng
        )
    else:
        delays = np.asarray(delays, dtype=int)
        powers = np.asarray(powers, dtype=float)
        powers = powers / np.sum(powers)

    T_sym = (N_sc + cp_len) / fs

    if fading_model == 'jakes':
        g_scatter = _jakes_fading(num_path, N_sy, f_max, T_sym, n_sin=n_sin, rng=rng)
    elif fading_model == 'ar1':
        if ar_rho is None:
            x = 2.0 * np.pi * abs(f_max) * T_sym
            ar_rho = _j0_safe(x)
            ar_rho = float(np.clip(abs(ar_rho), 0.0, 0.999999))
        g_scatter = _ar1_fading(num_path, N_sy, ar_rho, rng=rng)
    else:
        raise ValueError("Unsupported fading_model. Choose 'jakes' or 'ar1'.")

    if channel_type == 'Rician':
        g = g_scatter.copy()
        theta_los = rng.uniform(0.0, 2.0 * np.pi)
        nu_los = f_max * np.cos(theta_los)
        phi0 = rng.uniform(0.0, 2.0 * np.pi)
        t = np.arange(N_sy, dtype=float) * T_sym
        los = np.exp(1j * (2.0 * np.pi * nu_los * t + phi0))
        g[0, :] = np.sqrt(rician_k / (rician_k + 1.0)) * los + \
                  np.sqrt(1.0 / (rician_k + 1.0)) * g_scatter[0, :]
    else:
        g = g_scatter

    Ht = np.zeros((N_sy, tau_max_samples), dtype=complex)
    for l in range(num_path):
        Ht[:, delays[l]] += np.sqrt(powers[l]) * g[l, :]

    Hf = np.fft.fft(Ht, n=N_sc, axis=1)

    if normalize:
        avg_power = np.mean(np.abs(Hf) ** 2)
        Hf = Hf / np.sqrt(avg_power + EPS)
        Ht = Ht / np.sqrt(avg_power + EPS)

    return Hf, Ht, delays, powers


def generate_channel_dataset(config):
    """
    Generate a channel dataset according to the input configuration.

    Input:
    - config: dict, required/optional fields include:
        'N_sample'        : int, number of independent channel realizations.
        'N_sy'            : int, number of OFDM symbols per realization.
        'N_sc'            : int, number of subcarriers.
        'tau_max_samples' : int, maximum delay spread in samples.
        'num_path'        : int, number of paths.
        'channel_type'    : 'Rayleigh', 'Rician', or 'TDL'.
        'fading_model'    : 'jakes' or 'ar1'.
        'rician_k'        : float, Rician K factor.
        'f_max'           : float, maximum Doppler frequency in Hz.
        'fs'              : float, sampling rate in Hz.
        'tau_rms'         : float or None, RMS delay spread used by TDL.
        'cp_len'          : int, CP length in samples.
        'n_sin'           : int, number of sinusoids for Jakes.
        'ar_rho'          : float or None, AR(1) coefficient.
        'normalize'       : bool, whether to normalize average power.
        'seed'            : int or None, random seed.
        'fixed_profile'   : bool, whether to use one fixed delay/PDP profile for all realizations.

    Output:
    - dataset: dict containing
        'Hf'      : ndarray, shape (N_sample, N_sy, N_sc)
        'Ht'      : ndarray, shape (N_sample, N_sy, tau_max_samples)
        'delays'  : ndarray, shape (N_sample, num_path)
        'powers'  : ndarray, shape (N_sample, num_path)
        'config'  : dict, a shallow copy of input config
    """
    N_sample = int(config.get('N_sample', 100))
    N_sy = int(config.get('N_sy', 14))
    N_sc = int(config.get('N_sc', 64))
    tau_max_samples = int(config.get('tau_max_samples', 12))
    num_path = int(config.get('num_path', 6))

    channel_type = config.get('channel_type', 'Rayleigh')
    fading_model = config.get('fading_model', 'jakes')
    rician_k = float(config.get('rician_k', 6.0))
    f_max = float(config.get('f_max', 50.0))
    fs = float(config.get('fs', 960e3))
    tau_rms = config.get('tau_rms', None)
    cp_len = int(config.get('cp_len', 0))
    n_sin = int(config.get('n_sin', 16))
    ar_rho = config.get('ar_rho', None)
    normalize = bool(config.get('normalize', True))
    seed = config.get('seed', None)
    fixed_profile = bool(config.get('fixed_profile', True))

    if N_sample <= 0:
        raise ValueError("N_sample must be positive.")

    Hf_set = np.zeros((N_sample, N_sy, N_sc), dtype=complex)
    Ht_set = np.zeros((N_sample, N_sy, tau_max_samples), dtype=complex)
    delays_set = np.zeros((N_sample, num_path), dtype=int)
    powers_set = np.zeros((N_sample, num_path), dtype=float)

    ref_delays = ref_powers = None
    if fixed_profile:
        ref_delays, ref_powers = get_reference_profile(config)

    for s in range(N_sample):
        local_seed = None if seed is None else int(seed) + s
        Hf, Ht, delays, powers = _gen_multipath(
            N_sy=N_sy,
            N_sc=N_sc,
            tau_max_samples=tau_max_samples,
            num_path=num_path,
            rician_k=rician_k,
            f_max=f_max,
            fs=fs,
            channel_type=channel_type,
            fading_model=fading_model,
            tau_rms=tau_rms,
            cp_len=cp_len,
            n_sin=n_sin,
            ar_rho=ar_rho,
            normalize=normalize,
            seed=local_seed,
            delays=ref_delays,
            powers=ref_powers
        )
        Hf_set[s] = Hf
        Ht_set[s] = Ht
        delays_set[s] = delays
        powers_set[s] = powers

    return {
        'Hf': Hf_set,
        'Ht': Ht_set,
        'delays': delays_set,
        'powers': powers_set,
        'config': dict(config),
    }


def _to_3d_channel_array(arr, name='arr'):
    """
    Convert channel array to a 3D array.

    Input:
    - arr: ndarray of shape (N_sample, N_a, N_b) or (N_a, N_b)
    - name: variable name for error message.

    Output:
    - arr_3d: ndarray of shape (N_sample, N_a, N_b)
    """
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"{name} must have shape (N_sample, N_a, N_b) or (N_a, N_b).")
    return arr


def _normalize_matrix(R):
    """
    Normalize a correlation matrix so that its average diagonal value equals 1.

    Input:
    - R: ndarray, square correlation matrix.

    Output:
    - Rn: normalized correlation matrix.
    """
    R = np.asarray(R, dtype=complex)
    diag_mean = np.mean(np.real(np.diag(R)))
    if abs(diag_mean) < EPS:
        return R
    return R / diag_mean


def _autocorr_lag_time(Hf, normalize=True):
    """
    Estimate nonnegative-lag time autocorrelation from CFR dataset.

    Input:
    - Hf: ndarray, shape (N_sample, N_sy, N_sc)
    - normalize: bool, normalize by zero-lag value.

    Output:
    - stats: dict with key 'R_time'.
    """
    N_sample, N_sy, N_sc = Hf.shape
    R = np.zeros(N_sy, dtype=complex)
    for dm in range(N_sy):
        block_a = Hf[:, :N_sy - dm, :]
        block_b = Hf[:, dm:, :]
        R[dm] = np.mean(block_a * np.conj(block_b))
    if normalize and abs(R[0]) > EPS:
        R = R / R[0]
    return {'R_time': R}


def _autocorr_lag_freq(Hf, normalize=True):
    """
    Estimate nonnegative-lag frequency autocorrelation from CFR dataset.

    Input:
    - Hf: ndarray, shape (N_sample, N_sy, N_sc)
    - normalize: bool, normalize by zero-lag value.

    Output:
    - stats: dict with key 'R_freq'.
    """
    N_sample, N_sy, N_sc = Hf.shape
    R = np.zeros(N_sc, dtype=complex)
    for dk in range(N_sc):
        block_a = Hf[:, :, :N_sc - dk]
        block_b = Hf[:, :, dk:]
        R[dk] = np.mean(block_a * np.conj(block_b))
    if normalize and abs(R[0]) > EPS:
        R = R / R[0]
    return {'R_freq': R}


def _autocorr_lag_delay(Ht, normalize=True):
    """
    Estimate nonnegative-lag delay autocorrelation from CIR dataset.

    Input:
    - Ht: ndarray, shape (N_sample, N_sy, N_tap)
    - normalize: bool, normalize by zero-lag value.

    Output:
    - stats: dict with key 'R_delay'.
    """
    N_sample, N_sy, N_tap = Ht.shape
    R = np.zeros(N_tap, dtype=complex)
    for dn in range(N_tap):
        block_a = Ht[:, :, :N_tap - dn]
        block_b = Ht[:, :, dn:]
        R[dn] = np.mean(block_a * np.conj(block_b))
    if normalize and abs(R[0]) > EPS:
        R = R / R[0]
    return {'R_delay': R}


def _autocorr_time_matrix(Hf, normalize=True):
    """
    Estimate full time autocorrelation matrix from CFR dataset.

    Input:
    - Hf: ndarray, shape (N_sample, N_sy, N_sc)
    - normalize: bool, normalize matrix so the average diagonal is 1.

    Output:
    - stats: dict with key 'R_time_matrix'
    """
    N_sample, N_sy, N_sc = Hf.shape
    R = np.einsum('smk,snk->mn', Hf, np.conj(Hf)) / (N_sample * N_sc)
    if normalize:
        R = _normalize_matrix(R)
    return {'R_time_matrix': R}


def _autocorr_freq_matrix(Hf, normalize=True):
    """
    Estimate full frequency autocorrelation matrix from CFR dataset.

    Input:
    - Hf: ndarray, shape (N_sample, N_sy, N_sc)
    - normalize: bool, normalize matrix so the average diagonal is 1.

    Output:
    - stats: dict with key 'R_freq_matrix'
    """
    N_sample, N_sy, N_sc = Hf.shape
    R = np.einsum('smk,sml->kl', Hf, np.conj(Hf)) / (N_sample * N_sy)
    if normalize:
        R = _normalize_matrix(R)
    return {'R_freq_matrix': R}


def _autocorr_delay_matrix(Ht, normalize=True):
    """
    Estimate full delay autocorrelation matrix from CIR dataset.

    Input:
    - Ht: ndarray, shape (N_sample, N_sy, N_tap)
    - normalize: bool, normalize matrix so the average diagonal is 1.

    Output:
    - stats: dict with key 'R_delay_matrix'
    """
    N_sample, N_sy, N_tap = Ht.shape
    R = np.einsum('smn,smp->np', Ht, np.conj(Ht)) / (N_sample * N_sy)
    if normalize:
        R = _normalize_matrix(R)
    return {'R_delay_matrix': R}


def _autocorr_full_matrix(Hf, normalize=True):
    """
    Estimate full 2D covariance matrix of vec(Hf).

    Input:
    - Hf: ndarray, shape (N_sample, N_sy, N_sc)
    - normalize: bool, normalize matrix so the average diagonal is 1.

    Output:
    - stats: dict with key 'R_full_matrix'
    """
    N_sample, N_sy, N_sc = Hf.shape
    X = Hf.reshape(N_sample, N_sy * N_sc)
    R = (X.conj().T @ X) / N_sample
    if normalize:
        R = _normalize_matrix(R)
    return {'R_full_matrix': R}


def generate_channel_autocorrelation(Hf, Ht=None, mode='lag', normalize=True):
    """
    Estimate channel autocorrelation functions or autocorrelation matrices from the dataset.

    Input:
    - Hf: ndarray, shape (N_sample, N_sy, N_sc) or (N_sy, N_sc),
          frequency-domain CFR dataset.
    - Ht: ndarray, shape (N_sample, N_sy, N_tap) or (N_sy, N_tap), optional,
          time-domain CIR dataset. It is required when delay-domain correlation is requested.
    - mode: str, autocorrelation output mode. Supported modes are:
        'lag'          : return nonnegative-lag time/frequency autocorrelation, and delay-lag if Ht is given.
        'time_matrix'  : return the full time autocorrelation matrix of shape (N_sy, N_sy).
        'freq_matrix'  : return the full frequency autocorrelation matrix of shape (N_sc, N_sc).
        'delay_matrix' : return the full delay autocorrelation matrix of shape (N_tap, N_tap).
        'full_matrix'  : return the full 2D autocorrelation matrix of vec(Hf),
                         with shape (N_sy*N_sc, N_sy*N_sc).
        'all'          : return all available lag functions and matrices.
    - normalize: bool, whether to normalize the returned correlation function or matrix.

    Output:
    - stats: dict containing the requested autocorrelation functions or matrices.
    """
    Hf = _to_3d_channel_array(Hf, 'Hf')
    Ht_3d = None
    if Ht is not None:
        Ht_3d = _to_3d_channel_array(Ht, 'Ht')

    mode = str(mode).lower()
    valid_modes = {'lag', 'time_matrix', 'freq_matrix', 'delay_matrix', 'full_matrix', 'all'}
    if mode not in valid_modes:
        raise ValueError(f"Unsupported mode: {mode}. Choose from {sorted(valid_modes)}.")

    stats = {}
    if mode in {'lag', 'all'}:
        stats.update(_autocorr_lag_time(Hf, normalize=normalize))
        stats.update(_autocorr_lag_freq(Hf, normalize=normalize))
        if Ht_3d is not None:
            stats.update(_autocorr_lag_delay(Ht_3d, normalize=normalize))

    if mode in {'time_matrix', 'all'}:
        stats.update(_autocorr_time_matrix(Hf, normalize=normalize))
    if mode in {'freq_matrix', 'all'}:
        stats.update(_autocorr_freq_matrix(Hf, normalize=normalize))
    if mode in {'delay_matrix', 'all'}:
        if Ht_3d is None:
            raise ValueError("Ht must be provided when mode is 'delay_matrix' or 'all'.")
        stats.update(_autocorr_delay_matrix(Ht_3d, normalize=normalize))
    if mode in {'full_matrix', 'all'}:
        stats.update(_autocorr_full_matrix(Hf, normalize=normalize))
    return stats


def _build_freq_covariance_from_profile(n_fft: int, delays: np.ndarray, powers: np.ndarray) -> np.ndarray:
    """
    Build the full frequency covariance matrix from a fixed delay profile.

    Input:
    - n_fft: FFT size.
    - delays: delay taps in samples, shape (L,)
    - powers: average powers, shape (L,)

    Output:
    - Rf: frequency covariance matrix, shape (n_fft, n_fft)
    """
    idx = np.arange(n_fft, dtype=int)
    delta_k = idx.reshape(-1, 1) - idx.reshape(1, -1)
    delays = np.asarray(delays, dtype=float).reshape(-1, 1, 1)
    powers = np.asarray(powers, dtype=float).reshape(-1, 1, 1)
    Rf = np.sum(powers * np.exp(-1j * 2.0 * np.pi * delays * delta_k / n_fft), axis=0)
    return _normalize_matrix(Rf)


def _build_time_covariance_from_model(config: Dict) -> np.ndarray:
    """
    Build the full time covariance matrix from the fading model assumption.

    Input:
    - config: channel configuration dictionary.

    Output:
    - Rt: time covariance matrix, shape (N_sy, N_sy)
    """
    N_sy = int(config.get('N_sy', 14))
    N_sc = int(config.get('N_sc', 64))
    cp_len = int(config.get('cp_len', 0))
    fs = float(config.get('fs', 960e3))
    f_max = float(config.get('f_max', 0.0))
    fading_model = str(config.get('fading_model', 'jakes')).lower()
    ar_rho = config.get('ar_rho', None)

    T_sym = (N_sc + cp_len) / fs
    idx = np.arange(N_sy, dtype=float)
    lag = np.abs(idx.reshape(-1, 1) - idx.reshape(1, -1))

    if fading_model == 'jakes':
        Rt = _j0_safe(2.0 * np.pi * abs(f_max) * T_sym * lag)
    elif fading_model == 'ar1':
        if ar_rho is None:
            ar_rho = _j0_safe(2.0 * np.pi * abs(f_max) * T_sym)
        rho = float(np.clip(abs(ar_rho), 0.0, 0.999999))
        Rt = rho ** lag
    else:
        raise ValueError("Unsupported fading_model. Choose 'jakes' or 'ar1'.")
    return _normalize_matrix(Rt.astype(complex))


def derive_channel_autocorrelation_from_config(config: Dict, normalize=True) -> Dict:
    """
    Derive correlation matrices analytically from the channel configuration.

    Input:
    - config: channel configuration dictionary.
    - normalize: bool, whether to normalize the matrices.

    Output:
    - stats: dict containing reference delay/PDP and correlation matrices.
    """
    delays, powers = get_reference_profile(config)
    n_fft = int(config.get('N_sc', 64))
    Rt = _build_time_covariance_from_model(config)
    Rf = _build_freq_covariance_from_profile(n_fft, delays, powers)
    Rfull = np.kron(Rt, Rf)
    if normalize:
        Rt = _normalize_matrix(Rt)
        Rf = _normalize_matrix(Rf)
        Rfull = _normalize_matrix(Rfull)
    return {
        'source': 'theory',
        'delays': delays,
        'powers': powers,
        'R_time_matrix': Rt,
        'R_freq_matrix': Rf,
        'R_full_matrix': Rfull,
    }


def _corr_filename(config: Dict, source: str, mc_frames: int, out_dir: str) -> str:
    """
    Build the correlation-matrix cache filename.

    Input:
    - config: channel configuration dictionary.
    - source: 'theory' or 'mc'
    - mc_frames: Monte Carlo frame count.
    - out_dir: output directory.

    Output:
    - path: cache file path ending with .npz
    """
    tag = 'theory' if source == 'theory' else f'mc{int(mc_frames)}'
    fname = (
        f"corr_{tag}_"
        f"{config.get('channel_type','TDL')}_{config.get('fading_model','jakes')}_"
        f"Nsy{int(config.get('N_sy',14))}_Nsc{int(config.get('N_sc',64))}_"
        f"L{int(config.get('num_path',4))}_tau{int(config.get('tau_max_samples',8))}_"
        f"fmax{float(config.get('f_max',0.0)):.3f}_fs{float(config.get('fs',0.0)):.1f}_"
        f"seed{config.get('seed',0)}.npz"
    )
    return os.path.join(out_dir, fname)


def save_correlation_stats(path: str, stats: Dict) -> None:
    """
    Save correlation statistics to an .npz file.

    Input:
    - path: output file path.
    - stats: correlation statistics dictionary.

    Output:
    - None.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {}
    meta = {}
    for k, v in stats.items():
        if isinstance(v, np.ndarray):
            payload[k] = v
        else:
            meta[k] = v
    payload['meta_json'] = np.array(json.dumps(meta), dtype=object)
    np.savez(path, **payload)


def load_correlation_stats(path: str) -> Dict:
    """
    Load correlation statistics from an .npz file.

    Input:
    - path: saved correlation file path.

    Output:
    - stats: correlation statistics dictionary.
    """
    with np.load(path, allow_pickle=True) as data:
        stats = {k: data[k] for k in data.files if k != 'meta_json'}
        if 'meta_json' in data.files:
            meta = json.loads(str(data['meta_json'].item()))
            stats.update(meta)
    return stats


def get_or_create_correlation_stats(config: Dict,
                                    source: str = 'theory',
                                    mc_frames: int = 1000,
                                    out_dir: str = './corr_stats',
                                    force_regen: bool = False) -> Tuple[Dict, str]:
    """
    Get correlation matrices from cache, or create and save them.

    Input:
    - config: channel configuration dictionary.
    - source: 'theory' or 'mc'
    - mc_frames: Monte Carlo frame count used when source='mc'
    - out_dir: cache directory
    - force_regen: whether to ignore existing cache and regenerate

    Output:
    - stats: correlation statistics dictionary
    - path: cache file path
    """
    source = str(source).lower()
    if source not in {'theory', 'mc'}:
        raise ValueError("source must be 'theory' or 'mc'.")
    path = _corr_filename(config, source, mc_frames, out_dir)
    if (not force_regen) and os.path.exists(path):
        return load_correlation_stats(path), path

    if source == 'theory':
        stats = derive_channel_autocorrelation_from_config(config, normalize=True)
        stats['path'] = path
        save_correlation_stats(path, stats)
        return stats, path

    mc_config = dict(config)
    mc_config['N_sample'] = int(mc_frames)
    dataset = generate_channel_dataset(mc_config)
    stats = generate_channel_autocorrelation(dataset['Hf'], dataset['Ht'], mode='all', normalize=True)
    stats['source'] = f'mc{int(mc_frames)}'
    # Save a reference profile too for convenience.
    delays, powers = get_reference_profile(config)
    stats['delays'] = delays
    stats['powers'] = powers
    stats['path'] = path
    save_correlation_stats(path, stats)
    return stats, path


def test_channel_model():
    """
    Test the channel model generator and the autocorrelation estimator.

    Input:
    - None. Internal test parameters are used.

    Output:
    - dataset: generated channel dataset.
    - stats: estimated channel autocorrelation functions.
    """
    config = {
        'N_sample': 8,
        'N_sy': 32,
        'N_sc': 64,
        'tau_max_samples': 12,
        'num_path': 6,
        'channel_type': 'Rician',
        'fading_model': 'jakes',
        'rician_k': 6.0,
        'f_max': 50.0,
        'fs': 960e3,
        'tau_rms': 4.0,
        'cp_len': 16,
        'n_sin': 16,
        'normalize': True,
        'seed': 42,
        'fixed_profile': True,
    }

    dataset = generate_channel_dataset(config)
    stats = generate_channel_autocorrelation(dataset['Hf'], dataset['Ht'], normalize=True)
    theory, path = get_or_create_correlation_stats(config, source='theory', out_dir='./corr_stats_test', force_regen=True)

    print("==== channel_model.py test ====")
    print("Hf shape:", dataset['Hf'].shape)
    print("Ht shape:", dataset['Ht'].shape)
    print("delays shape:", dataset['delays'].shape)
    print("powers shape:", dataset['powers'].shape)
    print("R_time shape:", stats['R_time'].shape)
    print("R_freq shape:", stats['R_freq'].shape)
    print("R_delay shape:", stats['R_delay'].shape)
    print("Theory corr path:", path)
    return dataset, stats, theory


if __name__ == "__main__":
    test_channel_model()
