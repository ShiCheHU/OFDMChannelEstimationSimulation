"""
Description:
    SISO OFDM pilot-domain channel estimation and equalization utilities. This
    module contains LS, MMSE, and LMMSE pilot CFR estimators, pilot-domain
    theoretical NMSE formulas, one-tap SISO equalization, and BER calculation.

Args:
    - Received pilot grid y_p and transmitted pilot grid x_p.
    - Optional pilot-position frequency covariance r_pp.
    - Full received grid and estimated full-grid CFR for equalization.

Returns:
    - Pilot-position CFR estimates.
    - Equalized data symbols.
    - BER and theoretical pilot-domain NMSE values.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def ls_estimate(y_p: np.ndarray, x_p: np.ndarray) -> np.ndarray:
    """
    Estimate pilot-position CFR using least squares.

    Args:
        y_p: Received pilot symbols, shape (n_pilot_sym, n_pilot_sc) or any
            broadcast-compatible pilot grid shape.
        x_p: Transmitted pilot symbols with the same shape as y_p.

        Pilot observation model y_p = h_p x_p + w_p.

    Returns:
        LS pilot CFR estimate h_ls with the same shape as y_p.
    """
    return np.asarray(y_p, dtype=complex) / (np.asarray(x_p, dtype=complex) + EPS)


def mmse_estimate(y_p: np.ndarray, x_p: np.ndarray, r_pp: np.ndarray, noise_var: float) -> np.ndarray:
    """
    Estimate pilot-position CFR using covariance-aided MMSE filtering.

    Args:
        y_p: Received pilot symbols, shape (..., n_pilot_sc).
        x_p: Transmitted pilot symbols, shape compatible with y_p.
        r_pp: Pilot-subcarrier covariance matrix, shape (n_pilot_sc, n_pilot_sc).
        noise_var: AWGN variance per complex received symbol.

        LS pilot estimates and the channel covariance over pilot subcarriers.

    Returns:
        MMSE-filtered pilot CFR estimates with the same shape as y_p.
    """
    h_ls = ls_estimate(y_p, x_p)
    pilot_power = float(np.mean(np.abs(np.asarray(x_p, dtype=complex)) ** 2))
    eta = float(noise_var) / (pilot_power + EPS)
    r_pp = np.asarray(r_pp, dtype=complex)
    filt = r_pp @ np.linalg.solve(r_pp + eta * np.eye(r_pp.shape[0], dtype=complex), np.eye(r_pp.shape[0]))
    return np.asarray(h_ls) @ filt.T


def lmmse_estimate(y_p: np.ndarray, x_p: np.ndarray, r_pp: np.ndarray, noise_var: float) -> np.ndarray:
    """
    Estimate pilot-position CFR using the LMMSE estimator.

    Args:
        y_p: Received pilot symbols, shape (..., n_pilot_sc).
        x_p: Transmitted pilot symbols, shape compatible with y_p.
        r_pp: Pilot-subcarrier covariance matrix, shape (n_pilot_sc, n_pilot_sc).
        noise_var: AWGN variance per complex received symbol.

        Pilot observations and second-order channel statistics.

    Returns:
        LMMSE pilot CFR estimates with the same shape as y_p.
    """
    return mmse_estimate(y_p, x_p, r_pp, noise_var)


def estimate_pilot_channel(
    method: str,
    y_p: np.ndarray,
    x_p: np.ndarray,
    noise_var: float,
    r_pp: np.ndarray | None = None,
) -> np.ndarray:
    """
    Dispatch pilot-position CFR estimation by method name.

    Args:
        method: Estimator name, "ls", "mmse", or "lmmse".
        y_p: Received pilot symbols, shape (..., n_pilot_sc).
        x_p: Transmitted pilot symbols, shape compatible with y_p.
        noise_var: AWGN variance per complex received symbol.
        r_pp: Optional pilot-subcarrier covariance matrix required by MMSE and
            LMMSE, shape (n_pilot_sc, n_pilot_sc).

        Pilot observation grid and estimator selection.

    Returns:
        Pilot CFR estimate with the same leading shape as y_p.
    """
    method = method.lower()
    if method == "ls":
        return ls_estimate(y_p, x_p)
    if r_pp is None:
        raise ValueError(f"{method} requires r_pp.")
    if method == "mmse":
        return mmse_estimate(y_p, x_p, r_pp, noise_var)
    if method == "lmmse":
        return lmmse_estimate(y_p, x_p, r_pp, noise_var)
    raise ValueError("Unsupported estimator. Choose ls, mmse, or lmmse.")


def equalize_siso(y_grid: np.ndarray, h_est: np.ndarray) -> np.ndarray:
    """
    Equalize a SISO OFDM grid with one-tap frequency-domain division.

    Args:
        y_grid: Received OFDM grid, shape (n_sym, n_fft).
        h_est: Estimated full-grid CFR, shape (n_sym, n_fft).

        Frequency-domain SISO model Y = H X + W.

    Returns:
        Equalized symbol grid X_hat with shape (n_sym, n_fft).
    """
    return np.asarray(y_grid, dtype=complex) / (np.asarray(h_est, dtype=complex) + EPS)


def bit_error_rate(tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
    """
    Compute bit error rate between transmitted and detected bits.

    Args:
        tx_bits: Transmitted bit array.
        rx_bits: Detected bit array with the same shape as tx_bits.

        Two integer bit arrays.

    Returns:
        Scalar BER in [0, 1].
    """
    tx_bits = np.asarray(tx_bits, dtype=int)
    rx_bits = np.asarray(rx_bits, dtype=int)
    if tx_bits.shape != rx_bits.shape:
        raise ValueError(f"Bit array shape mismatch: {tx_bits.shape} vs {rx_bits.shape}")
    return float(np.mean(tx_bits != rx_bits))


def pilot_nmse_ls_theory(r_pp: np.ndarray, noise_var: float, pilot_power: float) -> float:
    """
    Compute theoretical pilot-domain LS NMSE.

    Args:
        r_pp: Pilot-subcarrier covariance matrix, shape (n_pilot_sc, n_pilot_sc).
        noise_var: AWGN variance per complex received symbol.
        pilot_power: Average pilot-symbol power.

        Pilot-domain channel covariance and noise/pilot power.

    Returns:
        Scalar theoretical LS NMSE at pilot positions.
    """
    r_pp = np.asarray(r_pp, dtype=complex)
    c_ls = (float(noise_var) / (float(pilot_power) + EPS)) * np.eye(r_pp.shape[0], dtype=complex)
    return float(np.real(np.trace(c_ls)) / (np.real(np.trace(r_pp)) + EPS))


def pilot_nmse_mmse_theory(r_pp: np.ndarray, noise_var: float, pilot_power: float) -> float:
    """
    Compute theoretical pilot-domain MMSE/LMMSE NMSE.

    Args:
        r_pp: Pilot-subcarrier covariance matrix, shape (n_pilot_sc, n_pilot_sc).
        noise_var: AWGN variance per complex received symbol.
        pilot_power: Average pilot-symbol power.

        Pilot-domain channel covariance and noise/pilot power.

    Returns:
        Scalar theoretical MMSE/LMMSE NMSE at pilot positions.
    """
    r_pp = np.asarray(r_pp, dtype=complex)
    eta = float(noise_var) / (float(pilot_power) + EPS)
    c_mmse = r_pp - r_pp @ np.linalg.solve(r_pp + eta * np.eye(r_pp.shape[0], dtype=complex), r_pp)
    return float(np.real(np.trace(c_mmse)) / (np.real(np.trace(r_pp)) + EPS))


def ls_nmse_lower_bound(noise_var_eff: float,
                        n_fft: int | None = None,
                        n_pilot_subcarrier: int | None = None,
                        l_h: int | None = None,
                        n_pilot_symbol: int | None = None) -> float:
    """
    Compute a theoretical LS channel-estimation NMSE lower bound.

    Args:
        noise_var_eff: Effective LS pilot-CFR noise variance, equal to
            sigma_w^2 / pilot_power for SISO pilots.
        n_fft: Number of OFDM subcarriers. If omitted, the function returns the
            pilot-domain LS NMSE bound noise_var_eff.
        n_pilot_subcarrier: Number of pilot subcarriers.
        l_h: Effective CIR support length used by finite-delay reconstruction.
        n_pilot_symbol: Number of OFDM symbols containing pilots.

        Noise variance and optional finite-delay OFDM/pilot dimensions.

    Returns:
        Scalar normalized LS estimation-error lower bound. With full dimension
        arguments, the value approximates an ideal finite-delay LS full-grid
        reconstruction bound.
    """
    if n_fft is None or n_pilot_subcarrier is None or l_h is None:
        return float(noise_var_eff)
    if n_pilot_subcarrier <= 0:
        raise ValueError("n_pilot_subcarrier must be positive.")
    n_time = 1 if n_pilot_symbol is None else max(int(n_pilot_symbol), 1)
    n_obs = n_time * int(n_pilot_subcarrier)
    return float(noise_var_eff * min(int(l_h), int(n_pilot_subcarrier)) / n_obs)


def lmmse_nmse_lower_bound(r_time: np.ndarray,
                           r_freq: np.ndarray,
                           q_idx: np.ndarray,
                           k_idx: np.ndarray,
                           noise_var_eff: float) -> float:
    """
    Compute the theoretical full-grid LMMSE/Wiener NMSE lower bound.

    Args:
        r_time: OFDM-symbol time covariance matrix, shape (n_sym, n_sym).
        r_freq: Subcarrier frequency covariance matrix, shape (n_fft, n_fft).
        q_idx: Pilot OFDM symbol indices, shape (n_pilot_sym,).
        k_idx: Pilot subcarrier indices, shape (n_pilot_sc,).
        noise_var_eff: Effective pilot-CFR noise variance.

        Separable time/frequency channel covariance and pilot-grid positions.

    Returns:
        Scalar normalized full-grid LMMSE estimation-error lower bound.
    """
    r_time = np.asarray(r_time, dtype=complex)
    r_freq = np.asarray(r_freq, dtype=complex)
    q_idx = np.asarray(q_idx, dtype=int)
    k_idx = np.asarray(k_idx, dtype=int)
    n_sym = r_time.shape[0]
    n_fft = r_freq.shape[0]
    n_pilot = len(q_idx) * len(k_idx)
    all_q = np.arange(n_sym, dtype=int)
    all_k = np.arange(n_fft, dtype=int)
    r_pp = np.kron(r_time[np.ix_(q_idx, q_idx)], r_freq[np.ix_(k_idx, k_idx)])
    r_dp = np.kron(r_time[np.ix_(all_q, q_idx)], r_freq[np.ix_(all_k, k_idx)])
    r_dd_trace = np.trace(np.kron(r_time, r_freq)).real
    c_yy = r_pp + float(noise_var_eff) * np.eye(n_pilot, dtype=complex)
    gain_trace = np.trace(r_dp @ np.linalg.solve(c_yy, r_dp.conj().T)).real
    mse = max(r_dd_trace - gain_trace, 0.0)
    return float(mse / (n_sym * n_fft + EPS))


def test_ls_noiseless() -> None:
    """
    Test LS estimation under noiseless pilot observations.

    Args:
        None.

        Hard-coded pilot channel and unit pilot symbols.

    Returns:
        None. Raises AssertionError on failure.
    """
    h = np.array([[1 + 2j, 2 - 1j]])
    x = np.ones_like(h)
    y = h * x
    assert np.allclose(ls_estimate(y, x), h)


def test_lmmse_shapes() -> None:
    """
    Test LMMSE output shape.

    Args:
        None.

        Small synthetic pilot grid and identity covariance matrix.

    Returns:
        None. Raises AssertionError on failure.
    """
    y = np.ones((2, 4), dtype=complex)
    x = np.ones((2, 4), dtype=complex)
    r = np.eye(4, dtype=complex)
    h = lmmse_estimate(y, x, r, 0.1)
    assert h.shape == y.shape


def test_equalize_siso_noiseless() -> None:
    """
    Test SISO equalization with perfect CSI and no noise.

    Args:
        None.

        Hard-coded transmitted symbols and channel coefficients.

    Returns:
        None. Raises AssertionError on failure.
    """
    x = np.array([1 + 1j, -1 + 0.5j])
    h = np.array([2 - 1j, 0.5 + 0.2j])
    y = h * x
    assert np.allclose(equalize_siso(y, h), x)


def test_theory_nmse_bounds() -> None:
    """
    Test theoretical LS and LMMSE NMSE bound helpers.

    Args:
        None.

        Small identity time/frequency covariance matrices and simple pilot grid.

    Returns:
        None. Raises AssertionError if bounds are non-finite or negative.
    """
    r_time = np.eye(2, dtype=complex)
    r_freq = np.eye(4, dtype=complex)
    ls_bound = ls_nmse_lower_bound(0.1, n_fft=4, n_pilot_subcarrier=2, l_h=2, n_pilot_symbol=2)
    lmmse_bound = lmmse_nmse_lower_bound(r_time, r_freq, np.array([0, 1]), np.array([0, 2]), 0.1)
    assert np.isfinite(ls_bound) and ls_bound >= 0.0
    assert np.isfinite(lmmse_bound) and lmmse_bound >= 0.0


if __name__ == "__main__":
    test_ls_noiseless()
    test_lmmse_shapes()
    test_equalize_siso_noiseless()
    test_theory_nmse_bounds()
    print("estimation.py tests passed.")
