"""
Description:
    SISO OFDM channel interpolation over frequency and time. The module starts
    from pilot-position CFR estimates and reconstructs the full OFDM resource
    grid using linear, quadratic, Wiener/LMMSE, or DFT-based interpolation.

Args:
    - Pilot CFR estimates h_p_grid with shape (n_pilot_sym, n_pilot_sc).
    - Pilot OFDM symbol indices q_idx and pilot subcarrier indices k_idx.
    - Optional time/frequency covariance matrices for Wiener interpolation.

Returns:
    - Full-grid CFR estimate h_est with shape (n_sym, n_fft).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def interpolate_linear_1d(h_p: np.ndarray, pilot_idx: np.ndarray, n_fft: int) -> np.ndarray:
    """
    Interpolate CFR values over frequency using linear interpolation.

    Args:
        h_p: CFR values at pilot subcarriers, shape (n_pilot_sc,).
        pilot_idx: Pilot subcarrier indices, shape (n_pilot_sc,).
        n_fft: Total number of OFDM subcarriers.

        Sparse frequency-domain channel samples for one OFDM symbol.

    Returns:
        Interpolated CFR over all subcarriers, shape (n_fft,).
    """
    pilot_idx = np.asarray(pilot_idx, dtype=int)
    h_p = np.asarray(h_p, dtype=complex)
    all_k = np.arange(n_fft)
    real = np.interp(all_k, pilot_idx, np.real(h_p))
    imag = np.interp(all_k, pilot_idx, np.imag(h_p))
    return real + 1j * imag


def _quadratic_window_interp(x: float, x_pts: np.ndarray, y_pts: np.ndarray) -> complex:
    """
    Evaluate a three-point quadratic Lagrange interpolation.

    Args:
        x: Target coordinate.
        x_pts: Three support coordinates, shape (3,).
        y_pts: Three complex support values, shape (3,).

        Local interpolation window.

    Returns:
        Complex interpolated value at x.
    """
    x0, x1, x2 = x_pts
    y0, y1, y2 = y_pts
    l0 = (x - x1) * (x - x2) / ((x0 - x1) * (x0 - x2) + EPS)
    l1 = (x - x0) * (x - x2) / ((x1 - x0) * (x1 - x2) + EPS)
    l2 = (x - x0) * (x - x1) / ((x2 - x0) * (x2 - x1) + EPS)
    return l0 * y0 + l1 * y1 + l2 * y2


def interpolate_quadratic_1d(h_p: np.ndarray, pilot_idx: np.ndarray, n_fft: int) -> np.ndarray:
    """
    Interpolate CFR values over frequency using quadratic interpolation.

    Args:
        h_p: CFR values at pilot subcarriers, shape (n_pilot_sc,).
        pilot_idx: Pilot subcarrier indices, shape (n_pilot_sc,).
        n_fft: Total number of OFDM subcarriers.

        Sparse frequency-domain channel samples for one OFDM symbol.

    Returns:
        Interpolated CFR over all subcarriers, shape (n_fft,). Falls back to
        linear interpolation when fewer than three pilot subcarriers exist.
    """
    pilot_idx = np.asarray(pilot_idx, dtype=int)
    h_p = np.asarray(h_p, dtype=complex)
    if pilot_idx.size < 3:
        return interpolate_linear_1d(h_p, pilot_idx, n_fft)
    h_est = np.zeros(n_fft, dtype=complex)
    for k in range(n_fft):
        if k <= pilot_idx[1]:
            win = np.array([0, 1, 2])
        elif k >= pilot_idx[-2]:
            win = np.array([pilot_idx.size - 3, pilot_idx.size - 2, pilot_idx.size - 1])
        else:
            g = np.searchsorted(pilot_idx, k, side="right") - 1
            g = int(np.clip(g, 1, pilot_idx.size - 2))
            win = np.array([g - 1, g, g + 1])
        h_est[k] = _quadratic_window_interp(float(k), pilot_idx[win].astype(float), h_p[win])
    return h_est


def interpolate_wiener_1d(
    h_p: np.ndarray,
    pilot_idx: np.ndarray,
    n_fft: int,
    r_freq: np.ndarray,
    noise_var: float = 0.0,
    pilot_power: float = 1.0,
) -> np.ndarray:
    """
    Interpolate CFR values over frequency using Wiener/LMMSE filtering.

    Args:
        h_p: CFR values at pilot subcarriers, shape (n_pilot_sc,).
        pilot_idx: Pilot subcarrier indices, shape (n_pilot_sc,).
        n_fft: Total number of OFDM subcarriers.
        r_freq: Full frequency covariance matrix, shape (n_fft, n_fft).
        noise_var: Effective pilot-estimation noise variance.
        pilot_power: Average pilot-symbol power.

        Pilot CFR estimates and frequency-domain second-order statistics.

    Returns:
        Wiener-interpolated CFR over all subcarriers, shape (n_fft,).
    """
    pilot_idx = np.asarray(pilot_idx, dtype=int)
    all_idx = np.arange(n_fft, dtype=int)
    r_dp = r_freq[np.ix_(all_idx, pilot_idx)]
    r_pp = r_freq[np.ix_(pilot_idx, pilot_idx)]
    r_pp = r_pp + (noise_var / (pilot_power + EPS)) * np.eye(pilot_idx.size, dtype=complex)
    return r_dp @ np.linalg.solve(r_pp, np.asarray(h_p, dtype=complex))


def interpolate_dft_1d(h_p: np.ndarray, pilot_idx: np.ndarray, n_fft: int, l_h: int) -> np.ndarray:
    """
    Reconstruct full-frequency CFR using a finite-delay DFT method.

    Args:
        h_p: CFR values at equally spaced pilot subcarriers, shape (n_pilot_sc,).
        pilot_idx: Pilot subcarrier indices, shape (n_pilot_sc,).
        n_fft: Total number of OFDM subcarriers.
        l_h: Assumed effective CIR support length.

        Comb-pilot CFR samples for one OFDM symbol.

    Returns:
        Full CFR estimate, shape (n_fft,). Falls back to linear interpolation
        if pilot positions are not compatible with the DFT method.
    """
    pilot_idx = np.asarray(pilot_idx, dtype=int)
    h_p = np.asarray(h_p, dtype=complex)
    if pilot_idx.size < 2:
        return np.ones(n_fft, dtype=complex) * h_p[0]
    spacing = np.diff(pilot_idx)
    if not np.all(spacing == spacing[0]) or pilot_idx[0] != 0:
        return interpolate_linear_1d(h_p, pilot_idx, n_fft)
    n_p = pilot_idx.size
    h_alias = np.fft.ifft(h_p, n=n_p)
    support = int(min(l_h, n_p))
    h_trunc = np.zeros(n_p, dtype=complex)
    h_trunc[:support] = h_alias[:support]
    h_pad = np.zeros(n_fft, dtype=complex)
    h_pad[:n_p] = h_trunc
    return np.fft.fft(h_pad, n=n_fft)


def _interp_time_linear(h_known: np.ndarray, known_idx: np.ndarray, n_sym: int) -> np.ndarray:
    """
    Interpolate CFR values along the OFDM symbol axis linearly.

    Args:
        h_known: Known channel values over time, shape (n_known,).
        known_idx: OFDM symbol indices for known values, shape (n_known,).
        n_sym: Total number of OFDM symbols.

        One subcarrier's channel values at pilot-bearing OFDM symbols.

    Returns:
        Time-interpolated channel sequence, shape (n_sym,).
    """
    t = np.arange(n_sym)
    real = np.interp(t, known_idx, np.real(h_known))
    imag = np.interp(t, known_idx, np.imag(h_known))
    return real + 1j * imag


def _interp_time_quadratic(h_known: np.ndarray, known_idx: np.ndarray, n_sym: int) -> np.ndarray:
    """
    Interpolate CFR values along the OFDM symbol axis quadratically.

    Args:
        h_known: Known channel values over time, shape (n_known,).
        known_idx: OFDM symbol indices for known values, shape (n_known,).
        n_sym: Total number of OFDM symbols.

        One subcarrier's channel values at pilot-bearing OFDM symbols.

    Returns:
        Time-interpolated channel sequence, shape (n_sym,). Falls back to
        linear interpolation when fewer than three known symbols exist.
    """
    known_idx = np.asarray(known_idx, dtype=int)
    h_known = np.asarray(h_known, dtype=complex)
    if known_idx.size < 3:
        return _interp_time_linear(h_known, known_idx, n_sym)
    h_est = np.zeros(n_sym, dtype=complex)
    for m in range(n_sym):
        if m <= known_idx[1]:
            win = np.array([0, 1, 2])
        elif m >= known_idx[-2]:
            win = np.array([known_idx.size - 3, known_idx.size - 2, known_idx.size - 1])
        else:
            q = np.searchsorted(known_idx, m, side="right") - 1
            q = int(np.clip(q, 1, known_idx.size - 2))
            win = np.array([q - 1, q, q + 1])
        h_est[m] = _quadratic_window_interp(float(m), known_idx[win].astype(float), h_known[win])
    return h_est


def _interp_time_wiener(
    h_known: np.ndarray,
    known_idx: np.ndarray,
    n_sym: int,
    r_time: np.ndarray,
    noise_var: float = 0.0,
) -> np.ndarray:
    """
    Interpolate CFR values along time using Wiener/LMMSE filtering.

    Args:
        h_known: Known channel values over time, shape (n_known,).
        known_idx: OFDM symbol indices for known values, shape (n_known,).
        n_sym: Total number of OFDM symbols.
        r_time: Full time covariance matrix, shape (n_sym, n_sym).
        noise_var: Effective estimation noise variance at known positions.

        Per-subcarrier channel sequence samples and time-domain statistics.

    Returns:
        Wiener-interpolated channel sequence, shape (n_sym,).
    """
    known_idx = np.asarray(known_idx, dtype=int)
    target_idx = np.arange(n_sym, dtype=int)
    r_tk = r_time[np.ix_(target_idx, known_idx)]
    r_kk = r_time[np.ix_(known_idx, known_idx)] + noise_var * np.eye(known_idx.size, dtype=complex)
    return r_tk @ np.linalg.solve(r_kk, np.asarray(h_known, dtype=complex))


def interpolate_2d(
    h_p_grid: np.ndarray,
    q_idx: np.ndarray,
    k_idx: np.ndarray,
    n_sym: int,
    n_fft: int,
    method: str = "linear",
    l_h: int | None = None,
    r_time: np.ndarray | None = None,
    r_freq: np.ndarray | None = None,
    noise_var_eff: float = 0.0,
    pilot_power: float = 1.0,
) -> np.ndarray:
    """
    Perform separable two-dimensional time-frequency interpolation.

    Args:
        h_p_grid: Pilot CFR estimates, shape (n_pilot_sym, n_pilot_sc).
        q_idx: OFDM symbol indices containing pilots, shape (n_pilot_sym,).
        k_idx: Pilot subcarrier indices, shape (n_pilot_sc,).
        n_sym: Total number of OFDM symbols.
        n_fft: Total number of OFDM subcarriers.
        method: Interpolation method: "linear", "quadratic", "wiener", or "dft".
        l_h: Effective CIR support length, required by "dft".
        r_time: Time covariance matrix, required by "wiener".
        r_freq: Frequency covariance matrix, required by "wiener".
        noise_var_eff: Effective pilot-estimation noise variance.
        pilot_power: Average pilot-symbol power.

        Pilot-position channel estimates and optional covariance statistics.

    Returns:
        Full-grid CFR estimate, shape (n_sym, n_fft).
    """
    method = method.lower()
    q_idx = np.asarray(q_idx, dtype=int)
    k_idx = np.asarray(k_idx, dtype=int)
    freq_est = np.zeros((q_idx.size, n_fft), dtype=complex)
    for i in range(q_idx.size):
        if method == "linear":
            freq_est[i] = interpolate_linear_1d(h_p_grid[i], k_idx, n_fft)
        elif method == "quadratic":
            freq_est[i] = interpolate_quadratic_1d(h_p_grid[i], k_idx, n_fft)
        elif method == "wiener":
            if r_freq is None:
                raise ValueError("wiener interpolation requires r_freq.")
            freq_est[i] = interpolate_wiener_1d(
                h_p_grid[i], k_idx, n_fft, r_freq, noise_var_eff, pilot_power
            )
        elif method == "dft":
            if l_h is None:
                raise ValueError("dft interpolation requires l_h.")
            freq_est[i] = interpolate_dft_1d(h_p_grid[i], k_idx, n_fft, l_h)
        else:
            raise ValueError("Unsupported interpolation method.")

    h_est = np.zeros((n_sym, n_fft), dtype=complex)
    for k in range(n_fft):
        if q_idx.size == 1:
            h_est[:, k] = freq_est[0, k]
        elif method == "quadratic":
            h_est[:, k] = _interp_time_quadratic(freq_est[:, k], q_idx, n_sym)
        elif method == "wiener":
            if r_time is None:
                raise ValueError("wiener interpolation requires r_time.")
            h_est[:, k] = _interp_time_wiener(freq_est[:, k], q_idx, n_sym, r_time, noise_var_eff)
        else:
            h_est[:, k] = _interp_time_linear(freq_est[:, k], q_idx, n_sym)
    return h_est


def interpolate_channel(method: str, h_pilot: np.ndarray, q_idx: np.ndarray, k_idx: np.ndarray,
                        n_sym: int, n_fft: int, l_h: int, **kwargs) -> np.ndarray:
    """
    Public wrapper for full-grid channel interpolation.

    Args:
        method: Interpolation method name.
        h_pilot: Pilot CFR estimates, shape (n_pilot_sym, n_pilot_sc).
        q_idx: Pilot OFDM symbol indices.
        k_idx: Pilot subcarrier indices.
        n_sym: Total number of OFDM symbols.
        n_fft: Total number of OFDM subcarriers.
        l_h: Effective CIR support length for DFT interpolation.
        **kwargs: Optional covariance and noise arguments forwarded to
            interpolate_2d.

        Pilot CFR estimate and interpolation configuration.

    Returns:
        Full-grid CFR estimate, shape (n_sym, n_fft).
    """
    return interpolate_2d(h_pilot, q_idx, k_idx, n_sym, n_fft, method=method, l_h=l_h, **kwargs)


def test_linear_interpolation_shape() -> None:
    """
    Test 1D linear interpolation output shape.

    Args:
        None.

        Two synthetic pilot samples.

    Returns:
        None. Raises AssertionError on failure.
    """
    h = interpolate_linear_1d(np.array([1 + 0j, 2 + 0j]), np.array([0, 3]), 4)
    assert h.shape == (4,)


def test_dft_interpolation_shape() -> None:
    """
    Test DFT interpolation output shape.

    Args:
        None.

        Four synthetic equally spaced pilot samples.

    Returns:
        None. Raises AssertionError on failure.
    """
    h = interpolate_dft_1d(np.ones(4, dtype=complex), np.array([0, 2, 4, 6]), 8, 2)
    assert h.shape == (8,)


def test_interpolate_2d_constant_channel() -> None:
    """
    Test 2D interpolation on a constant channel.

    Args:
        None.

        Constant pilot CFR grid.

    Returns:
        None. Raises AssertionError if the interpolated full grid is not
        constant.
    """
    h_p = np.ones((2, 4), dtype=complex) * (1 + 1j)
    h = interpolate_2d(h_p, np.array([0, 2]), np.array([0, 2, 4, 6]), 4, 8, "linear")
    assert h.shape == (4, 8)
    assert np.allclose(h, 1 + 1j)


if __name__ == "__main__":
    test_linear_interpolation_shape()
    test_dft_interpolation_shape()
    test_interpolate_2d_constant_channel()
    print("interpolation.py tests passed.")
