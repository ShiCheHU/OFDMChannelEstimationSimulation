"""
Description:
    SISO OFDM pilot placement and transmit-grid generation. This module builds
    block, comb, and scattered pilot masks, then creates one OFDM resource grid
    containing pilot symbols and randomly modulated data symbols.

Args:
    - OFDM grid dimensions n_sym and n_fft.
    - Pilot pattern and spacing.
    - Modulation name, pilot value, and random generator.

Returns:
    - Pilot symbol/subcarrier indices.
    - Boolean pilot/data masks.
    - Transmit resource grid and data bits.
"""

from __future__ import annotations

import numpy as np

from modulation import generate_bits, modulate


def pilot_positions(
    n_sym: int,
    n_fft: int,
    pattern: str = "comb",
    spacing_t: int = 4,
    spacing_f: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return OFDM symbol and subcarrier indices carrying pilots.

    Args:
        n_sym: Number of OFDM symbols in one frame.
        n_fft: Number of OFDM subcarriers.
        pattern: Pilot pattern, "block", "comb", or "scattered".
        spacing_t: Pilot spacing along OFDM symbols.
        spacing_f: Pilot spacing along subcarriers.

        Pilot-grid design parameters.

    Returns:
        Tuple (q_idx, k_idx), where q_idx contains pilot OFDM symbol indices
        and k_idx contains pilot subcarrier indices.
    """
    pattern = pattern.lower()
    if pattern == "block":
        q_idx = np.arange(0, n_sym, spacing_t, dtype=int)
        k_idx = np.arange(n_fft, dtype=int)
    elif pattern == "comb":
        q_idx = np.arange(n_sym, dtype=int)
        k_idx = np.arange(0, n_fft, spacing_f, dtype=int)
    elif pattern == "scattered":
        q_idx = np.arange(0, n_sym, spacing_t, dtype=int)
        k_idx = np.arange(0, n_fft, spacing_f, dtype=int)
    else:
        raise ValueError("pattern must be block, comb, or scattered.")
    return q_idx, k_idx


def pilot_mask(n_sym: int, n_fft: int, q_idx: np.ndarray, k_idx: np.ndarray) -> np.ndarray:
    """
    Build a boolean pilot mask over the OFDM resource grid.

    Args:
        n_sym: Number of OFDM symbols.
        n_fft: Number of OFDM subcarriers.
        q_idx: Pilot OFDM symbol indices.
        k_idx: Pilot subcarrier indices.

        Resource-grid size and pilot coordinate vectors.

    Returns:
        Boolean mask with shape (n_sym, n_fft), True at pilot REs.
    """
    mask = np.zeros((n_sym, n_fft), dtype=bool)
    mask[np.ix_(np.asarray(q_idx, dtype=int), np.asarray(k_idx, dtype=int))] = True
    return mask


def build_pilot_grid(
    n_sym: int,
    n_fft: int,
    pattern: str = "comb",
    spacing_t: int = 4,
    spacing_f: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build pilot coordinate vectors and the corresponding mask.

    Args:
        n_sym: Number of OFDM symbols.
        n_fft: Number of OFDM subcarriers.
        pattern: Pilot pattern, "block", "comb", or "scattered".
        spacing_t: Pilot spacing along OFDM symbols.
        spacing_f: Pilot spacing along subcarriers.

        Pilot-grid design parameters.

    Returns:
        Tuple (q_idx, k_idx, mask), where mask has shape (n_sym, n_fft).
    """
    q_idx, k_idx = pilot_positions(n_sym, n_fft, pattern, spacing_t, spacing_f)
    return q_idx, k_idx, pilot_mask(n_sym, n_fft, q_idx, k_idx)


def generate_frame_symbols(
    n_sym: int,
    n_fft: int,
    modulation: str,
    mask: np.ndarray,
    pilot_value: complex,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate one SISO OFDM transmit frame containing pilots and data.

    Args:
        n_sym: Number of OFDM symbols.
        n_fft: Number of OFDM subcarriers.
        modulation: Data modulation name, such as "QPSK" or "16QAM".
        mask: Boolean pilot mask, shape (n_sym, n_fft).
        pilot_value: Complex pilot symbol inserted at pilot REs.
        rng: NumPy random generator used for data-bit generation.

        Pilot mask and modulation parameters.

    Returns:
        Tuple (tx_grid, bits, data_mask):
        - tx_grid: Complex transmit grid, shape (n_sym, n_fft).
        - bits: Data bits on data REs, shape (n_data, bits_per_symbol).
        - data_mask: Boolean data mask, shape (n_sym, n_fft).
    """
    data_mask = ~np.asarray(mask, dtype=bool)
    n_data = int(np.sum(data_mask))
    bits = generate_bits(n_data, modulation, rng=rng)
    data_symbols = modulate(bits, modulation)
    tx_grid = np.zeros((n_sym, n_fft), dtype=complex)
    tx_grid[mask] = pilot_value
    tx_grid[data_mask] = data_symbols
    return tx_grid, bits, data_mask


def test_pilot_patterns() -> None:
    """
    Test all supported pilot pattern builders.

    Args:
        None.

        Hard-coded OFDM grid and pilot spacing.

    Returns:
        None. Raises AssertionError on invalid pilot index/mask shapes.
    """
    for pattern in ("block", "comb", "scattered"):
        q_idx, k_idx, mask = build_pilot_grid(8, 16, pattern, 2, 4)
        assert q_idx.size > 0
        assert k_idx.size > 0
        assert mask.shape == (8, 16)
        assert np.sum(mask) == q_idx.size * k_idx.size


def test_frame_symbol_generation() -> None:
    """
    Test transmit-grid generation with pilots and data.

    Args:
        None.

        Small QPSK frame with comb pilots.

    Returns:
        None. Raises AssertionError if pilot/data masks overlap or shapes are
        invalid.
    """
    rng = np.random.default_rng(0)
    _, _, mask = build_pilot_grid(4, 8, "comb", 2, 2)
    tx, bits, data_mask = generate_frame_symbols(4, 8, "QPSK", mask, 1 + 0j, rng)
    assert tx.shape == (4, 8)
    assert bits.shape[0] == int(np.sum(data_mask))
    assert np.all(tx[mask] == 1 + 0j)
    assert not np.any(mask & data_mask)


if __name__ == "__main__":
    test_pilot_patterns()
    test_frame_symbol_generation()
    print("pilot_design.py tests passed.")
