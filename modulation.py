"""
Description:
    Digital modulation and hard-demodulation utilities for the SISO OFDM
    simulation. The module provides normalized constellations, random bit
    generation, bit-to-symbol mapping, nearest-neighbor demodulation, and local
    tests.

Args:
    - Modulation name: BPSK, QPSK, 16QAM, or 64QAM.
    - Bit arrays with shape (num_symbols, bits_per_symbol).
    - Complex symbol arrays with arbitrary leading shape.

Returns:
    - Normalized constellation symbols.
    - Random bit arrays.
    - Hard-demodulated bit arrays.
"""

from __future__ import annotations

import numpy as np


EPS = 1e-12


def get_constellation(modulation: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Return normalized constellation points and corresponding bit labels.

    Args:
    - modulation: Modulation name, one of 'BPSK', 'QPSK', '16QAM', or '64QAM'.

    - modulation: modulation name, support 'BPSK', 'QPSK', '16QAM', '64QAM'.

    Returns:
    - constellation: complex-valued constellation points, shape (M,)
    - bit_labels: bit labels corresponding to each constellation point, shape (M, bits_per_symbol)
    """
    modulation = modulation.upper()

    if modulation == 'BPSK':
        constellation = np.array([-1 + 0j, 1 + 0j], dtype=complex)
        bit_labels = np.array([[0], [1]], dtype=int)
        return constellation, bit_labels

    if modulation == 'QPSK':
        # Gray mapping: 00, 01, 11, 10
        constellation = np.array([
            (1 + 1j) / np.sqrt(2),
            (-1 + 1j) / np.sqrt(2),
            (-1 - 1j) / np.sqrt(2),
            (1 - 1j) / np.sqrt(2),
        ], dtype=complex)
        bit_labels = np.array([
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 0],
        ], dtype=int)
        return constellation, bit_labels

    def gray_pam_constellation(M_pam: int) -> tuple[np.ndarray, np.ndarray]:
        levels = np.arange(-(M_pam - 1), M_pam, 2, dtype=float)
        bits_per_dim = int(np.log2(M_pam))
        gray_order = np.array([i ^ (i >> 1) for i in range(M_pam)], dtype=int)
        bits = (((gray_order[:, None]) >> np.arange(bits_per_dim - 1, -1, -1)) & 1).astype(int)
        return levels, bits

    if modulation == '16QAM':
        levels, bits2 = gray_pam_constellation(4)
        norm = np.sqrt((2 / 3) * (16 - 1))  # sqrt(10)
        constellation = []
        bit_labels = []
        # Use [I bits, Q bits]. Q axis positive to negative for standard plotting orientation.
        for i_idx, i_bits in enumerate(bits2):
            for q_idx, q_bits in enumerate(bits2[::-1]):
                point = (levels[i_idx] + 1j * levels[::-1][q_idx]) / norm
                constellation.append(point)
                bit_labels.append(np.concatenate([i_bits, q_bits]))
        return np.array(constellation, dtype=complex), np.array(bit_labels, dtype=int)

    if modulation == '64QAM':
        levels, bits3 = gray_pam_constellation(8)
        norm = np.sqrt((2 / 3) * (64 - 1))  # sqrt(42)
        constellation = []
        bit_labels = []
        for i_idx, i_bits in enumerate(bits3):
            for q_idx, q_bits in enumerate(bits3[::-1]):
                point = (levels[i_idx] + 1j * levels[::-1][q_idx]) / norm
                constellation.append(point)
                bit_labels.append(np.concatenate([i_bits, q_bits]))
        return np.array(constellation, dtype=complex), np.array(bit_labels, dtype=int)

    raise ValueError("Unsupported modulation. Choose 'BPSK', 'QPSK', '16QAM', or '64QAM'.")


def bits_per_symbol(modulation: str) -> int:
    """
    Get number of bits per symbol.

    Args:
    - modulation: Modulation name.

    - modulation: modulation name.

    Returns:
    - bps: number of bits carried by one modulation symbol.
    """
    constellation, labels = get_constellation(modulation)
    _ = constellation
    return labels.shape[1]


def generate_bits(num_symbols: int, modulation: str, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Generate random bit sequence for modulation.

    Args:
    - num_symbols: Number of modulation symbols.
    - modulation: Modulation name.
    - rng: Optional NumPy random generator.

    - num_symbols: number of modulation symbols.
    - modulation: modulation name.
    - rng: optional numpy random generator.

    Returns:
    - bits: random bits, shape (num_symbols, bits_per_symbol)
    """
    if rng is None:
        rng = np.random.default_rng()
    bps = bits_per_symbol(modulation)
    return rng.integers(0, 2, size=(num_symbols, bps), dtype=int)


def modulate(bits: np.ndarray, modulation: str) -> np.ndarray:
    """
    Map bit sequence to constellation symbols.

    Args:
    - bits: Input bit array, shape (num_symbols, bits_per_symbol).
    - modulation: Modulation name.

    - bits: input bit array, shape (num_symbols, bits_per_symbol)
    - modulation: modulation name.

    Returns:
    - symbols: complex-valued modulation symbols, shape (num_symbols,)
    """
    constellation, labels = get_constellation(modulation)
    bits = np.asarray(bits, dtype=int)
    if bits.ndim != 2 or bits.shape[1] != labels.shape[1]:
        raise ValueError("Input bits shape must be (num_symbols, bits_per_symbol).")

    # Find exact matching bit labels.
    eq = (bits[:, None, :] == labels[None, :, :]).all(axis=2)
    indices = np.argmax(eq, axis=1)
    if not np.all(eq[np.arange(bits.shape[0]), indices]):
        raise ValueError("Invalid bit labels for the selected modulation.")
    return constellation[indices]


def demodulate(symbols: np.ndarray, modulation: str) -> np.ndarray:
    """
    Demodulate complex symbols by nearest-neighbor detection.

    Args:
    - symbols: Complex-valued input symbols, shape (...,).
    - modulation: Modulation name.

    - symbols: complex-valued input symbols, shape (...,)
    - modulation: modulation name.

    Returns:
    - bits_hat: detected bits, shape (..., bits_per_symbol)
    """
    constellation, labels = get_constellation(modulation)
    flat_symbols = np.asarray(symbols, dtype=complex).reshape(-1)
    distances = np.abs(flat_symbols[:, None] - constellation[None, :]) ** 2
    indices = np.argmin(distances, axis=1)
    bits_hat = labels[indices]
    out_shape = np.asarray(symbols).shape + (labels.shape[1],)
    return bits_hat.reshape(out_shape)


def test_bits_per_symbol() -> None:
    """
    Test bits-per-symbol lookup.

    Args:
    - None.

    - None.

    Returns:
    - None.
    """
    assert bits_per_symbol('BPSK') == 1
    assert bits_per_symbol('QPSK') == 2
    assert bits_per_symbol('16QAM') == 4
    assert bits_per_symbol('64QAM') == 6


def test_modulation_roundtrip() -> None:
    """
    Test modulation and demodulation modules.

    Args:
    - None.

    - None.

    Returns:
    - None. Raises AssertionError on round-trip failure.
    """
    rng = np.random.default_rng(0)
    for modulation_name in ['BPSK', 'QPSK', '16QAM', '64QAM']:
        bits = generate_bits(256, modulation_name, rng=rng)
        symbols = modulate(bits, modulation_name)
        bits_hat = demodulate(symbols, modulation_name)
        ber = np.mean(bits != bits_hat)
        assert ber == 0.0


def test_modulation() -> None:
    """
    Test modulation and demodulation modules.

    Args:
    - None.

    - None.

    Returns:
    - Print test results for all supported modulation schemes.
    """
    test_bits_per_symbol()
    rng = np.random.default_rng(0)
    for modulation_name in ['BPSK', 'QPSK', '16QAM', '64QAM']:
        bits = generate_bits(256, modulation_name, rng=rng)
        symbols = modulate(bits, modulation_name)
        bits_hat = demodulate(symbols, modulation_name)
        ber = np.mean(bits != bits_hat)
        print(f"[{modulation_name}] test BER = {ber:.3e}")


if __name__ == '__main__':
    test_modulation()
