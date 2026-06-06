"""
Description:
    Plotting helpers for SISO OFDM channel-estimation metrics. This module
    saves BER, full-grid NMSE, and pilot-domain NMSE figures for all configured
    estimator/interpolator combinations.

Args:
    - Result dictionaries and optional theoretical NMSE curves from main.simulate.
    - SNR sweep values.
    - Output directory and filename base.

Returns:
    - PNG figures saved under the configured output directory. The full-grid
      NMSE figure can include theoretical LS/LMMSE estimation-error curves.
    - Optional monotonic-envelope arrays for noisy Monte Carlo curves.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def curve_style(name: str) -> dict[str, object]:
    """
    Choose a stable Matplotlib style from an algorithm name.

    Args:
        name: Algorithm label such as "ls-linear" or "mmse-wiener".

        Estimator/interpolator result name.

    Returns:
        Dictionary of Matplotlib keyword arguments for marker and line style.
    """
    key = name.lower()
    style: dict[str, object] = {"marker": "o", "linestyle": "-", "markersize": 5}
    if key.startswith("mmse"):
        style.update({"marker": "s", "markerfacecolor": "none", "markeredgewidth": 1.4})
    elif key.startswith("lmmse"):
        style.update({"marker": "^", "markerfacecolor": "none", "markeredgewidth": 1.4})
    if key.endswith("-wiener"):
        style["linestyle"] = "--"
    elif key.endswith("-linear"):
        style["linestyle"] = "-."
    elif key.endswith("-quadratic"):
        style["linestyle"] = ":"
    return style


def monotone_decreasing(values: np.ndarray) -> np.ndarray:
    """
    Return the non-increasing envelope of a metric curve.

    Args:
        values: One-dimensional metric values ordered by increasing SNR.

        BER or NMSE curve that may contain Monte Carlo fluctuations.

    Returns:
        Array where each element is no greater than all previous elements.
    """
    return np.minimum.accumulate(np.asarray(values, dtype=float))


def plot_metrics(
    results: dict[str, dict[str, np.ndarray]],
    snr_db: np.ndarray,
    output_dir: str | Path,
    basename: str,
    bounds: dict[str, np.ndarray] | None = None,
) -> tuple[Path, Path, Path]:
    """
    Plot BER, full-grid NMSE, and pilot-domain NMSE curves.

    Args:
        results: Mapping from algorithm name to metric arrays. Each value may
            contain "ber", "nmse", "pilot_nmse", and "pilot_nmse_theory".
        snr_db: SNR sweep values in dB, shape (n_snr,).
        output_dir: Directory where figures are saved.
        basename: Base filename for generated PNG files.
        bounds: Optional theoretical full-grid NMSE curves, such as LS and
            LMMSE estimation-error bounds. These are shown on the NMSE figure.

        Simulation metrics produced by main.simulate.

    Returns:
        Tuple (nmse_path, ber_path, pilot_nmse_path) containing paths to the
        saved figure files.
    """
    import matplotlib.pyplot as plt

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nmse_path = out_dir / f"{basename}_nmse.png"
    ber_path = out_dir / f"{basename}_ber.png"
    pilot_nmse_path = out_dir / f"{basename}_pilot_nmse.png"

    plt.figure(figsize=(8, 5))
    for name, res in results.items():
        plt.semilogy(snr_db, np.maximum(res["nmse"], 1e-12), label=name, **curve_style(name))
    if bounds:
        for name, values in bounds.items():
            plt.semilogy(snr_db, np.maximum(values, 1e-12), linestyle=":", linewidth=2.0, label=name)
    plt.grid(True, which="both", alpha=0.3)
    plt.xlabel("SNR (dB)")
    plt.ylabel("NMSE")
    plt.title("SISO OFDM Channel Estimation NMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(nmse_path, dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    for name, res in results.items():
        plt.semilogy(snr_db, np.maximum(res["ber"], 1e-8), label=name, **curve_style(name))
    plt.grid(True, which="both", alpha=0.3)
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("SISO OFDM BER After Channel Estimation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ber_path, dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    for name, res in results.items():
        plt.semilogy(snr_db, np.maximum(res["pilot_nmse"], 1e-12), label=f"{name} sim", **curve_style(name))
        if "pilot_nmse_theory" in res:
            plt.semilogy(snr_db, np.maximum(res["pilot_nmse_theory"], 1e-12), linestyle=":", linewidth=2.0,
                         label=f"{name} theory")
    plt.grid(True, which="both", alpha=0.3)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Pilot-domain NMSE")
    plt.title("SISO OFDM Pilot-domain NMSE")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(pilot_nmse_path, dpi=160, bbox_inches="tight")
    plt.close()
    return nmse_path, ber_path, pilot_nmse_path


def test_monotone_decreasing() -> None:
    """
    Test monotone envelope generation.

    Args:
        None.

        Hard-coded metric curve with one upward fluctuation.

    Returns:
        None. Raises AssertionError on failure.
    """
    out = monotone_decreasing(np.array([3.0, 2.0, 2.5, 1.0]))
    assert np.array_equal(out, np.array([3.0, 2.0, 2.0, 1.0]))


def test_curve_style() -> None:
    """
    Test style selection for representative algorithm names.

    Args:
        None.

        Hard-coded algorithm labels.

    Returns:
        None. Raises AssertionError on failure.
    """
    assert "marker" in curve_style("ls-linear")
    assert curve_style("mmse-wiener")["linestyle"] == "--"


if __name__ == "__main__":
    test_monotone_decreasing()
    test_curve_style()
    print("visualization.py tests passed.")
