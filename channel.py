"""
Description:
    SISO OFDM channel generation and correlation helpers. This module uses the
    YAML channel configuration dictionary directly and only converts key names
    when calling the legacy channel_model.py implementation.

Args:
    - Channel dictionaries loaded from config.yaml.
    - Correlation source selection, either "theory" or "mc".

Returns:
    - Frequency-domain CFR dataset Hf with shape (n_frame, n_sym, n_fft).
    - Time-domain CIR dataset Ht with shape (n_frame, n_sym, l_h).
    - Time/frequency correlation matrices used by MMSE/LMMSE/Wiener methods.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from channel_model import (
    generate_channel_dataset,
    get_or_create_correlation_stats as _get_or_create_correlation_stats,
    load_correlation_stats,
    save_correlation_stats,
)

EPS = 1e-12


def channel_config_to_legacy_dict(cfg: dict[str, Any], n_sample: int | None = None) -> dict[str, Any]:
    """
    Convert YAML channel config to the legacy channel_model dictionary format.

    Args:
        cfg: YAML channel configuration dictionary.
        n_sample: Optional override for the number of channel realizations.

        Channel section loaded from config.yaml.

    Returns:
        Dictionary with legacy keys such as N_sample, N_sy, N_sc, and
        tau_max_samples.
    """
    return {
        "N_sample": int(cfg["n_frame"] if n_sample is None else n_sample),
        "N_sy": int(cfg["n_sym"]),
        "N_sc": int(cfg["n_fft"]),
        "tau_max_samples": int(cfg["l_h"]),
        "num_path": int(cfg["num_path"]),
        "channel_type": str(cfg["channel_type"]),
        "fading_model": str(cfg["fading_model"]),
        "rician_k": float(cfg["rician_k"]),
        "f_max": float(cfg["f_max"]),
        "fs": float(cfg["fs"]),
        "tau_rms": float(cfg["tau_rms"]),
        "cp_len": int(cfg["cp_len"]),
        "normalize": bool(cfg["normalize"]),
        "seed": int(cfg["seed"]),
        "fixed_profile": bool(cfg["fixed_profile"]),
    }


def generate_channel_matrices(cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    """
    Generate SISO channel matrices and default theoretical correlations.

    Args:
        cfg: YAML channel configuration dictionary.

        Channel section specifying OFDM dimensions and channel model parameters.

    Returns:
        Dictionary containing:
        - Hf: CFR dataset, shape (n_frame, n_sym, n_fft).
        - Ht: CIR dataset, shape (n_frame, n_sym, l_h).
        - delays: path delay indices, shape (n_frame, num_path).
        - powers: path powers, shape (n_frame, num_path).
        - R_time: time covariance matrix, shape (n_sym, n_sym).
        - R_freq: frequency covariance matrix, shape (n_fft, n_fft).
    """
    dataset = generate_channel_dataset(channel_config_to_legacy_dict(cfg))
    stats, _ = get_or_create_correlation_stats(cfg, source="theory", mc_frames=4, out_dir="./output/corr_stats")
    return {
        "Hf": dataset["Hf"],
        "Ht": dataset["Ht"],
        "delays": dataset["delays"],
        "powers": dataset["powers"],
        "R_time": stats["R_time_matrix"],
        "R_freq": stats["R_freq_matrix"],
    }


def get_or_create_correlation_stats(
    cfg: dict[str, Any],
    source: str = "theory",
    mc_frames: int = 1000,
    out_dir: str | Path = "./output/corr_stats",
    force_regen: bool = False,
) -> tuple[dict[str, Any], str]:
    """
    Load or generate channel correlation statistics.

    Args:
        cfg: YAML channel configuration dictionary.
        source: Correlation source, "theory" or "mc".
        mc_frames: Number of Monte Carlo frames used when source is "mc".
        out_dir: Directory for correlation-cache .npz files.
        force_regen: If True, regenerate even when a cache file exists.

        Channel configuration and correlation-cache policy.

    Returns:
        Tuple (stats, path), where stats contains correlation matrices such as
        R_time_matrix and R_freq_matrix, and path is the cache file path.
    """
    legacy = channel_config_to_legacy_dict(cfg, n_sample=max(int(mc_frames), 4))
    return _get_or_create_correlation_stats(
        legacy,
        source=source,
        mc_frames=int(mc_frames),
        out_dir=str(out_dir),
        force_regen=force_regen,
    )


def large_scale_label(cfg: dict[str, Any]) -> str:
    """
    Build a compact channel label for output filenames.

    Args:
        cfg: YAML channel configuration dictionary.

        Large-scale channel descriptors such as channel type, fading model,
        delay support, and Doppler.

    Returns:
        String label used in result basenames.
    """
    return (
        f"{cfg['channel_type']}_{cfg['fading_model']}_"
        f"tau{float(cfg['l_h']):g}_fmax{float(cfg['f_max']):g}"
    )


def nmse(h_est: np.ndarray, h_true: np.ndarray) -> float:
    """
    Compute normalized mean-square error.

    Args:
        h_est: Estimated complex channel array.
        h_true: Reference complex channel array with the same shape.

        Estimated and true CFR/CIR values.

    Returns:
        Scalar NMSE = E[|h_est - h_true|^2] / E[|h_true|^2].
    """
    h_est = np.asarray(h_est, dtype=complex)
    h_true = np.asarray(h_true, dtype=complex)
    return float(np.mean(np.abs(h_est - h_true) ** 2) / (np.mean(np.abs(h_true) ** 2) + EPS))


def test_channel_shapes_and_power() -> None:
    """
    Test generated SISO channel shapes and power normalization.

    Args:
        None.

        Small hard-coded YAML-style channel dictionary.

    Returns:
        None. Raises AssertionError if shape or power checks fail.
    """
    cfg = {
        "n_frame": 3,
        "n_sym": 4,
        "n_fft": 16,
        "cp_len": 4,
        "l_h": 4,
        "num_path": 2,
        "channel_type": "TDL",
        "fading_model": "jakes",
        "tau_rms": 2.5,
        "rician_k": 6.0,
        "f_max": 10.0,
        "fs": 960e3,
        "fixed_profile": True,
        "normalize": True,
        "seed": 1,
    }
    data = generate_channel_matrices(cfg)
    assert data["Hf"].shape == (3, 4, 16)
    assert data["Ht"].shape == (3, 4, 4)
    assert data["R_time"].shape == (4, 4)
    assert data["R_freq"].shape == (16, 16)
    assert np.isfinite(data["Hf"]).all()
    assert 0.5 < np.mean(np.abs(data["Hf"]) ** 2) < 1.5


def test_correlation_shapes() -> None:
    """
    Test theoretical correlation matrix generation and caching.

    Args:
        None.

        Small hard-coded YAML-style channel dictionary and output/corr_stats_test cache path.

    Returns:
        None. Raises AssertionError if expected cache or matrix shapes are
        missing.
    """
    cfg = {
        "n_frame": 2,
        "n_sym": 4,
        "n_fft": 16,
        "cp_len": 4,
        "l_h": 4,
        "num_path": 2,
        "channel_type": "TDL",
        "fading_model": "jakes",
        "tau_rms": 2.5,
        "rician_k": 6.0,
        "f_max": 10.0,
        "fs": 960e3,
        "fixed_profile": True,
        "normalize": True,
        "seed": 2,
    }
    stats, path = get_or_create_correlation_stats(
        cfg, source="theory", mc_frames=4, out_dir="./output/corr_stats_test", force_regen=True
    )
    assert Path(path).exists()
    assert stats["R_time_matrix"].shape == (4, 4)
    assert stats["R_freq_matrix"].shape == (16, 16)


if __name__ == "__main__":
    test_channel_shapes_and_power()
    test_correlation_shapes()
    print("channel.py tests passed.")
