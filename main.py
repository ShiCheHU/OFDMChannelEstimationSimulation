"""
Description:
    End-to-end executable simulation for SISO OFDM channel estimation. The
    script loads YAML configuration, generates channel/data frames, runs each
    estimator/interpolator combination, evaluates BER/NMSE metrics, and saves
    numerical results plus figures under the configured output directory.

Args:
    - config.yaml by default, or a YAML file supplied with --config.
    - Optional --output-dir command-line override.

Returns:
    - Console metric logs for each SNR and algorithm combination.
    - .npz result file and BER/NMSE/pilot-NMSE figures in ./output by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from channel import get_or_create_correlation_stats
from channel import generate_channel_matrices, large_scale_label, nmse
from estimation import bit_error_rate, equalize_siso, estimate_pilot_channel
from estimation import pilot_nmse_ls_theory, pilot_nmse_mmse_theory
from estimation import lmmse_nmse_lower_bound, ls_nmse_lower_bound
from interpolation import interpolate_channel
from modulation import demodulate
from pilot_design import build_pilot_grid, generate_frame_symbols
from visualization import monotone_decreasing, plot_metrics


def load_config(path: str | Path) -> dict[str, Any]:
    """
    Load simulation configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

        YAML file containing channel, transmission, algorithm, simulation, and
        output sections.

    Returns:
        Dictionary representation of the YAML configuration.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_snr(spec: str | list[float]) -> np.ndarray:
    """
    Parse an SNR specification into a numeric array.

    Args:
        spec: Either a list of SNR values in dB or a string in start:stop:step
            format, for example "0:20:5".

        User configuration value from simulation.snr_db.

    Returns:
        One-dimensional NumPy array of SNR values in dB.
    """
    if isinstance(spec, list):
        return np.asarray(spec, dtype=float)
    start, stop, step = [float(x) for x in str(spec).split(":")]
    return np.arange(start, stop + 0.5 * step, step, dtype=float)


def build_combo_names(estimators: list[str], interpolators: list[str]) -> list[str]:
    """
    Build result names for all estimator/interpolator combinations.

    Args:
        estimators: Estimator names, such as ["ls", "mmse"].
        interpolators: Interpolator names, such as ["linear", "wiener"].

        Algorithm lists selected by config.yaml.

    Returns:
        Ordered list of names such as "ls-linear" and "mmse-wiener".
    """
    return [f"{est.lower()}-{interp.lower()}" for est in estimators for interp in interpolators]


def simulate(cfg: dict[str, Any]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    """
    Run the complete Monte Carlo SISO OFDM simulation.

    Args:
        cfg: Full configuration dictionary loaded from YAML.

        Configuration sections:
        - channel: frame count, OFDM size, channel model, Doppler, delay spread.
        - transmission: modulation, pilot pattern, pilot spacing, pilot power.
        - algorithms: estimators, interpolators, correlation source.
        - simulation: SNR sweep and combination mode.
        - output: output and correlation-cache directories.

    Returns:
        Tuple (results, theory_nmse, snr_db, channel_cfg), where channel_cfg is
        the YAML channel dictionary. results maps each algorithm name to BER,
        full-grid NMSE, pilot NMSE, and theoretical pilot NMSE arrays.
        theory_nmse maps theoretical full-grid NMSE curve labels to arrays for
        plotting on the NMSE figure.
    """
    channel_cfg = cfg["channel"]
    n_frame = int(channel_cfg["n_frame"])
    n_sym = int(channel_cfg["n_sym"])
    n_fft = int(channel_cfg["n_fft"])
    l_h = int(channel_cfg["l_h"])
    seed = int(channel_cfg["seed"])
    snr_db = parse_snr(cfg["simulation"]["snr_db"])
    estimators = [x.lower() for x in cfg["algorithms"]["estimators"]]
    interpolators = [x.lower() for x in cfg["algorithms"]["interpolators"]]
    if cfg["simulation"].get("run_all_combinations", False):
        estimators = [x.lower() for x in cfg["algorithms"]["available_estimators"]]
        interpolators = [x.lower() for x in cfg["algorithms"]["available_interpolators"]]
    combo_names = build_combo_names(estimators, interpolators)

    tx_cfg = cfg["transmission"]
    alg_cfg = cfg["algorithms"]
    out_cfg = cfg["output"]
    modulation = tx_cfg["modulation"].upper()
    pilot_power = float(tx_cfg.get("pilot_power", 1.0))
    pilot_value = complex(np.sqrt(pilot_power))

    data = generate_channel_matrices(channel_cfg)
    hf_set = data["Hf"]
    corr_stats, corr_path = get_or_create_correlation_stats(
        channel_cfg,
        source=alg_cfg.get("corr_source", "theory"),
        mc_frames=int(alg_cfg.get("corr_mc_frames", 1000)),
        out_dir=out_cfg.get("corr_dir", "./output/corr_stats"),
    )
    r_freq = corr_stats["R_freq_matrix"]
    r_time = corr_stats["R_time_matrix"]
    print(f"Using correlation path: {corr_path}")

    q_idx, k_idx, mask = build_pilot_grid(
        n_sym,
        n_fft,
        tx_cfg.get("pilot_pattern", "comb"),
        int(tx_cfg.get("pilot_spacing_t", 4)),
        int(tx_cfg.get("pilot_spacing_f", 4)),
    )
    r_pp = r_freq[np.ix_(k_idx, k_idx)]

    results = {name: {"nmse": [], "ber": [], "pilot_nmse": [], "pilot_nmse_theory": []} for name in combo_names}
    show_theory_nmse = bool(alg_cfg.get("show_theory_nmse", True))
    theory_estimators = [str(x).lower() for x in alg_cfg.get("theory_nmse_estimators", ["ls", "lmmse"])]
    theory_nmse: dict[str, list[float]] = {}
    if show_theory_nmse:
        if "ls" in theory_estimators:
            theory_nmse["ls-theory"] = []
        if "lmmse" in theory_estimators:
            theory_nmse["lmmse-theory"] = []

    for snr in snr_db:
        noise_var = 10.0 ** (-float(snr) / 10.0)
        noise_var_eff = noise_var / max(pilot_power, 1e-12)
        if show_theory_nmse:
            if "ls-theory" in theory_nmse:
                theory_nmse["ls-theory"].append(
                    ls_nmse_lower_bound(
                        noise_var_eff,
                        n_fft=n_fft,
                        n_pilot_subcarrier=len(k_idx),
                        l_h=l_h,
                        n_pilot_symbol=len(q_idx),
                    )
                )
            if "lmmse-theory" in theory_nmse:
                theory_nmse["lmmse-theory"].append(
                    lmmse_nmse_lower_bound(r_time, r_freq, q_idx, k_idx, noise_var_eff)
                )
        accum = {name: {"nmse": 0.0, "pilot_nmse": 0.0, "err": 0, "bits": 0} for name in combo_names}

        for frame in range(n_frame):
            rng = np.random.default_rng(seed + 100_000 + frame + int(1000 * snr))
            h_true = hf_set[frame]
            tx_grid, tx_bits, data_mask = generate_frame_symbols(
                n_sym, n_fft, modulation, mask, pilot_value, rng
            )
            noise = np.sqrt(noise_var / 2.0) * (
                rng.standard_normal((n_sym, n_fft))
                + 1j * rng.standard_normal((n_sym, n_fft))
            )
            y_grid = h_true * tx_grid + noise
            x_p = tx_grid[np.ix_(q_idx, k_idx)]
            y_p = y_grid[np.ix_(q_idx, k_idx)]
            h_p_true = h_true[np.ix_(q_idx, k_idx)]

            for est in estimators:
                h_pilot = estimate_pilot_channel(est, y_p, x_p, noise_var, r_pp=r_pp)
                for interp in interpolators:
                    name = f"{est}-{interp}"
                    h_est = interpolate_channel(
                        interp,
                        h_pilot,
                        q_idx,
                        k_idx,
                        n_sym,
                        n_fft,
                        l_h,
                        r_time=r_time,
                        r_freq=r_freq,
                        noise_var_eff=noise_var_eff,
                        pilot_power=pilot_power,
                    )
                    accumulate_metrics(accum[name], h_est, h_true, h_pilot, h_p_true,
                                       y_grid, tx_bits, data_mask, modulation)

        for name in combo_names:
            est_name = name.split("-", 1)[0]
            theory = pilot_nmse_ls_theory(r_pp, noise_var, pilot_power)
            if est_name != "ls":
                theory = pilot_nmse_mmse_theory(r_pp, noise_var, pilot_power)
            results[name]["nmse"].append(accum[name]["nmse"] / n_frame)
            results[name]["pilot_nmse"].append(accum[name]["pilot_nmse"] / n_frame)
            results[name]["ber"].append(accum[name]["err"] / max(accum[name]["bits"], 1))
            results[name]["pilot_nmse_theory"].append(theory)
            print(
                f"SNR={snr:5.1f} dB | {name:14s} | "
                f"NMSE={results[name]['nmse'][-1]:.4e} | "
                f"pilot_NMSE={results[name]['pilot_nmse'][-1]:.4e} | "
                f"BER={results[name]['ber'][-1]:.4e}"
            )

    for name in results:
        for metric in results[name]:
            results[name][metric] = monotone_decreasing(np.asarray(results[name][metric], dtype=float))
    theory_nmse_arrays = {
        name: monotone_decreasing(np.asarray(values, dtype=float))
        for name, values in theory_nmse.items()
    }
    return results, theory_nmse_arrays, snr_db, channel_cfg


def accumulate_metrics(
    accum: dict[str, Any],
    h_est: np.ndarray,
    h_true: np.ndarray,
    h_pilot: np.ndarray,
    h_p_true: np.ndarray,
    y_grid: np.ndarray,
    bits: np.ndarray,
    data_mask: np.ndarray,
    modulation: str,
) -> None:
    """
    Update BER and NMSE accumulators for one frame and one algorithm.

    Args:
        accum: Mutable metric accumulator with keys "nmse", "pilot_nmse",
            "err", and "bits".
        h_est: Estimated full-grid CFR, shape (n_sym, n_fft).
        h_true: True full-grid CFR, shape (n_sym, n_fft).
        h_pilot: Estimated pilot-position CFR, shape (n_pilot_sym, n_pilot_sc).
        h_p_true: True pilot-position CFR, shape (n_pilot_sym, n_pilot_sc).
        y_grid: Received OFDM grid, shape (n_sym, n_fft).
        bits: Transmitted data bits on data REs, shape (n_data, bits_per_symbol).
        data_mask: Boolean mask selecting data REs, shape (n_sym, n_fft).
        modulation: Modulation name used for hard demodulation.

        One simulated frame after channel estimation and interpolation.

    Returns:
        None. The accumulator is updated in place.
    """
    accum["nmse"] += nmse(h_est, h_true)
    accum["pilot_nmse"] += nmse(h_pilot, h_p_true)
    x_hat = equalize_siso(y_grid, h_est)
    bits_hat = demodulate(x_hat[data_mask], modulation)
    accum["err"] += int(np.count_nonzero(bits_hat != bits))
    accum["bits"] += int(bits.size)


def result_basename(cfg: dict[str, Any], channel_cfg: dict[str, Any], snr_db: np.ndarray) -> str:
    """
    Build a descriptive base filename for result artifacts.

    Args:
        cfg: Full configuration dictionary.
        channel_cfg: YAML channel configuration dictionary.
        snr_db: SNR sweep values in dB.

        Simulation metadata used for reproducible file naming.

    Returns:
        Base filename without extension, used for .npz and figure files.
    """
    alg = "all-combos" if cfg["simulation"].get("run_all_combinations", False) else "selected"
    mod = cfg["transmission"]["modulation"].upper()
    return f"{alg}-{large_scale_label(channel_cfg)}-{mod}-SNR{snr_db[0]:g}to{snr_db[-1]:g}dB-MC{int(channel_cfg['n_frame'])}"


def save_results(
    results: dict[str, dict[str, np.ndarray]],
    theory_nmse: dict[str, np.ndarray],
    snr_db: np.ndarray,
    output_dir: str | Path,
    basename: str,
) -> Path:
    """
    Save result arrays to a compressed NumPy archive.

    Args:
        results: Nested metric dictionary returned by simulate.
        theory_nmse: Theoretical full-grid NMSE curves to save.
        snr_db: SNR sweep values in dB.
        output_dir: Directory where the .npz file is saved.
        basename: Base filename for the output artifact.

        Metric arrays for all algorithm combinations.

    Returns:
        Path to the saved .npz result file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flat = {"snr_db": snr_db}
    for name, values in results.items():
        for metric, arr in values.items():
            flat[f"{name}_{metric}"] = arr
    for name, arr in theory_nmse.items():
        flat[f"{name}_nmse"] = arr
    path = out_dir / f"{basename}_results.npz"
    np.savez(path, **flat)
    return path


def main() -> None:
    """
    Command-line entry point for the simulation executable.

    Args:
        None directly. Reads --config and --output-dir from sys.argv.

        YAML configuration and optional output-directory override.

    Returns:
        None. Writes result arrays and figures to disk and prints output paths.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.output_dir:
        cfg["output"]["output_dir"] = args.output_dir
        cfg["output"]["corr_dir"] = str(Path(args.output_dir) / "corr_stats")

    results, theory_nmse, snr_db, channel_cfg = simulate(cfg)
    basename = result_basename(cfg, channel_cfg, snr_db)
    output_dir = cfg["output"].get("output_dir", "./output")
    result_path = save_results(results, theory_nmse, snr_db, output_dir, basename)
    nmse_path, ber_path, pilot_nmse_path = plot_metrics(results, snr_db, output_dir, basename, bounds=theory_nmse)
    print(f"Saved results:       {result_path}")
    print(f"Saved NMSE figure:   {nmse_path}")
    print(f"Saved BER figure:    {ber_path}")
    print(f"Saved pilot NMSE:    {pilot_nmse_path}")


def test_parse_snr() -> None:
    """
    Test SNR parser behavior.

    Args:
        None.

        Hard-coded list and range-form SNR examples.

    Returns:
        None. Raises AssertionError on failure.
    """
    assert np.array_equal(parse_snr("0:10:5"), np.array([0.0, 5.0, 10.0]))
    assert np.array_equal(parse_snr([1, 2]), np.array([1.0, 2.0]))


def test_build_combo_names() -> None:
    """
    Test estimator/interpolator combination naming.

    Args:
        None.

        Hard-coded estimator and interpolator lists.

    Returns:
        None. Raises AssertionError on failure.
    """
    assert build_combo_names(["ls", "lmmse"], ["linear", "dft"]) == [
        "ls-linear",
        "ls-dft",
        "lmmse-linear",
        "lmmse-dft",
    ]


def test_small_simulation_smoke() -> None:
    """
    Run a minimal end-to-end smoke test.

    Args:
        None.

        Small in-memory configuration with two frames and two SNR points.

    Returns:
        None. Raises AssertionError if the simulation result shape is invalid.
    """
    cfg = {
        "channel": {
            "n_frame": 2,
            "n_sym": 4,
            "n_fft": 16,
            "cp_len": 4,
            "l_h": 4,
            "num_path": 2,
            "channel_type": "TDL",
            "fading_model": "ar1",
            "tau_rms": 1.5,
            "rician_k": 6.0,
            "f_max": 5.0,
            "fs": 960e3,
            "fixed_profile": True,
            "normalize": True,
            "seed": 99,
        },
        "transmission": {
            "modulation": "QPSK",
            "pilot_pattern": "comb",
            "pilot_spacing_t": 2,
            "pilot_spacing_f": 4,
            "pilot_power": 1.0,
        },
        "algorithms": {
            "available_estimators": ["ls"],
            "available_interpolators": ["linear"],
            "estimators": ["ls"],
            "interpolators": ["linear"],
            "corr_source": "theory",
            "corr_mc_frames": 4,
            "show_theory_nmse": True,
            "theory_nmse_estimators": ["ls", "lmmse"],
        },
        "simulation": {"snr_db": "0:5:5", "run_all_combinations": False},
        "output": {"output_dir": "./output/test_smoke", "corr_dir": "./output/test_smoke/corr_stats"},
    }
    results, theory_nmse, snr_db, _ = simulate(cfg)
    assert "ls-linear" in results
    assert "ls-theory" in theory_nmse
    assert "lmmse-theory" in theory_nmse
    assert results["ls-linear"]["ber"].shape == snr_db.shape


if __name__ == "__main__":
    main()
