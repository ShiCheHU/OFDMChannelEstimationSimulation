
"""
SISO OFDM channel estimation simulation.
Support functions:
- build pilot patterns for block-type / comb-type / scattered-type pilots
- generate transmit grid and received grid
- estimate channel from pilots and interpolate to data positions
- evaluate BER / pilot-domain NMSE / full-grid NMSE over SNR
- compare analytic and Monte Carlo covariance matrices for MMSE/LMMSE/Wiener methods
- plot and save comparison figures

Dependency:
- modulation.py
- channelestimation.py
- channel_model.py

Note:
- channel_model.py should be placed in the same directory as this file,
  or be available in the Python import path.
- This simulation uses the standard per-subcarrier frequency-domain OFDM model
  Y[m,k] = H[m,k] X[m,k] + W[m,k], assuming CP is sufficient and ICI is negligible.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from modulation import generate_bits, modulate, demodulate
from channelestimation import estimate_channel_from_pilots
from channel_model import generate_channel_dataset, get_or_create_correlation_stats

EPS = 1e-12


@dataclass
class SimulationConfig:
    """
    Simulation configuration.

    Input:
    - n_frame: number of independent channel realizations.
    - n_sym: number of OFDM symbols in one frame.
    - n_fft: number of subcarriers.
    - cp_len: cyclic prefix length in samples.
    - modulation: modulation type, e.g., 'QPSK', '16QAM', '64QAM'.
    - channel_type: 'Rayleigh', 'Rician', or 'TDL'.
    - fading_model: 'jakes' or 'ar1'.
    - tau_max_samples: maximum delay spread in samples.
    - num_path: number of channel paths.
    - tau_rms: RMS delay spread used by TDL.
    - rician_k: Rician K factor.
    - f_max: maximum Doppler in Hz.
    - fs: sampling rate in Hz.
    - fixed_profile: whether all frames use the same delay/PDP profile.
    - pilot_pattern: 'block', 'comb', or 'scattered'.
    - pilot_spacing_f: pilot spacing in frequency for comb/scattered pilots.
    - pilot_spacing_t: pilot spacing in time for block/scattered pilots.
    - est_method: 'LS', 'MMSE', or 'LMMSE'.
    - interp_method: 'linear', 'quadratic', 'wiener', or 'dft'.
    - corr_source: correlation source for MMSE/LMMSE/Wiener, 'theory' or 'mc'.
    - corr_mc_frames: MC frame count used to estimate correlation matrices when corr_source='mc'.
    - corr_dir: directory used to save/load correlation matrices.
    - corr_path: optional explicit correlation-matrix path. If None, it is created from corr_source/corr_mc_frames.
    - pilot_value: complex pilot symbol value.
    - seed: random seed.
    """
    n_frame: int = 200
    n_sym: int = 14
    n_fft: int = 64
    cp_len: int = 8
    modulation: str = 'QPSK'
    channel_type: str = 'TDL'
    fading_model: str = 'jakes'
    tau_max_samples: int = 8
    num_path: int = 4
    tau_rms: float = 3.0
    rician_k: float = 6.0
    f_max: float = 20.0
    fs: float = 960e3
    fixed_profile: bool = True
    pilot_pattern: str = 'comb'
    pilot_spacing_f: int = 4
    pilot_spacing_t: int = 4
    est_method: str = 'LS'
    interp_method: str = 'linear'
    corr_source: str = 'theory'
    corr_mc_frames: int = 1000
    corr_dir: str = './corr_stats'
    corr_path: str | None = None
    pilot_value: complex = 1.0 + 0.0j
    seed: int = 7


def bit_error_rate(tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
    """
    Compute BER between transmitted and detected bits.

    Input:
    - tx_bits: transmitted bits, shape (..., bits_per_symbol)
    - rx_bits: detected bits, shape (..., bits_per_symbol)

    Output:
    - ber: bit error rate.
    """
    tx_bits = np.asarray(tx_bits, dtype=int)
    rx_bits = np.asarray(rx_bits, dtype=int)
    if tx_bits.shape != rx_bits.shape:
        raise ValueError(f"Bit array shape mismatch: {tx_bits.shape} vs {rx_bits.shape}")
    return float(np.mean(tx_bits != rx_bits))


def nmse(h_hat: np.ndarray, h_true: np.ndarray) -> float:
    """
    Compute normalized mean-square error.

    Input:
    - h_hat: estimated channel coefficients, shape (...)
    - h_true: true channel coefficients, shape (...)

    Output:
    - nmse_value: normalized mean-square error.
    """
    h_hat = np.asarray(h_hat, dtype=complex)
    h_true = np.asarray(h_true, dtype=complex)
    return float(np.mean(np.abs(h_hat - h_true) ** 2) / (np.mean(np.abs(h_true) ** 2) + EPS))


def build_pilot_grid(n_sym: int,
                     n_fft: int,
                     pilot_pattern: str,
                     pilot_spacing_f: int = 4,
                     pilot_spacing_t: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build pilot positions and pilot mask for a 2D OFDM resource grid.

    Input:
    - n_sym: number of OFDM symbols.
    - n_fft: number of subcarriers.
    - pilot_pattern: 'block', 'comb', or 'scattered'.
    - pilot_spacing_f: pilot spacing in frequency.
    - pilot_spacing_t: pilot spacing in time.

    Output:
    - pilot_symbol_idx: pilot symbol indices, shape (Nt_p,)
    - pilot_subcarrier_idx: pilot subcarrier indices, shape (Nf_p,)
    - pilot_mask: boolean mask on the whole grid, shape (n_sym, n_fft)
    """
    pilot_pattern = pilot_pattern.lower()

    if pilot_pattern == 'block':
        pilot_symbol_idx = np.arange(0, n_sym, pilot_spacing_t, dtype=int)
        pilot_subcarrier_idx = np.arange(n_fft, dtype=int)
    elif pilot_pattern == 'comb':
        pilot_symbol_idx = np.arange(n_sym, dtype=int)
        pilot_subcarrier_idx = np.arange(0, n_fft, pilot_spacing_f, dtype=int)
    elif pilot_pattern == 'scattered':
        pilot_symbol_idx = np.arange(0, n_sym, pilot_spacing_t, dtype=int)
        pilot_subcarrier_idx = np.arange(0, n_fft, pilot_spacing_f, dtype=int)
    else:
        raise ValueError("pilot_pattern must be 'block', 'comb', or 'scattered'.")

    pilot_mask = np.zeros((n_sym, n_fft), dtype=bool)
    pilot_mask[np.ix_(pilot_symbol_idx, pilot_subcarrier_idx)] = True
    return pilot_symbol_idx, pilot_subcarrier_idx, pilot_mask


def generate_frame_symbols(n_sym: int,
                           n_fft: int,
                           modulation_name: str,
                           pilot_mask: np.ndarray,
                           pilot_value: complex,
                           rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate one OFDM frame including data and pilots.

    Input:
    - n_sym: number of OFDM symbols.
    - n_fft: number of subcarriers.
    - modulation_name: modulation type for data symbols.
    - pilot_mask: pilot mask, shape (n_sym, n_fft).
    - pilot_value: pilot symbol value.
    - rng: numpy random generator.

    Output:
    - tx_grid: transmit resource grid, shape (n_sym, n_fft)
    - data_bits: transmitted data bits on data positions, shape (N_data, bits_per_symbol)
    - data_mask: boolean data mask, shape (n_sym, n_fft)
    """
    data_mask = ~pilot_mask
    n_data = int(np.sum(data_mask))
    data_bits = generate_bits(n_data, modulation_name, rng=rng)
    data_symbols = modulate(data_bits, modulation_name)

    tx_grid = np.zeros((n_sym, n_fft), dtype=complex)
    tx_grid[pilot_mask] = pilot_value
    tx_grid[data_mask] = data_symbols
    return tx_grid, data_bits, data_mask


def _config_to_channel_dict(cfg: SimulationConfig, n_sample: int) -> Dict:
    """
    Convert SimulationConfig to channel-model dictionary.

    Input:
    - cfg: simulation configuration.
    - n_sample: number of frames/realizations.

    Output:
    - config: channel-model configuration dictionary.
    """
    return {
        'N_sample': int(n_sample),
        'N_sy': cfg.n_sym,
        'N_sc': cfg.n_fft,
        'tau_max_samples': cfg.tau_max_samples,
        'num_path': cfg.num_path,
        'channel_type': cfg.channel_type,
        'fading_model': cfg.fading_model,
        'rician_k': cfg.rician_k,
        'f_max': cfg.f_max,
        'fs': cfg.fs,
        'tau_rms': cfg.tau_rms,
        'cp_len': cfg.cp_len,
        'normalize': True,
        'seed': cfg.seed,
        'fixed_profile': cfg.fixed_profile,
    }


def _resolve_corr_stats(cfg: SimulationConfig) -> Tuple[Dict | None, str | None]:
    """
    Resolve correlation statistics and correlation-matrix path for MMSE/LMMSE/Wiener methods.

    Input:
    - cfg: simulation configuration.

    Output:
    - corr_stats: loaded/generated correlation statistics dictionary, or None for LS+non-Wiener cases.
    - corr_path: saved correlation-matrix file path, or None.
    """
    need_corr = (cfg.est_method.upper() in {'MMSE', 'LMMSE'}) or (cfg.interp_method.lower() == 'wiener')
    if not need_corr:
        return None, None

    if cfg.corr_path is not None and os.path.exists(cfg.corr_path):
        from channel_model import load_correlation_stats
        return load_correlation_stats(cfg.corr_path), cfg.corr_path

    channel_cfg = _config_to_channel_dict(cfg, n_sample=max(cfg.corr_mc_frames, 4))
    corr_stats, corr_path = get_or_create_correlation_stats(
        channel_cfg,
        source=cfg.corr_source,
        mc_frames=cfg.corr_mc_frames,
        out_dir=cfg.corr_dir,
        force_regen=False,
    )
    return corr_stats, corr_path


def pilot_theoretical_nmse_ls(r_pp: np.ndarray, noise_var: float, pilot_power: float) -> float:
    """
    Compute the theoretical pilot-domain NMSE of LS estimation.

    Input:
    - r_pp: pilot-position channel covariance matrix, shape (Np, Np).
    - noise_var: AWGN variance.
    - pilot_power: average pilot-symbol power.

    Output:
    - nmse_theory: theoretical pilot-domain NMSE.
    """
    c_ls = (noise_var / (pilot_power + EPS)) * np.eye(r_pp.shape[0], dtype=complex)
    return float(np.real(np.trace(c_ls)) / (np.real(np.trace(r_pp)) + EPS))


def pilot_theoretical_nmse_mmse(r_pp: np.ndarray, noise_var: float, pilot_power: float) -> float:
    """
    Compute the theoretical pilot-domain NMSE of MMSE/LMMSE estimation.

    Input:
    - r_pp: pilot-position channel covariance matrix, shape (Np, Np).
    - noise_var: AWGN variance.
    - pilot_power: average pilot-symbol power.

    Output:
    - nmse_theory: theoretical pilot-domain NMSE.
    """
    eta = noise_var / (pilot_power + EPS)
    c_mmse = r_pp - r_pp @ np.linalg.solve(r_pp + eta * np.eye(r_pp.shape[0], dtype=complex), r_pp)
    return float(np.real(np.trace(c_mmse)) / (np.real(np.trace(r_pp)) + EPS))


def simulate_one_snr(cfg: SimulationConfig, snr_db: float) -> Dict:
    """
    Run one SNR-point simulation.

    Input:
    - cfg: simulation configuration.
    - snr_db: SNR in dB.

    Output:
    - metrics: dict containing BER, full-grid NMSE, pilot-domain NMSE, and theory NMSE.
    """
    rng = np.random.default_rng(cfg.seed + int(100 * (snr_db + 100)))
    pilot_symbol_idx, pilot_subcarrier_idx, pilot_mask = build_pilot_grid(
        cfg.n_sym, cfg.n_fft, cfg.pilot_pattern, cfg.pilot_spacing_f, cfg.pilot_spacing_t
    )
    data_mask = ~pilot_mask

    dataset = generate_channel_dataset(_config_to_channel_dict(cfg, n_sample=cfg.n_frame))
    Hf_set = dataset['Hf']

    corr_stats, corr_path = _resolve_corr_stats(cfg)
    if corr_stats is None:
        # LS theory still needs a channel covariance reference; use theory covariance.
        theory_stats, _ = get_or_create_correlation_stats(
            _config_to_channel_dict(cfg, n_sample=max(cfg.corr_mc_frames, 4)),
            source='theory',
            mc_frames=cfg.corr_mc_frames,
            out_dir=cfg.corr_dir,
            force_regen=False,
        )
        r_freq_full_theory = theory_stats['R_freq_matrix']
    else:
        r_freq_full_theory = corr_stats['R_freq_matrix']

    noise_var = 10.0 ** (-snr_db / 10.0)
    pilot_power = float(np.abs(cfg.pilot_value) ** 2)

    ber_acc = 0.0
    nmse_acc = 0.0
    pilot_nmse_acc = 0.0

    for i in range(cfg.n_frame):
        H_true = Hf_set[i]
        tx_grid, tx_bits_data, data_mask = generate_frame_symbols(
            cfg.n_sym, cfg.n_fft, cfg.modulation, pilot_mask, cfg.pilot_value, rng
        )

        noise = np.sqrt(noise_var / 2.0) * (
            rng.standard_normal((cfg.n_sym, cfg.n_fft)) + 1j * rng.standard_normal((cfg.n_sym, cfg.n_fft))
        )
        y_grid = H_true * tx_grid + noise

        x_p_grid = tx_grid[np.ix_(pilot_symbol_idx, pilot_subcarrier_idx)]
        y_p_grid = y_grid[np.ix_(pilot_symbol_idx, pilot_subcarrier_idx)]
        h_p_true = H_true[np.ix_(pilot_symbol_idx, pilot_subcarrier_idx)]

        h_p_hat, h_hat = estimate_channel_from_pilots(
            y_p_grid=y_p_grid,
            x_p_grid=x_p_grid,
            pilot_symbol_idx=pilot_symbol_idx,
            pilot_subcarrier_idx=pilot_subcarrier_idx,
            n_sym=cfg.n_sym,
            n_fft=cfg.n_fft,
            est_method=cfg.est_method,
            interp_method=cfg.interp_method,
            corr_stats=corr_stats,
            corr_path=corr_path,
            noise_var=noise_var,
            cp_len=cfg.cp_len,
        )

        pilot_nmse_acc += nmse(h_p_hat, h_p_true)
        nmse_acc += nmse(h_hat, H_true)

        x_hat = y_grid / (h_hat + EPS)
        rx_bits_data = demodulate(x_hat[data_mask], cfg.modulation)
        ber_acc += bit_error_rate(tx_bits_data, rx_bits_data)

    r_pp_theory = r_freq_full_theory[np.ix_(pilot_subcarrier_idx, pilot_subcarrier_idx)]
    pilot_nmse_ls_theory = pilot_theoretical_nmse_ls(r_pp_theory, noise_var, pilot_power)
    if cfg.est_method.upper() == 'LS':
        pilot_nmse_theory = pilot_nmse_ls_theory
    else:
        pilot_nmse_theory = pilot_theoretical_nmse_mmse(r_pp_theory, noise_var, pilot_power)

    return {
        'ber': ber_acc / cfg.n_frame,
        'nmse': nmse_acc / cfg.n_frame,
        'pilot_nmse': pilot_nmse_acc / cfg.n_frame,
        'pilot_nmse_theory': pilot_nmse_theory,
        'pilot_nmse_ls_theory': pilot_nmse_ls_theory,
        'corr_path': corr_path or '',
    }


def run_snr_sweep(cfg: SimulationConfig, snr_db_list: np.ndarray) -> Dict:
    """
    Run SNR sweep for one pilot pattern / estimation / interpolation setting.

    Input:
    - cfg: simulation configuration.
    - snr_db_list: SNR list, shape (N_snr,)

    Output:
    - result: dict containing arrays of BER, full-grid NMSE, pilot-domain NMSE, and theory NMSE.
    """
    ber_list: List[float] = []
    nmse_list: List[float] = []
    pilot_nmse_list: List[float] = []
    pilot_nmse_theory_list: List[float] = []
    pilot_nmse_ls_theory_list: List[float] = []

    corr_stats, corr_path = _resolve_corr_stats(cfg)
    if corr_path is not None:
        print(f'Using covariance path: {corr_path}')

    print('-' * 100)
    print(
        f"pattern={cfg.pilot_pattern:9s} | est={cfg.est_method:6s} | interp={cfg.interp_method:9s} | "
        f"corr={cfg.corr_source:6s} | mod={cfg.modulation:6s} | channel={cfg.channel_type:7s} | fading={cfg.fading_model}"
    )

    for snr_db in snr_db_list:
        metrics = simulate_one_snr(cfg, float(snr_db))
        ber_list.append(metrics['ber'])
        nmse_list.append(metrics['nmse'])
        pilot_nmse_list.append(metrics['pilot_nmse'])
        pilot_nmse_theory_list.append(metrics['pilot_nmse_theory'])
        pilot_nmse_ls_theory_list.append(metrics['pilot_nmse_ls_theory'])
        print(
            f"SNR={snr_db:>5.1f} dB | BER={metrics['ber']:.4e} | "
            f"NMSE={metrics['nmse']:.4e} | pilot_NMSE={metrics['pilot_nmse']:.4e} | "
            f"theory={metrics['pilot_nmse_theory']:.4e}"
        )

    return {
        'snr_db': np.asarray(snr_db_list, dtype=float),
        'ber': np.asarray(ber_list, dtype=float),
        'nmse': np.asarray(nmse_list, dtype=float),
        'pilot_nmse': np.asarray(pilot_nmse_list, dtype=float),
        'pilot_nmse_theory': np.asarray(pilot_nmse_theory_list, dtype=float),
        'pilot_nmse_ls_theory': np.asarray(pilot_nmse_ls_theory_list, dtype=float),
        'corr_path': corr_path or '',
    }


def plot_results(results: Dict[str, Dict[str, np.ndarray]],
                 out_prefix: str = 'siso_ofdm_channel_estimation') -> Tuple[str, str, str]:
    """
    Plot BER, full-grid NMSE, and pilot-domain NMSE with theory curves.

    Input:
    - results: dict of simulation results.
    - out_prefix: output file prefix.

    Output:
    - ber_path: BER figure path.
    - nmse_path: full-grid NMSE figure path.
    - pilot_nmse_path: pilot-domain NMSE figure path.
    """
    ber_path = f'{out_prefix}_ber.png'
    nmse_path = f'{out_prefix}_nmse.png'
    pilot_nmse_path = f'{out_prefix}_pilot_nmse.png'

    plt.figure(figsize=(10, 6))
    for label, val in results.items():
        plt.semilogy(val['snr_db'], val['ber'], marker='o', linewidth=2, label=label)
    plt.grid(True, which='both', alpha=0.3)
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.title('SISO OFDM Channel Estimation: BER')
    plt.legend()
    plt.tight_layout()
    plt.savefig(ber_path, dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    for label, val in results.items():
        plt.semilogy(val['snr_db'], val['nmse'], marker='s', linewidth=2, label=label)
    plt.grid(True, which='both', alpha=0.3)
    plt.xlabel('SNR (dB)')
    plt.ylabel('NMSE')
    plt.title('SISO OFDM Channel Estimation: Full-grid NMSE')
    plt.legend()
    plt.tight_layout()
    plt.savefig(nmse_path, dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    for label, val in results.items():
        plt.semilogy(val['snr_db'], val['pilot_nmse'], marker='o', linewidth=2, label=f'{label} (sim)')
        plt.semilogy(val['snr_db'], val['pilot_nmse_theory'], linestyle='--', linewidth=2, label=f'{label} (theory)')
    plt.grid(True, which='both', alpha=0.3)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Pilot-domain NMSE')
    plt.title('SISO OFDM Channel Estimation: Pilot-domain NMSE (Simulation vs Theory)')
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(pilot_nmse_path, dpi=150, bbox_inches='tight')
    plt.close()

    return ber_path, nmse_path, pilot_nmse_path


def demo_compare() -> Dict[str, Dict[str, np.ndarray]]:
    """
    Demo simulation comparing covariance sources and theoretical NMSE.

    Input:
    - None.

    Output:
    - results: dict of simulation results for all compared settings.
    """
    snr_db_list = np.arange(0, 21, 5)

    base = SimulationConfig(
        n_frame=1000,
        n_sym=14,
        n_fft=64,
        cp_len=8,
        modulation='QPSK',
        channel_type='TDL',
        fading_model='jakes',
        tau_max_samples=8,
        num_path=4,
        tau_rms=2.5,
        f_max=10.0,
        fs=960e3,
        fixed_profile=True,
        seed=2026,
        corr_dir='./corr_stats',
        corr_mc_frames=1000,
    )
    
    # compute velocity and delay parameters for annotation
    fc = 2.4e9  # carrier frequency
    f_sc = 15e3  # subcarrier spacing
    lda = 3e8 / fc  # wavelength
    vs = base.f_max * lda  # mobile speed in m/s
    max_velocity = vs * 3.6  # convert to km/h

    T_sym = 1/f_sc  # OFDM symbol duration without CP
    max_delay = base.tau_max_samples * T_sym/base.n_fft * 1e6  # maximum delay in us

    compare_list = [
        ('LS-linear',            SimulationConfig(**{**base.__dict__, 'pilot_pattern': 'comb', 'pilot_spacing_f': 4, 'est_method': 'LS',    'interp_method': 'linear'})),
        ('LS-wiener[theory]',    SimulationConfig(**{**base.__dict__, 'pilot_pattern': 'comb', 'pilot_spacing_f': 4, 'est_method': 'LS',    'interp_method': 'wiener', 'corr_source': 'theory'})),
        ('MMSE-wiener[theory]',  SimulationConfig(**{**base.__dict__, 'pilot_pattern': 'comb', 'pilot_spacing_f': 4, 'est_method': 'MMSE',  'interp_method': 'wiener', 'corr_source': 'theory'})),
        ('MMSE-wiener[mc1000]',  SimulationConfig(**{**base.__dict__, 'pilot_pattern': 'comb', 'pilot_spacing_f': 4, 'est_method': 'MMSE',  'interp_method': 'wiener', 'corr_source': 'mc'})),
        ('LMMSE-wiener[theory]', SimulationConfig(**{**base.__dict__, 'pilot_pattern': 'comb', 'pilot_spacing_f': 4, 'est_method': 'LMMSE', 'interp_method': 'wiener', 'corr_source': 'theory'})),
        ('LMMSE-wiener[mc1000]', SimulationConfig(**{**base.__dict__, 'pilot_pattern': 'comb', 'pilot_spacing_f': 4, 'est_method': 'LMMSE', 'interp_method': 'wiener', 'corr_source': 'mc'})),
        ('LS-dft',               SimulationConfig(**{**base.__dict__, 'pilot_pattern': 'comb', 'pilot_spacing_f': 4, 'est_method': 'LS',    'interp_method': 'dft'})),
    ]

    results: Dict[str, Dict[str, np.ndarray]] = {}
    for label, cfg in compare_list:
        results[label] = run_snr_sweep(cfg, snr_db_list)

    out_prefix = f'siso_ofdm_channel_estimation_compare_' \
                f"{base.modulation}_{f_sc: .4f}Hz_{base.fs: .4f}Hz_" \
                f"{max_velocity: .4f}kmh_{max_delay: .4f}us_" \

    ber_path, nmse_path, pilot_nmse_path = plot_results(results, out_prefix=out_prefix)
    print(f'Saved BER figure to:        {os.path.abspath(ber_path)}')
    print(f'Saved full-grid NMSE to:    {os.path.abspath(nmse_path)}')
    print(f'Saved pilot NMSE figure to: {os.path.abspath(pilot_nmse_path)}')
    return results


if __name__ == '__main__':
    demo_compare()
