# SISO OFDM Channel Estimation Simulation

This project provides an end-to-end **SISO OFDM channel-estimation simulation**. It follows the module layout of `MIMOOFDMChanelEstimation`, while keeping the signal model SISO:

```text
Y[q, k] = H[q, k] X[q, k] + W[q, k]
```

## Features

- Rayleigh, Rician, and TDL fading channels.
- Jakes and AR(1) time-selective fading.
- Block, comb, and scattered pilot patterns.
- LS, MMSE, and LMMSE pilot-domain channel estimation.
- Linear, quadratic, Wiener, and DFT interpolation.
- SISO one-tap equalization.
- BER, full-grid NMSE, and pilot-domain NMSE evaluation.
- Optional LS and LMMSE theoretical estimation-error curves on the NMSE plot.
- YAML-driven simulation configuration.
- All generated results are saved under `./output`.

## Structure

| File | Description |
|---|---|
| `config.yaml` | Channel, transmission, algorithm, SNR, and output configuration |
| `main.py` | End-to-end simulation entry point |
| `channel.py` | SISO channel generation and correlation helpers |
| `estimation.py` | LS/MMSE/LMMSE estimation and equalization |
| `interpolation.py` | Time-frequency channel interpolation |
| `pilot_design.py` | Pilot positions, masks, and transmit-grid generation |
| `modulation.py` | Bit generation, modulation, and demodulation |
| `visualization.py` | BER/NMSE plotting |
| `reference.txt` | Notes about the reference framework |
| `output/` | Generated figures, result arrays, and correlation caches |

## Run

```bash
python main.py
```

Use a custom config or output directory:

```bash
python main.py --config config.yaml --output-dir ./output
```

## Outputs

The simulation writes files to `./output`:

```text
*_results.npz
*_ber.png
*_nmse.png
*_pilot_nmse.png
output/corr_stats/*.npz
```

The default `config.yaml` enables theoretical NMSE curves:

```yaml
algorithms:
  show_theory_nmse: true
  theory_nmse_estimators: [ls, lmmse]
```

These curves are saved in `*_results.npz` and drawn on `*_nmse.png`.

## Tests

Each Python module has local test functions and a direct test entry:

```bash
python modulation.py
python channel.py
python pilot_design.py
python estimation.py
python interpolation.py
python visualization.py
python -c "import main; main.test_parse_snr(); main.test_build_combo_names(); main.test_small_simulation_smoke()"
```
