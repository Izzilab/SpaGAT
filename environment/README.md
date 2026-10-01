# Recorded environments
Mouse_recorded.json and SEA_AD_recorded.json preserve the benchmark runtime reports (Python 3.13.15, PyTorch 2.11.0+cu128, NumPy 2.1.3, pandas 2.2.3, LightGBM 4.6.0, torch-geometric 2.8.0.post1; CUDA 12.8 / Tesla T4 where recorded).

component_ablation_full_freeze.txt is the saved package freeze for the completed component ablations. component_ablation_selected_pins.txt extracts only directly relevant packages. These are observed environment records, not a claim that a fresh installation has been validated. Colab-specific or platform-specific entries in the full freeze should not be blindly installed on macOS.

requirements-plots.txt is a lightweight convenience dependency list for table/figure regeneration; it is not the training environment. The current assembly checks use the local environment reported in validation_report.json, which differs from the training environment. Original liver runtime details are not sufficiently complete here for an exact environment reconstruction.

## Fresh CPU installation checked on 2026-10-01

`macos_cpu_smoke_freeze.txt` records a new Python 3.12.14 / macOS arm64 environment
with PyTorch 2.14.1, NumPy 2.5.3, pandas 3.0.6 and PyG 2.8.0.post1. It differs
from the original CUDA training environment. The checks completed were:

- Synthetic CLI training, validation selection, checkpoint save/reload and test
  export for full, distance removal and uniform routing (`synthetic_training_validation.json`).
- Loading all original brain inputs, reproducing every test-target hash, and two
  optimizer steps plus checkpoint roundtrip on four real training receivers for
  both datasets and all four Table 1 configurations (`real_input_training_validation.json`).
- LightGBM synthetic fit/predict (`lightgbm_cpu_validation.json`). macOS required
  an OpenMP runtime; the check used an existing llvm-openmp 22.1.0 installation.
  A plain pip install without that system dependency initially failed.

For the tested Python dependencies use the CPU freeze file in an isolated
environment. On Linux/CUDA, select the appropriate PyTorch build and consult
the original recorded versions. These checks do not establish bit-identical
training across platforms or reproduce complete paper-training runs. The
six original trained SpaGAT brain checkpoints were subsequently verified and
evaluated on every test receiver; see the inference check below.

## Full test inference from original trained checkpoints

`checkpoint_inference_validation.json` records six complete runs: three
seeds (123, 456, 789) on each brain dataset, with 117,257 Mouse and 54,762
SEA-AD test receivers per seed. The same fresh macOS CPU environment
loaded the original weights and fixed partitions without retraining.
Checkpoint hashes and all test-target hashes match the archived records.
PCC, MSE and zero-reference EV match the saved Figure 2 values within
the declared cross-platform numerical tolerances. Individual differences
and prediction hashes are reported; byte-identical predictions are not
claimed. Baseline, component-control and liver weights were not evaluated.
