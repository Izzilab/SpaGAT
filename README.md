# SpaGAT

SpaGAT predicts cell-type-centered expression residuals from spatial
neighborhoods and evaluates model dependence on neighboring cell populations.
This repository contains the code and recorded results accompanying the
revised manuscript. The maintained repository is
[WuBoFu/SpaGAT](https://github.com/WuBoFu/SpaGAT).

## Start here

| Task | Entry point | External inputs |
|---|---|---|
| Check included files and split records | `python scripts/verify_bundle.py` | None; Python standard library |
| Regenerate numerical summaries | `scripts/reproduce_tables.py` | None beyond the included records |
| Redraw Figure 2 and Figure S2 | `figures/plot_Figure2.py`, `figures/plot_FigureS2.py` | None beyond the included records |
| Download verified brain inputs | `scripts/download_data.py` | Public release assets |
| Download and evaluate the six SpaGAT checkpoints | `scripts/download_checkpoints.py`, `scripts/evaluate_checkpoint.py` | Public brain inputs and checkpoint release |
| Train and evaluate a brain benchmark or component control | `scripts/run_benchmark.py` | Downloaded inputs and a training environment |
| Inspect or adapt the supplementary analyses | [analysis/README.md](analysis/README.md) | Original inputs/predictions/checkpoints as specified in each workflow |

## Reproduce included results without training

Run from this repository's root:

```bash
python -m pip install -r requirements.txt
python scripts/verify_bundle.py
python scripts/reproduce_tables.py --output-dir reproduced_tables
python figures/plot_Figure2.py --output reproduced_figures/Figure2.pdf
python figures/plot_FigureS2.py --output reproduced_figures/FigureS2.pdf
```

These commands use the included numerical records. For inference from the
released checkpoints, use the commands below. Figure S2 describes
the original Figure 3 analysis cohorts, which differ from the brain test sets.

## Reproduce SpaGAT test predictions without retraining

The [checkpoint release](https://github.com/WuBoFu/SpaGAT/releases/tag/benchmark-checkpoints-v1)
contains the six original full-model weights for SEA-AD and Mouse, with seeds
123, 456 and 789. Install PyTorch and `requirements-training.txt`, then run:

```bash
python scripts/download_data.py --dataset all --output-dir EXTERNAL_DATA
python scripts/download_checkpoints.py --dataset all --output-dir EXTERNAL_DATA
python scripts/evaluate_checkpoint.py --dataset all --seed all --data-root EXTERNAL_DATA --checkpoint-dir EXTERNAL_DATA/checkpoints --output-dir checkpoint_test --device cpu
```

Use `--device cuda` on a supported GPU. The evaluator checks checkpoint hashes,
saved partitions and every test target, loads weights strictly, and exports
predictions and gene-level PCC for all test receivers. It compares metrics with
the recorded Figure 2 values using declared numerical tolerances. It neither
trains nor selects a checkpoint. The output directory must be new. These six
weights cover SpaGAT; baseline, ablation and liver weights are not included.

## Independent-test brain benchmarks

SEA-AD and Mouse use the saved training, validation and test partitions, with
training seeds **123, 456 and 789**. Cell-type baselines are training-derived.
Checkpoint selection uses validation median gene-wise PCC; the selected model
is then evaluated on the held-out test set. Neighborhoods remain within
sections, and receiver/homotypic-neighbor residuals are masked in the inputs.

The five methods are SpaGAT, GITIII, GAT, SPICE-adapted and LightGBM. Use the
same verified model-ready inputs for every method. Download them once:

```bash
python scripts/download_data.py --dataset all --output-dir EXTERNAL_DATA
python scripts/run_benchmark.py --dataset SEA_AD --method SpaGAT --seed 123 --data-dir EXTERNAL_DATA/SEA_AD/data/processed --output-dir runs/SEA_AD_SpaGAT_123
```

Install PyTorch and `requirements-training.txt` before training; see the environment notes below.
The downloader checks archive and per-file SHA-256 values and preserves cell/gene order.
The launcher validates required paths and creates a separate run directory.
It stages the fixed partitions rather than generating a new random split.
The underlying `scripts/train.py` requires all three partitions and includes
the final test export; there is no root validation-only `evaluate.py` command.
See [docs/REPRODUCTION.md](docs/REPRODUCTION.md) for all methods and controls,
and [environment/README.md](environment/README.md) for recorded versions.

## Component and supplementary analyses

Revised Table 1 compares the full model with distance removal, four-channel
uniform routing, and matched random edge-gene replacement. Per-seed values and
sample SDs are in `results/table1/`. One fixed matched gene set is used per
dataset across training seeds; these runs do not measure between-gene-set
variability. The comparison does not establish ligand-receptor specificity.

The liver analyses use their separately documented within-specimen cohorts;
they are not independent-patient tests. Masking results measure model-derived
prediction dependence rather than causal signaling. The additional notebooks
and frozen source snapshots are indexed in [analysis/README.md](analysis/README.md).
They retain author paths and require external inputs; they are not a universal
run-all workflow. The component code and results in this package cover the
configurations reported in revised Table 1.

The file-to-manuscript mapping is in [docs/PAPER_SCOPE.md](docs/PAPER_SCOPE.md).

## Repository contents

- `spagat/`: model and data-loading code; SpaGAT is the public name, with
  historical SpaGP identifiers retained for checkpoint compatibility.
- `scripts/`, `baselines/`: fixed-partition training, controls and summary tools.
- `splits/`, `configs/`, `environment/`: saved indices, settings, matching maps
  and recorded environments.
- `results/`, `figures/`: numerical records and selected plotting scripts.
- `analysis/`: supplementary-analysis notebooks and recovered source snapshots.
- `docs/`, `data_records/`, `provenance/`, `assets/`: protocol, source coverage,
  filtering counts and external-input/checkpoint records.

## Data and reproducibility scope

Public dataset sources and the model-ready input contract are documented in
[docs/DATA_AND_PREPROCESSING.md](docs/DATA_AND_PREPROCESSING.md).
**Verified SEA-AD and Mouse model-ready matrices are available in the
[benchmark input release](https://github.com/WuBoFu/SpaGAT/releases/tag/benchmark-inputs-v1).**
The download manifest includes individual-file hashes and complete test-target
hash checks against Figure 2. Original CSV/baseline bytes are retained.
See [data licenses and attribution](docs/DATA_LICENSES.md).
The six SpaGAT brain checkpoints are available in the separate
[checkpoint release](https://github.com/WuBoFu/SpaGAT/releases/tag/benchmark-checkpoints-v1);
liver matrices and other checkpoints are not included. Complete raw-to-input
conversion pipelines remain incomplete. The Mouse LightGBM runner is
a documented recovered adapter. A fresh macOS CPU environment passed a synthetic train/validation/test CLI
check and two optimizer steps on four real training receivers for each brain
dataset and Table 1 configuration. These small training checks establish execution;
full-test verification from the original weights is reported separately below.
The tested environment and reports
are in [environment/README.md](environment/README.md). Complete Figure 1/3/4 workflows and
the original Figure 5 composite are not provided.

[docs/COVERAGE.csv](docs/COVERAGE.csv) and [docs/KNOWN_GAPS.md](docs/KNOWN_GAPS.md)
identify what can be reproduced from the included records and what still needs
external inputs. `validation_report.json` records local checks; it is not a
claim of fresh full-data GPU training. Checkpoint inference has its own report
in `environment/checkpoint_inference_validation.json`: all six full test runs
match the archived Figure 2 metrics within the declared numerical tolerances.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for source attribution and
the retained GITIII license. Historical repository versions remain in Git
history; the main entry points above describe the revised experiments.
