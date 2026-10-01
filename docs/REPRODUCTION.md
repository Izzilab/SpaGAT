# Commands and execution scope

## Figures and numeric tables from included records
Run the nontraining commands in README.md. They regenerate Figure 2, Figure S2 and numeric CSV summaries without checkpoints. Figure S2 describes the original Figure 3 analysis cohorts, not the independent-test benchmark. It reproduces saved gene-level statistics; it does not establish checkpoint provenance beyond the supplied archived verification records.

`scripts/reproduce_tables.py` aggregates the 30 independent-test runs and the 24 revised Table 1 rows with sample SD (ddof=1), checks the six k-size rows, and regenerates Figure 5 numeric masking means/SD. Donor bootstrap intervals are included as recorded output; they are not recomputed from donor-level median PCCs, which would be a different statistic.

## Training with the verified model-ready inputs
First run `python scripts/download_data.py --dataset all --output-dir EXTERNAL_DATA`.
The path passed below must contain section CSVs and _TypeExp.npz files; its parent must contain genes.pth and ligands.pth. Install PyTorch and `requirements-training.txt` first; tested CPU scope and original CUDA versions are distinguished in environment/README.md.

```bash
python scripts/run_benchmark.py --dataset SEA_AD --method SpaGAT --seed 123 --data-dir EXTERNAL_DATA/SEA_AD/data/processed --output-dir run_SEA_AD_SpaGAT_123
python scripts/run_benchmark.py --dataset Mouse --method GITIII --seed 123 --data-dir EXTERNAL_DATA/Mouse/data/processed --output-dir run_Mouse_GITIII_123
```

Methods: SpaGAT, GAT, SPICE-adapted, GITIII, LightGBM. Repeat explicitly for seeds 123, 456, 789. Each command launches one training run; it refuses an existing output directory. The wrapper creates a new working copy with relocated data paths and the exact saved partition indices. It does not modify original inputs or silently regenerate partitions. All methods require the stored 50-cell neighborhoods.

For the revised Table 1 use SpaGAT with `--variant no_distance`,
`--variant uniform_routing`, or `--variant matched_random_edge`. All three use
four routing channels. The wrapper uses uniform weights for uniform_routing
and passes the frozen dataset-specific gene map for matched_random_edge.
Use the same inputs and each of seeds 123, 456 and 789. For example:

```bash
python scripts/run_benchmark.py --dataset Mouse --method SpaGAT --variant uniform_routing --seed 123 --data-dir EXTERNAL_DATA/Mouse/data/processed --output-dir Mouse_uniform_123
python scripts/run_benchmark.py --dataset SEA_AD --method SpaGAT --variant matched_random_edge --seed 123 --data-dir EXTERNAL_DATA/SEA_AD/data/processed --output-dir SEA_AD_random_123
```

The stored maps were chosen by fixed expression-only matching criteria using
training inputs. Do not redraw gene sets based on prediction results.
`scripts/control_support.py` preserves the matching implementation; its original
Colab orchestration functions expect their recorded directory layout and are
not the public training entry point. Use run_benchmark.py for a single run.

The frozen baseline runners validate the split, sample order and dimensions. GITIII includes the pinned source representation used by the archived runner; its provenance is in THIRD_PARTY_NOTICES.md. Mouse LightGBM is a recovered adapter of the archived SEA-AD runner, with labels/count assertions changed; it is not claimed to be the original missing script.

The command wrapper, source syntax, full real input loading, all test-target hashes, synthetic CLI training/test export, and real-input optimizer/checkpoint steps were checked locally. Full-data GPU training was not rerun for this assembly, and stochastic results are not promised to be bit-identical across hardware/software. Do not infer independent patients from liver cell-level splits or independent animals from Mouse test sections.

## Full test inference from the released SpaGAT checkpoints

```bash
python scripts/download_data.py --dataset all --output-dir EXTERNAL_DATA
python scripts/download_checkpoints.py --dataset all --output-dir EXTERNAL_DATA
python scripts/evaluate_checkpoint.py --dataset all --seed all --data-root EXTERNAL_DATA --checkpoint-dir EXTERNAL_DATA/checkpoints --output-dir checkpoint_test --device cpu
```

The six weights belong to the full SpaGAT model in Figure 2. Use `--dataset Mouse`
or `--dataset SEA_AD` and `--seed 123`, `456`, or `789` for a single run; use
`--device cuda` for GPU inference. Test sizes are 117,257 Mouse receivers and
54,762 SEA-AD receivers. No test observations are used for model selection.

The evaluator verifies each checkpoint SHA-256 before loading, checks all three
partition index lists against the saved checkpoint, and checks the complete
test-target hash. It uses the original inference code and batch size 32. It
exports predictions, target/index/gene order, gene-wise PCC, and a JSON validation
record. Summary metrics use float64 postprocessing and the lower-middle median
for even gene counts, matching the archived Figure 2 calculation. EV is
`100 * (1 - MSE / mean(target**2))`, relative to the zero-residual reference.

Absolute cross-platform tolerances are 0.0001 for median/mean PCC, 0.00001 for
MSE, 0.0000001 for MSE0, and 0.001 percentage points for EV. They are numerical
checks, not confidence intervals. Prediction hashes are also recorded; metric
agreement does not imply byte-identical predictions across CPU/CUDA environments.
The script records failures and stops if a comparison exceeds its tolerance.
It requires a new output directory and does not overwrite previous predictions.

Baseline, component-control and liver checkpoints are not provided. This
workflow therefore checks the six SpaGAT Figure 2 results, not all methods or
every figure in the paper. See `environment/checkpoint_inference_validation.json`
for the actual rerun results and environment.

## Optional collection of existing author checkpoints
After placing this bundle on a Colab runtime with Drive already mounted:
```bash
python scripts/collect_checkpoints.py --output-dir /content/SpaGAT_existing_checkpoints
```
This only copies six existing full SpaGAT checkpoints whose hashes are recorded. It does not collect all historical/baseline checkpoints, infer missing files, rerun experiments, or publish anything. Readers should use the public download command above.

## Validate or stage a command without training
Use `--dry-run` to print the planned command without creating a run. Use
`--prepare-only` to stage source, the fixed split and any frozen gene map in
a new run directory without executing training. Neither mode reads the full
expression matrices or establishes that they match the historical targets.
The full training loader performs the additional split/sample checks.

The random-gene launcher stages control_support.py and ablation_support.py
alongside train.py; the gene map is copied into the run's own configs/.
Supplementary author workflows are indexed in analysis/README.md.

## Check execution before a long run

```bash
python scripts/smoke_train.py --output-dir smoke_synthetic
python scripts/verify_data.py --dataset Mouse --data-dir EXTERNAL_DATA/Mouse/data/processed --output Mouse_input_check.json
python scripts/check_training_inputs.py --data-root EXTERNAL_DATA --output-dir smoke_real_inputs
```

The synthetic check uses one epoch on tiny generated cohorts. The real-input
check uses four training receivers and two optimizer steps for each reported
configuration on each brain dataset. Neither reports a new benchmark or changes
the saved partition. Each requires a new output directory and uses CPU.
