"""Evaluate released SpaGAT checkpoints on the unchanged full brain test sets.

No training or checkpoint selection. Output includes predictions and an explicit
comparison with the archived Figure 2 metrics. Requires a new output directory.
"""
import argparse
import gc
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from data_assets import ROOT, array_sha, build_model, file_sha, validate_inputs
from train import evaluate, set_all_seeds
from spagat.gene_program_model import SpaGP_Loss

# Absolute tolerances for cross-platform inference, declared before evaluation.
# EV tolerance is in percentage points. These are not statistical intervals.
TOLERANCES = {'median_gene_pcc': 1e-4, 'mean_gene_pcc': 1e-4,
              'mse': 1e-5, 'mse_zero': 1e-7, 'ev_zero_percent': 1e-3}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def postprocess(prediction, target):
    """Float64 summaries and lower-middle median, as in Figure 2 postprocessing."""
    p = prediction.numpy().astype(np.float64)
    y = target.numpy().astype(np.float64)
    mse = float(np.mean((p-y)**2))
    mse0 = float(np.mean(y*y))
    p -= p.mean(axis=0)
    y -= y.mean(axis=0)
    denominator = np.sqrt(np.maximum(np.sum(p*p, axis=0)*np.sum(y*y, axis=0), 1e-12))
    corr = np.clip(np.sum(p*y, axis=0)/denominator, -1, 1)
    if mse0 <= 0 or not np.isfinite(corr).all():
        raise ValueError('Undefined metrics')
    return {'median_gene_pcc': float(np.sort(corr)[(len(corr)-1)//2]),
            'mean_gene_pcc': float(corr.mean()), 'mse': mse, 'mse_zero': mse0,
            'ev_zero_percent': 100*(1-mse/mse0)}, corr


class Progress:
    def __init__(self, loader, label):
        self.loader, self.label = loader, label

    def __iter__(self):
        tick = time.monotonic()
        for i, batch in enumerate(self.loader, 1):
            yield batch
            if i == 1 or time.monotonic()-tick >= 30 or i == len(self.loader):
                print(f'{self.label}: {i}/{len(self.loader)} test batches', flush=True)
                tick = time.monotonic()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['Mouse', 'SEA_AD', 'all'], default='all')
    parser.add_argument('--seed', choices=['123', '456', '789', 'all'], default='all')
    parser.add_argument('--data-root', type=Path, default=Path('EXTERNAL_DATA'))
    parser.add_argument('--checkpoint-dir', type=Path, default=Path('EXTERNAL_DATA/checkpoints'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--cpu-threads', type=int, default=2)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        raise ValueError('--cpu-threads must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(args.cpu_threads)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    weights = json.loads((ROOT/'assets/checkpoint_manifest.json').read_text())
    archived = pd.read_csv(ROOT/'results/brain_benchmarks/per_seed_metrics.csv')
    records = json.loads((ROOT/'provenance/brain_prediction_records.json').read_text())
    datasets = ['Mouse', 'SEA_AD'] if args.dataset == 'all' else [args.dataset]
    seeds = [123, 456, 789] if args.seed == 'all' else [int(args.seed)]
    report = {'complete': False, 'training_performed': False, 'device': args.device,
              'python': platform.python_version(), 'platform': platform.platform(),
              'torch': str(torch.__version__), 'numpy': np.__version__,
              'batch_size': 32, 'absolute_tolerances': TOLERANCES,
              'scope': 'Full test inference from original released SpaGAT checkpoints; no baseline or liver inference.',
              'runs': []}
    save_json(args.output_dir/'validation.json', report)
    try:
        for dataset in datasets:
            print('Validating all inputs and test targets:', dataset, flush=True)
            data, split, _, input_report = validate_inputs(args.data_root/dataset/'data/processed', dataset)
            for seed in seeds:
                start = time.monotonic()
                row = next(r for r in weights if r['dataset'] == dataset and r['seed'] == seed)
                path = args.checkpoint_dir/f'{dataset}_SpaGAT_seed{seed}.pth'
                digest = file_sha(path)
                if digest != row['sha256']:
                    raise ValueError(f'Checkpoint checksum mismatch: {path}')
                saved = torch.load(path, map_location='cpu', weights_only=True)
                if saved['seed'] != seed:
                    raise ValueError('Checkpoint seed mismatch')
                for key in ('train_indices', 'val_indices', 'test_indices'):
                    if saved['shared_split'][key] != split[key]:
                        raise ValueError('Checkpoint partition mismatch: '+key)
                set_all_seeds(seed)
                model = build_model(data, saved).to(args.device)
                loss = SpaGP_Loss(data.genes, data.interactions, lambda_orth=.1, lambda_sparse=.01)
                loader = DataLoader(Subset(data, split['test_indices']), batch_size=32,
                                    shuffle=False, drop_last=False)
                raw, prediction, target, _ = evaluate(model, Progress(loader, f'{dataset}/{seed}'),
                                                      loss, args.device, return_predictions=True)
                target_sha = array_sha(target.numpy())
                if target_sha != input_report['test_target_sha256_float32']:
                    raise ValueError('Inference target order differs from validated input targets')
                metrics, corr = postprocess(prediction, target)
                reference = archived[(archived.dataset == dataset) & (archived.model == 'SpaGAT') &
                                     (archived.seed == seed)]
                if len(reference) != 1:
                    raise ValueError('Missing or ambiguous reference metric row')
                reference = {key: float(reference.iloc[0][key]) for key in TOLERANCES}
                differences = {key: abs(metrics[key]-reference[key]) for key in TOLERANCES}
                passed = all(differences[key] <= tol for key, tol in TOLERANCES.items())
                original = next(r for r in records if r['dataset'] == dataset and
                                r['model'] == 'SpaGAT' and r['seed'] == seed)
                prediction_sha = array_sha(prediction.numpy())
                result = {'dataset': dataset, 'seed': seed, 'checkpoint_sha256': digest,
                          'best_epoch': saved['best_epoch'], 'n_cells': raw['n_cells'],
                          'n_genes': raw['n_genes'], 'metrics': metrics, 'reference_metrics': reference,
                          'absolute_differences': differences, 'metrics_match_within_tolerance': passed,
                          'test_target_sha256_float32': target_sha, 'all_test_targets_match': True,
                          'prediction_sha256_float32': prediction_sha,
                          'archived_prediction_sha256_float32': original['prediction_sha256_float32'],
                          'prediction_bytes_identical': prediction_sha == original['prediction_sha256_float32'],
                          'seconds': time.monotonic()-start}
                dest = args.output_dir/f'{dataset}_seed{seed}'
                dest.mkdir()
                save_json(dest/'metrics.json', result)
                pd.DataFrame({'gene': data.genes, 'PCC': corr}).to_csv(dest/'gene_PCC.csv', index=False)
                np.savez_compressed(dest/'test_predictions.npz', prediction=prediction.numpy(),
                                    target=target.numpy(), genes=np.asarray(data.genes, dtype=str),
                                    test_indices=np.asarray(split['test_indices'], dtype=np.int64))
                report['runs'].append(result)
                save_json(args.output_dir/'validation.json', report)
                print(f'{dataset}/{seed}: PCC={metrics["median_gene_pcc"]:.7f}; metric check={passed}', flush=True)
                if not passed:
                    raise ValueError('Metrics differ from the archived result beyond the declared tolerances')
                del saved, model, loss, loader, prediction, target
                gc.collect()
                if args.device == 'cuda':
                    torch.cuda.empty_cache()
            del data
            gc.collect()
        report['complete'] = True
        save_json(args.output_dir/'validation.json', report)
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        save_json(args.output_dir/'validation.json', report)
        raise


if __name__ == '__main__':
    main()
