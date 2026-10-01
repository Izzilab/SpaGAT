"""Small synthetic CLI training/checkpoint/export check; not a paper experiment."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    data = out/'data/processed'
    data.mkdir(parents=True)
    rng = np.random.default_rng(123)
    genes = [f'g{i}' for i in range(8)]
    ligands = ([['g0'], ['g1']], [[0], [0]])
    torch.save(genes, data.parent/'genes.pth')
    torch.save(ligands, data.parent/'ligands.pth')
    # Three distinct sections and four eligible receivers per section. All 50
    # cells remain in each section's neighborhood context.
    for section in ['section0', 'section1', 'section2']:
        positions = rng.normal(size=(50, 2))*10
        distance = ((positions[:, None]-positions[None, :])**2).sum(axis=2)
        indices = np.argsort(distance, axis=1, kind='stable')
        frame = pd.DataFrame(rng.normal(scale=.1, size=(50, 8)), columns=genes)
        frame['centerx'], frame['centery'] = positions.T
        frame['subclass'] = ['A', 'B']*25
        frame['flag'] = [True]*4+[False]*46
        for i in range(50):
            frame[f'index_{i}'] = indices[:, i]
        frame.to_csv(data/f'{section}.csv', index=False)
        np.savez(data/f'{section}_TypeExp.npz', A=np.ones(8), B=np.ones(8)*1.2)
    split = {'train_indices': list(range(4)), 'val_indices': list(range(4, 8)),
             'test_indices': list(range(8, 12)), 'processed_dir': str(data)}
    split_path = out/'synthetic_split.json'
    split_path.write_text(json.dumps(split))
    reports = []
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    for variant in ['full', 'no_distance', 'uniform_routing']:
        cmd = [sys.executable, str(ROOT/'scripts/train.py'), '--data-dir', str(data),
               '--shared-split', str(split_path), '--output-dir', str(out/variant),
               '--device', 'cpu', '--max-epochs', '1', '--batch-size', '4',
               '--component-ablation', variant,
               '--routing-mode', 'uniform' if variant == 'uniform_routing' else 'program']
        t0 = time.monotonic()
        with (out/f'{variant}.log').open('w') as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        if proc.returncode:
            raise RuntimeError(f'Training failed: {out / (variant + ".log")}')
        target_dir = out/variant/'seed123_test'
        ckpt = torch.load(out/variant/'spagat_seed123_best.pth', map_location='cpu', weights_only=False)
        metrics = json.loads((target_dir/'test_metrics.json').read_text())
        with np.load(target_dir/'test_predictions.npz', allow_pickle=False) as z:
            assert z['prediction'].shape == z['target'].shape == (4, 8)
            assert np.isfinite(z['prediction']).all()
            assert z['test_indices'].tolist() == split['test_indices']
            assert z['genes'].tolist() == genes
        assert ckpt['shared_split']['test_indices'] == split['test_indices']
        assert ckpt['epoch'] == metrics['best_epoch'] == 0
        assert all(np.isfinite(metrics[key]) for key in ['test_mse', 'median_gene_pcc', 'ev_percent'])
        reports.append({'variant': variant, 'training_checkpoint_reload_test_export': 'passed',
                        'seconds': time.monotonic()-t0})
    report = {'scope': 'Synthetic CPU execution check only; no paper-data performance validation.',
              'system': platform.platform(), 'python': platform.python_version(),
              'torch': torch.__version__, 'numpy': np.__version__, 'pandas': pd.__version__,
              'runs': reports, 'paper_data_used': False, 'full_benchmark_retrained': False}
    (out/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
