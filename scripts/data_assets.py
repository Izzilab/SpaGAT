"""Validate model-ready input order and targets against the saved brain benchmark."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spagat.dataloader import SPAGAT_dataset
from train import validate_sample_order


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(value):
    array = np.ascontiguousarray(value, dtype=np.float32)
    h = hashlib.sha256(str(array.shape).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def validate_inputs(data_dir, dataset, repo=ROOT):
    """Read original data without rewriting cells, genes, neighborhoods or targets."""
    data_dir, repo = Path(data_dir).resolve(), Path(repo)
    split_path = repo/'splits'/f'{dataset}_inductive_split_indices.json'
    split = json.loads(split_path.read_text())
    audit = pd.read_csv(split_path.with_name(split_path.stem+'_samples.csv'))
    expected_samples = audit['sample'].tolist()
    actual_samples = sorted(p.name[:-len('_TypeExp.npz')] for p in data_dir.glob('*_TypeExp.npz'))
    if actual_samples != expected_samples:
        raise ValueError(f'{dataset}: section inventory/order differs from the saved split')
    files = [data_dir.parent/'genes.pth', data_dir.parent/'ligands.pth']
    for sample in expected_samples:
        files.extend([data_dir/f'{sample}.csv', data_dir/f'{sample}_TypeExp.npz'])
    if any(not p.is_file() or p.is_symlink() for p in files):
        raise ValueError('Required original input file missing or a symbolic link')
    data = SPAGAT_dataset(str(data_dir), num_neighbors=50)
    # Catch the historical loader's skip-on-error behavior before indexing.
    if len(data.exps) != len(expected_samples) or len(data.flags) != len(expected_samples):
        raise ValueError('Loader skipped a section; inputs cannot be exported')
    validate_sample_order(str(split_path), data, split)
    flat = [i for k in ('train_indices', 'val_indices', 'test_indices') for i in split[k]]
    if sorted(flat) != list(range(len(data))):
        raise ValueError('Saved split does not partition the actual receiver set')
    ix = np.asarray(split['test_indices'], dtype=np.int64)
    counts = np.asarray(data.meta_counts, dtype=np.int64)
    ends = np.cumsum(counts)
    starts = ends-counts
    section_id = np.searchsorted(ends, ix, side='right')
    target = np.empty((len(ix), len(data.genes)), dtype=np.float32)
    for j, name in enumerate(data.samples):
        indices = data.indexes[j]
        if indices.shape != (len(data.exps[j]), 50):
            raise ValueError(f'{name}: incorrect 50-cell graph shape')
        if indices.min() < 0 or indices.max() >= len(data.exps[j]):
            raise ValueError(f'{name}: out-of-range neighbor index')
        if not torch.equal(indices[:, 0], torch.arange(len(indices))):
            raise ValueError(f'{name}: center column does not identify the receiver')
        if not torch.isfinite(data.exps[j]).all() or not torch.isfinite(data.type_exp[j]).all():
            raise ValueError(f'{name}: nonfinite expression/baseline')
        chosen = np.where(section_id == j)[0]
        if len(chosen):
            rows = data.arg_meta[j][torch.as_tensor(ix[chosen]-starts[j])]
            target[chosen] = data.exps[j][rows].numpy()
    records = json.loads((repo/'provenance/brain_prediction_records.json').read_text())
    hashes = {r['target_sha256_float32'] for r in records if r['dataset'] == dataset}
    target_sha = array_sha(target)
    if hashes != {target_sha}:
        raise ValueError(f'{dataset}: actual test targets differ from Figure 2; stop without substitution')
    report = {'dataset': dataset, 'sections': len(data.samples), 'genes': len(data.genes),
              'train_receivers': len(split['train_indices']), 'validation_receivers': len(split['val_indices']),
              'test_receivers': len(ix), 'test_target_sha256_float32': target_sha,
              'test_mse0_float64': float(np.mean(target.astype(np.float64)**2)),
              'all_test_targets_match_paper': True, 'sample_order_and_counts_match': True,
              'neighbor_indices_in_bounds': True, 'receiver_first': True,
              'gene_order': list(map(str, data.genes))}
    return data, split, files, report


def build_model(data, checkpoint, variant='full'):
    from spagat.gene_program_model import SpaGP
    config = checkpoint.get('config', {})
    required = {'node_dim': 256, 'edge_dim': 48, 'num_heads': 2, 'n_layers': 1,
                'att_dim': 8, 'n_programs': 4, 'program_token_dim': 32, 'num_neighbors': 50}
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f'Checkpoint setting mismatch: {key}')
    model = SpaGP(data.genes, data.interactions, node_dim=256, edge_dim=48,
                  num_heads=2, n_layers=1, att_dim=8, n_programs=4,
                  program_aware=True, program_token_dim=32,
                  message_decoder='free', routing_mode='program')
    model.embeddings.component_ablation = variant
    model.load_state_dict(checkpoint['model'], strict=True)
    return model
