# Reconstructed Mouse metadata adapter; computational functions retained from archived SEA-AD runner.
# Mouse adaptation of the archived Mouse revision runner.
# Only dataset metadata, split counts, dimensions checked in main, and log labels change.
# Network classes, features, losses, optimizers, selection and scoring are retained.
# Historical class identifiers (MouseNoLog) do NOT perform normalization.
SEED = 123
"""Mouse LightGBM: fixed inductive split, existing 100-tree settings.

No hyperparameter search. Only training-derived residuals are used. The original
49-neighbor mean feature is retained, with homotypic residuals masked to zero
before averaging to obey the same residual-information restriction as SpaGAT.
No new baseline/type/distance features are introduced.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
DEFAULT_REVISION = '/content/drive/MyDrive/SpaGAT_revision/inductive_training/20260922_104353_552891_UTC_f291de00'
PARAMETERS = dict(objective='regression', n_estimators=100, learning_rate=0.05, num_leaves=31, max_depth=-1, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbosity=-1)

def write_json(path, content):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(content, handle, indent=2, ensure_ascii=False, allow_nan=False)

def load_split(path, dataset, data_dir):
    split = json.loads(Path(path).read_text())
    keys = ('train_indices', 'val_indices', 'test_indices')
    for key in keys:
        if not isinstance(split.get(key), list) or not split[key] or any((type(i) is not int for i in split[key])):
            raise ValueError(f'{key}: expected a nonempty integer list')
    indices = [i for key in keys for i in split[key]]
    if len(indices) != len(dataset) or set(indices) != set(range(len(dataset))):
        raise ValueError('Split is not disjoint, in range and complete')
    if Path(split['processed_dir']).resolve() != Path(data_dir).resolve():
        raise ValueError('Split processed_dir mismatch')
    for key, count_key in zip(keys, ('n_train', 'n_validation', 'n_test')):
        if count_key in split and split[count_key] != len(split[key]):
            raise ValueError(f'{count_key} mismatch')
    audit = pd.read_csv(Path(path).with_name(Path(path).stem + '_samples.csv'))
    if audit['sample'].tolist() != list(dataset.samples):
        raise ValueError('Sample order differs from verified audit')
    counts = np.asarray(dataset.meta_counts, dtype=np.int64)
    ends = np.cumsum(counts)
    starts = ends - counts
    for key, expected in [('eligible_cells', counts), ('global_start', starts), ('global_end_exclusive', ends)]:
        if not np.array_equal(audit[key].to_numpy(), expected):
            raise ValueError(f'Sample audit {key} mismatch')
    membership = np.empty(len(dataset), dtype=np.int8)
    for label, key in enumerate(keys):
        membership[split[key]] = label
    labels = ('train', 'validation', 'test')
    for row, start, end in zip(audit.itertuples(index=False), starts, ends):
        if row.split not in labels or not np.all(membership[start:end] == labels.index(row.split)):
            raise ValueError(f'{row.sample}: sample crosses partitions or audit mismatch')
    return split

def batch_features(batch):
    x = batch['x'].numpy()
    ct = batch['cell_types'].numpy()
    if x.ndim != 3 or x.shape[1] != 50 or ct.shape != x.shape[:2]:
        raise ValueError('Expected 50-cell neighborhoods with aligned cell types')
    if 'neighbor_mask' in batch and (not bool(torch.all(batch['neighbor_mask'] == 1))):
        raise ValueError('Variable neighborhoods require explicit handling; expected all 50 valid nodes')
    heterotypic = ct[:, 1:] != ct[:, :1]
    features = np.where(heterotypic[..., None], x[:, 1:, :], 0.0).mean(axis=1, dtype=np.float32)
    target = batch['y'].numpy()
    if not np.array_equal(target, x[:, 0, :]):
        raise ValueError('Dataset center target and residual disagree')
    if not np.isfinite(features).all() or not np.isfinite(target).all():
        raise ValueError('Nonfinite features/targets')
    return (features, target)

def build_features(dataset, indices, genes, label, batch_size=128):
    x = np.empty((len(indices), len(genes)), dtype=np.float32)
    y = np.empty_like(x)
    cursor = 0
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    for batch in loader:
        features, target = batch_features(batch)
        n = len(target)
        x[cursor:cursor + n] = features
        y[cursor:cursor + n] = target
        cursor += n
        if cursor == n or cursor % (batch_size * 80) == 0 or cursor == len(indices):
            print(f'{label} features: {cursor}/{len(indices)}', flush=True)
    if cursor != len(indices):
        raise ValueError('Incomplete feature construction')
    return (x, y)

def evaluate_arrays(prediction, target):
    """Match the established SpaGAT PCC/MSE conventions; use paper EV_zero."""
    if prediction.shape != target.shape or not np.isfinite(prediction).all() or (not np.isfinite(target).all()):
        raise ValueError('Invalid prediction/target arrays')
    prediction = torch.as_tensor(prediction).float()
    target = torch.as_tensor(target).float()
    pc = prediction - prediction.mean(dim=0, keepdim=True)
    tc = target - target.mean(dim=0, keepdim=True)
    denominator = torch.sqrt((pc.square().sum(dim=0) * tc.square().sum(dim=0)).clamp_min(1e-12))
    pcc = ((pc * tc).sum(dim=0) / denominator).clamp(-1, 1)
    mse = torch.mean((prediction - target).square()).item()
    mse0 = torch.mean(target.square()).item()
    metrics = dict(median_gene_pcc=torch.median(pcc).item(), mean_gene_pcc=torch.mean(pcc).item(), mse=mse, mse_zero=mse0, ev_zero_percent=None if mse0 == 0 else 100 * (1 - mse / mse0), n_cells=int(target.shape[0]), n_genes=int(target.shape[1]))
    return (metrics, pcc.numpy())

def run_experiment(dataset, split, genes, out_dir, lgb, batch_size=128):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / 'models').mkdir()
    write_json(out_dir / 'shared_split.json', split)
    write_json(out_dir / 'run_config.json', {'dataset': 'Mouse', 'model': 'LightGBM', 'seed': SEED, 'parameters': PARAMETERS, 'features': '49-neighbor residual mean; homotypic residuals zeroed; center excluded; denominator 49', 'feature_count': len(genes), 'new_features_added': False, 'selection': 'Fixed 100 trees per gene; no early stopping or hyperparameter search', 'pcc': 'SpaGAT float32 centered PCC, clamp denominator product at 1e-12, torch.median lower-middle convention; constants yield zero', 'ev': '100*(1-MSE/MSE0), zero-residual reference', 'target_space': 'residual relative to training-derived cell-type baseline', 'versions': {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__, 'pandas': pd.__version__, 'lightgbm': getattr(lgb, '__version__', 'test double')}})
    started = time.perf_counter()
    x_train, y_train = build_features(dataset, split['train_indices'], genes, 'train', batch_size)
    x_val, y_val = build_features(dataset, split['val_indices'], genes, 'validation', batch_size)
    val_pred = np.empty_like(y_val)
    records = []
    for g, gene in enumerate(genes):
        t = time.perf_counter()
        model = lgb.LGBMRegressor(**PARAMETERS)
        model.fit(x_train, y_train[:, g])
        val_pred[:, g] = model.predict(x_val)
        model.booster_.save_model(str(out_dir / 'models' / f'gene_{g:04d}.txt'))
        records.append(dict(gene_index=g, gene=gene, runtime_seconds=time.perf_counter() - t))
        pd.DataFrame(records).to_csv(out_dir / 'training_progress.csv', index=False)
        if g == 0 or (g + 1) % 10 == 0 or g + 1 == len(genes):
            print(f'Trained {g + 1}/{len(genes)} genes; elapsed {(time.perf_counter() - started) / 60:.1f} min', flush=True)
        del model
    pd.DataFrame({'gene_index': range(len(genes)), 'gene': genes}).to_csv(out_dir / 'gene_order.csv', index=False)
    val_metrics, _ = evaluate_arrays(val_pred, y_val)
    write_json(out_dir / 'validation_metrics.json', val_metrics)
    del x_train, y_train, x_val, y_val, val_pred
    print('All models fixed and saved. Starting the single final test evaluation.', flush=True)
    x_test, y_test = build_features(dataset, split['test_indices'], genes, 'test', batch_size)
    prediction = np.empty_like(y_test)
    for g in range(len(genes)):
        model = lgb.Booster(model_file=str(out_dir / 'models' / f'gene_{g:04d}.txt'))
        prediction[:, g] = model.predict(x_test)
    metrics, pcc = evaluate_arrays(prediction, y_test)
    metrics.update(dataset='Mouse', model='LightGBM', seed=SEED, runtime_seconds=time.perf_counter() - started)
    np.savez_compressed(out_dir / 'test_predictions.npz', prediction=prediction, target=y_test, test_indices=np.asarray(split['test_indices'], dtype=np.int64), genes=np.asarray(genes, dtype=str))
    pd.DataFrame({'gene': genes, 'PCC': pcc}).to_csv(out_dir / 'test_per_gene_PCC.csv', index=False)
    write_json(out_dir / 'test_metrics.json', metrics)
    pd.DataFrame([metrics]).to_csv(out_dir / 'test_metrics.csv', index=False)
    print(json.dumps(metrics, indent=2), flush=True)
    print('Completed; saved to:', out_dir, flush=True)
    return metrics

def main():
    global SEED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision-dir', default=DEFAULT_REVISION)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    SEED = args.seed
    PARAMETERS["random_state"] = SEED
    revision = Path(args.revision_dir)
    out = Path(args.output_dir) if args.output_dir else revision / 'baseline_results' / f'Mouse_LightGBM_seed{SEED}'
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite existing results: {out}')
    split_path = revision / 'splits' / 'Mouse_inductive_split_indices.json'
    metadata = json.loads(split_path.read_text())
    data_dir = Path(metadata['processed_dir'])
    if tuple((len(metadata[key]) for key in ('train_indices', 'val_indices', 'test_indices'))) != (122886, 36193, 117257):
        raise ValueError('Mouse split counts differ from the verified experiment')
    loader_path = revision / 'spagat' / 'dataloader.py'
    manifest = json.loads((revision / 'revision_manifest.json').read_text())
    digest = hashlib.sha256(loader_path.read_bytes()).hexdigest()
    if digest != manifest['files_sha256']['spagat/dataloader.py']:
        raise ValueError('Frozen SpaGAT dataloader changed since revision installation')
    spec = importlib.util.spec_from_file_location('frozen_spagat_data', loader_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import lightgbm as lgb
    print('Loading the verified Mouse inductive processed data...', flush=True)
    dataset = module.SPAGAT_dataset(processed_dir=str(data_dir), num_neighbors=50)
    split = load_split(split_path, dataset, data_dir)
    genes = list(dataset.genes)
    if len(genes) != 254 or len(set(genes)) != len(genes):
        raise ValueError('Expected 254 unique Mouse genes')
    print('Fixed split and sample audit passed. Training fixed 100-tree models.', flush=True)
    run_experiment(dataset, split, genes, out, lgb)
    write_json(out / 'source_provenance.json', {'revision': str(revision), 'split_sha256': hashlib.sha256(split_path.read_bytes()).hexdigest(), 'dataloader_sha256': digest, 'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
if __name__ == '__main__':
    main()
