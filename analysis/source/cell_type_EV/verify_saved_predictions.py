"""Verify archived cell-type MSE/EV using saved tensors only; no model execution."""
from pathlib import Path
from datetime import datetime, timezone
import gc
import hashlib
import json
import traceback
import zipfile
import numpy as np
import pandas as pd
import torch

BASE = Path('/content/drive/MyDrive')
TABLE_DIR = BASE / 'spagatv2/analysis/celltype_performance'
DATA_DIR = BASE / 'GITIII-main/liver_workdir/gitiii/data/processed'
TENSOR_DIR = BASE / 'GITIII-main/liver_workdir/gitiii/influence_tensor'
EXPECTED_HASHES = {
    'CancerousLiver': '921828f7b76aa38e60c22148bf86865bbf3d96d370038b2540708983f41ce5dc',
    'NormalLiver': '5671673b04e1bf278c9b99513bde490ccfb50d7570ddc013cb0e5726a58564ab',
}


def accumulate_metrics(pred, target, labels, block_size=2048):
    """Bounded float64 reductions; never build a full error matrix."""
    names, group = np.unique(np.asarray(labels, dtype=str), return_inverse=True)
    counts = np.zeros(len(names), dtype=np.int64)
    error_sums = np.zeros(len(names), dtype=np.float64)
    zero_sums = np.zeros(len(names), dtype=np.float64)
    for start in range(0, len(labels), block_size):
        stop = min(start + block_size, len(labels))
        y = np.asarray(target[start:stop], dtype=np.float64)
        yp = np.asarray(pred[start:stop], dtype=np.float64)
        if not np.isfinite(y).all() or not np.isfinite(yp).all():
            raise ValueError('Saved predictions or targets contain nonfinite entries')
        error = yp - y
        keys = group[start:stop]
        counts += np.bincount(keys, minlength=len(names))
        error_sums += np.bincount(keys, weights=np.einsum('ij,ij->i', error, error), minlength=len(names))
        zero_sums += np.bincount(keys, weights=np.einsum('ij,ij->i', y, y), minlength=len(names))
        if start % (block_size * 40) == 0:
            print(f'  Reading saved arrays: {stop:,}/{len(labels):,} cells', flush=True)
    entries = counts * target.shape[1]
    mse, mse0 = error_sums / entries, zero_sums / entries
    ev = np.full(len(names), np.nan)
    np.divide(mse, mse0, out=ev, where=mse0 > 0)
    ev = 100 * (1 - ev)
    return pd.DataFrame({'cell_type': names, 'n_cells': counts,
                         'audited_MSE': mse, 'MSE0': mse0, 'audited_EV_percent': ev})


def compare_table(reference, recomputed):
    if reference.cell_type.duplicated().any() or recomputed.cell_type.duplicated().any():
        raise ValueError('Duplicate cell-type rows')
    merged = reference.merge(recomputed, on='cell_type', how='outer', suffixes=('_reported', '_audited'), indicator=True)
    if not (merged['_merge'] == 'both').all():
        raise ValueError('Saved tensor and summary table have different cell-type populations')
    merged['count_matches'] = merged.n_cells_reported == merged.n_cells_audited
    merged['MSE_matches'] = np.isclose(merged.MSE, merged.audited_MSE, atol=1e-6, rtol=1e-5)
    merged['EV_matches_zero_reference'] = np.isclose(merged.EV_percent, merged.audited_EV_percent, atol=1e-3, rtol=2e-5)
    merged['included_n_ge_100'] = merged.n_cells_reported >= 100
    return merged.drop(columns='_merge')


def recover_labels_from_processed(sample, archived_target):
    """Recover metadata only when saved labels are absent; check row alignment against 16 genes."""
    genes = torch.load(DATA_DIR.parent / 'genes.pth', map_location='cpu', weights_only=True)
    genes = [str(g) for g in genes]
    if len(genes) != archived_target.shape[1]:
        raise ValueError('Gene count differs from saved targets')
    sentinels = np.unique(np.linspace(0, len(genes) - 1, min(16, len(genes)), dtype=int))
    sentinel_genes = [genes[i] for i in sentinels]
    with np.load(DATA_DIR / f'{sample}_TypeExp.npz', allow_pickle=False) as z:
        valid_types = set(z.files)
    csv_path = DATA_DIR / f'{sample}.csv'
    header = list(pd.read_csv(csv_path, nrows=0).columns)
    # Historical GITIII_dataset uses col_cell_type='subclass'.
    label_column = 'subclass'
    required = [label_column, 'index_0'] + sentinel_genes
    missing = set(required) - set(header)
    if missing:
        raise ValueError(f'Processed metadata missing columns: {sorted(missing)}; '
                         f'available metadata: {[c for c in header if c not in genes][:80]}')
    selected_columns = required + (['flag'] if 'flag' in header else [])
    frames = []
    print('  Restoring missing labels from processed metadata; checking saved target alignment.', flush=True)
    for frame in pd.read_csv(csv_path, usecols=selected_columns, chunksize=20000):
        frame = frame[frame[label_column].isin(valid_types)].copy()
        frame = frame.rename(columns={label_column: 'cell_type'})
        if 'flag' not in frame:
            frame['flag'] = True
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    flags = frame.flag
    if flags.dtype == object:
        mapping = {'true': True, 'false': False, '1': True, '0': False, '1.0': True, '0.0': False}
        flags = flags.astype(str).str.lower().map(mapping)
        if flags.isna().any():
            raise ValueError('Unrecognized receiver eligibility flags')
    elif flags.isna().any() or not flags.isin([0, 1, False, True]).all():
        raise ValueError('Invalid receiver eligibility flags')
    eligible = np.flatnonzero(flags.to_numpy(dtype=bool))
    if len(archived_target) == len(eligible):
        positions = eligible
    elif sample == 'CancerousLiver' and len(archived_target) == 30000 and len(eligible) == 460429:
        # Exact historical sampling expression: np.random.seed(42); np.random.choice(...).
        positions = eligible[np.random.RandomState(42).choice(len(eligible), 30000, replace=False)]
    else:
        raise ValueError('Saved array length does not match a documented evaluation ordering')
    center_rows = frame.index_0.to_numpy(dtype=np.int64)[positions]
    if center_rows.min() < 0 or center_rows.max() >= len(frame):
        raise ValueError('Center-node row index out of bounds')
    expected = frame[sentinel_genes].to_numpy(dtype=np.float32)[center_rows]
    matches_float32 = True
    matches_half_roundtrip = True
    for start in range(0, len(expected), 4096):
        stop = min(start + 4096, len(expected))
        observed = np.asarray(archived_target[start:stop])[:, sentinels]
        source = expected[start:stop]
        matches_float32 &= bool(np.allclose(observed, source, rtol=1e-6, atol=1e-6))
        # Historical exports explicitly converted target/predictions to half then back to float.
        matches_half_roundtrip &= bool(np.allclose(observed, source.astype(np.float16).astype(np.float32), rtol=1e-6, atol=1e-6))
    if not (matches_float32 or matches_half_roundtrip):
        raise ValueError('Saved target rows do not match processed metadata under the documented ordering')
    return frame.cell_type.to_numpy(dtype=str)[center_rows], {
        'source': 'processed metadata with documented evaluation ordering',
        'cell_type_column': label_column,
        'processed_csv_path': str(csv_path),
        'target_alignment_genes_checked': len(sentinels),
        'matches_float32': matches_float32,
        'matches_documented_half_roundtrip': matches_half_roundtrip,
    }


def safe_tensor_load(path):
    # Permit only NumPy array/dtype constructors used in the documented saved metadata.
    core = getattr(np, '_core', np.core)
    allowed = [np.ndarray, np.dtype,
               (core.multiarray._reconstruct, 'numpy.core.multiarray._reconstruct'),
               (core.multiarray._reconstruct, 'numpy._core.multiarray._reconstruct')]
    for name in ['Int32DType', 'Int64DType', 'Float16DType', 'Float32DType', 'Float64DType', 'StrDType', 'BoolDType']:
        if hasattr(np.dtypes, name):
            allowed.append(getattr(np.dtypes, name))
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location='cpu', mmap=True, weights_only=True)


def run_audit(samples=None):
    samples = tuple(EXPECTED_HASHES) if samples is None else tuple(samples)
    if not samples or len(set(samples)) != len(samples) or not set(samples).issubset(EXPECTED_HASHES):
        raise ValueError('Invalid requested samples')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')
    folder = Path('/content') / ('R1_Minor9_EV_audit_' + stamp)
    folder.mkdir(exist_ok=False)
    manifest = {'training_performed': False, 'inference_performed': False,
                'original_files_modified': False, 'source': 'existing prediction/target tensors',
                'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'torch': torch.__version__},
                'tolerances': {'MSE_atol': 1e-6, 'MSE_rtol': 1e-5, 'EV_atol_percentage_points': 1e-3, 'EV_rtol': 2e-5},
                'requested_samples': list(samples),
                'verification_scope': 'all_verified refers only to requested_samples',
                'samples': []}
    for sample in samples:
        entry = {'sample': sample, 'status': 'not_verified'}
        try:
            table_path = TABLE_DIR / f'{sample}_celltype_performance.csv'
            raw = table_path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != EXPECTED_HASHES[sample]:
                raise ValueError('Summary CSV differs from the uploaded version; no automatic substitution')
            reference = pd.read_csv(table_path)
            tensor_path = TENSOR_DIR / f'edges_{sample}.pth'
            entry.update(table_sha256=EXPECTED_HASHES[sample], tensor_path=str(tensor_path))
            if not tensor_path.is_file():
                raise FileNotFoundError('Historical saved tensor not found: ' + str(tensor_path))
            entry['tensor_bytes'] = tensor_path.stat().st_size
            print(f'Checking {sample}: reading saved tensors only (no model loading).', flush=True)
            saved = safe_tensor_load(tensor_path)
            if not isinstance(saved, dict) or not all(k in saved for k in ['y', 'y_pred']):
                raise ValueError('Saved file does not contain the documented y/y_pred arrays')
            pred = saved['y_pred'].detach().cpu().numpy()
            target = saved['y'].detach().cpu().numpy()
            if pred.shape != target.shape or target.ndim != 2 or target.shape[1] != 1000:
                raise ValueError('Unexpected saved prediction shape')
            if len(target) != int(reference.n_cells.sum()):
                raise ValueError(f'Saved tensor has {len(target)} cells; uploaded table has {reference.n_cells.sum()}')
            if 'cell_type_name' in saved:
                labels = np.asarray([v if isinstance(v, str) else v[0] for v in saved['cell_type_name']], dtype=str)
                entry['label_alignment'] = {'source': 'cell_type_name stored with the prediction arrays'}
            else:
                labels, entry['label_alignment'] = recover_labels_from_processed(sample, target)
            if len(labels) != len(target):
                raise ValueError('Label length differs from prediction length')
            recomputed = accumulate_metrics(pred, target, labels)
            comparison = compare_table(reference, recomputed)
            comparison.to_csv(folder / f'{sample}_MSE_EV_comparison.csv', index=False)
            valid = bool(comparison[['count_matches', 'MSE_matches', 'EV_matches_zero_reference']].all().all())
            entry.update(status='verified' if valid else 'metric_mismatch', n_cells=len(target), n_genes=target.shape[1],
                         all_celltype_counts_match=bool(comparison.count_matches.all()),
                         all_MSE_match=bool(comparison.MSE_matches.all()),
                         all_EV_match_zero_reference=bool(comparison.EV_matches_zero_reference.all()))
            print(sample, entry['status'], flush=True)
            del saved, pred, target, labels
            gc.collect()
        except Exception as exc:
            entry.update(error=str(exc), traceback=traceback.format_exc())
            print(sample, 'not verified:', str(exc), flush=True)
        manifest['samples'].append(entry)
    manifest['all_verified'] = len(manifest['samples']) == len(samples) and all(x['status'] == 'verified' for x in manifest['samples'])
    (folder / 'audit.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    archive = folder.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for p in folder.iterdir():
            if p.is_file():
                z.write(p, arcname=p.name)
    print('All verified:', manifest['all_verified'])
    print('Only small audit tables are exported. Original arrays remain unchanged.')
    return archive
