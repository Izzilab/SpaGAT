"""Author-side, verified export of original brain inputs and six SpaGAT checkpoints.

No retraining, source-data modification, recursive Drive scan, or automatic upload.
Requires the saved author split/prediction/checkpoint paths in the repository.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from data_assets import ROOT, array_sha, file_sha, validate_inputs, build_model


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def pack_group(items, dest, label, max_bytes):
    """Independent ZIP parts, bounded by uncompressed bytes; never split a file."""
    parts, batch, size = [], [], 0
    batches = []
    for row in items:
        if row['bytes'] > max_bytes:
            raise ValueError(f"Single file exceeds ZIP part limit: {row['path']}")
        if batch and size+row['bytes'] > max_bytes:
            batches.append(batch)
            batch, size = [], 0
        batch.append(row)
        size += row['bytes']
    if batch:
        batches.append(batch)
    for i, rows in enumerate(batches, 1):
        target = dest/f'{label}.part{i:02d}.zip'
        print(f'Packaging {target.name} ({len(rows)} files)', flush=True)
        with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
            for row in rows:
                source = Path(row['_source'])
                # Check again immediately before packing; never mix changed files.
                if source.stat().st_size != row['bytes'] or file_sha(source) != row['sha256']:
                    raise ValueError(f'File changed after audit: {source}')
                archive.write(source, row['path'])
        # Reading every member checks both ZIP CRCs and archived SHA-256s.
        import hashlib
        with zipfile.ZipFile(target) as archive:
            for row in rows:
                h = hashlib.sha256()
                with archive.open(row['path']) as handle:
                    for block in iter(lambda: handle.read(8*1024*1024), b''):
                        h.update(block)
                if h.hexdigest() != row['sha256']:
                    raise ValueError('Archive member verification failed')
        parts.append({'file': target.name, 'bytes': target.stat().st_size,
                      'sha256': file_sha(target), 'members': [r['path'] for r in rows], 'public_url': ''})
    return parts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    p.add_argument('--part-mib', type=int, default=1024)
    a = p.parse_args()
    if not 64 <= a.part_mib <= 1800:
        raise ValueError('Use a part limit between 64 and 1800 MiB')
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    all_records = json.loads((ROOT/'provenance/brain_prediction_records.json').read_text())
    weights = json.loads((ROOT/'assets/checkpoint_manifest.json').read_text())
    files, checks = [], []
    status = {'complete': False, 'uploaded': False, 'training_performed': False}
    write_json(out/'status.json', status)
    try:
        for dataset in ['Mouse', 'SEA_AD']:
            records = [r for r in all_records if r['dataset'] == dataset and r['model'] == 'SpaGAT']
            if sorted(r['seed'] for r in records) != [123, 456, 789]:
                raise ValueError('Expected exactly three full-model references')
            original_split = json.loads(Path(records[0]['split_path']).read_text())
            print('Checking original data:', dataset, flush=True)
            data, split, inputs, report = validate_inputs(original_split['processed_dir'], dataset)
            for key in ['train_indices', 'val_indices', 'test_indices']:
                if original_split[key] != split[key]:
                    raise ValueError('Author/public fixed partitions differ')
            for source in inputs:
                rel = source.relative_to(Path(original_split['processed_dir']).parent)
                files.append({'path': f'{dataset}/data/{rel.as_posix()}', 'bytes': source.stat().st_size,
                              'sha256': file_sha(source), 'group': dataset, '_source': str(source)})
            batch = next(iter(DataLoader(Subset(data, split['test_indices'][:32]), batch_size=32)))
            batch = {k: v.to(a.device) for k, v in batch.items()}
            report['checkpoints'] = []
            for rec in sorted(records, key=lambda r: r['seed']):
                row = next(w for w in weights if w['dataset'] == dataset and w['seed'] == rec['seed'])
                checkpoint_path = Path(row['author_drive_path'])
                prediction_path = Path(rec['npz'])
                if file_sha(checkpoint_path) != row['sha256']:
                    raise ValueError('Original checkpoint hash differs; no replacement selected')
                saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                if saved['seed'] != rec['seed']:
                    raise ValueError('Checkpoint seed mismatch')
                for key in ['train_indices', 'val_indices', 'test_indices']:
                    if saved['shared_split'][key] != split[key]:
                        raise ValueError('Checkpoint fixed partition mismatch')
                with np.load(prediction_path, allow_pickle=False) as z:
                    if z['genes'].astype(str).tolist() != list(map(str, data.genes)):
                        raise ValueError('Prediction/input gene order differs')
                    if not np.array_equal(z['test_indices'], split['test_indices']):
                        raise ValueError('Saved test receiver order differs')
                    if array_sha(z['target']) != rec['target_sha256_float32']:
                        raise ValueError('Saved test target hash differs')
                    if array_sha(z['prediction']) != rec['prediction_sha256_float32']:
                        raise ValueError('Saved prediction hash differs')
                    reference = z['prediction'][:32].copy()
                model = build_model(data, saved).to(a.device).eval()
                with torch.no_grad():
                    predicted = model(batch)[0].cpu().numpy()
                error = float(np.max(np.abs(predicted-reference)))
                if not np.allclose(predicted, reference, rtol=1e-4, atol=1e-4):
                    raise ValueError(f'{dataset}/{rec["seed"]}: prediction check failed (max error {error}); no archive published')
                report['checkpoints'].append({'seed': rec['seed'], 'sha256': row['sha256'],
                    'first_32_predictions_match': True, 'maximum_absolute_error': error,
                    'rtol': 1e-4, 'atol': 1e-4, 'device': a.device})
                for source, rel in [(checkpoint_path, f'checkpoints/{dataset}_SpaGAT_seed{rec["seed"]}.pth'),
                                    (prediction_path, f'reference_predictions/{dataset}_SpaGAT_seed{rec["seed"]}.npz')]:
                    files.append({'path': rel, 'bytes': source.stat().st_size, 'sha256': file_sha(source),
                                  'group': 'checkpoints_and_predictions', '_source': str(source)})
                del saved, model
                gc.collect()
                if a.device == 'cuda':
                    torch.cuda.empty_cache()
                print('Verified checkpoint:', dataset, rec['seed'], flush=True)
            checks.append(report)
            write_json(out/f'{dataset}_validation.json', report)
            del data, batch
            gc.collect()
        # Only after BOTH datasets and all six weights pass, produce archives.
        public_files = [{k: v for k, v in f.items() if k != '_source'} for f in files]
        write_json(out/'asset_files.json', public_files)
        parts = []
        for group in ['Mouse', 'SEA_AD', 'checkpoints_and_predictions']:
            parts.extend(pack_group([f for f in files if f['group'] == group], out,
                                    f'SpaGAT_{group}', a.part_mib*1024*1024))
        write_json(out/'download_manifest.json', {'archives': parts, 'files': public_files,
                    'unpack_into': 'EXTERNAL_DATA', 'published': False})
        status.update(complete=True, datasets=checks, archives=parts,
                      note='Inputs and six SpaGAT checkpoints audited; public download links still need depositing.')
        write_json(out/'status.json', status)
    except Exception as exc:
        status.update(error_type=type(exc).__name__, error=str(exc))
        write_json(out/'status.json', status)
        raise
    finally:
        # Small report is separate from large data: practical for review/upload.
        with zipfile.ZipFile(out.with_name(out.name+'_audit.zip'), 'x', zipfile.ZIP_DEFLATED) as z:
            for path in sorted(out.glob('*.json')):
                z.write(path, path.name)
        print('Audit report:', out.with_name(out.name+'_audit.zip'), flush=True)
    print('COMPLETE. Original files unchanged. No training or automatic publication.', flush=True)
    print('Asset folder:', out, flush=True)


if __name__ == '__main__':
    main()
