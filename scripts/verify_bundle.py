"""Verify bundle bytes, Python syntax, seed statistics and fixed index membership."""
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]

def main():
    manifest = json.loads((ROOT/'MANIFEST.json').read_text())
    for rel, expected in manifest['sha256'].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == expected, rel
    sources = list(ROOT.rglob('*.py'))
    for p in sources:
        ast.parse(p.read_text(), filename=str(p))
    rows = list(csv.DictReader((ROOT/'figures/Figure2_source_values.csv').open()))
    assert len(rows) == 30
    for r in rows:
        values = [float(r[f'seed{s}']) for s in [123,456,789]]
        assert math.isclose(statistics.mean(values),float(r['mean']),abs_tol=1e-12)
        assert math.isclose(statistics.stdev(values),float(r['sample_SD']),abs_tol=1e-12)
    for ds in ['SEA_AD','Mouse']:
        d = json.loads((ROOT/f'splits/{ds}_inductive_split_indices.json').read_text())
        keys = ['train_indices','val_indices','test_indices']
        flat = [i for k in keys for i in d[k]]
        assert len(set(flat)) == len(flat) and set(flat) == set(range(len(flat)))
        mapping = {i: label for k,label in zip(keys,['train','validation','test']) for i in d[k]}
        audits = list(csv.DictReader((ROOT/f'splits/{ds}_inductive_split_indices_samples.csv').open()))
        cursor = 0
        for r in audits:
            start, end = int(r['global_start']), int(r['global_end_exclusive'])
            assert start == cursor and end-start == int(r['eligible_cells'])
            assert all(mapping[i] == r['split'] for i in range(start,end))
            cursor = end
        assert cursor == len(flat)
    print(json.dumps({'integrity': 'passed', 'manifest_files': len(manifest['sha256']),
                      'python_syntax_files': len(sources), 'Figure2_seed_statistics': 'passed',
                      'split_index_membership': 'passed', 'full_training_rerun': False},indent=2))

if __name__ == '__main__':
    main()
