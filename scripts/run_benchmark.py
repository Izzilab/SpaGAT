"""Launch one recorded benchmark/ablation run using existing model-ready data."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
METHODS = {'SpaGAT': None, 'GAT': 'GAT', 'GITIII': 'GITIII_official',
           'SPICE-adapted': 'SPICE_adapted', 'LightGBM': 'LightGBM'}

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['SEA_AD','Mouse'], required=True)
    p.add_argument('--method', choices=list(METHODS), required=True)
    p.add_argument('--seed', type=int, choices=[123,456,789], required=True)
    p.add_argument('--variant', choices=['full','no_distance','matched_random_edge','uniform_routing'], default='full')
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='Check paths and print the planned command without creating files or training.')
    mode.add_argument('--prepare-only', action='store_true', help='Stage the run and fixed split without starting training.')
    return p.parse_args()

def main():
    a = parse_args()
    data = a.data_dir.resolve()
    out = a.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f'Refusing existing output directory: {out}')
    if a.method != 'SpaGAT' and a.variant != 'full':
        raise ValueError('Component variants apply only to SpaGAT.')
    for p in [data, data.parent/'genes.pth', data.parent/'ligands.pth']:
        if not p.exists():
            raise FileNotFoundError(f'Required external model-ready input missing: {p}')
    if not list(data.glob('*_TypeExp.npz')):
        raise FileNotFoundError('No section baseline .npz files found. Raw input CSVs alone are insufficient.')
    runtime = out/'runtime'
    split_name = f'{a.dataset}_inductive_split_indices.json'
    method = METHODS[a.method]
    if method is None:
        cmd = [sys.executable, str(runtime/'scripts/train.py'), '--data-dir', str(data),
               '--shared-split', str(runtime/'splits'/split_name), '--output-dir', str(out/'results'),
               '--seeds', str(a.seed), '--max-epochs', '10', '--patience', '5', '--batch-size', '32',
               '--component-ablation', a.variant, '--n-programs', '4',
               '--routing-mode', 'uniform' if a.variant == 'uniform_routing' else 'program']
        if a.variant == 'matched_random_edge':
            cmd += ['--edge-gene-map', str(runtime/'configs/edge_gene_map.json')]
    else:
        cmd = [sys.executable, str(runtime/'baselines'/f'{method}.py'), '--revision-dir', str(runtime),
               '--output-dir', str(out/'results'), '--seed', str(a.seed)]
    print(json.dumps({'dataset': a.dataset, 'method': a.method, 'seed': a.seed,
                     'variant': a.variant, 'command': cmd, 'dry_run': a.dry_run, 'prepare_only': a.prepare_only}, indent=2))
    if a.dry_run:
        return
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(ROOT/'spagat', runtime/'spagat', ignore=shutil.ignore_patterns('__pycache__'))
    (runtime/'scripts').mkdir()
    for filename in ['train.py', 'control_support.py', 'ablation_support.py']:
        shutil.copyfile(ROOT/'scripts'/filename, runtime/'scripts'/filename)
    if a.variant == 'matched_random_edge':
        (runtime/'configs').mkdir()
        shutil.copyfile(ROOT/'configs/matched_genes'/a.dataset/'random_set_1.json', runtime/'configs/edge_gene_map.json')
    (runtime/'splits').mkdir()
    split = json.loads((ROOT/'splits'/split_name).read_text())
    split['processed_dir'] = str(data)
    split['split_manifest'] = str(ROOT/'data_records/counts_by_section.csv')
    (runtime/'splits'/split_name).write_text(json.dumps(split))
    sample_name = split_name.replace('.json','_samples.csv')
    shutil.copyfile(ROOT/'splits'/sample_name, runtime/'splits'/sample_name)
    loader_hash = hashlib.sha256((runtime/'spagat/dataloader.py').read_bytes()).hexdigest()
    (runtime/'revision_manifest.json').write_text(json.dumps({'files_sha256': {'spagat/dataloader.py': loader_hash}}))
    if method is not None:
        (runtime/'baselines').mkdir()
        shutil.copyfile(ROOT/'baselines'/a.dataset/f'{method}.py', runtime/'baselines'/f'{method}.py')
    (out/'launch.json').write_text(json.dumps({'command': cmd, 'source_bundle': str(ROOT),
                                            'data_dir': str(data)}, indent=2))
    if not a.prepare_only:
        subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
