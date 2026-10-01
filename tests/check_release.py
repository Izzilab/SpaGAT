"""Check command staging and protocol boundaries without training or real data."""
from pathlib import Path
import ast
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    cases = 0
    with tempfile.TemporaryDirectory(prefix='spagat_release_check_') as folder:
        fixture = Path(folder).resolve()
        data = fixture / 'data/processed'
        data.mkdir(parents=True)
        # These are path-only placeholders. No claim is made about a data loader run.
        for name in ['genes.pth', 'ligands.pth']:
            (data.parent / name).write_text('path-only placeholder')
        (data / 'example_TypeExp.npz').write_text('path-only placeholder')
        for dataset in ['SEA_AD', 'Mouse']:
            routes = [('SpaGAT',v) for v in ['full','no_distance','uniform_routing','matched_random_edge']]
            routes += [(m,'full') for m in ['GAT','GITIII','SPICE-adapted','LightGBM']]
            original = json.loads((ROOT/f'splits/{dataset}_inductive_split_indices.json').read_text())
            for method, variant in routes:
                out = fixture / f'{dataset}_{method}_{variant}'
                base = [sys.executable, str(ROOT/'scripts/run_benchmark.py'),
                        '--dataset', dataset, '--method', method, '--variant', variant,
                        '--seed', '456', '--data-dir', str(data), '--output-dir', str(out)]
                dry = subprocess.run(base+['--dry-run'], capture_output=True, text=True)
                assert dry.returncode == 0, dry.stderr
                assert not out.exists()
                prep = subprocess.run(base+['--prepare-only'], capture_output=True, text=True)
                assert prep.returncode == 0, prep.stderr
                runtime = out/'runtime'
                split = json.loads((runtime/f'splits/{dataset}_inductive_split_indices.json').read_text())
                for key in ['train_indices','val_indices','test_indices']:
                    assert split[key] == original[key]
                assert Path(split['processed_dir']) == data
                assert (runtime/f'splits/{dataset}_inductive_split_indices_samples.csv').is_file()
                command = json.loads((out/'launch.json').read_text())['command']
                assert Path(command[1]).is_file()
                assert not (out/'results').exists()
                if method == 'SpaGAT':
                    assert command[command.index('--seeds')+1] == '456'
                    for name in ['train.py','control_support.py','ablation_support.py']:
                        assert (runtime/'scripts'/name).read_bytes() == (ROOT/'scripts'/name).read_bytes()
                else:
                    assert command[command.index('--seed')+1] == '456'
                if variant == 'matched_random_edge':
                    mapping = Path(command[command.index('--edge-gene-map')+1])
                    assert mapping.is_relative_to(runtime)
                    assert mapping.read_bytes() == (ROOT/f'configs/matched_genes/{dataset}/random_set_1.json').read_bytes()
                if variant == 'uniform_routing':
                    assert command[command.index('--routing-mode')+1] == 'uniform'
                    assert command[command.index('--n-programs')+1] == '4'
                refused = subprocess.run(base+['--prepare-only'], capture_output=True, text=True)
                assert refused.returncode != 0 and 'Refusing existing output' in refused.stderr
                cases += 1

        missing = fixture/'not_present'
        out = fixture/'missing_input_output'
        failed = subprocess.run([sys.executable,str(ROOT/'scripts/run_benchmark.py'),
            '--dataset','Mouse','--method','SpaGAT','--seed','123',
            '--data-dir',str(missing),'--output-dir',str(out),'--prepare-only'],capture_output=True,text=True)
        assert failed.returncode != 0 and not out.exists()

        # Exercise the actual standalone split-validation function without importing torch.
        tree = ast.parse((ROOT/'scripts/train.py').read_text())
        fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_and_validate_shared_split')
        namespace = {'json':json,'os':os}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<split validation>','exec'),namespace)
        split_path = fixture/'small_split.json'
        split_path.write_text(json.dumps({'train_indices':[0],'val_indices':[1],'test_indices':[2]}))
        namespace[fn.name](split_path,3)
        split_path.write_text(json.dumps({'train_indices':[0,1],'val_indices':[2]}))
        try:
            namespace[fn.name](split_path,3)
        except ValueError:
            pass
        else:
            raise AssertionError('Validation-only split accepted by root training entry')

    # Maps are fixed and must retain their original integrity checks.
    spec = importlib.util.spec_from_file_location('release_control',ROOT/'scripts/control_support.py')
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)
    for dataset in ['SEA_AD','Mouse']:
        d = ROOT/'configs/matched_genes'/dataset
        mapping = json.loads((d/'random_set_1.json').read_text())
        genes = [r['gene'] for r in csv.DictReader((d/'training_gene_statistics.csv').open())]
        ligands = ([[g] for g in mapping['original_genes']], [[0] for _ in mapping['original_genes']])
        control.validate_map(mapping,genes,ligands)
    print(json.dumps({'prepared_routes':cases,'dry_runs':cases,
        'matching_control_dependencies_staged':True,'fixed_split_indices_preserved':True,
        'old_two_way_split_rejected':True,'existing_outputs_preserved':True,
        'missing_inputs_rejected_before_writes':True,'frozen_maps_checked':True,
        'training_started':False,'real_input_loading_tested':False},indent=2))


if __name__ == '__main__':
    main()
