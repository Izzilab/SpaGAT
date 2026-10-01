"""A few real-data optimizer steps; validates execution, not benchmark accuracy."""
import argparse
import gc
import json
from pathlib import Path
import platform

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from data_assets import ROOT, validate_inputs
from ablation_support import build_model
from control_support import apply_gene_map
from spagat.gene_program_model import SpaGP_Loss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    reports = []
    for ds in ['Mouse', 'SEA_AD']:
        data, split, _, audit = validate_inputs(a.data_root/ds/'data/processed', ds)
        batch = next(iter(DataLoader(Subset(data, split['train_indices'][:4]), batch_size=4)))
        for variant in ['full', 'no_distance', 'uniform_routing', 'matched_random_edge']:
            torch.manual_seed(123)
            if variant == 'uniform_routing':
                from spagat.gene_program_model import SpaGP
                model = SpaGP(data.genes, data.interactions, node_dim=256, edge_dim=48,
                              num_heads=2, n_layers=1, att_dim=8, n_programs=4,
                              program_aware=True, program_token_dim=32,
                              message_decoder='free', routing_mode='uniform')
            else:
                model = build_model(data.genes, data.interactions, variant)
            if variant == 'matched_random_edge':
                mapping = json.loads((ROOT/'configs/matched_genes'/ds/'random_set_1.json').read_text())
                apply_gene_map(model, data.genes, data.interactions, mapping)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.99, .999))
            loss_fn = SpaGP_Loss(data.genes, data.interactions, lambda_orth=.1, lambda_sparse=.01)
            loss_values = []
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(batch), batch['y'])[0]
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite loss on actual training inputs')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError('Nonfinite gradients')
                optimizer.step()
                loss_values.append(float(loss.detach()))
            model.eval()
            with torch.no_grad():
                before = model(batch)[0].clone()
            checkpoint = a.output_dir/f'{ds}_{variant}_smoke.pth'
            torch.save(model.state_dict(), checkpoint)
            # Exercise serialization/reload independently of optimizer state.
            model.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
            with torch.no_grad():
                after = model(batch)[0]
            assert torch.equal(before, after)
            reports.append({'dataset': ds, 'variant': variant, 'training_receivers': 4,
                            'optimizer_steps': 2, 'finite_loss_and_gradients': True,
                            'checkpoint_roundtrip_equal': True, 'losses': loss_values})
            print('PASSED', ds, variant, flush=True)
            del model, optimizer
        del data, batch
        gc.collect()
    report = {'scope': 'Real-input CPU smoke check, not full-epoch or paper-performance reproduction.',
              'system': platform.platform(), 'python': platform.python_version(),
              'torch': torch.__version__, 'numpy': np.__version__, 'runs': reports,
              'full_benchmark_retrained': False}
    (a.output_dir/'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
