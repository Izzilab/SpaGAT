SEED = 123
"""Mouse SPICE-adapted revision: fixed splits, validation PCC selection, one final test."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as tnn
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
DEFAULT_REVISION = '/content/drive/MyDrive/SpaGAT_revision/inductive_training/20260922_104353_552891_UTC_f291de00'

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

class SpiceAdaptedMouseNoLog(nn.Module):

    def __init__(self, gene_dim=254, num_celltypes=24, hidden_dim=128, type_dim=16):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.expr_proj = nn.Sequential(nn.Linear(gene_dim, hidden_dim), nn.GELU())
        self.type_emb = nn.Embedding(num_celltypes, type_dim)
        node_dim = hidden_dim + type_dim
        self.gcn = GCNConv(node_dim, hidden_dim, add_self_loops=True, normalize=True)
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, gene_dim))

    def forward(self, x, celltypes):
        B, K, G = x.shape
        h_expr = self.expr_proj(x)
        h_type = self.type_emb(celltypes)
        h = torch.cat([h_expr, h_type], dim=-1)
        h = h.reshape(B * K, -1)
        src_all = []
        dst_all = []
        for b in range(B):
            offset = b * K
            center = offset
            neighbors = offset + torch.arange(1, K, device=x.device, dtype=torch.long)
            centers = torch.full((K - 1,), center, device=x.device, dtype=torch.long)
            src_all.append(neighbors)
            dst_all.append(centers)
            src_all.append(centers)
            dst_all.append(neighbors)
        edge_index = torch.stack([torch.cat(src_all), torch.cat(dst_all)], dim=0)
        h = self.gcn(h, edge_index)
        h = torch.relu(h)
        center_idx = torch.arange(B, device=x.device) * K
        h_center = h[center_idx]
        return self.decoder(h_center)

def prepare_batch(batch, device):
    x, ct, y = (batch['x'], batch['cell_types'], batch['y'])
    if x.ndim != 3 or x.shape[1] != 50 or ct.shape != x.shape[:2]:
        raise ValueError('Expected aligned 50-cell neighborhoods')
    if 'neighbor_mask' in batch and (not bool(torch.all(batch['neighbor_mask'] == 1))):
        raise ValueError('Expected all 50 nodes valid')
    if not torch.equal(y, x[:, 0, :]):
        raise ValueError('Center residual differs from target')
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError('Nonfinite data')
    x = x.masked_fill((ct == ct[:, :1]).unsqueeze(-1), 0)
    return (x.to(device), ct.to(device), y.to(device))

def predict(model, dataset, indices, device, batch_size, label):
    model.eval()
    predictions, targets = ([], [])
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        for step, batch in enumerate(loader, 1):
            x, ct, y = prepare_batch(batch, device)
            predictions.append(model(x, ct).cpu().numpy())
            targets.append(y.cpu().numpy())
            if step == 1 or step % 200 == 0 or step == len(loader):
                print(f'{label}: batch {step}/{len(loader)}', flush=True)
    return (np.concatenate(predictions), np.concatenate(targets))

def run_experiment(dataset, split, genes, out, model, device, max_epochs=10, patience=5, batch_size=32):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    import shutil
    shutil.copyfile(__file__, out / 'runner_source.py')
    write_json(out / 'shared_split.json', split)
    config = dict(dataset='Mouse', model='SPICE-adapted', seed=SEED, max_epochs=max_epochs, patience=patience, batch_size=batch_size, learning_rate=0.0001, optimizer='AdamW', weight_decay=0.01, betas=[0.9, 0.999], hidden_dim=128, type_dim=16, add_self_loops=True, normalize=True, edges='bidirectional center-neighbor star; GCN adds self-loops', input='residual expression plus learned type embedding; center and homotypic residuals zeroed; no baseline or distance features added', selection='validation median_gene_pcc; strict improvement; first epoch wins ties', epoch_numbering='zero-based', monitored_metric='median_gene_pcc', target_space='residual relative to training-derived cell-type baseline', pcc='float32 centered; denominator product clamped at 1e-12; torch.median; constant columns score zero', ev='100*(1-MSE/MSE0)', parameter_count=sum((p.numel() for p in model.parameters())), device=str(device), gpu=torch.cuda.get_device_name() if device.type == 'cuda' else None, versions=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__))
    if 'torch_geometric' in sys.modules:
        config['versions']['torch_geometric'] = sys.modules['torch_geometric'].__version__
    write_json(out / 'run_config.json', config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    train_loader = DataLoader(Subset(dataset, split['train_indices']), batch_size=batch_size, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(SEED))
    best_score, best_epoch, stale = (-float('inf'), None, 0)
    records = []
    checkpoint = out / f'spice_adapted_seed{SEED}_best.pth'
    started = time.perf_counter()
    for epoch in range(max_epochs):
        model.train()
        total, count = (0.0, 0)
        for step, batch in enumerate(train_loader, 1):
            x, ct, y = prepare_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, ct)
            loss = tnn.functional.mse_loss(prediction, y)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss; test will not be evaluated')
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
            count += len(y)
            if step == 1 or step % 200 == 0 or step == len(train_loader):
                print(f'Epoch {epoch}: train batch {step}/{len(train_loader)}, MSE {total / count:.6f}', flush=True)
        pred, target = predict(model, dataset, split['val_indices'], device, batch_size, 'validation')
        metrics, _ = evaluate_arrays(pred, target)
        score = metrics['median_gene_pcc']
        if not np.isfinite(score):
            raise ValueError('Nonfinite validation PCC; test will not be evaluated')
        improved = score > best_score
        if improved:
            best_score, best_epoch, stale = (score, epoch, 0)
            torch.save(dict(model_state_dict=model.state_dict(), best_epoch=epoch, best_validation_pcc=score), checkpoint)
        else:
            stale += 1
        records.append(dict(epoch=epoch, train_mse=total / count, validation_median_pcc=score, validation_mse=metrics['mse'], best_epoch=best_epoch, elapsed_seconds=time.perf_counter() - started))
        pd.DataFrame(records).to_csv(out / 'training_history.csv', index=False)
        print(f'Epoch {epoch} completed: validation median PCC={score:.6f}; best epoch={best_epoch}', flush=True)
        del pred, target
        if stale >= patience:
            print('Validation early stopping.', flush=True)
            break
    if best_epoch is None:
        raise RuntimeError('No valid checkpoint; test will not be evaluated')
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved['model_state_dict'])
    print(f"Loaded best checkpoint (epoch {saved['best_epoch']}). Starting final test once.", flush=True)
    pred, target = predict(model, dataset, split['test_indices'], device, batch_size, 'test')
    metrics, pcc = evaluate_arrays(pred, target)
    metrics.update(dataset='Mouse', model='SPICE-adapted', seed=SEED, best_epoch=saved['best_epoch'], best_validation_pcc=saved['best_validation_pcc'], checkpoint=checkpoint.name, runtime_seconds=time.perf_counter() - started, target_space=config['target_space'])
    np.savez_compressed(out / 'test_predictions.npz', prediction=pred, target=target, test_indices=np.asarray(split['test_indices'], dtype=np.int64), genes=np.asarray(genes, dtype=str))
    pd.DataFrame({'gene': genes, 'PCC': pcc}).to_csv(out / 'test_per_gene_PCC.csv', index=False)
    write_json(out / 'test_metrics.json', metrics)
    pd.DataFrame([metrics]).to_csv(out / 'test_metrics.csv', index=False)
    pd.DataFrame([dict(seed=SEED, best_epoch=best_epoch, best_validation_pcc=best_score, monitored_metric='median_gene_pcc', parameter_count=config['parameter_count'], max_epochs=max_epochs, patience=patience)]).to_csv(out / 'best_runs.csv', index=False)
    print(json.dumps(metrics, indent=2), flush=True)
    print('Completed; saved to:', out, flush=True)
    return metrics

def main():
    global SEED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision-dir', default=DEFAULT_REVISION)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    SEED = args.seed
    revision = Path(args.revision_dir)
    out = Path(args.output_dir) if args.output_dir else revision / 'baseline_results' / f'Mouse_SPICE_adapted_seed{SEED}'
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite: {out}')
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device, flush=True)
    from torch_geometric.nn import GCNConv
    split_path = revision / 'splits' / 'Mouse_inductive_split_indices.json'
    metadata = json.loads(split_path.read_text())
    if tuple((len(metadata[k]) for k in ('train_indices', 'val_indices', 'test_indices'))) != (122886, 36193, 117257):
        raise ValueError('Unexpected Mouse split counts')
    loader_path = revision / 'spagat' / 'dataloader.py'
    manifest = json.loads((revision / 'revision_manifest.json').read_text())
    digest = hashlib.sha256(loader_path.read_bytes()).hexdigest()
    if digest != manifest['files_sha256']['spagat/dataloader.py']:
        raise ValueError('Frozen dataloader hash mismatch')
    spec = importlib.util.spec_from_file_location('frozen_spagat_data', loader_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print('Loading verified processed data...', flush=True)
    dataset = module.SPAGAT_dataset(processed_dir=metadata['processed_dir'], num_neighbors=50)
    split = load_split(split_path, dataset, metadata['processed_dir'])
    genes = list(dataset.genes)
    if len(genes) != 254 or len(set(genes)) != 254:
        raise ValueError('Expected 254 unique genes')
    model = SpiceAdaptedMouseNoLog(gene_dim=len(genes), num_celltypes=len(dataset.cell_types_dict)).to(device)
    print('Split audit passed. Starting Mouse SPICE-adapted, seed 123, max 10 epochs, patience 5.', flush=True)
    run_experiment(dataset, split, genes, out, model, device)
    write_json(out / 'source_provenance.json', dict(revision=str(revision), split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(), dataloader_sha256=digest, runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
if __name__ == '__main__':
    main()
