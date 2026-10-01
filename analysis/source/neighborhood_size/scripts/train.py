"""SpaGAT inductive revision using a fixed train/validation/test split.

Select checkpoints on validation median gene-wise PCC only, then evaluate the
best checkpoint once on test per seed. Model and metric definitions are retained.
"""

import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from spagat.dataloader import SPAGAT_dataset
from spagat.gene_program_model import SpaGP, SpaGP_Loss


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def percentile_summary(tensor):
    """JSON-serializable distribution summary; None denotes unavailable q."""
    if tensor is None or tensor.numel() == 0:
        return None
    values = tensor.detach().float().flatten().cpu()
    quantiles = torch.quantile(values, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
    return {
        "mean": values.mean().item(),
        "std": values.std(unbiased=False).item(),
        "min": values.min().item(),
        "p05": quantiles[0].item(),
        "p25": quantiles[1].item(),
        "median": quantiles[2].item(),
        "p75": quantiles[3].item(),
        "p95": quantiles[4].item(),
        "max": values.max().item(),
        "positive_fraction": (values > 0).float().mean().item(),
        "negative_fraction": (values < 0).float().mean().item(),
    }


def gene_wise_pcc(prediction, target):
    prediction = prediction.float()
    target = target.float()
    prediction = prediction - prediction.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)
    denominator = torch.sqrt((prediction.square().sum(dim=0) * target.square().sum(dim=0)).clamp_min(1e-12))
    pcc = (prediction * target).sum(dim=0) / denominator
    return pcc.clamp(-1, 1)


def explained_variance_percent(prediction, target):
    residual_sum = (target - prediction).square().sum()
    total_sum = target.square().sum().clamp_min(1e-12)  # zero-residual reference; logging only
    return (100 * (1 - residual_sum / total_sum)).item()


def routing_statistics(attention_batches):
    """Compute conditional neighbor entropy and program-routing cosine similarity."""
    if not attention_batches:
        return None, None
    attention = torch.cat(attention_batches, dim=0).float()  # (cells, N-1, K)

    # The public tensor excludes the center self-edge.  Renormalize it so entropy
    # describes the conditional distribution over retained neighboring cells.
    conditional = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-12)
    entropy = -(conditional.clamp_min(1e-12) * conditional.clamp_min(1e-12).log()).sum(dim=1)
    entropy_per_program = entropy.mean(dim=0)

    # K=1 has a well-defined entropy but no pair of programs to compare.
    # Return N/A for routing similarity; these metrics are logging-only.
    if conditional.shape[-1] < 2:
        return entropy_per_program, None

    routing_vectors = conditional.permute(2, 0, 1).reshape(conditional.shape[-1], -1)
    routing_vectors = routing_vectors / routing_vectors.norm(dim=1, keepdim=True).clamp_min(1e-12)
    similarity = routing_vectors @ routing_vectors.T
    return entropy_per_program, similarity


def evaluate(model, loader, loss_function, device, return_predictions=False):
    model.eval()
    predictions, targets = [], []
    activations, coefficients, attention_batches = [], [], []
    loss_sum, steps = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            batch = {name: value.to(device) for name, value in batch.items()}
            model_output = model(batch)
            loss, _, _, _ = loss_function(model_output, batch["y"])
            prediction, info = model_output

            loss_sum += loss.item()
            steps += 1
            predictions.append(prediction.cpu())
            targets.append(batch["y"].cpu())
            if not return_predictions:
                activations.append(info["activations"].cpu())
            if not return_predictions and info.get("program_message_coefficients") is not None:
                coefficients.append(info["program_message_coefficients"].cpu())
            if not return_predictions and info.get("program_attention") is not None:
                attention_batches.append(info["program_attention"].cpu())

    prediction = torch.cat(predictions, dim=0)
    target = torch.cat(targets, dim=0)
    pcc = gene_wise_pcc(prediction, target)
    # Final test uses the same metric functions, in one inference pass. Avoid
    # retaining diagnostic attention tensors for the much larger held-out set.
    if return_predictions:
        return {
            "test_loss": loss_sum / steps,
            "test_mse": torch.mean((prediction - target).square()).item(),
            "test_mse_zero": torch.mean(target.square()).item(),
            "median_gene_pcc": torch.median(pcc).item(),
            "mean_gene_pcc": torch.mean(pcc).item(),
            "ev_percent": explained_variance_percent(prediction, target),
            "n_cells": int(target.shape[0]),
            "n_genes": int(target.shape[1]),
        }, prediction, target, pcc
    entropy_per_program, routing_similarity = routing_statistics(attention_batches)
    off_diagonal_similarity = None
    if routing_similarity is not None and routing_similarity.shape[0] > 1:
        mask = ~torch.eye(routing_similarity.shape[0], dtype=torch.bool)
        off_diagonal_similarity = routing_similarity[mask].mean().item()

    return {
        "val_loss": loss_sum / steps,
        "val_mse": torch.mean((prediction - target).square()).item(),
        "median_gene_pcc": torch.median(pcc).item(),
        "mean_gene_pcc": torch.mean(pcc).item(),
        "ev_percent": explained_variance_percent(prediction, target),
        "g_distribution": percentile_summary(torch.cat(activations, dim=0)),
        "q_distribution": percentile_summary(torch.cat(coefficients, dim=0) if coefficients else None),
        "attention_entropy_per_program": None if entropy_per_program is None else entropy_per_program.tolist(),
        "attention_entropy_mean": None if entropy_per_program is None else entropy_per_program.mean().item(),
        "routing_similarity": None if routing_similarity is None else routing_similarity.tolist(),
        "routing_similarity_off_diagonal_mean": off_diagonal_similarity,
    }


def load_and_validate_shared_split(split_path, dataset_size, data_dir=None):
    """Load a previously saved split and verify it is valid for this dataset."""
    with open(split_path, "r", encoding="utf-8") as handle:
        split = json.load(handle)
    keys = ("train_indices", "val_indices", "test_indices")
    if not isinstance(split, dict) or not set(keys).issubset(split):
        raise ValueError(
            f"{split_path} must contain train_indices, val_indices and test_indices."
        )

    if any(not isinstance(split[key], list) or not split[key] for key in keys):
        raise ValueError("Each split must be a non-empty JSON list.")
    all_indices = [index for key in keys for index in split[key]]
    if any(type(index) is not int for index in all_indices):
        raise ValueError("Shared split must contain non-empty integer index lists.")
    if any(index < 0 or index >= dataset_size for index in all_indices):
        raise ValueError(
            f"Shared split contains an index outside the current dataset range [0, {dataset_size - 1}]."
        )
    if len(set(all_indices)) != len(all_indices):
        raise ValueError("Shared split contains duplicate or overlapping indices.")
    if len(all_indices) != dataset_size:
        raise ValueError("Fixed split must cover the current dataset exactly once.")
    if data_dir is not None and "processed_dir" in split:
        if os.path.realpath(split["processed_dir"]) != os.path.realpath(data_dir):
            raise ValueError("--data-dir does not match the split's processed_dir.")
    for key, count_key in zip(keys, ("n_train", "n_validation", "n_test")):
        if count_key in split and split[count_key] != len(split[key]):
            raise ValueError(f"{count_key} does not match {key}.")
    return split  # Preserve the inductive split's provenance metadata.


def validate_sample_order(split_path, dataset, split):
    """Check the saved sample audit when present, without fetching any cells."""
    audit_path = os.path.splitext(split_path)[0] + "_samples.csv"
    if not os.path.isfile(audit_path):
        return
    audit = pd.read_csv(audit_path)
    if audit["sample"].tolist() != list(dataset.samples):
        raise ValueError("Dataset sample order differs from the saved split audit.")
    counts = np.asarray(dataset.meta_counts, dtype=np.int64)
    ends = np.cumsum(counts)
    starts = ends - counts
    for column, expected in (("eligible_cells", counts),
                             ("global_start", starts),
                             ("global_end_exclusive", ends)):
        if not np.array_equal(audit[column].to_numpy(), expected):
            raise ValueError(f"Dataset {column} differs from the saved split audit.")
    membership = np.empty(len(dataset), dtype=np.int8)
    labels = ("train", "validation", "test")
    for label, key in enumerate(("train_indices", "val_indices", "test_indices")):
        membership[split[key]] = label
    for row, start, end in zip(audit.itertuples(index=False), starts, ends):
        if row.split not in labels or not np.all(membership[start:end] == labels.index(row.split)):
            raise ValueError(f"Sample {row.sample} differs from the fixed split.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the SpaGAT model for residual cell-state prediction.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[123])
    parser.add_argument("--split-seed", type=int, default=123)
    

    parser.add_argument(
        "--shared-split", required=True,
        help="Fixed inductive JSON with train_indices, val_indices and test_indices.",
    )
    parser.add_argument(
    "--max-epochs", "--epochs", dest="max_epochs", type=int, default=10,
    help="Maximum training epochs (default: 10). --epochs is a compatibility alias.",
)
    parser.add_argument(
        "--patience", type=int, default=5,
        help="Stop after this many consecutive epochs without improving validation median gene-wise PCC.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-neighbors", type=int, default=50)
    parser.add_argument("--node-dim", type=int, default=256)
    parser.add_argument("--edge-dim", type=int, default=48)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--att-dim", type=int, default=8)
    parser.add_argument("--n-programs", type=int, default=4)
    parser.add_argument("--program-token-dim", type=int, default=32)
    
    parser.add_argument("--lambda-orth", type=float, default=0.1)
    parser.add_argument("--lambda-sparse", type=float, default=0.01)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--component-ablation", choices=["full","no_distance"], default="full")
    parser.add_argument("--routing-mode", choices=["program"], default="program")
    parser.add_argument("--neighbor-cache", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_epochs < 1:
        raise ValueError("--max-epochs must be at least 1.")
    if args.patience < 0:
        raise ValueError("--patience must be non-negative.")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Duplicate seeds would overwrite results.")
    
    expected = (4, "program")
    if (args.n_programs, args.routing_mode) != expected:
        raise ValueError("Ablation definition/configuration mismatch")
    device = torch.device(args.device)
    # Each invocation owns a new output directory; never reuse old results.
    os.makedirs(args.output_dir, exist_ok=False)
    data_dir = args.data_dir.rstrip("/")
    data_root = os.path.dirname(data_dir)
    genes = torch.load(os.path.join(data_root, "genes.pth"), weights_only=False)
    ligands_info = torch.load(os.path.join(data_root, "ligands.pth"), weights_only=False)

    # Construct once; deterministic index sets are reused for every selected
    # mode/seed.  A supplied split is loaded exactly instead of regenerated.
    if args.neighbor_cache is None:
        dataset = SPAGAT_dataset(processed_dir=data_dir, num_neighbors=args.num_neighbors)
    else:
        from neighborhood_support import dataset_for_k
        dataset = dataset_for_k(data_dir, args.num_neighbors, args.neighbor_cache)
    split = load_and_validate_shared_split(args.shared_split, len(dataset), data_dir)
    validate_sample_order(args.shared_split, dataset, split)
    with open(os.path.join(args.output_dir, "shared_split.json"), "w", encoding="utf-8") as handle:
        json.dump(split, handle)

    records = []
    best_run_records = []
   
    for seed in args.seeds:

            set_all_seeds(seed)

            model = SpaGP(
                genes=genes,
                ligands_info=ligands_info,
                node_dim=args.node_dim,
                edge_dim=args.edge_dim,
                num_heads=args.num_heads,
                n_layers=args.n_layers,
                att_dim=args.att_dim,
                n_programs=args.n_programs,
                program_aware=True,
                program_token_dim=args.program_token_dim,
                message_decoder="free",
                routing_mode=args.routing_mode,
            ).to(device)
            model.embeddings.component_ablation = args.component_ablation
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.99, 0.999))
            loss_function = SpaGP_Loss(
                genes, ligands_info, lambda_orth=args.lambda_orth, lambda_sparse=args.lambda_sparse
            ).to(device)

            # Each mode starts with the same generator seed, giving identical
            # shuffled batch order for every epoch under the same split.
            loader_generator = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(
                Subset(dataset, split["train_indices"]), batch_size=args.batch_size,
                shuffle=True, generator=loader_generator,
            )
            val_loader = DataLoader(Subset(dataset, split["val_indices"]), batch_size=args.batch_size, shuffle=False)
            parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            run_start = time.perf_counter()
            best_metric = float("-inf")
            best_epoch = None
            best_metrics = None
            epochs_without_improvement = 0

            for epoch in range(args.max_epochs):
                epoch_start = time.perf_counter()
                model.train()
                train_loss, train_steps = 0.0, 0
                for batch in train_loader:
                    batch = {name: value.to(device) for name, value in batch.items()}
                    optimizer.zero_grad(set_to_none=True)
                    model_output = model(batch)
                    loss, _, _, _ = loss_function(model_output, batch["y"])
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    train_steps += 1

                metrics = evaluate(model, val_loader, loss_function, device)
                monitored_metric = metrics["median_gene_pcc"]
                improved = np.isfinite(monitored_metric) and monitored_metric > best_metric
                if improved:
                    best_metric = monitored_metric
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    best_metrics = dict(metrics)
                else:
                    epochs_without_improvement += 1
                early_stopped = (
                    not improved and epochs_without_improvement >= args.patience
                )
                metrics.update(
                    {
                        "seed": seed,
                        "epoch": epoch,
                        "train_loss": train_loss / train_steps,
                        "parameter_count": parameter_count,
                        "epoch_runtime_seconds": time.perf_counter() - epoch_start,
                        "elapsed_runtime_seconds": time.perf_counter() - run_start,
                        "best_epoch": best_epoch,
                        "best_validation_pcc": best_metric,
                        "epochs_without_improvement": epochs_without_improvement,
                        "early_stopped": early_stopped,
                    }
                )

                records.append(metrics)

                pd.DataFrame(records).to_csv(
                    os.path.join(args.output_dir, "training_metrics.csv"),
                    index=False,
                )

                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": vars(args),
                    "model_config": {
                        "program_aware": True,
                        "message_decoder": "free",
                        "routing_mode": args.routing_mode,
                        "component_ablation": args.component_ablation,
                        "n_programs": args.n_programs,
                        "program_token_dim": args.program_token_dim,
                        "node_dim": args.node_dim,
                        "edge_dim": args.edge_dim,
                        "num_heads": args.num_heads,
                        "n_layers": args.n_layers,
                        "att_dim": args.att_dim,
                    },
                    "seed": seed,
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_validation_pcc": best_metric,
                    "early_stopped": early_stopped,
                    "epochs_without_improvement": epochs_without_improvement,
                    "parameter_count": parameter_count,
                    "shared_split": split,
                    "metrics": metrics,
                }

                checkpoint_stem = f"spagat_seed{seed}"

                checkpoint_path = os.path.join(
                    args.output_dir,
                    f"{checkpoint_stem}_latest.pth",
                )

                torch.save(
                    checkpoint,
                    checkpoint_path,
                )

                if improved:
                    best_checkpoint_path = os.path.join(
                        args.output_dir,
                        f"{checkpoint_stem}_best.pth",
                    )

                    torch.save(
                        checkpoint,
                        best_checkpoint_path,
                    )

                print(
                    f"SpaGAT seed={seed} epoch={epoch} "
                    f"MSE={metrics['val_mse']:.5f} "
                    f"PCC={metrics['median_gene_pcc']:.4f} "
                    f"EV={metrics['ev_percent']:.2f}% "
                    f"best_epoch={best_epoch} "
                    f"best_PCC={best_metric:.4f}"
                )

                if early_stopped:
                    print(
                        f"SpaGAT seed={seed} early stopping at epoch={epoch}: "
                        f"no validation median gene-wise PCC improvement for "
                        f"{epochs_without_improvement} epoch(s)."
                    )
                    break

            if best_epoch is None:
                raise RuntimeError("No finite validation PCC; test evaluation is not allowed.")
            best_checkpoint = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
            model.load_state_dict(best_checkpoint["model"])
            del best_checkpoint
            # Test is evaluated exactly once per seed, only after selection ends.
            test_loader = DataLoader(
                Subset(dataset, split["test_indices"]), batch_size=args.batch_size,
                shuffle=False, drop_last=False,
            )
            test_metrics, prediction, target, pcc = evaluate(
                model, test_loader, loss_function, device, return_predictions=True
            )
            test_metrics.update({
                "seed": seed, "best_epoch": best_epoch,
                "best_validation_pcc": best_metric,
                "checkpoint": os.path.basename(best_checkpoint_path),
                "target_space": "residual relative to training-derived cell-type baseline",
            })
            test_dir = os.path.join(args.output_dir, f"seed{seed}_test")
            os.makedirs(test_dir, exist_ok=False)
            with open(os.path.join(test_dir, "test_metrics.json"), "x", encoding="utf-8") as handle:
                json.dump(test_metrics, handle, indent=2)
            pd.DataFrame([test_metrics]).to_csv(os.path.join(test_dir, "test_metrics.csv"), index=False)
            pd.DataFrame({"gene": genes, "PCC": pcc.numpy()}).to_csv(
                os.path.join(test_dir, "test_per_gene_PCC.csv"), index=False,
            )
            np.savez_compressed(
                os.path.join(test_dir, "test_predictions.npz"),
                prediction=prediction.numpy(), target=target.numpy(),
                test_indices=np.asarray(split["test_indices"], dtype=np.int64),
                genes=np.asarray(genes, dtype=str),
            )
            print(f"SpaGAT seed={seed} final test PCC={test_metrics['median_gene_pcc']:.4f}; saved to {test_dir}")
            del prediction, target, pcc

            best_run_records.append(
                {
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "best_validation_pcc": best_metric,
                    "monitored_metric": "median_gene_pcc",
                    "parameter_count": parameter_count,
                    "max_epochs": args.max_epochs,
                    "patience": args.patience,
                }
            )

            pd.DataFrame(best_run_records).to_csv(
                os.path.join(args.output_dir, "best_runs.csv"),
                index=False,
            )
if __name__ == "__main__":
    main()
                
                
                
                
                
                
