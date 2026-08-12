
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset


# ============================================================
# Import SpaGAT package
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from spagat.dataloader import SPAGAT_dataset
from spagat.gene_program_model import SpaGP


# ============================================================
# Metrics
# ============================================================

def gene_wise_pcc(prediction, target, eps=1e-8):
    prediction = prediction.float()
    target = target.float()

    prediction_centered = (
        prediction
        - prediction.mean(dim=0, keepdim=True)
    )

    target_centered = (
        target
        - target.mean(dim=0, keepdim=True)
    )

    numerator = torch.sum(
        prediction_centered * target_centered,
        dim=0,
    )

    denominator = torch.sqrt(
        torch.sum(prediction_centered ** 2, dim=0)
        *
        torch.sum(target_centered ** 2, dim=0)
    )

    return numerator / (denominator + eps)


def explained_variance_percent(prediction, target):
    mse = torch.mean(
        (prediction - target) ** 2
    )

    mse_zero = torch.mean(
        target ** 2
    )

    return (
        (mse_zero - mse)
        / mse_zero
        * 100.0
    ).item()


# ============================================================
# Arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SpaGAT model."
    )

    parser.add_argument(
        "--data-dir",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--num-neighbors",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    device = torch.device(
        args.device
    )

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    data_dir = args.data_dir.rstrip("/")
    data_root = os.path.dirname(data_dir)

    # --------------------------------------------------------
    # Dataset metadata
    # --------------------------------------------------------

    genes = list(
        torch.load(
            os.path.join(
                data_root,
                "genes.pth",
            ),
            weights_only=False,
        )
    )

    ligands_info = torch.load(
        os.path.join(
            data_root,
            "ligands.pth",
        ),
        weights_only=False,
    )

    dataset = SPAGAT_dataset(
        processed_dir=data_dir,
        num_neighbors=args.num_neighbors,
    )

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    if "shared_split" not in checkpoint:
        raise ValueError(
            "Checkpoint does not contain shared_split."
        )

    val_indices = checkpoint[
        "shared_split"
    ]["val_indices"]

    # --------------------------------------------------------
    # Model configuration
    # --------------------------------------------------------

    config = checkpoint.get(
        "model_config",
        checkpoint.get(
            "config",
            {},
        ),
    )

    state = checkpoint["model"]

    # Infer K/token dimension directly from checkpoint
    n_programs = state[
        "program_tokens"
    ].shape[0]

    program_token_dim = state[
        "program_tokens"
    ].shape[1]

    model = SpaGP(
        genes=genes,
        ligands_info=ligands_info,
        node_dim=config.get(
            "node_dim",
            256,
        ),
        edge_dim=config.get(
            "edge_dim",
            48,
        ),
        num_heads=config.get(
            "num_heads",
            2,
        ),
        n_layers=config.get(
            "n_layers",
            1,
        ),
        att_dim=config.get(
            "att_dim",
            8,
        ),
        n_programs=n_programs,
        program_aware=True,
        program_token_dim=program_token_dim,
        message_decoder="free",
        routing_mode="program",
    )

    model.load_state_dict(
        state
    )

    model = model.to(
        device
    ).eval()

    # --------------------------------------------------------
    # Validation loader
    # --------------------------------------------------------

    loader = DataLoader(
        Subset(
            dataset,
            val_indices,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions = []
    targets = []

    with torch.no_grad():

        for batch in loader:

            batch = {
                k: v.to(device)
                for k, v in batch.items()
            }

            prediction, _ = model(
                batch
            )

            predictions.append(
                prediction.cpu()
            )

            targets.append(
                batch["y"].cpu()
            )

    prediction = torch.cat(
        predictions,
        dim=0,
    )

    target = torch.cat(
        targets,
        dim=0,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    pcc = gene_wise_pcc(
        prediction,
        target,
    )

    mse = torch.mean(
        (prediction - target) ** 2
    ).item()

    mse_zero = torch.mean(
        target ** 2
    ).item()

    ev = explained_variance_percent(
        prediction,
        target,
    )

    median_pcc = torch.median(
        pcc
    ).item()

    mean_pcc = torch.mean(
        pcc
    ).item()

    summary = pd.DataFrame(
        [
            {
                "n_cells": len(val_indices),
                "n_genes": len(genes),
                "median_gene_PCC": median_pcc,
                "mean_gene_PCC": mean_pcc,
                "MSE": mse,
                "MSE0": mse_zero,
                "EV_percent": ev,
            }
        ]
    )

    summary.to_csv(
        os.path.join(
            args.output_dir,
            "summary_metrics.csv",
        ),
        index=False,
    )

    per_gene = pd.DataFrame(
        {
            "gene": genes,
            "PCC": pcc.numpy(),
        }
    )

    per_gene.to_csv(
        os.path.join(
            args.output_dir,
            "per_gene_PCC.csv",
        ),
        index=False,
    )

    np.savez_compressed(
        os.path.join(
            args.output_dir,
            "predictions.npz",
        ),
        prediction=prediction.numpy(),
        target=target.numpy(),
        val_indices=np.asarray(
            val_indices
        ),
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print("\nSpaGAT evaluation")
    print("=" * 50)

    print(
        f"Validation cells: {len(val_indices)}"
    )

    print(
        f"Genes: {len(genes)}"
    )

    print(
        f"Median gene-wise PCC: {median_pcc:.6f}"
    )

    print(
        f"Mean gene-wise PCC:   {mean_pcc:.6f}"
    )

    print(
        f"MSE:                  {mse:.6f}"
    )

    print(
        f"MSE0:                 {mse_zero:.6f}"
    )

    print(
        f"EV (%):               {ev:.4f}"
    )

    print(
        "\nSaved results to:",
        args.output_dir,
    )


if __name__ == "__main__":
    main()
