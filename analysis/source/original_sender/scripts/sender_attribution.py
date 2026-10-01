
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
# Arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Matched sender-masking analysis for SpaGAT "
            "sender-origin attribution."
        )
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing processed SpaGAT samples.",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Trained SpaGAT checkpoint.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for sender-attribution results.",
    )

    parser.add_argument(
        "--receiver-types",
        nargs="+",
        default=["tumor_1", "tumor_2"],
        help="Receiver cell types to analyze.",
    )

    parser.add_argument(
        "--sender-types",
        nargs=2,
        default=["tumor_1", "tumor_2"],
        help="Two sender populations used for matched-count masking.",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of matched-removal repeats.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123,
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
# Metadata helpers
# ============================================================

def load_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return torch.load(
                p,
                map_location="cpu",
                weights_only=False,
            )

    raise FileNotFoundError(
        "Could not find any of:\n"
        + "\n".join(paths)
    )


def load_metadata(data_dir):
    data_dir = data_dir.rstrip("/")
    data_root = os.path.dirname(data_dir)

    genes = list(
        load_first_existing(
            [
                os.path.join(data_root, "genes.pth"),
                os.path.join(data_dir, "genes.pth"),
            ]
        )
    )

    ligands_info = load_first_existing(
        [
            os.path.join(data_root, "ligands.pth"),
            os.path.join(data_dir, "ligands.pth"),
        ]
    )

    cell_type_names = list(
        load_first_existing(
            [
                os.path.join(data_root, "cell_types.pth"),
                os.path.join(data_root, "cell_type_names.pth"),
                os.path.join(data_root, "celltypes.pth"),
                os.path.join(data_dir, "cell_types.pth"),
                os.path.join(data_dir, "cell_type_names.pth"),
                os.path.join(data_dir, "celltypes.pth"),
            ]
        )
    )

    return genes, ligands_info, cell_type_names


# ============================================================
# Model loading
# ============================================================

def load_model(
    checkpoint_path,
    genes,
    ligands_info,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "model" not in checkpoint:
        raise ValueError(
            "Expected a SpaGAT training checkpoint containing "
            "the key 'model'."
        )

    state = checkpoint["model"]

    config = checkpoint.get(
        "model_config",
        checkpoint.get("config", {}),
    )

    if "program_tokens" not in state:
        raise ValueError(
            "Checkpoint does not contain program_tokens and "
            "does not appear to be the program-aware SpaGAT model."
        )

    n_programs = state["program_tokens"].shape[0]
    program_token_dim = state["program_tokens"].shape[1]

    model = SpaGP(
        genes=genes,
        ligands_info=ligands_info,
        node_dim=config.get("node_dim", 256),
        edge_dim=config.get("edge_dim", 48),
        num_heads=config.get("num_heads", 2),
        n_layers=config.get("n_layers", 1),
        att_dim=config.get("att_dim", 8),
        n_programs=n_programs,
        program_aware=True,
        program_token_dim=program_token_dim,
        message_decoder="free",
        routing_mode="program",
    )

    model.load_state_dict(state)

    model = model.to(device)
    model.eval()

    return model, checkpoint


# ============================================================
# Mask construction
# ============================================================

def build_matched_mask(
    cell_types,
    cell_type_names,
    receiver_name,
    sender_names,
    condition,
    n_programs,
    rng,
    device,
):
    """
    Construct inference-time masks for the center receiver.

    For each eligible receiver:
        m_i = min(n_sender1, n_sender2)

    Conditions:
        random   -> remove m_i arbitrary neighbors
        sender 1 -> remove m_i sender-1 neighbors
        sender 2 -> remove m_i sender-2 neighbors

    The receiver itself (position 0) is never removed.
    """

    types_np = cell_types.detach().cpu().numpy()

    batch_size, n_nodes = types_np.shape

    edge_mask = torch.ones(
        (
            batch_size,
            n_nodes,
            n_nodes,
            n_programs,
        ),
        dtype=torch.bool,
        device=device,
    )

    selected_rows = []
    removed_counts = []

    sender_1, sender_2 = sender_names

    for row in range(batch_size):

        receiver_type = cell_type_names[
            int(types_np[row, 0])
        ]

        if receiver_type != receiver_name:
            continue

        neighbor_names = np.asarray(
            [
                cell_type_names[int(x)]
                for x in types_np[row, 1:]
            ],
            dtype=object,
        )

        sender1_positions = np.flatnonzero(
            neighbor_names == sender_1
        )

        sender2_positions = np.flatnonzero(
            neighbor_names == sender_2
        )

        # Eligible receiver must contain both populations.
        if (
            len(sender1_positions) == 0
            or len(sender2_positions) == 0
        ):
            continue

        m = min(
            len(sender1_positions),
            len(sender2_positions),
        )

        if m <= 0:
            continue

        if condition == sender_1:

            chosen = rng.choice(
                sender1_positions,
                size=m,
                replace=False,
            )

        elif condition == sender_2:

            chosen = rng.choice(
                sender2_positions,
                size=m,
                replace=False,
            )

        elif condition == "random":

            # Match only the number of removed neighbors.
            all_neighbor_positions = np.arange(
                n_nodes - 1
            )

            chosen = rng.choice(
                all_neighbor_positions,
                size=m,
                replace=False,
            )

        else:
            raise ValueError(
                f"Unknown masking condition: {condition}"
            )

        # neighbor_names index 0 corresponds to graph position 1.
        graph_positions = chosen + 1

        # Remove sender -> center-receiver edges for ALL latent channels.
        edge_mask[
            row,
            0,
            torch.as_tensor(
                graph_positions,
                dtype=torch.long,
                device=device,
            ),
            :
        ] = False

        selected_rows.append(row)
        removed_counts.append(m)

    return (
        edge_mask,
        np.asarray(selected_rows, dtype=int),
        np.asarray(removed_counts, dtype=float),
    )


# ============================================================
# Full prediction
# ============================================================

def collect_full_predictions(
    model,
    loader,
    device,
):
    records = []

    with torch.no_grad():

        offset = 0

        for batch in loader:

            batch = {
                k: v.to(device)
                for k, v in batch.items()
            }

            prediction, _ = model(batch)

            prediction = prediction.cpu()
            target = batch["y"].cpu()
            cell_types = batch["cell_types"].cpu()

            batch_size = prediction.shape[0]

            for row in range(batch_size):
                records.append(
                    {
                        "loader_index": offset + row,
                        "prediction": prediction[row],
                        "target": target[row],
                        "cell_types": cell_types[row],
                    }
                )

            offset += batch_size

    return records


# ============================================================
# One masking condition
# ============================================================

def evaluate_masking_condition(
    model,
    loader,
    device,
    cell_type_names,
    receiver_name,
    sender_names,
    condition,
    n_programs,
    rng,
):
    full_predictions = []
    masked_predictions = []
    targets = []
    removed_all = []

    with torch.no_grad():

        for batch in loader:

            batch = {
                k: v.to(device)
                for k, v in batch.items()
            }

            # Full inference
            full_prediction, _ = model(batch)

            edge_mask, selected_rows, removed_counts = (
                build_matched_mask(
                    cell_types=batch["cell_types"],
                    cell_type_names=cell_type_names,
                    receiver_name=receiver_name,
                    sender_names=sender_names,
                    condition=condition,
                    n_programs=n_programs,
                    rng=rng,
                    device=device,
                )
            )

            if len(selected_rows) == 0:
                continue

            # Masked inference
            masked_prediction, _ = model(
                batch,
                program_edge_mask=edge_mask,
            )

            selected = torch.as_tensor(
                selected_rows,
                dtype=torch.long,
                device=device,
            )

            full_predictions.append(
                full_prediction[selected].cpu()
            )

            masked_predictions.append(
                masked_prediction[selected].cpu()
            )

            targets.append(
                batch["y"][selected].cpu()
            )

            removed_all.extend(
                removed_counts.tolist()
            )

    if len(full_predictions) == 0:
        raise RuntimeError(
            f"No eligible receivers found for "
            f"receiver={receiver_name}."
        )

    full_prediction = torch.cat(
        full_predictions,
        dim=0,
    )

    masked_prediction = torch.cat(
        masked_predictions,
        dim=0,
    )

    target = torch.cat(
        targets,
        dim=0,
    )

    mean_abs_change = torch.mean(
        torch.abs(
            masked_prediction
            - full_prediction
        )
    ).item()

    full_mse = torch.mean(
        (full_prediction - target) ** 2
    ).item()

    masked_mse = torch.mean(
        (masked_prediction - target) ** 2
    ).item()

    delta_mse = masked_mse - full_mse

    relative_mse = (
        100.0
        * delta_mse
        / full_mse
    )

    return {
        "n_receivers": full_prediction.shape[0],
        "mean_edges_removed": float(
            np.mean(removed_all)
        ),
        "mean_abs_prediction_change": mean_abs_change,
        "full_MSE": full_mse,
        "masked_MSE": masked_mse,
        "delta_MSE": delta_mse,
        "relative_MSE_increase_percent": relative_mse,
    }


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    device = torch.device(args.device)

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    genes, ligands_info, cell_type_names = (
        load_metadata(args.data_dir)
    )

    print("Cell types:")
    print(cell_type_names)

    for name in (
        list(args.receiver_types)
        + list(args.sender_types)
    ):
        if name not in cell_type_names:
            raise ValueError(
                f"Cell type '{name}' was not found in "
                f"cell-type metadata."
            )

    dataset = SPAGAT_dataset(
        processed_dir=args.data_dir,
        num_neighbors=args.num_neighbors,
    )

    model, checkpoint = load_model(
        args.checkpoint,
        genes,
        ligands_info,
        device,
    )

    # Reuse validation split from training when available.
    if "shared_split" in checkpoint:

        val_indices = checkpoint[
            "shared_split"
        ]["val_indices"]

        eval_dataset = Subset(
            dataset,
            val_indices,
        )

        print(
            f"Using checkpoint validation split: "
            f"{len(val_indices)} cells"
        )

    else:

        eval_dataset = dataset

        print(
            "Checkpoint contains no shared_split; "
            "using the complete processed dataset."
        )

    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    n_programs = model.n_programs

    rows = []

    conditions = [
        args.sender_types[0],
        args.sender_types[1],
        "random",
    ]

    for repeat in range(args.repeats):

        print(
            f"\nRepeat {repeat + 1}/{args.repeats}"
        )

        for receiver_name in args.receiver_types:

            for condition_index, condition in enumerate(
                conditions
            ):

                # Deterministic but distinct RNG stream.
                rng = np.random.default_rng(
                    args.seed
                    + repeat * 1000
                    + condition_index * 100
                    + args.receiver_types.index(
                        receiver_name
                    )
                )

                result = evaluate_masking_condition(
                    model=model,
                    loader=loader,
                    device=device,
                    cell_type_names=cell_type_names,
                    receiver_name=receiver_name,
                    sender_names=args.sender_types,
                    condition=condition,
                    n_programs=n_programs,
                    rng=rng,
                )

                row = {
                    "repeat": repeat,
                    "receiver": receiver_name,
                    "condition": condition,
                    **result,
                }

                rows.append(row)

                print(
                    receiver_name,
                    condition,
                    "|Δpred| =",
                    f"{result['mean_abs_prediction_change']:.6f}",
                    "relative MSE =",
                    f"{result['relative_MSE_increase_percent']:.3f}%",
                )

    # ========================================================
    # Save repeat-level results
    # ========================================================

    repeats = pd.DataFrame(rows)

    repeats_path = os.path.join(
        args.output_dir,
        "matched_sender_random_control_repeats.csv",
    )

    repeats.to_csv(
        repeats_path,
        index=False,
    )

    # ========================================================
    # Summary across repeats
    # ========================================================

    summary = (
        repeats
        .groupby(
            ["receiver", "condition"],
            as_index=False,
        )
        .agg(
            n_receivers=(
                "n_receivers",
                "first",
            ),
            mean_edges_removed=(
                "mean_edges_removed",
                "mean",
            ),
            mean_abs_prediction_change=(
                "mean_abs_prediction_change",
                "mean",
            ),
            std_abs_prediction_change=(
                "mean_abs_prediction_change",
                "std",
            ),
            relative_MSE_increase_percent=(
                "relative_MSE_increase_percent",
                "mean",
            ),
            std_relative_MSE=(
                "relative_MSE_increase_percent",
                "std",
            ),
        )
    )

    # Preserve Figure-5-style ordering.
    condition_order = {
        "random": 0,
        args.sender_types[0]: 1,
        args.sender_types[1]: 2,
    }

    receiver_order = {
        name: i
        for i, name
        in enumerate(args.receiver_types)
    }

    summary["_receiver_order"] = (
        summary["receiver"]
        .map(receiver_order)
    )

    summary["_condition_order"] = (
        summary["condition"]
        .map(condition_order)
    )

    summary = (
        summary
        .sort_values(
            [
                "_receiver_order",
                "_condition_order",
            ]
        )
        .drop(
            columns=[
                "_receiver_order",
                "_condition_order",
            ]
        )
        .reset_index(drop=True)
    )

    summary_path = os.path.join(
        args.output_dir,
        "matched_sender_random_control_summary.csv",
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("Sender-origin attribution summary")
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(repeats_path)
    print(summary_path)


if __name__ == "__main__":
    main()
