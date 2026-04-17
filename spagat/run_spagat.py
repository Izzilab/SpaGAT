import argparse
import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from spagat.estimator import SPAGAT_estimator
from spagat.spatial_visualizer import Spatial_visualizer
from spagat.subtyping_analyzer import Subtyping_anlayzer

def downstream_analysis(sample, output_dir):
    """
    Run downstream spatial and subtype analysis, and network visualization for a given sample.
    """
    # Spatial analysis
    spatial_visualizer = Spatial_visualizer(sample=sample)
    spatial_visualizer.plot_distance_scaler(rank_or_distance='distance', proportion_or_abs='abs', bins=300, frac=0.003)
    spatial_visualizer.visualize_prediction(target_gene='APOE', plot_state=True)
    spatial_visualizer.visualize_information_flow(target_gene='APOE', select_topk=5, use_neuron_layer=False, cutoff=0.005)
    spatial_visualizer.visualize_CCI_function(select_topk=5, num_type_pair=10)

    # Subtype analysis
    subtyping_analyzer = Subtyping_anlayzer(sample=sample, normalize_to_1=True, use_abs=True, noise_threshold=2e-2)
    subtyping_analyzer.subtyping(COI='tumor_1', resolution=0.05)
    subtyping_analyzer.subtyping_filter_groups(['0', '1'])
    subtyping_analyzer.subtyping_DE()
    subtyping_analyzer.subtyping_get_aggregated_influence()

    # Network significance heatmap and statistics
    z_df = pd.read_csv(f'./network/significant_network/{sample}.csv', index_col=0)
    for target_gene in ['APOE', 'VEGFA', 'SERPINA1']:
        cell_types = sorted(set([p.split('__')[0] for p in z_df.index]))
        n_ct = len(cell_types)
        ct_to_idx = {ct: i for i, ct in enumerate(cell_types)}
        z_matrix_gene = np.zeros((n_ct, n_ct))
        for pair in z_df.index:
            sender, receiver = pair.split('__')
            if sender in ct_to_idx and receiver in ct_to_idx:
                z_matrix_gene[ct_to_idx[sender], ct_to_idx[receiver]] = z_df.loc[pair, target_gene]
       
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(z_matrix_gene, xticklabels=cell_types, yticklabels=cell_types,
                    cmap='RdBu_r', center=0, vmin=-5, vmax=5, ax=ax,
                    linewidths=0.5, linecolor='gray')
        ax.set_title(f'CCI Network Z-scores targeting {target_gene}\n(z>2: significant, z>3: strong)', fontsize=18)
        ax.set_xlabel('Receiver cell type', fontsize=12)
        ax.set_ylabel('Sender cell type', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'network_{target_gene}_{sample}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    # Top active pairs
    sig_counts = (z_df.abs() > 2).sum(axis=1).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    top20 = sig_counts.head(20)
    colors = ['red' if v > 10 else 'steelblue' for v in top20.values]
    ax.barh(range(len(top20)), top20.values, color=colors)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20.index, fontsize=8)
    ax.set_xlabel('Number of significantly influenced genes (|z|>2)')
    ax.set_title(f'Most Active CCI Pairs in {sample}')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'network_top_pairs_{sample}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Run SPAGAT on a spatial transcriptomics dataset.")
    parser.add_argument('--data_csv', type=str, required=True, help='Path to your input CSV file')
    parser.add_argument('--genes', type=str, nargs='+', required=True, help='List of gene names (columns in CSV)')
    parser.add_argument('--species', type=str, choices=['human', 'mouse'], required=True, help='Species')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save results')
    parser.add_argument('--use_log_normalize', action='store_true', help='Whether to log-normalize expression')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--num_neighbors', type=int, default=50, help='Number of neighbors')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use (cuda or cpu)')
    parser.add_argument('--sample', type=str, required=True, help='Sample name for downstream analysis (e.g., CancerousLiver)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available, switching to CPU.')
        device = 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
    torch_device = torch.device(device)

    # Initialize estimator
    estimator = SPAGAT_estimator(
        df_path=args.data_csv,
        genes=args.genes,
        use_log_normalize=args.use_log_normalize,
        species=args.species,
        visualize_when_preprocessing=False,
        distance_threshold=80,
        process_num_neighbors=args.num_neighbors,
        num_neighbors=args.num_neighbors,
        batch_size_train=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        node_dim=256,
        edge_dim=48,
        att_dim=8,
        batch_size_val=args.batch_size,
        use_cell_type_embedding=True
    )

    print("Step 1: Preprocessing dataset...")
    estimator.preprocess_dataset()

    print("Step 2: Training model...")
    estimator.train()

    print("Step 3: Calculating influence tensor...")
    estimator.calculate_influence_tensor()

    print(f"Step 4: Downstream analysis for sample: {args.sample}")
    downstream_analysis(sample=args.sample, output_dir=args.output_dir)
    print("All steps completed. Results are saved in:", args.output_dir)

if __name__ == '__main__':
    main()
