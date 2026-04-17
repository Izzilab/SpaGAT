"""
Training loop for Spatial Gene Program (SpaGP) model.

Drop-in replacement for trainer.py — same data loading, same evaluation,
different model.
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np

from .dataloader import SPAGAT_dataset
from .gene_program_model import SpaGP, SpaGP_Loss
from .calculate_PCC import Calculate_PCC


def train_SpaGP(num_neighbors=50, batch_size=256, lr=1e-4, data_dir=None, epochs=50,
                node_dim=256, edge_dim=48, att_dim=8, use_cell_type_embedding=True,
                n_programs=16, lambda_orth=0.1, lambda_sparse=0.01, lambda_div=0.01):
    """
    Train the Spatial Gene Program model.
    
    Key differences from train_GITIII:
        - Uses SpaGP model instead of GITIII
        - Uses SpaGP_Loss with orthogonality and sparsity penalties
        - Logs program-specific metrics (orth_loss, sparse_loss, activation stats)
        - Saves program vectors for downstream analysis
    
    Args:
        n_programs: number of gene programs K (recommended: 10-20)
        lambda_orth: weight for orthogonality penalty (recommended: 0.01-1.0)
        lambda_sparse: weight for sparsity penalty (recommended: 0.001-0.1)
        (other args same as train_GITIII)
    """
    # Preparation
    if data_dir is None:
        data_dir = os.path.join(os.getcwd(), "data", "processed")
    if data_dir[-1] != "/":
        data_dir = data_dir + "/"
    torch.cuda.empty_cache()

    # Load dataset (identical to GITIII)
    print("=" * 60)
    print("Spatial Gene Program (SpaGP) Training")
    print(f"  n_programs={n_programs}, λ_orth={lambda_orth}, λ_sparse={lambda_sparse}")
    print("=" * 60)
    
    print("Loading dataset...")
    dataset = SPAGAT_dataset(processed_dir=data_dir, num_neighbors=num_neighbors)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    validation_size = total_size - train_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    print(f"Train: {train_size} cells, Val: {validation_size} cells")

    # Load gene/ligand info
    ligands_info = torch.load("/".join(data_dir.split("/")[:-2]) + "/ligands.pth", weights_only=False)
    genes = torch.load("/".join(data_dir.split("/")[:-2]) + "/genes.pth", weights_only=False)

    # Define model
    my_model = SpaGP(
        genes, ligands_info,
        node_dim=node_dim, edge_dim=edge_dim, num_heads=2,
        n_layers=1, att_dim=att_dim,
        use_cell_type_embedding=use_cell_type_embedding,
        n_programs=n_programs
    )
    my_model = my_model.cuda()
    
    n_params = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    optimizer = torch.optim.AdamW(my_model.parameters(), lr=lr, betas=(0.99, 0.999))
    loss_func = SpaGP_Loss(genes, ligands_info, 
                           lambda_orth=lambda_orth, 
                           lambda_sparse=lambda_sparse,
                           lambda_div=lambda_div).cuda()
    evaluator = Calculate_PCC(genes, ligands_info)

    records = []
    best_val = 1e10
    
    save_prefix = f"SpaGP_K{n_programs}"
    checkpoint_name = os.path.join(os.getcwd(), f"{save_prefix}.pth")
    best_name = os.path.join(os.getcwd(), f"{save_prefix}_best.pth")
    record_name = os.path.join(os.getcwd(), f"record_{save_prefix}.csv")

    # Resume if checkpoint exists
    if os.path.exists(checkpoint_name):
        print(f"Resuming from {checkpoint_name}")
        checkpoint = torch.load(checkpoint_name)
        my_model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_val = checkpoint['best_val']
        records = checkpoint['records']

    print("Start training")
    for epochi in range(epochs):
        my_model.train()
        loss_total_sum = 0
        loss_mse_not_interact_sum = 0
        loss_orth_sum = 0
        loss_sparse_sum = 0
        activation_mean_sum = 0
        activation_sparsity_sum = 0

        for (stepi, x) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            x = {k: v.cuda() for k, v in x.items()}
            
            y_pred_tuple = my_model(x)
            y = x["y"]
            
            lossi_total, lossi_not_interact, lossi_orth, lossi_sparse = loss_func(y_pred_tuple, y)
            
            # For PCC evaluation, wrap y_pred to match GITIII's format: (y_pred, anything)
            evaluator.add_input(y_pred_tuple, y)
            
            lossi_total.backward()
            optimizer.step()
            
            loss_total_sum += lossi_total.cpu().item()
            loss_mse_not_interact_sum += lossi_not_interact.item()
            loss_orth_sum += lossi_orth.item()
            loss_sparse_sum += lossi_sparse.item()
            
            # Track activation statistics
            with torch.no_grad():
                g = y_pred_tuple[1]['activations']  # (B, K)
                activation_mean_sum += g.mean().item()
                # Sparsity: fraction of activations < 0.1
                activation_sparsity_sum += (g < 0.1).float().mean().item()
            
            if stepi % 500 == 0:
                PCC1, PCC2 = evaluator.calculate_pcc()
                print(f"  Train epoch:{epochi} step:{stepi} "
                      f"loss:{loss_total_sum/stepi:.4f} "
                      f"orth:{loss_orth_sum/stepi:.4f} "
                      f"sparse:{loss_sparse_sum/stepi:.4f} "
                      f"median_PCC:{torch.median(PCC1):.4f} "
                      f"act_mean:{activation_mean_sum/stepi:.3f} "
                      f"act_sparsity:{activation_sparsity_sum/stepi:.2%}")

        PCC1_train, PCC2_train = evaluator.calculate_pcc(clear=True)
        n_steps = len(train_loader)
        print(f"[Epoch {epochi}] Train | "
              f"loss:{loss_total_sum/n_steps:.4f} "
              f"orth:{loss_orth_sum/n_steps:.4f} "
              f"sparse:{loss_sparse_sum/n_steps:.4f} "
              f"median_PCC_all:{torch.median(PCC1_train):.4f} "
              f"median_PCC_not_interact:{torch.median(PCC2_train):.4f}")

        # Save checkpoint
        checkpoints = {
            'model': my_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'records': records,
            'best_val': best_val,
            'n_programs': n_programs,
            'genes': genes
        }
        torch.save(checkpoints, checkpoint_name)

        # Validation
        loss_val_sum = 0
        loss_val_not_interact_sum = 0
        my_model.eval()
        with torch.no_grad():
            for (stepi, x) in enumerate(val_loader, start=1):
                x = {k: v.cuda() for k, v in x.items()}
                y_pred_tuple = my_model(x)
                y = x["y"]
                
                evaluator.add_input(y_pred_tuple, y)
                lossi_total, lossi_not_interact, _, _ = loss_func(y_pred_tuple, y)
                loss_val_sum += lossi_total.cpu().item()
                loss_val_not_interact_sum += lossi_not_interact.item()
                
                torch.cuda.empty_cache()

        PCC1_val, PCC2_val = evaluator.calculate_pcc(clear=True)
        n_val_steps = len(val_loader)
        print(f"[Epoch {epochi}] Val   | "
              f"loss:{loss_val_sum/n_val_steps:.4f} "
              f"median_PCC_all:{torch.median(PCC1_val):.4f} "
              f"median_PCC_not_interact:{torch.median(PCC2_val):.4f}")

        # Record
        records.append({
            'epoch': epochi,
            'train_loss': loss_total_sum / n_steps,
            'train_orth_loss': loss_orth_sum / n_steps,
            'train_sparse_loss': loss_sparse_sum / n_steps,
            'train_PCC1_median': torch.median(PCC1_train).item(),
            'train_PCC2_median': torch.median(PCC2_train).item(),
            'val_loss': loss_val_sum / n_val_steps,
            'val_PCC1_median': torch.median(PCC1_val).item(),
            'val_PCC2_median': torch.median(PCC2_val).item(),
        })
        pd.DataFrame(records).to_csv(record_name, index=False)

        # Save best
        val_loss_avg = loss_val_sum / n_val_steps
        if best_val > val_loss_avg:
            best_val = val_loss_avg
            torch.save(my_model.state_dict(), best_name)
            # Also save programs for analysis
            with torch.no_grad():
                programs = F.normalize(my_model.program_head.programs, dim=-1)
                torch.save({
                    'programs': programs.cpu(),
                    'genes': genes,
                    'n_programs': n_programs
                }, os.path.join(os.getcwd(), f"{save_prefix}_programs.pth"))
            print(f"  ★ New best val loss: {best_val:.6f} (programs saved)")
        else:
            print(f"  Best val loss: {best_val:.6f}")
    
    print("=" * 60)
    print("Training complete!")
    print(f"Best val loss: {best_val:.6f}")
    print(f"Records saved to: {record_name}")
    print(f"Best model saved to: {best_name}")
    print(f"Programs saved to: {save_prefix}_programs.pth")
    print("=" * 60)
from .gene_program_model import SpaGP, SpaGP_Loss, SpaGP_NoSpatial


def train_SpaGP_NoSpatial(num_neighbors=50, batch_size=256, lr=1e-4, data_dir=None, 
                           epochs=50, n_programs=16, lambda_orth=0.1, 
                           lambda_sparse=0.01, lambda_div=0.01):
    """
    Ablation: train SpaGP without spatial context.
    Identical to train_SpaGP except uses SpaGP_NoSpatial model.
    """
    if data_dir is None:
        data_dir = os.path.join(os.getcwd(), "data", "processed")
    if data_dir[-1] != "/":
        data_dir = data_dir + "/"
    torch.cuda.empty_cache()

    print("=" * 60)
    print("SpaGP NoSpatial ABLATION")
    print("=" * 60)

    dataset = SPAGAT_dataset(processed_dir=data_dir, num_neighbors=num_neighbors)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    validation_size = total_size - train_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    ligands_info = torch.load("/".join(data_dir.split("/")[:-2]) + "/ligands.pth", weights_only=False)
    genes = torch.load("/".join(data_dir.split("/")[:-2]) + "/genes.pth", weights_only=False)

    # KEY DIFFERENCE: use NoSpatial model
    my_model = SpaGP_NoSpatial(
        genes, ligands_info, n_programs=n_programs
    ).cuda()

    n_params = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    print(f"NoSpatial model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(my_model.parameters(), lr=lr, betas=(0.99, 0.999))
    loss_func = SpaGP_Loss(genes, ligands_info, 
                           lambda_orth=lambda_orth, 
                           lambda_sparse=lambda_sparse,
                           lambda_div=lambda_div).cuda()
    evaluator = Calculate_PCC(genes, ligands_info)

    records = []
    best_val = 1e10
    save_prefix = f"SpaGP_NoSpatial_K{n_programs}"
    best_name = os.path.join(os.getcwd(), f"{save_prefix}_best.pth")
    record_name = os.path.join(os.getcwd(), f"record_{save_prefix}.csv")

    for epochi in range(epochs):
        my_model.train()
        loss_total_sum = 0

        for (stepi, x) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            x = {k: v.cuda() for k, v in x.items()}
            y_pred_tuple = my_model(x)
            y = x["y"]
            lossi_total, lossi_not_interact, lossi_orth, lossi_sparse = loss_func(y_pred_tuple, y)
            evaluator.add_input(y_pred_tuple, y)
            lossi_total.backward()
            optimizer.step()
            loss_total_sum += lossi_total.cpu().item()

        PCC1_train, PCC2_train = evaluator.calculate_pcc(clear=True)
        n_steps = len(train_loader)
        print(f"[Epoch {epochi}] Train | loss:{loss_total_sum/n_steps:.4f} "
              f"median_PCC:{torch.median(PCC1_train):.4f}")

        # Validation
        loss_val_sum = 0
        my_model.eval()
        with torch.no_grad():
            for (stepi, x) in enumerate(val_loader, start=1):
                x = {k: v.cuda() for k, v in x.items()}
                y_pred_tuple = my_model(x)
                y = x["y"]
                evaluator.add_input(y_pred_tuple, y)
                lossi_total, _, _, _ = loss_func(y_pred_tuple, y)
                loss_val_sum += lossi_total.cpu().item()

        PCC1_val, PCC2_val = evaluator.calculate_pcc(clear=True)
        n_val_steps = len(val_loader)
        print(f"[Epoch {epochi}] Val   | loss:{loss_val_sum/n_val_steps:.4f} "
              f"median_PCC:{torch.median(PCC1_val):.4f}")

        records.append({
            'epoch': epochi,
            'train_loss': loss_total_sum / n_steps,
            'val_loss': loss_val_sum / n_val_steps,
            'val_PCC1_median': torch.median(PCC1_val).item(),
        })
        pd.DataFrame(records).to_csv(record_name, index=False)

        val_loss_avg = loss_val_sum / n_val_steps
        if best_val > val_loss_avg:
            best_val = val_loss_avg
            torch.save(my_model.state_dict(), best_name)
            print(f"  ★ New best: {best_val:.6f}")

    print(f"NoSpatial ablation complete. Best val loss: {best_val:.6f}")