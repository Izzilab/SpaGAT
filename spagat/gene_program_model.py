"""
Spatial Gene Program Model (SpaGP)

Core idea:
    y_i = μ(cell_type_i) + Σ_k g_k(z_i) · p_k + ε

where:
    - μ(cell_type) is cell-type baseline expression (from type_exp)
    - p_k ∈ R^genes is the k-th gene program (global, shared, orthogonal)
    - g_k(z_i) ∈ R+ is scalar activation of program k for cell i (non-negative)
    - z_i ∈ R^d is spatial context embedding from neighborhood via graph encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import GRIT_encoder
from .embedding import Embedding, FFN


class GeneProgramHead(nn.Module):
    """
    The program-based prediction head.
    
    Given spatial context embedding z_i (from graph encoder),
    predicts gene expression as:
        y_i = baseline_i + Σ_k g_k(z_i) · p_k
    """
    def __init__(self, node_dim, n_genes, n_programs=16, n_neighbors=50):
        super().__init__()
        self.n_genes = n_genes
        self.n_programs = n_programs
        self.n_neighbors = n_neighbors
        
        # ===== Gene Programs p_k ∈ R^{K × genes} =====
        # Initialized with Xavier, will be orthogonalized during training via loss penalty
        self.programs = nn.Parameter(torch.empty(n_programs, n_genes))
        nn.init.xavier_uniform_(self.programs)
        
        # ===== Context Aggregator: aggregate neighborhood into z_i =====
        # Attention-weighted aggregation of neighbor node features → z_i
        self.context_query = nn.Linear(node_dim, node_dim)
        self.context_key = nn.Linear(node_dim, node_dim)
        self.context_proj = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, node_dim)
        )
        self.scale = node_dim ** 0.5
        
        # ===== Activation Network: z_i → g(z_i) ∈ R+^K =====
        self.activation_net = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, n_programs),
            nn.Softplus()  # ensures g_k ≥ 0
        )
        
        # ===== Baseline: cell-type expression → gene-level baseline =====
        self.baseline_head = nn.Sequential(
            nn.Linear(n_genes, n_genes),
            nn.GELU(),
            nn.Linear(n_genes, n_genes)
        )
    
    def forward(self, node_features, type_exp_center):
        """
        Args:
            node_features: (B, N, node_dim) — encoder output for center + neighbors
            type_exp_center: (B, n_genes) — cell-type average expression for center cell
        
        Returns:
            y_pred: (B, n_genes)
            program_info: dict with activations and programs for downstream analysis
        """
        B, N, C = node_features.shape
        
        # --- Step 1: Aggregate neighborhood → context vector z_i ---
        # Center cell as query, neighbors as keys
        center = node_features[:, 0:1, :]     # (B, 1, C)
        neighbors = node_features[:, 1:, :]    # (B, N-1, C)
        
        Q = self.context_query(center)          # (B, 1, C)
        K = self.context_key(neighbors)         # (B, N-1, C)
        
        attn = (Q @ K.transpose(-1, -2)) / self.scale  # (B, 1, N-1)
        attn = F.softmax(attn, dim=-1)
        
        context = attn @ neighbors              # (B, 1, C)
        z_i = self.context_proj(context).squeeze(1)  # (B, C) — spatial context embedding
        
        # --- Step 2: Program activation g_k(z_i) ---
        g = self.activation_net(z_i)            # (B, K), non-negative
        
        # --- Step 3: Normalize programs (unit norm per program) ---
        programs_normalized = F.normalize(self.programs, dim=-1)  # (K, genes)
        
        # --- Step 4: Program contribution ---
        # Σ_k g_k(z_i) · p_k = g @ P
        program_output = g @ programs_normalized   # (B, genes)
        
        # --- Step 5: Baseline from cell-type expression ---
        baseline = self.baseline_head(type_exp_center)  # (B, genes)
        
        # --- Step 6: Final prediction ---
        y_pred = baseline + program_output
        
        program_info = {
            'activations': g,                    # (B, K) — program activation per cell
            'programs': programs_normalized,      # (K, genes) — gene program vectors
            'context': z_i,                       # (B, C) — spatial context embedding
            'attn_weights': attn.squeeze(1),      # (B, N-1) — neighbor attention weights
            'baseline': baseline,                 # (B, genes) — cell-type baseline
            'program_output': program_output      # (B, genes) — program contribution
        }
        
        return y_pred, program_info


class SpaGP(nn.Module):
    """
    Spatial Gene Program Model.
    """
    def __init__(self, genes, ligands_info, node_dim=256, edge_dim=48, num_heads=2,
                 n_layers=1, att_dim=8, use_cell_type_embedding=True,
                 n_programs=16):
        super().__init__()
        self.genes = genes
        self.n_programs = n_programs
        
        self.embeddings = Embedding(genes, ligands_info, node_dim, edge_dim,
                                    use_cell_type_embedding=use_cell_type_embedding)
        self.encoders = nn.ModuleList([
            GRIT_encoder(node_dim, edge_dim, num_heads, att_dim) 
            for i in range(n_layers)  
        ])
        self.program_head = GeneProgramHead(
            node_dim=node_dim,
            n_genes=len(genes),
            n_programs=n_programs
        )
    
    def forward(self, x):
        """
        Args:
            x: dict from dataloader with keys:
                "x": (B, N, genes), "type_exp": (B, N, genes),
                "cell_types": (B, N), "position_x": (B, N), "position_y": (B, N),
                "y": (B, genes)
        
        Returns:
            y_pred: (B, genes)
            program_info: dict with activations, programs, context, etc.
        """
        # Center cell's type expression (for baseline)
        type_exp_center = x["type_exp"][:, 0, :]  # (B, genes)
        encoded = self.embeddings(x)   # [node, edge, distance]
        for encoder in self.encoders:
            encoded = encoder(encoded)
        node_features = encoded[0]     # (B, N, node_dim)
        
        # Program-based prediction
        y_pred, program_info = self.program_head(node_features, type_exp_center)
        
        return y_pred, program_info


class SpaGP_Loss(nn.Module):
    """
    Loss function for SpaGP model:
        L = L_mse + λ_orth * L_orthogonality + λ_sparse * L_sparsity
    """
    def __init__(self, gene_list, interaction_gene_list1, 
                 lambda_orth=0.1, lambda_sparse=0.01):
        super().__init__()
        interaction_gene_list = list(set(elem for sublist in interaction_gene_list1[0] for elem in sublist))
        
        self.interaction_gene_index = []
        self.not_interaction_gene_index = []
        for i in range(len(gene_list)):
            if gene_list[i] in interaction_gene_list:
                self.interaction_gene_index.append(i)
            else:
                self.not_interaction_gene_index.append(i)
        self.interaction_gene_index = torch.LongTensor(self.interaction_gene_index)
        self.not_interaction_gene_index = torch.LongTensor(self.not_interaction_gene_index)
        
        self.mse = nn.MSELoss()
        self.lambda_orth = lambda_orth
        self.lambda_sparse = lambda_sparse
    
    def orthogonality_loss(self, programs):
        """
        Soft orthogonality penalty: Σ_{i≠j} (p_i · p_j)²
        
        Args:
            programs: (K, genes) — normalized program vectors
        """
        # Gram matrix: (K, K)
        gram = programs @ programs.T  # p_i · p_j for all pairs
        # Zero out diagonal (self-similarity = 1, we don't penalize that)
        K = programs.shape[0]
        mask = 1.0 - torch.eye(K, device=programs.device)
        off_diag = gram * mask
        # Penalty: sum of squared off-diagonal entries
        return torch.sum(off_diag ** 2) / (K * (K - 1))
    
    def sparsity_loss(self, activations):
        """
        L1 sparsity on program activations.
        Encourages each cell to activate only a few programs.
        
        Args:
            activations: (B, K) — non-negative program activations
        """
        return torch.mean(activations)  # L1 of non-negative = mean
    
    def forward(self, y_pred_tuple, y):
        """
        Args:
            y_pred_tuple: (y_pred, program_info) from model
            y: (B, genes) ground truth
        
        Returns:
            total_loss: scalar (for backward)
            mse_not_interact: scalar (for monitoring, detached)
            orth_loss: scalar (for monitoring, detached)
            sparse_loss: scalar (for monitoring, detached)
        """
        y_pred, program_info = y_pred_tuple
        
        # Main prediction loss
        mse_loss = self.mse(y_pred, y)
        
        # Orthogonality penalty
        orth_loss = self.orthogonality_loss(program_info['programs'])
        
        # Sparsity penalty
        sparse_loss = self.sparsity_loss(program_info['activations'])
        
        # Total
        total_loss = mse_loss + self.lambda_orth * orth_loss + self.lambda_sparse * sparse_loss
        
        # Monitoring: MSE on non-interaction genes
        mse_not_interact = self.mse(
            y_pred[:, self.not_interaction_gene_index],
            y[:, self.not_interaction_gene_index]
        ).detach().cpu()
        
        return (total_loss, mse_not_interact, 
                orth_loss.detach().cpu(), sparse_loss.detach().cpu())
class SpaGP_NoSpatial(nn.Module):
    """
    Ablation model: g_k predicted from center cell expression only.
    No neighborhood information → if VE ≈ full model, spatial story fails.
    """
    def __init__(self, genes, ligands_info, node_dim=256, edge_dim=48, num_heads=2,
                 n_layers=1, att_dim=8, use_cell_type_embedding=True,
                 n_programs=16):
        super().__init__()
        self.genes = genes
        self.n_programs = n_programs
        n_genes = len(genes)
        
        # Programs (same as SpaGP)
        self.programs = nn.Parameter(torch.empty(n_programs, n_genes))
        nn.init.xavier_uniform_(self.programs)
        
        # Activation from center cell expression ONLY (no spatial context)
        self.activation_net = nn.Sequential(
            nn.Linear(n_genes, 128),
            nn.GELU(),
            nn.Linear(128, n_programs),
            nn.Softplus()
        )
    
    def forward(self, x):
        # Only use center cell's raw expression — NO neighbors
        center_exp = x["type_exp"][:, 0, :] + x["x"][:, 0, :]  # (B, genes)
        
        g = self.activation_net(center_exp)  # (B, K)
        programs_norm = F.normalize(self.programs, dim=-1)
        y_pred = g @ programs_norm
        
        program_info = {
            'activations': g,
            'programs': programs_norm,
            'context': center_exp,
        }
        return y_pred, program_info
