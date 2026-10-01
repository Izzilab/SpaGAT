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
    def __init__(self, node_dim, n_genes, n_programs=16, n_neighbors=50,
                 program_aware=False, message_decoder="free"):
        super().__init__()
        self.n_genes = n_genes
        self.n_programs = n_programs
        self.n_neighbors = n_neighbors
        self.program_aware = program_aware
        self.message_decoder = message_decoder
        if message_decoder not in {"free", "program_aligned"}:
            raise ValueError("message_decoder must be 'free' or 'program_aligned'.")
        
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

        # Only program-aware SpaGP uses this decoder branch.  Keeping it absent
        # from the legacy construction preserves existing checkpoint layouts.
        if program_aware:
            if message_decoder == "free":
                self.message_to_genes = nn.Linear(node_dim, n_genes)
            else:
                # A shared scalar decoder yields q_k from each program message
                # m_k.  Its gene effect is constrained to that program's P_k.
                self.message_to_program = nn.Linear(node_dim, 1)
    
    def forward(self, node_features, type_exp_center, encoder_program_state=None, program_mask=None):
        """
        Args:
            node_features: (B, N, node_dim) — encoder output for center + neighbors
            type_exp_center: (B, n_genes) — cell-type average expression for center cell
            encoder_program_state: Optional state returned by a program-aware
                GRIT encoder.  When supplied, its center-cell activations replace
                this head's legacy context/activation inference.
            program_mask: Optional inference-only ``(K,)`` 0/1 mask. It is
                applied only to program-specific decoder contributions; the
                shared cell-type baseline remains unchanged.
        
        Returns:
            y_pred: (B, n_genes)
            program_info: dict with activations and programs for downstream analysis
        """
        B, N, C = node_features.shape
        
        if encoder_program_state is None:
            # Legacy decoder path: retain the original head computation and
            # parameter layout so existing SpaGP checkpoints remain usable.
            center = node_features[:, 0:1, :]
            neighbors = node_features[:, 1:, :]
            Q = self.context_query(center)
            K = self.context_key(neighbors)
            attn = F.softmax((Q @ K.transpose(-1, -2)) / self.scale, dim=-1)
            z_i = self.context_proj(attn @ neighbors).squeeze(1)
            g = self.activation_net(z_i)
            program_attention = None
            program_messages = None
        else:
            # Program-aware decoder path: graph message passing has already
            # produced non-negative activations for every node/program pair.
            # The center-cell row is the activation vector used to decode genes.
            g = encoder_program_state['activations'][:, 0, :]
            z_i = node_features[:, 0, :]
            program_attention = encoder_program_state['attention'][:, 0, 1:, :]
            program_messages = encoder_program_state['messages'][:, 0, :, :]

            # Keep the historical (B, N-1) attn_weights field as an activation-
            # weighted summary of the full program-specific attention tensor.
            gate_weights = g / (torch.sum(g, dim=-1, keepdim=True) + 1e-8)
            attn = torch.sum(program_attention * gate_weights.unsqueeze(dim=1), dim=-1).unsqueeze(dim=1)
        
        # --- Step 3: Normalize programs (unit norm per program) ---
        programs_normalized = F.normalize(self.programs, dim=-1)  # (K, genes)
        if program_mask is not None:
            program_mask = torch.as_tensor(
                program_mask, device=programs_normalized.device, dtype=programs_normalized.dtype
            )
            if program_mask.ndim != 1 or program_mask.shape[0] != self.n_programs:
                raise ValueError(
                    f"program_mask must have shape ({self.n_programs},); got {tuple(program_mask.shape)}."
                )
        mask_is_identity = program_mask is None or bool(torch.all(program_mask == 1))
        
        # --- Step 4: Program contribution ---
        # Σ_k g_k(z_i) · p_k = g @ P
        # Keep the default expression byte-for-byte on the existing path.  For
        # an intervention, mask each program's gene-program contribution before
        # summing across k; no re-normalization occurs.
        masked_g = g if mask_is_identity else g * program_mask.unsqueeze(0)
        program_output = masked_g @ programs_normalized   # (B, genes)

        if encoder_program_state is None:
            message_output = None
            message_coefficients = None
        else:
            if not self.program_aware:
                raise RuntimeError(
                    "GeneProgramHead received encoder program state but was not "
                    "constructed with program_aware=True."
                )
            if self.message_decoder == "free":
                # Legacy program-aware ablation: a flexible shared gene decoder
                # consumes the gate-weighted sum of program messages.
                if mask_is_identity:
                    fused_program_message = torch.sum(
                        program_messages * gate_weights.unsqueeze(dim=-1), dim=1
                    )
                    message_output = self.message_to_genes(fused_program_message)
                else:
                    # Distribute the free decoder's shared bias by the original
                    # program weights: sum_k w_k (W m_k + b) equals the legacy
                    # W sum_k(w_k m_k) + b when every mask is one, while an
                    # all-zero mask removes the complete program message path.
                    message_weights = gate_weights * program_mask.unsqueeze(0)
                    message_components = F.linear(
                        program_messages, self.message_to_genes.weight, self.message_to_genes.bias
                    )
                    message_output = torch.sum(message_components * message_weights.unsqueeze(dim=-1), dim=1)
                message_coefficients = None
            else:
                # Program-aligned decoder: q_k is derived from m_k, then its
                # contribution is restricted to the corresponding gene program
                # P_k rather than passing through an unrestricted gene decoder.
                message_coefficients = self.message_to_program(program_messages).squeeze(dim=-1)
                masked_coefficients = (
                    message_coefficients if mask_is_identity
                    else message_coefficients * program_mask.unsqueeze(0)
                )
                message_output = masked_coefficients @ programs_normalized
        
        # --- Step 5: Baseline from cell-type expression ---
        baseline = self.baseline_head(type_exp_center)  # (B, genes)
        
        # --- Step 6: Final prediction ---
        y_pred = baseline + program_output
        if message_output is not None:
            y_pred = y_pred + message_output
        
        program_info = {
            'activations': g,                    # (B, K) — program activation per cell
            'programs': programs_normalized,      # (K, genes) — gene program vectors
            'context': z_i,                       # (B, C) — spatial context embedding
            'attn_weights': attn.squeeze(1),      # (B, N-1) — neighbor attention weights
            'baseline': baseline,                 # (B, genes) — cell-type baseline
            'program_output': program_output      # (B, genes) — program contribution
        }
        if program_attention is not None:
            program_info['program_attention'] = program_attention
            program_info['program_messages'] = program_messages
            program_info['program_message_output'] = message_output
            program_info['program_message_coefficients'] = message_coefficients
        if program_mask is not None:
            program_info['program_mask'] = program_mask
        
        return y_pred, program_info


class SpaGP(nn.Module):
    """
    Spatial Gene Program Model.
    """
    def __init__(self, genes, ligands_info, node_dim=256, edge_dim=48, num_heads=2,
                 n_layers=1, att_dim=8, use_cell_type_embedding=True,
                 n_programs=16, program_aware=False, program_token_dim=32,
                 message_decoder="free", routing_mode="program"):
        super().__init__()
        self.genes = genes
        self.n_programs = n_programs
        self.program_aware = program_aware
        self.program_token_dim = program_token_dim
        self.message_decoder = message_decoder
        self.routing_mode = routing_mode

        if program_aware and n_layers < 1:
            raise ValueError("SpaGP program-aware mode requires at least one GRIT encoder layer.")
        
        self.embeddings = Embedding(genes, ligands_info, node_dim, edge_dim,
                                    use_cell_type_embedding=use_cell_type_embedding)
        self.encoders = nn.ModuleList([
            GRIT_encoder(node_dim, edge_dim, num_heads, att_dim, program_aware=program_aware, routing_mode=routing_mode)
            for i in range(n_layers)  
        ])

        if program_aware:
            # The learned bank is intentionally in a separate latent space P.
            # Its shared projection supplies (K, edge_dim) tokens required by
            # every GRIT layer, keeping program identity fixed across depth.
            self.program_tokens = nn.Parameter(torch.empty(n_programs, program_token_dim))
            nn.init.xavier_uniform_(self.program_tokens)
            self.program_token_projection = nn.Linear(program_token_dim, edge_dim, bias=False)

        self.program_head = GeneProgramHead(
            node_dim=node_dim,
            n_genes=len(genes),
            n_programs=n_programs,
            program_aware=program_aware,
            message_decoder=message_decoder,
        )
    
    def forward(self, x, program_mask=None, program_edge_mask=None,ligand_feature_mask=None):
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
        encoded = self.embeddings(x,ligand_feature_mask=ligand_feature_mask)   # [node, edge, distance]
        if self.program_aware:
            program_tokens = self.program_token_projection(self.program_tokens)
            final_program_state = None
            for encoder in self.encoders:
                # Program state is analysis metadata; only graph tensors are
                # forwarded to the next GRIT layer.
                encoded = encoder(
                    encoded[:3], program_tokens=program_tokens, program_edge_mask=program_edge_mask
                )
                final_program_state = encoded[3]
        else:
            for encoder in self.encoders:
                encoded = encoder(encoded)
            final_program_state = None
        node_features = encoded[0]     # (B, N, node_dim)
        
        # Program-based prediction
        y_pred, program_info = self.program_head(
            node_features, type_exp_center, encoder_program_state=final_program_state,
            program_mask=program_mask,
        )
        
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
        if K < 2:
            # A single program has no program pair, so orthogonality/diversity
            # is defined exactly as zero rather than evaluating 0 / (K * (K-1)).
            return programs.new_zeros(())
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
