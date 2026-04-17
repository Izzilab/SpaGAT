import torch
import torch.nn as nn
import numpy as np
import random
import os
from .attention import GRIT_encoder, GRIT_encoder_last_layer, ProgramLastLayer
from .embedding import Embedding

class SPAGAT(nn.Module):
    def __init__(self, genes, ligands_info, node_dim, edge_dim, num_heads, n_layers,
                 node_dim_small=16, att_dim=8, use_cell_type_embedding=True):
        super().__init__()
        self.embeddings = Embedding(genes, ligands_info, node_dim, edge_dim,
                                   use_cell_type_embedding=use_cell_type_embedding)
        self.encoders = nn.ModuleList([
            GRIT_encoder(node_dim, edge_dim, num_heads, att_dim) for _ in range(n_layers - 1)
        ])
        self.last_layer = ProgramLastLayer(node_dim, len(genes), edge_dim, n_programs=8, att_dim=att_dim)

    def forward(self, x):
        x = self.embeddings(x)
        for encoderi in self.encoders:
            x = encoderi(x)
        x = self.last_layer(x)
        return x

class Loss_function(nn.Module):
    def __init__(self, gene_list, interaction_gene_list1):
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

    def forward(self, y_pred, y, neighbor_mask=None):
        output, edges = y_pred
        mse_loss = self.mse(output, y)
        total_loss = mse_loss
        return (total_loss,
                self.mse(output[:, self.not_interaction_gene_index],
                         y[:, self.not_interaction_gene_index]).detach().cpu())

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed_value)

set_seed(123)

class ProgramSPAGAT(nn.Module):
    def __init__(self, genes, ligands_info, node_dim, edge_dim, num_heads, n_layers,
                 n_programs=8, node_dim_small=16, att_dim=8, use_cell_type_embedding=True):
        super().__init__()
        self.embeddings = Embedding(genes, ligands_info, node_dim, edge_dim,
                                    use_cell_type_embedding=use_cell_type_embedding)
        self.encoders = nn.ModuleList([
            GRIT_encoder(node_dim, edge_dim, num_heads, att_dim) for _ in range(n_layers - 1)
        ])
        self.last_layer = ProgramLastLayer(
            node_dim, len(genes), edge_dim,
            n_programs=n_programs, node_dim_small=node_dim_small, att_dim=att_dim
        )

    def forward(self, x):
        x = self.embeddings(x)
        for encoderi in self.encoders:
            x = encoderi(x)
        x = self.last_layer(x)
        return x