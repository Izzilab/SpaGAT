import torch
import torch.nn as nn
import torch.nn.functional as F
from .embedding import FFN

def rho(x):
    return torch.sqrt(F.relu(x)) - torch.sqrt(F.relu(-x))

def get_diagonal(x):
    index = torch.arange(0, x.shape[1], 1)
    return x[:, index, index, :]


def _normalize_pairwise_tensors(node, edge, embedding, edge_dim=None):
    """Normalize Embedding's initial singleton receiver axis.

    ``Embedding.forward`` supplies pairwise tensors as ``(B, 1, N, E)``: one
    sender-wise edge representation shared by every receiver.  GRIT attention
    operates on an explicit receiver/sender representation ``(B, N, N, E)``.
    Expanding the singleton receiver axis before Q/K--edge arithmetic prevents
    PyTorch from broadcasting the batch dimension into an unintended
    ``(B, B, N, N, E)`` tensor.  Later GRIT layers already use the expanded
    four-dimensional convention.
    """
    if node.ndim != 3:
        raise ValueError(f"node must have shape (B, N, node_dim); got {tuple(node.shape)}.")
    B, N, _ = node.shape

    def normalize(name, tensor):
        if tensor.ndim != 4:
            raise ValueError(
                f"{name} must have shape (B, 1, N, edge_dim) or (B, N, N, edge_dim); "
                f"got {tuple(tensor.shape)}."
            )
        if tensor.shape[0] != B or tensor.shape[2] != N:
            raise ValueError(
                f"{name} batch and sender dimensions must be B={B} and N={N}; "
                f"got {tuple(tensor.shape)}."
            )
        if tensor.shape[1] == 1:
            tensor = tensor.expand(B, N, N, tensor.shape[-1])
        elif tensor.shape[1] != N:
            raise ValueError(
                f"{name} receiver dimension must be 1 or N={N}; got {tuple(tensor.shape)}."
            )
        return tensor

    edge = normalize("edge", edge)
    embedding = normalize("embedding", embedding)
    if edge.shape[-1] != embedding.shape[-1]:
        raise ValueError(
            "edge and embedding must have the same feature dimension; got "
            f"{edge.shape[-1]} and {embedding.shape[-1]}."
        )
    if edge_dim is not None and edge.shape[-1] != edge_dim:
        raise ValueError(
            f"edge and embedding feature dimension must be edge_dim={edge_dim}; "
            f"got {edge.shape[-1]}."
        )
    return edge, embedding

class GRIT_attention(nn.Module):
    def __init__(self, node_dim, edge_dim, att_dim=8, program_aware=False, routing_mode="program"):
        super().__init__()
        self.program_aware = program_aware
        if routing_mode not in {"grit", "program"}:
            raise ValueError("routing_mode must be 'grit' or 'program'.")
        self.routing_mode = routing_mode
        self.node_dim = node_dim
        self.base_edge_dim = edge_dim
        self.edge_dim = edge_dim * 2
        self.W_Q = nn.Linear(node_dim, self.edge_dim * att_dim)
        self.W_K = nn.Linear(node_dim, self.edge_dim * att_dim)
        self.W_V = nn.Linear(node_dim, node_dim)
        self.W_Ew = nn.Linear(edge_dim, edge_dim, bias=False)
        self.W_Eb = nn.Linear(edge_dim, edge_dim, bias=False)
        self.W_En = nn.Linear(edge_dim, node_dim)
        self.W_A = nn.Linear(edge_dim, 1, bias=False)
        self.W_No = nn.Linear(node_dim, node_dim)
        self.W_Eo = nn.Linear(edge_dim, edge_dim, bias=False)

        # Program tokens live in the common edge-feature space (edge_dim).  They
        # are supplied by a parent model so that token k has the same biological
        # identity in every graph layer; these projections remain layer-specific.
        if self.program_aware:
            self.W_program_edge = nn.Linear(edge_dim, edge_dim, bias=False)
            self.W_program_receiver = nn.Linear(node_dim, edge_dim, bias=False)
            self.W_program_gate = nn.Linear(node_dim, edge_dim, bias=False)

    def forward(self, x, program_tokens=None, program_edge_mask=None):
        """Run edge-aware attention, optionally routed through communication programs.

        Args:
            x: ``[node, edge, embedding]`` with node ``(B, N, node_dim)`` and
                broadcastable edge/embedding tensors ``(B, 1 or N, N, edge_dim)``.
            program_tokens: Optional shared token bank ``(K, edge_dim)``.  A
                token is a globally named latent communication program; the
                same bank should be passed to every program-aware GRIT layer.
            program_edge_mask: Optional inference-only ``(B, N, N, K)`` mask.
                Zero entries remove sender edges from a program's softmax.

        Returns:
            ``[node_updated, edge_updated, program_state]``.  ``program_state``
            is ``None`` in baseline mode, preserving the original computation.
        """
        node, edge, embedding = x
        B, N, C = node.shape

        # Always normalize the initial Embedding output, including the legacy
        # path.  The routing mode must not determine whether Q/K arithmetic is
        # performed on a correctly shaped pairwise edge tensor.
        edge, embedding = _normalize_pairwise_tensors(
            node, edge, embedding, edge_dim=self.base_edge_dim
        )

        if program_tokens is not None and not self.program_aware:
            raise ValueError(
                "program_tokens were provided, but this GRIT_attention was "
                "constructed with program_aware=False."
            )
        if program_tokens is not None and program_tokens.ndim != 2:
            raise ValueError(
                "program_tokens must have shape (n_programs, edge_dim); got "
                f"{tuple(program_tokens.shape)}."
            )
        if program_tokens is not None and program_tokens.shape[1] != self.base_edge_dim:
            raise ValueError(
                "program_tokens must have shape (n_programs, edge_dim), where "
                f"edge_dim={self.base_edge_dim}; got {tuple(program_tokens.shape)}."
            )
        if program_edge_mask is not None and program_tokens is None:
            raise ValueError("program_edge_mask requires program-aware routing tokens.")

        Q = self.W_Q(node).reshape(B, N, -1, self.edge_dim).permute(0, 3, 1, 2)
        K = self.W_K(node).reshape(B, N, -1, self.edge_dim).permute(0, 3, 2, 1)
        QK = (Q @ K).permute(0, 2, 3, 1)
        edge = F.gelu(rho((QK[:, :, :, :self.edge_dim // 2]) * self.W_Ew(edge)) + self.W_Eb(edge) + QK[:, :, :, self.edge_dim // 2:] + embedding)
        V = self.W_V(node)

        if program_tokens is None:
            # This is the original GRIT message-passing path, kept numerically
            # unchanged when no shared program bank is supplied.
            alpha = self.W_A(edge).squeeze(dim=-1)
            alpha = F.softmax(alpha, dim=-1)
            node = alpha @ V + self.W_En(torch.sum(edge * alpha.unsqueeze(dim=-1), dim=-2))
            node = self.W_No(node)
            edge = self.W_Eo(edge)
            return [node, edge, None]

        if self.routing_mode == "grit":
            if program_tokens.shape[0] != 1:
                raise ValueError("routing_mode='grit' is defined for the controlled K=1 ablation only.")
            # Controlled routing ablation: retain GRIT's original scalar edge
            # routing and expose it as the K=1 program-state interface.  The
            # downstream program gate/messages/free decoder are unchanged.
            program_attention = F.softmax(self.W_A(edge).squeeze(dim=-1), dim=-1).unsqueeze(dim=-1)
        else:
            # Program routing: the edge, receiver, and shared program token
            # interact multiplicatively. This is the pre-existing SpaGP score.
            edge_program = self.W_program_edge(edge)
            receiver_program = self.W_program_receiver(node)
            program_scores = torch.einsum(
                "bije,bie,ke->bijk", edge_program, receiver_program, program_tokens
            ) / (self.base_edge_dim ** 0.5)
            program_attention = None
        if program_edge_mask is not None:
            expected_shape = (B, N, N, program_tokens.shape[0])
            if tuple(program_edge_mask.shape) != expected_shape:
                raise ValueError(
                    f"program_edge_mask must have shape {expected_shape}; got "
                    f"{tuple(program_edge_mask.shape)}."
                )
            allowed = program_edge_mask.to(device=program_scores.device, dtype=torch.bool)
            if not torch.all(allowed.any(dim=2)):
                raise ValueError("program_edge_mask removes every sender for at least one receiver/program.")
            # This is inference-time graph intervention: masked edges receive
            # exactly zero attention and remaining edges are renormalized by
            # the existing neighbor softmax.
            if self.routing_mode == "grit":
                grit_scores = self.W_A(edge).squeeze(dim=-1).masked_fill(~allowed[..., 0], float("-inf"))
                program_attention = F.softmax(grit_scores, dim=-1).unsqueeze(dim=-1)
            else:
                program_scores = program_scores.masked_fill(~allowed, float("-inf"))
                program_attention = F.softmax(program_scores, dim=2)
        elif program_attention is None:
            program_attention = F.softmax(program_scores, dim=2)

        # Aggregate node and edge evidence independently for each program:
        # m[i, k] = sum_j alpha[i, j, k] * message[i, j].
        node_messages = torch.einsum("bijk,bjd->bikd", program_attention, V)
        edge_messages = torch.einsum("bijk,bije->bike", program_attention, edge)
        program_messages = node_messages + self.W_En(edge_messages)

        # Receiver-specific, non-negative gates fuse the K program messages back
        # into the original node_dim, keeping the downstream tensor shape intact.
        gate_scores = torch.einsum(
            "bin,kn->bik", self.W_program_gate(node), program_tokens
        ) / (self.base_edge_dim ** 0.5)
        program_activations = F.softplus(gate_scores)
        node = torch.sum(program_messages * program_activations.unsqueeze(dim=-1), dim=2)
        node = self.W_No(node)
        edge = self.W_Eo(edge)

        program_state = {
            "attention": program_attention,
            "messages": program_messages,
            "activations": program_activations,
        }
        return [node, edge, program_state]

class DegScaler(nn.Module):
    def __init__(self, node_dim):
        super().__init__()
        scaler = (2 / node_dim) ** 0.5
        self.theta1 = nn.Parameter(torch.randn(node_dim) * scaler)
        self.theta2 = nn.Parameter(torch.randn(node_dim) * scaler)

    def forward(self, node, degree):
        node = self.theta1 * node + torch.log(degree + 1).unsqueeze(dim=-1) * node * self.theta2
        return node

class Multi_Head_Attention(nn.Module):
    def __init__(self, node_dim, edge_dim, num_heads, att_dim=8):
        super().__init__()
        self.attentions = nn.ModuleList([
            GRIT_attention(node_dim, edge_dim, att_dim) for _ in range(num_heads)
        ])
        self.W_hn = nn.Parameter(torch.ones(num_heads, 1) / num_heads)
        self.W_he = nn.Parameter(torch.ones(num_heads, 1) / num_heads)

    def forward(self, x):
        results = [attentioni(x) for attentioni in self.attentions]
        node = (torch.stack([tmp[0] for tmp in results], dim=-1) @ self.W_hn).squeeze(dim=-1)
        edge = torch.stack([tmp[1] for tmp in results], dim=-1) @ self.W_he
        edge = edge.squeeze(dim=-1)
        return [node, edge, x[2]]

class GRIT_encoder(nn.Module):
    def __init__(self, node_dim, edge_dim, num_heads, att_dim=8, program_aware=False, routing_mode="program"):
        super().__init__()
        self.program_aware = program_aware
        self.edge_dim = edge_dim
        self.routing_mode = routing_mode
        self.attentions = Multi_Head_Attention(node_dim, edge_dim, num_heads, att_dim)
        if program_aware:
            # Keep Multi_Head_Attention unchanged for baseline callers, but replace
            # its heads with token-aware GRIT attention modules for this encoder.
            # The shared tokens themselves are supplied to forward(), not owned by
            # a layer, so program index k remains consistent across GNN layers.
            self.attentions.attentions = nn.ModuleList([
                GRIT_attention(node_dim, edge_dim, att_dim, program_aware=True, routing_mode=routing_mode)
                for _ in range(num_heads)
            ])
        self.FFN = FFN(node_dim)
        self.ln1 = nn.LayerNorm(node_dim)
        self.ln2 = nn.LayerNorm(node_dim)
        self.ln_edge = nn.LayerNorm(edge_dim)

    def forward(self, x, program_tokens=None, program_edge_mask=None):
        """Encode one graph layer, optionally with shared program-aware routing.

        Without ``program_tokens``, this follows the original GRIT encoder path
        and returns ``[node, edge, distance_embedding]``.  With a shared token
        bank, each head returns program-specific routing state; the encoder
        combines it across heads and returns it as a fourth element.
        """
        node, edge, distance = x
        # Keep the residual edge tensor in the same four-dimensional convention
        # as GRIT_attention.  This is essential for the baseline residual add.
        edge, distance = _normalize_pairwise_tensors(
            node, edge, distance, edge_dim=self.edge_dim
        )
        x = [node, edge, distance]
        if program_tokens is None:
            # Preserve the existing multi-head aggregation and output contract.
            x = self.attentions(x)
            edge = self.ln_edge(edge + x[1])
            node = self.ln1(node + x[0])
            node = self.ln2(node + self.FFN(node))
            return [node, edge, distance]

        if not self.program_aware:
            raise ValueError(
                "program_tokens were provided, but this GRIT_encoder was "
                "constructed with program_aware=False."
            )

        # Multi_Head_Attention's baseline forward intentionally remains unchanged.
        # Here, route the same shared program tokens through each token-aware head
        # and use its existing learned head weights for node and edge aggregation.
        results = [
            attentioni(x, program_tokens, program_edge_mask=program_edge_mask)
            for attentioni in self.attentions.attentions
        ]
        node_attention = (
            torch.stack([result[0] for result in results], dim=-1) @ self.attentions.W_hn
        ).squeeze(dim=-1)
        edge_attention = torch.stack([result[1] for result in results], dim=-1) @ self.attentions.W_he
        edge_attention = edge_attention.squeeze(dim=-1)

        # Combine per-head routing metadata with normalized head weights.  This
        # yields one program axis K shared by all heads and preserves alpha's
        # interpretation as a neighbor distribution for each receiver/program.
        head_weights = F.softmax(self.attentions.W_hn.squeeze(dim=-1), dim=0)
        program_state = {
            key: torch.sum(
                torch.stack([result[2][key] for result in results], dim=-1)
                * head_weights,
                dim=-1,
            )
            for key in ("attention", "messages", "activations")
        }

        # Pairwise tensors were normalized at entry, so the residual update is
        # always performed in the explicit ``(B, N, N, E)`` convention.
        edge = self.ln_edge(edge + edge_attention)
        node = self.ln1(node + node_attention)
        node = self.ln2(node + self.FFN(node))
        return [node, edge, distance, program_state]

class GRIT_encoder_last_layer(nn.Module):
    def __init__(self, node_dim, in_node, edge_dim, node_dim_small=16, att_dim=8):
        super().__init__()
        self.node_dim_small = node_dim_small
        self.in_node = in_node
        self.edge_dim = edge_dim * 2
        self.att_dim = att_dim
        self.W_Q = nn.Linear(node_dim, self.edge_dim * att_dim)
        self.W_K = nn.Linear(node_dim, self.edge_dim * att_dim)
        self.W_Ew = nn.Linear(edge_dim, edge_dim, bias=False)
        self.W_Eb = nn.Linear(edge_dim, edge_dim, bias=False)
        self.W_En = nn.Linear(edge_dim, node_dim)
        self.W_A = nn.Linear(edge_dim, in_node, bias=False)
        self.node_transform = nn.Linear(node_dim, in_node)
        self.edge_transform = nn.Linear(edge_dim, in_node)
        self.head = FFN(in_node, in_dim=in_node * 2, out_dim=in_node)
        self.head2 = nn.Sequential(nn.LayerNorm(in_node), FFN(in_node))
        self.scaler = 2 / (node_dim ** 0.5)

    def forward(self, x):
        node, edge, embedding = x
        B, N, C = node.shape
        Q = self.W_Q(node[:, 0:1, :]).reshape(B, 1, self.att_dim, self.edge_dim).permute(0, 3, 1, 2)
        K = self.W_K(node).reshape(B, N, self.att_dim, self.edge_dim).permute(0, 3, 2, 1)
        QK = (Q @ K).permute(0, 2, 3, 1)
        edge = edge[:, 0:1, :, :]
        edge = F.gelu(rho((QK[:, :, :, :self.edge_dim // 2]) * self.W_Ew(edge)) + self.W_Eb(edge) + QK[:, :, :, self.edge_dim // 2:] + embedding[:, 0:1, :, :])
        edge = edge[:, 0, 1:, :]
        alphas = F.softmax(self.W_A(edge).permute(0, 2, 1), dim=-1).permute(0, 2, 1)
        edge = self.edge_transform(edge) * alphas
        node = self.node_transform(node[:, 1:, :]) * alphas
        tmp = self.head(torch.concat([edge, node], dim=-1))
        node = torch.sum(tmp, dim=-2) * self.scaler
        return [node, [tmp.permute(0, 2, 1).unsqueeze(dim=-2), edge.permute(0, 2, 1).unsqueeze(dim=-2)]]

class ProgramLastLayer(nn.Module):
    def __init__(self, node_dim, in_node, edge_dim, n_programs=8, node_dim_small=16, att_dim=8):
        super().__init__()
        self.node_dim_small = node_dim_small
        self.in_node = in_node
        self.n_programs = n_programs
        self.edge_dim = edge_dim * 2
        self.att_dim = att_dim
        self.W_Q = nn.Linear(node_dim, self.edge_dim * att_dim)
        self.W_K = nn.Linear(node_dim, self.edge_dim * att_dim)
        self.W_Ew = nn.Linear(edge_dim, edge_dim, bias=False)
        self.W_Eb = nn.Linear(edge_dim, edge_dim, bias=False)
        self.W_A = nn.Linear(edge_dim, in_node * n_programs, bias=False)
        self.node_transform = nn.Linear(node_dim, in_node)
        self.edge_transform = nn.Linear(edge_dim, in_node)
        self.head = FFN(in_node, in_dim=in_node * 2, out_dim=in_node)
        self.scaler = 2 / (node_dim ** 0.5)

    def forward(self, x):
        node, edge, embedding = x
        B, N, C = node.shape
        Q = self.W_Q(node[:, 0:1, :]).reshape(B, 1, self.att_dim, self.edge_dim).permute(0, 3, 1, 2)
        K_mat = self.W_K(node).reshape(B, N, self.att_dim, self.edge_dim).permute(0, 3, 2, 1)
        QK = (Q @ K_mat).permute(0, 2, 3, 1)
        edge = edge[:, 0:1, :, :]
        edge = F.gelu(
            rho((QK[:, :, :, :self.edge_dim // 2]) * self.W_Ew(edge))
            + self.W_Eb(edge)
            + QK[:, :, :, self.edge_dim // 2:]
            + embedding[:, 0:1, :, :]
        )
        edge = edge[:, 0, 1:, :]
        alphas_all = self.W_A(edge)
        alphas_all = alphas_all.reshape(B, N - 1, self.in_node, self.n_programs)
        alphas_all = alphas_all.permute(0, 3, 2, 1)
        alphas_all = F.softmax(alphas_all, dim=-1)
        alphas_all = alphas_all.permute(0, 3, 1, 2)
        edge_feat = self.edge_transform(edge)
        node_feat = self.node_transform(node[:, 1:, :])
        program_outputs = []
        program_influences = []
        for k in range(self.n_programs):
            alpha_k = alphas_all[:, :, k, :]
            edge_k = edge_feat * alpha_k
            node_k = node_feat * alpha_k
            tmp_k = self.head(torch.cat([edge_k, node_k], dim=-1))
            output_k = torch.sum(tmp_k, dim=1) * self.scaler
            program_outputs.append(output_k)
            program_influences.append(tmp_k)
        y_pred = torch.stack(program_outputs, dim=0).sum(dim=0)
        total_influence = torch.stack(program_influences, dim=0).sum(dim=0)
        influence_for_compat = total_influence.permute(0, 2, 1).unsqueeze(dim=-2)
        program_info = {
            'per_program_influence': torch.stack(program_influences, dim=0),
            'per_program_output': torch.stack(program_outputs, dim=0),
            'per_program_attention': alphas_all,
        }
        return [y_pred, [influence_for_compat, program_info]]
