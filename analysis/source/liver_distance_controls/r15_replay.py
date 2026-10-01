"""Exact single-layer free-decoder replay, checked against full forwards at runtime."""
import torch
import torch.nn.functional as F


class MessageReplay:
    def __init__(self, model):
        if len(model.encoders) != 1 or not model.program_aware or model.message_decoder != 'free' or model.routing_mode != 'program':
            raise ValueError('Replay is restricted to the verified one-layer program-aware free-decoder checkpoint')
        self.model = model
        self.encoder = model.encoders[0]
        self.heads = list(self.encoder.attentions.attentions)
        self.handles = []
        self.cache = None
        for h, head in enumerate(self.heads):
            self.handles.append(head.W_V.register_forward_hook(self._value_hook(h)))
            self.handles.append(head.W_Eo.register_forward_pre_hook(self._edge_hook(h)))
            self.handles.append(head.W_program_edge.register_forward_hook(self._projection_hook(h, 'edge_projection')))
            self.handles.append(head.W_program_receiver.register_forward_hook(self._projection_hook(h, 'receiver_projection')))
            self.handles.append(head.register_forward_hook(self._state_hook(h)))

    def _value_hook(self, h):
        def hook(module, inputs, output):
            if self.cache is not None:
                self.cache[h]['V'] = output.detach()
        return hook

    def _edge_hook(self, h):
        def hook(module, inputs):
            if self.cache is not None:
                self.cache[h]['edge'] = inputs[0][:, 0].detach().clone()
        return hook

    def _projection_hook(self, h, key):
        def hook(module, inputs, output):
            if self.cache is not None:
                self.cache[h][key] = output[:, 0].detach().clone()
        return hook

    def _state_hook(self, h):
        def hook(module, inputs, output):
            if self.cache is not None:
                self.cache[h]['alpha'] = output[2]['attention'][:, 0].detach().clone()
                values = self.cache[h]
                values['logits'] = torch.einsum('bje,be,ke->bjk', values['edge_projection'],
                    values['receiver_projection'], inputs[1]) / (module.base_edge_dim ** 0.5)
                del values['edge_projection'], values['receiver_projection']
        return hook

    @torch.inference_mode()
    def capture(self, batch):
        self.cache = [{} for _ in self.heads]
        prediction, info = self.model(batch)
        cache = self.cache
        self.cache = None
        g = info['activations'].detach()
        return dict(heads=cache, gates=g / (g.sum(dim=-1, keepdim=True) + 1e-8),
                    offset=info['baseline'].detach() + info['program_output'].detach(),
                    prediction=prediction.detach(), reported_attention=info['program_attention'].detach(),
                    reported_activations=g, reported_baseline=info['baseline'].detach())

    @torch.inference_mode()
    def predict(self, cache, remove, operator):
        if operator not in ('edge_exclusion', 'fixed_weight_messages'):
            raise ValueError(operator)
        keep = (~remove).unsqueeze(-1)
        messages = []
        for head, values in zip(self.heads, cache['heads']):
            if operator == 'edge_exclusion':
                # Recompute softmax from cached scores, avoiding division by a tiny retained mass.
                alpha = F.softmax(values['logits'].masked_fill(~keep, float('-inf')), dim=1)
            else:
                alpha = values['alpha'] * keep
            node_message = torch.einsum('bjk,bjd->bkd', alpha, values['V'])
            edge_message = torch.einsum('bjk,bje->bke', alpha, values['edge'])
            # Shared affine bias is preserved; only sender-dependent evidence is neutralized.
            messages.append(node_message + head.W_En(edge_message))
        head_weights = F.softmax(self.encoder.attentions.W_hn.squeeze(-1), dim=0)
        combined = (torch.stack(messages, dim=-1) * head_weights).sum(dim=-1)
        fused = (combined * cache['gates'].unsqueeze(-1)).sum(dim=1)
        result = cache['offset'] + self.model.program_head.message_to_genes(fused)
        if not torch.isfinite(result).all():
            raise ValueError('Nonfinite replay prediction')
        return result

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def patched_neutral_forward(attention_source, attention_module):
    """Build a separately executed full-model control from the audited original method."""
    import ast
    import textwrap
    tree = ast.parse(attention_source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GRIT_attention')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    source = textwrap.dedent('\n'.join(attention_source.splitlines()[fn.lineno - 1:fn.end_lineno]))
    marker = '    # Aggregate node and edge evidence independently for each program:'
    if source.count(marker) != 1:
        raise ValueError('Frozen attention implementation differs from the audited version')
    insertion = '''    message_attention = program_attention
    neutral_mask = getattr(self, '_r15_remove', None)
    if neutral_mask is not None:
        if program_edge_mask is not None:
            raise ValueError('Do not combine interventions')
        message_attention = program_attention.clone()
        message_attention[:, 0] = message_attention[:, 0] * (~neutral_mask).unsqueeze(-1)

'''
    source = source.replace(marker, insertion + marker)
    source = source.replace('"bijk,bjd->bikd", program_attention, V', '"bijk,bjd->bikd", message_attention, V')
    source = source.replace('"bijk,bije->bike", program_attention, edge', '"bijk,bije->bike", message_attention, edge')
    if source.count('message_attention, V') != 1 or source.count('message_attention, edge') != 1:
        raise ValueError('Expected aggregation sites not found')
    namespace = dict(vars(attention_module))
    exec(compile(source, '<audited-neutral-message-forward>', 'exec'), namespace)
    return namespace['forward']


@torch.inference_mode()
def validate_replay(model, replay, batch, selection_masks, attention_source, attention_module):
    """Full original and separately patched forwards must agree before analyses run."""
    import types
    cache = replay.capture(batch)
    full = cache['prediction']
    reports = []

    def check(name, actual, expected):
        error = float((actual - expected).abs().max())
        ok = bool(torch.allclose(actual, expected, atol=3e-6, rtol=2e-5))
        reports.append(dict(check=name, maximum_absolute_difference=error, passed=ok))
        if not ok:
            raise ValueError(f'Forward-equivalence check failed: {name}; max error {error}')

    empty = torch.zeros(full.shape[0], 50, dtype=torch.bool, device=full.device)
    check('identity_edge_replay', replay.predict(cache, empty, 'edge_exclusion'), full)
    check('identity_message_replay', replay.predict(cache, empty, 'fixed_weight_messages'), full)
    full_mask = torch.ones(full.shape[0], 50, 50, model.n_programs, dtype=torch.bool, device=full.device)
    check('original_all_edges_present', model(batch, program_edge_mask=full_mask)[0], full)

    # Contains actual count-only and distance-bin masks, plus one fixed neighbor.
    cases = list(selection_masks)
    one = empty.clone(); one[:, 1] = True
    cases.append(('one_neighbor', one))
    for name, remove in cases:
        allowed = full_mask.clone()
        allowed[:, 0] = (~remove).unsqueeze(-1).expand(-1, -1, model.n_programs)
        reference = model(batch, program_edge_mask=allowed)[0]
        check('edge_' + name, replay.predict(cache, remove, 'edge_exclusion'), reference)

    # Patch only in-memory copies of methods, restore even on an exception.
    forward = patched_neutral_forward(attention_source, attention_module)
    originals = [head.forward for head in replay.heads]
    try:
        for head in replay.heads:
            head.forward = types.MethodType(forward, head)
        check('neutral_full_identity', model(batch)[0], full)
        for name, remove in cases:
            for head in replay.heads:
                head._r15_remove = remove
            reference, info = model(batch)
            check('message_' + name, replay.predict(cache, remove, 'fixed_weight_messages'), reference)
            for key, field in [('program_attention','reported_attention'), ('activations','reported_activations'), ('baseline','reported_baseline')]:
                unchanged = bool(torch.equal(info[key], cache[field]))
                reports.append(dict(check='neutral_preserves_' + key + '_' + name, passed=unchanged))
                if not unchanged:
                    raise ValueError('Fixed-weight control changed ' + key)
    finally:
        for head, original in zip(replay.heads, originals):
            head.forward = original
            if hasattr(head, '_r15_remove'):
                delattr(head, '_r15_remove')
    check('original_forward_restored', model(batch)[0], full)
    return reports


def synthetic_replay_test(model_class, attention_source, attention_module):
    """Runs in Colab before reading the large data file; no optimizer or training."""
    torch.manual_seed(20260928)
    genes = ['gene_a', 'gene_b', 'gene_c', 'gene_d']
    ligands = ([['gene_a'], ['gene_b']], [[0], [0]])
    model = model_class(genes, ligands, node_dim=8, edge_dim=4, num_heads=2, n_layers=1,
        att_dim=2, n_programs=3, program_aware=True, program_token_dim=5,
        message_decoder='free', routing_mode='program').cuda().eval()
    batch = dict(x=torch.randn(4,50,4,device='cuda') * .1,
        type_exp=torch.rand(4,50,4,device='cuda') + 1,
        cell_types=torch.randint(0,3,(4,50),device='cuda'),
        position_x=torch.rand(4,50,device='cuda') * 20,
        position_y=torch.rand(4,50,device='cuda') * 20,
        y=torch.zeros(4,4,device='cuda'))
    remove = torch.zeros(4,50,dtype=torch.bool,device='cuda'); remove[:, [2,4,7,11]] = True
    all_neighbors = torch.ones_like(remove); all_neighbors[:,0] = False
    replay = MessageReplay(model)
    try:
        reports = validate_replay(model, replay, batch, [('four_neighbors',remove),('all_noncenter_neighbors',all_neighbors)], attention_source, attention_module)
    finally:
        replay.close()
    del model, batch, replay
    torch.cuda.empty_cache()
    torch.manual_seed(123)
    return reports
