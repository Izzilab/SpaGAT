"""Distance-stratified mask selection. No training or outcome-dependent selection."""
import numpy as np

BIN_WIDTH = 0.05
N_BINS = 20
REPEATS = 10
MASK_SEED = 123
POLICIES = ('count_only_same_m', 'count_distance_bins')
CONDITIONS = ('tumor_1', 'tumor_2', 'random')
OPERATORS = ('edge_exclusion', 'fixed_weight_messages')


def original_batch_selection(n_original, eligible_positions, batch_start, batch_size):
    """Map a full, original export batch to positions in the eligible cohort."""
    positions = np.asarray(eligible_positions, dtype=np.int64)
    end = min(batch_start + batch_size, n_original)
    lo, hi = np.searchsorted(positions, [batch_start, end])
    local = positions[lo:hi] - batch_start
    return np.arange(lo, hi, dtype=np.int64), local


def discrepancy_report(actual, expected, atol, rtol):
    """Report discrepancies without changing the original acceptance threshold."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    close = np.isclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False)
    return dict(passed=bool(close.all()), n_failing_values=int((~close).sum()),
                n_failing_rows=int((~close).any(axis=1).sum()),
                maximum_absolute_difference=float(difference.max()),
                mean_absolute_difference=float(difference.mean()),
                atol=atol, rtol=rtol), ~close


def distance_plan(types, px, py, sender_ids):
    """All 50 positions include center 0. Bins are fixed before seeing predictions."""
    types = np.asarray(types)
    px, py = np.asarray(px, dtype=np.float64), np.asarray(py, dtype=np.float64)
    if types.ndim != 2 or types.shape[1] != 50 or px.shape != types.shape or py.shape != types.shape:
        raise ValueError('Expected 50-node neighborhoods')
    if not np.isfinite(px).all() or not np.isfinite(py).all():
        raise ValueError('Nonfinite coordinates')
    dist = np.hypot(px[:, 1:] - px[:, :1], py[:, 1:] - py[:, :1])
    radius = dist.max(axis=1)
    normalized = np.divide(dist, radius[:, None], out=np.zeros_like(dist), where=radius[:, None] > 0)
    bins = np.minimum(np.floor(normalized / BIN_WIDTH).astype(np.int64), N_BINS - 1)
    counts = np.zeros((len(types), N_BINS, 2), dtype=np.int64)
    for b in range(N_BINS):
        for k, ident in enumerate(sender_ids):
            counts[:, b, k] = ((bins == b) & (types[:, 1:] == ident)).sum(axis=1)
    per_bin = counts.min(axis=2)
    m = per_bin.sum(axis=1)
    original_m = counts.sum(axis=1).min(axis=1)
    assert np.all(m <= original_m)
    return dict(distance=dist, normalized=normalized, radius=radius, bins=bins,
                per_bin=per_bin, m=m, original_m=original_m)


def draw_masks(types, bins, per_bin, dataset_ids, sender_ids, repeat, policy):
    """Return remove masks (condition, receiver, node); shared by both operators."""
    if policy not in POLICIES:
        raise ValueError(policy)
    masks = np.zeros((3, len(types), 50), dtype=bool)
    for r, dataset_id in enumerate(dataset_ids):
        m = int(per_bin[r].sum())
        if m < 1:
            raise ValueError('Receiver has no overlapping distance bins')
        for c in range(3):
            rng = np.random.default_rng(np.random.SeedSequence([MASK_SEED, int(dataset_id), int(repeat), POLICIES.index(policy), c]))
            allowed = np.ones(49, dtype=bool) if c == 2 else types[r, 1:] == sender_ids[c]
            if policy == POLICIES[0]:
                chosen = rng.choice(np.flatnonzero(allowed), size=m, replace=False)
            else:
                chosen = np.concatenate([
                    rng.choice(np.flatnonzero(allowed & (bins[r] == b)), size=int(k), replace=False)
                    for b, k in enumerate(per_bin[r]) if k
                ])
            masks[c, r, chosen + 1] = True
            if len(chosen) != m or len(np.unique(chosen)) != m:
                raise AssertionError('Wrong removal count')
            if policy == POLICIES[1]:
                if not np.array_equal(np.bincount(bins[r, chosen], minlength=N_BINS), per_bin[r]):
                    raise AssertionError('Distance-bin counts differ')
    if masks[:, :, 0].any() or not np.all(masks.sum(axis=2) == per_bin.sum(axis=1)[None, :]):
        raise AssertionError('Receiver selected or unequal removal counts')
    return masks
