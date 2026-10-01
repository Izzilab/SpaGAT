"""R1.5 fixed-checkpoint controls; all outputs go to a new timestamped directory."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import gc
import hashlib
import json
import shutil
import time
import traceback
import uuid
import zipfile
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from r15_core import *
from r15_replay import MessageReplay, validate_replay, synthetic_replay_test
from spagat_r15.gene_program_model import SpaGP
import spagat_r15.attention as attention_module

BASE = Path('/content/drive/MyDrive')
DATA = BASE / 'GITIII-main/liver_workdir/gitiii/data/processed'
PRIOR = BASE / 'SpaGAT_revision/R3_5_7_gene_signed_20260928_090940_UTC'
CHECKPOINT = BASE / 'spagatv2/results/liver_free_k4_3epochs/free_seed123_best.pth'
CHECKPOINT_SHA = '2ab3dc98a627ca89c5b17179f71ce57ea47a5f2e36edd84eaab93434191228fe'
SPLIT_SHA = 'ac43227801123d1e2bceaa51ba230ec800062767124346c0d51753cf43eefaac'
BATCH_SIZE = 32
REPRO_ATOL = 1e-5
REPRO_RTOL = 2e-5
REPRO_MSE_ATOL = 1e-6
EXPECTED_COUNTS = {'tumor_1': 19659, 'tumor_2': 6892}
EXPECTED_FULL_MSE = {'tumor_1': 0.15171709428216743, 'tumor_2': 0.15173474350052685}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')


def load_torch(path):
    # No arbitrary pickle execution or unsafe fallback.
    core = np._core if hasattr(np, '_core') else np.core
    allowed = [argparse.Namespace, np.ndarray, np.dtype,
               (core.multiarray._reconstruct, 'numpy.core.multiarray._reconstruct'),
               (core.multiarray._reconstruct, 'numpy._core.multiarray._reconstruct'),
               (core.multiarray.scalar, 'numpy.core.multiarray.scalar'),
               (core.multiarray.scalar, 'numpy._core.multiarray.scalar')]
    if hasattr(np, 'dtypes'):
        for name in ('Int32DType', 'Int64DType', 'Float16DType', 'Float32DType', 'Float64DType', 'StrDType', 'BoolDType'):
            if hasattr(np.dtypes, name):
                allowed.append(getattr(np.dtypes, name))
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location='cpu', weights_only=True)


class CarcinomaData(Dataset):
    """Same CancerousLiver rows and global type IDs as the already verified export."""
    def __init__(self, cache, genes, names, meta):
        self.genes, self.names = list(genes), list(names)
        self.type_ids = {name: i for i, name in enumerate(names)}
        n = 460436
        self.exp = np.lib.format.open_memmap(cache / 'expression.npy', mode='w+', dtype='float32', shape=(n, len(genes)))
        self.neighbors = np.lib.format.open_memmap(cache / 'neighbors.npy', mode='w+', dtype='int64', shape=(n, 50))
        self.types = np.empty(n, np.int64)
        self.x, self.y = np.empty(n, np.float32), np.empty(n, np.float32)
        flags = np.empty(n, bool)
        path = DATA / 'CancerousLiver.csv'
        if path.stat().st_size != 9623364142:
            raise ValueError('CancerousLiver.csv size differs from the audited export')
        with np.load(DATA / 'CancerousLiver_TypeExp.npz', allow_pickle=False) as source:
            self.baseline = np.zeros((len(names), len(genes)), np.float32)
            available = set(source.files)
            for name in available:
                if name not in self.type_ids or source[name].shape != (len(genes),):
                    raise ValueError('Unexpected cell-type baseline')
                self.baseline[self.type_ids[name]] = source[name]
        if not np.isfinite(self.baseline).all():
            raise ValueError('Nonfinite baselines')
        header = list(pd.read_csv(path, nrows=0).columns)
        required = list(genes) + ['subclass', 'centerx', 'centery'] + [f'index_{j}' for j in range(50)]
        if not set(required).issubset(header):
            raise ValueError('Processed CSV schema differs from the verified source')
        # Use the two parsing passes from the verified R3 export exactly.
        # Metadata was read separately there, rather than alongside gene columns.
        metadata_columns = ['subclass', 'centerx', 'centery'] + (['flag'] if 'flag' in header else [])
        print('Reading metadata with the original R3 export parser.', flush=True)
        metadata = pd.read_csv(path, usecols=metadata_columns)
        if len(metadata) != n or metadata.subclass.isna().any() or not set(metadata.subclass).issubset(available):
            raise ValueError('Unexpected metadata rows or baseline coverage')
        flag = metadata['flag'] if 'flag' in metadata else pd.Series(True, index=metadata.index)
        if flag.isna().any() or not flag.isin([True, False, 0, 1]).all():
            raise ValueError('Unexpected receiver eligibility flag')
        flags[:] = flag.to_numpy(bool)
        self.types[:] = metadata.subclass.map(self.type_ids).to_numpy(np.int64)
        self.x[:], self.y[:] = metadata.centerx.to_numpy(np.float32), metadata.centery.to_numpy(np.float32)
        columns = list(genes) + [f'index_{j}' for j in range(50)]
        offset = 0
        print('Caching expression and neighbors with the original R3 export parser.', flush=True)
        for frame in pd.read_csv(path, usecols=columns, chunksize=4000):
            end = offset + len(frame)
            if end > n:
                raise ValueError('Unexpected source row count')
            values = frame.loc[:, genes].to_numpy(np.float32)
            neighbors = frame.loc[:, [f'index_{j}' for j in range(50)]].to_numpy()
            if not np.isfinite(values).all() or not np.isfinite(neighbors).all() or not np.equal(neighbors, np.floor(neighbors)).all():
                raise ValueError('Invalid expression or neighbor indices')
            neighbors = neighbors.astype(np.int64)
            if neighbors.min() < 0 or neighbors.max() >= n or not np.array_equal(neighbors[:, 0], np.arange(offset, end)):
                raise ValueError('Invalid center/neighbor row mapping')
            self.exp[offset:end] = values
            self.neighbors[offset:end] = neighbors
            offset = end
            if offset % 80000 == 0:
                print(f'  Cached {offset:,}/{n:,} source rows', flush=True)
        if offset != n or int(flags.sum()) != 460429:
            raise ValueError('CancerousLiver source/eligible row counts changed')
        self.exp.flush(); self.neighbors.flush()
        eligible = np.flatnonzero(flags)
        if not np.array_equal(eligible[meta.dataset_index.to_numpy()], meta.csv_row.to_numpy()):
            raise ValueError('Dataset index to CSV row mapping differs from prior export')
        if not np.array_equal(np.asarray(names)[self.types[meta.csv_row]], meta.cell_type.to_numpy()):
            raise ValueError('Receiver labels differ from prior export')
        self.meta = meta.reset_index(drop=True)
        self.rows = self.meta.csv_row.to_numpy(np.int64)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        nb = self.neighbors[self.rows[i]]
        types = self.types[nb]
        exp = np.asarray(self.exp[nb], dtype=np.float32)
        return dict(x=exp, type_exp=self.baseline[types], y=exp[0], cell_types=types,
                    position_x=self.x[nb], position_y=self.y[nb], neighbor_mask=np.ones(50, np.float32))


def check_saved_targets(ds, saved_target, genes, out):
    """Check every saved target before spending time on control inference."""
    for start in range(0, len(ds), 1024):
        end = min(start + 1024, len(ds))
        actual = np.asarray(ds.exp[ds.rows[start:end]])
        expected = np.asarray(saved_target[start:end])
        report, bad = discrepancy_report(actual, expected, atol=0., rtol=0.)
        if not report['passed']:
            report.update(stage='target_check', original_export_row_start=start)
            write_json(out / 'target_discrepancy.json', report)
            write_discrepancy_examples(actual, expected, bad, ds.meta.iloc[start:end], genes,
                                       out / 'target_discrepancy_examples.csv')
            raise ValueError('Saved targets differ: see target_discrepancy.json and examples; no predictions were accepted')
    return dict(passed=True, n_receivers=len(ds), n_genes=len(genes), exact_equality=True)


def write_discrepancy_examples(actual, expected, bad, metadata, genes, path):
    rows, columns = np.where(bad)
    examples = []
    for i, j in zip(rows[:100], columns[:100]):
        record = metadata.iloc[int(i)]
        examples.append(dict(dataset_index=int(record.dataset_index), csv_row=int(record.csv_row),
                             cell_type=record.cell_type, gene=genes[int(j)],
                             current=float(actual[i,j]), previous=float(expected[i,j]),
                             absolute_difference=float(abs(float(actual[i,j])-float(expected[i,j])))))
    pd.DataFrame(examples).to_csv(path, index=False)


@torch.inference_mode()
def verify_original_export(model, ds, saved_pred, saved_target, genes, out):
    """Keep all original export rows and original batch boundaries.

    Version 3 uses the empirically audited numerical reproduction tolerance,
    while separately checking exact targets and per-population aggregate MSE.
    The stricter original elementwise discrepancies are still reported.
    """
    reports = []
    first_failure_saved = False
    population_sums = {name: dict(current_sse=0., previous_sse=0., n_receivers=0)
                       for name in CONDITIONS[:2]}
    started = time.perf_counter()
    for bi, batch in enumerate(DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)):
        start = bi * BATCH_SIZE
        end = start + len(batch['y'])
        current = model({k:v.cuda() for k,v in batch.items()})[0].cpu().numpy()
        expected = np.asarray(saved_pred[start:end])
        report, bad = discrepancy_report(current, expected, atol=REPRO_ATOL, rtol=REPRO_RTOL)
        strict_report, _ = discrepancy_report(current, expected, atol=3e-6, rtol=REPRO_RTOL)
        report['original_tolerance_n_failing_values'] = strict_report['n_failing_values']
        report['original_tolerance_n_failing_rows'] = strict_report['n_failing_rows']
        report.update(batch=bi, original_export_row_start=start, n_receivers=end-start)
        reports.append(report)
        metadata = ds.meta.iloc[start:end]
        truth = np.asarray(saved_target[start:end]).astype(np.float64)
        has_both = (metadata.n_tumor1_neighbors.to_numpy() > 0) & (metadata.n_tumor2_neighbors.to_numpy() > 0)
        for name, sums in population_sums.items():
            selected = has_both & (metadata.cell_type.to_numpy() == name)
            sums['n_receivers'] += int(selected.sum())
            sums['current_sse'] += float(np.square(current[selected].astype(np.float64) - truth[selected]).sum())
            sums['previous_sse'] += float(np.square(expected[selected].astype(np.float64) - truth[selected]).sum())
        if not report['passed'] and not first_failure_saved:
            metadata = ds.meta.iloc[start:end]
            write_discrepancy_examples(current, expected, bad, metadata, genes,
                                       out / 'prediction_discrepancy_examples.csv')
            metadata.to_csv(out / 'first_failed_original_batch_metadata.csv', index=False)
            np.savez_compressed(out / 'first_failed_original_batch_predictions.npz',
                                current=current, previous=expected,
                                target=np.asarray(saved_target[start:end]),
                                dataset_indices=metadata.dataset_index.to_numpy(np.int64))
            first_failure_saved = True
        if (bi+1) % 250 == 0:
            print(f'  Original-order prediction check: {end:,}/{len(ds):,} rows', flush=True)
            pd.DataFrame(reports).to_csv(out / 'prediction_reproduction_batches.csv', index=False)
    frame = pd.DataFrame(reports)
    frame.to_csv(out / 'prediction_reproduction_batches.csv', index=False)
    populations = {}
    for name, sums in population_sums.items():
        n = sums['n_receivers']
        current_mse = sums['current_sse'] / (n * len(genes)) if n else float('nan')
        previous_mse = sums['previous_sse'] / (n * len(genes)) if n else float('nan')
        passed = (n == EXPECTED_COUNTS[name] and
                  np.isclose(current_mse, previous_mse, atol=REPRO_MSE_ATOL, rtol=0.) and
                  np.isclose(previous_mse, EXPECTED_FULL_MSE[name], atol=1e-6, rtol=1e-5))
        populations[name] = dict(n_receivers=n, current_MSE=current_mse,
                                previous_MSE=previous_mse, absolute_MSE_difference=abs(current_mse-previous_mse),
                                passed=bool(passed))
    mse_passed = all(item['passed'] for item in populations.values())
    result = dict(passed=bool(frame.passed.all()) and mse_passed,
                  elementwise_passed=bool(frame.passed.all()), population_MSE_passed=mse_passed,
                  population_MSE_checks=populations, population_MSE_absolute_tolerance=REPRO_MSE_ATOL,
                  n_receivers=int(frame.n_receivers.sum()),
                  n_failing_batches=int((~frame.passed).sum()),
                  n_failing_rows=int(frame.n_failing_rows.sum()),
                  n_failing_values=int(frame.n_failing_values.sum()),
                  maximum_absolute_difference=float(frame.maximum_absolute_difference.max()),
                  mean_absolute_difference=float(np.average(frame.mean_absolute_difference, weights=frame.n_receivers)),
                  original_tolerance_n_failing_values=int(frame.original_tolerance_n_failing_values.sum()),
                  original_tolerance_n_failing_rows=int(frame.original_tolerance_n_failing_rows.sum()),
                  elapsed_seconds=time.perf_counter()-started, atol=REPRO_ATOL, rtol=REPRO_RTOL,
                  batching='All original tumor export rows in their original order, batch size 32')
    write_json(out / 'prediction_reproduction.json', result)
    return result


def subset_cache(cache, indices):
    return dict(heads=[{k:v[indices] for k,v in h.items()} for h in cache['heads']],
                gates=cache['gates'][indices], offset=cache['offset'][indices],
                prediction=cache['prediction'][indices])


def summarize(acc, audit, out):
    rows = []
    values = acc.cpu().numpy()
    for ri, receiver in enumerate(CONDITIONS[:2]):
        for pi, policy in enumerate(POLICIES):
            for oi, operator in enumerate(OPERATORS):
                for ci, condition in enumerate(CONDITIONS):
                    for repeat in range(REPEATS):
                        sse, abs_change, signed_change, full_sse, n = values[ri, pi, oi, ci, repeat]
                        if n < 1:
                            raise ValueError('No receiver observations in a required analysis cell')
                        entries = n * 1000
                        rows.append(dict(receiver=receiver, selection=policy, operator=operator, condition=condition,
                            repeat=repeat, n_receivers=int(n), full_MSE=full_sse/entries, masked_MSE=sse/entries,
                            delta_MSE=(sse-full_sse)/entries, relative_MSE_increase_percent=100*(sse/full_sse-1),
                            mean_absolute_prediction_change=abs_change/entries, mean_signed_prediction_change=signed_change/entries))
    repeats = pd.DataFrame(rows)
    repeats.to_csv(out / 'control_repeats.csv', index=False)
    keys = ['receiver', 'selection', 'operator', 'condition']
    repeats.groupby(keys).agg(n_receivers=('n_receivers', 'first'),
        mean_delta_MSE=('delta_MSE', 'mean'), SD_delta_MSE=('delta_MSE', 'std'),
        mean_relative_MSE_increase_percent=('relative_MSE_increase_percent', 'mean'),
        SD_relative_MSE_increase_percent=('relative_MSE_increase_percent', 'std'),
        mean_absolute_prediction_change=('mean_absolute_prediction_change', 'mean'),
        SD_absolute_prediction_change=('mean_absolute_prediction_change', 'std'),
        mean_signed_prediction_change=('mean_signed_prediction_change', 'mean')).reset_index().to_csv(out / 'control_summary.csv', index=False)
    paired_keys = ['receiver', 'selection', 'operator', 'repeat']
    contrasts = []
    for field in ['relative_MSE_increase_percent', 'mean_absolute_prediction_change']:
        wide = repeats.pivot(index=paired_keys, columns='condition', values=field)
        for label, a, b in [('tumor2_minus_tumor1', 'tumor_2', 'tumor_1'),
                            ('tumor1_minus_random', 'tumor_1', 'random'), ('tumor2_minus_random', 'tumor_2', 'random')]:
            item = (wide[a] - wide[b]).rename('difference').reset_index()
            item['metric'], item['contrast'] = field, label
            contrasts.append(item)
    contrasts = pd.concat(contrasts, ignore_index=True)
    contrasts.to_csv(out / 'paired_contrast_repeats.csv', index=False)
    contrasts.groupby(['receiver', 'selection', 'operator', 'metric', 'contrast']).difference.agg(['mean', 'std', 'min', 'max']).reset_index().to_csv(out / 'paired_contrast_summary.csv', index=False)
    balance = []
    for (receiver, policy, condition, repeat), v in audit.items():
        nd, nr, rms, active_rms, count, min_d, max_d = v
        balance.append(dict(receiver=receiver, selection=policy, condition=condition, repeat=repeat,
            n_selected_sender_instances=int(count), mean_distance_raw=nd/count,
            mean_distance_over_neighborhood_radius=nr/count, mean_model_input_expression_RMS=rms/count,
            mean_model_visible_residual_RMS=active_rms/count, min_distance_raw=min_d, max_distance_raw=max_d))
    pd.DataFrame(balance).to_csv(out / 'selected_sender_balance.csv', index=False)
    return repeats


def plot_repeats(repeats, prior, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#20708A', '#C96E31', '#777777']
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ri, receiver in enumerate(CONDITIONS[:2]):
        for oi, operator in enumerate(OPERATORS):
            ax = axes[ri, oi]
            frame = repeats[(repeats.receiver == receiver) & (repeats.operator == operator)]
            for ci, condition in enumerate(CONDITIONS):
                for pi, policy in enumerate(POLICIES):
                    y = frame[(frame.condition == condition) & (frame.selection == policy)].sort_values('repeat').relative_MSE_increase_percent.to_numpy()
                    center = pi * 4 + ci
                    ax.scatter(center + np.linspace(-.13, .13, len(y)), y, s=14, color=colors[ci], alpha=.7)
                    ax.errorbar(center, y.mean(), yerr=y.std(ddof=1), color='black', fmt='_', capsize=3)
            ax.axhline(0, lw=.7, color='#999999')
            ax.set_title(receiver + ' | ' + operator.replace('_', ' '), fontsize=10)
            ax.set_xticks([0,1,2,4,5,6], ['T1','T2','Random','T1','T2','Random'])
            ax.set_xlabel('Count only (same m)        Count + distance bins')
            ax.set_ylabel('MSE increase relative to full prediction (%)')
    fig.suptitle('Same matched receiver cohort and removal counts; dots = 10 masking repeats', fontsize=11)
    fig.tight_layout(); fig.savefig(out / 'new_control_repeat_distributions.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    for ri, receiver in enumerate(CONDITIONS[:2]):
        frame = prior[prior.receiver == receiver]
        for mi, (metric, label) in enumerate([('relative_MSE_increase_percent','MSE increase (%)'),
                                            ('mean_abs_prediction_change','Mean absolute prediction change')]):
            ax = axes[ri, mi]
            for ci, condition in enumerate(CONDITIONS):
                y = frame[frame.condition == condition].sort_values('repeat')[metric].to_numpy()
                if len(y) != 10:
                    raise ValueError('Prior repeat table does not contain all ten repeats')
                ax.scatter(ci + np.linspace(-.12, .12, len(y)), y, color=colors[ci], s=20)
                ax.errorbar(ci, y.mean(), yerr=y.std(ddof=1), color='black', fmt='_', capsize=3)
            ax.set_title(receiver); ax.set_xticks([0,1,2], ['Tumor 1','Tumor 2','Random'])
            ax.set_ylabel(label)
    fig.suptitle('Original matched-count records: all 10 repeats, mean and sample SD', fontsize=11)
    fig.tight_layout(); fig.savefig(out / 'existing_matched_count_repeat_distributions.png', dpi=180); plt.close(fig)


def run_controls(source_hashes, frozen_attention, historical_repeats_text):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC') + '_' + uuid.uuid4().hex[:6]
    out = BASE / 'SpaGAT_revision' / ('R1_5_distance_message_controls_' + stamp)
    out.mkdir(parents=True, exist_ok=False)
    cache_dir = Path('/content') / ('R1_5_cache_' + stamp)
    cache_dir.mkdir(exist_ok=False)
    manifest = dict(status='running', analysis_version='3_audited_numerical_tolerance',
        new_training=False, new_inference=True, original_files_modified=False,
        checkpoint=str(CHECKPOINT), checkpoint_sha256=CHECKPOINT_SHA, split_sha256=SPLIT_SHA,
        prior_export=str(PRIOR), source_hashes=source_hashes,
        torch_version=torch.__version__, numpy_version=np.__version__, batch_size=BATCH_SIZE,
        repeats=REPEATS, training_seed=123, mask_seed=MASK_SEED,
        historical_prediction_tolerance=dict(atol=REPRO_ATOL, rtol=REPRO_RTOL,
                                             population_MSE_atol=REPRO_MSE_ATOL),
        numerical_tolerance_rationale='V2 verified all 76,753,000 targets exactly. Only 5 predictions in 2 rows exceeded atol=3e-6/rtol=2e-5; global max abs difference=7.718801498413086e-6, mean abs difference=4.369752660492709e-8. V3 uses atol=1e-5, retains original discrepancy counts, and adds per-population aggregate MSE checks before controls. No masking outcomes were available when this change was made.',
        distance_bin_width_fraction_of_radius=BIN_WIDTH, distance_bins=N_BINS,
        uncertainty='Repeat variation from neighbor selection; not biological confidence intervals.',
        scope='Within-specimen, existing-checkpoint sensitivity controls. Sender labels and model inputs are unchanged.',
        fixed_weight_control='Keep original softmax and gates; neutralize chosen sender-dependent node/edge evidence. Shared affine biases remain unchanged.',
        matching_scope='Distance bins and absolute counts; expression magnitude and removal fractions are audited but not matched.')
    started = time.perf_counter()
    replay = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('请选择 Colab GPU（例如 T4），本程序不在 CPU 上启动完整推理。')
        if BATCH_SIZE != 32:
            raise ValueError('Reproduction requires the original R3 export batch size of 32; do not change it')
        if shutil.disk_usage('/content').free < 6 * 1024**3:
            raise RuntimeError('Need at least 6 GiB free local runtime disk space')
        manifest['gpu'] = torch.cuda.get_device_name(0)
        torch.manual_seed(123)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        manifest['synthetic_forward_checks'] = synthetic_replay_test(SpaGP, frozen_attention, attention_module)
        print('Synthetic model replay/control checks passed.', flush=True)
        if sha(CHECKPOINT) != CHECKPOINT_SHA or sha(CHECKPOINT.parent / 'shared_split.json') != SPLIT_SHA:
            raise ValueError('Checkpoint/split hash differs; no automatic substitution')
        cp = load_torch(CHECKPOINT)
        split = json.loads((CHECKPOINT.parent / 'shared_split.json').read_text())
        val = np.asarray(split['val_indices'], np.int64)
        if not np.array_equal(val, np.asarray(cp['shared_split']['val_indices'])) or len(val) != 158652:
            raise ValueError('Validation split differs from the verified export')
        genes = list(load_torch(DATA.parent / 'genes.pth'))
        ligands = load_torch(DATA.parent / 'ligands.pth')
        prior_provenance = json.loads((PRIOR / 'provenance.json').read_text())
        if prior_provenance['selected_checkpoint'] != str(CHECKPOINT):
            raise ValueError('Prior export names a different checkpoint')
        if prior_provenance.get('batch_size') != BATCH_SIZE:
            raise ValueError('Prior export batch size differs')
        if genes != json.loads((PRIOR / 'genes.json').read_text()) or len(genes) != 1000:
            raise ValueError('Gene identities/order differ from prior export')
        meta_all = pd.read_csv(PRIOR / 'receiver_metadata.csv')
        old_ids = np.load(PRIOR / 'arrays/dataset_indices.npy', allow_pickle=False)
        if len(meta_all) != 76753 or not np.array_equal(old_ids, meta_all.dataset_index) or not np.array_equal(val[np.isin(val, old_ids)], old_ids):
            raise ValueError('Prior receiver indices/order changed')
        if meta_all.dataset_index.duplicated().any() or not (meta_all['sample'] == 'CancerousLiver').all():
            raise ValueError('Unexpected receiver metadata')
        eligible = (meta_all.n_tumor1_neighbors > 0) & (meta_all.n_tumor2_neighbors > 0)
        positions = np.flatnonzero(eligible)
        meta = meta_all.iloc[positions].copy().reset_index(drop=True)
        if meta.cell_type.value_counts().to_dict() != EXPECTED_COUNTS:
            raise ValueError('Eligible receiver counts differ from historical masking')
        manifest['prior_metadata_sha256'] = sha(PRIOR / 'receiver_metadata.csv')
        for name in ['prediction.npy', 'target.npy']:
            print('Copying existing ' + name + ' to runtime cache.', flush=True)
            shutil.copyfile(PRIOR / 'arrays' / name, cache_dir / name)
        saved_pred = np.load(cache_dir / 'prediction.npy', mmap_mode='r', allow_pickle=False)
        saved_target = np.load(cache_dir / 'target.npy', mmap_mode='r', allow_pickle=False)
        if saved_pred.shape != (76753, 1000) or saved_target.shape != saved_pred.shape:
            raise ValueError('Prior arrays have unexpected shapes')
        ds = CarcinomaData(cache_dir, genes, prior_provenance['cell_type_names'], meta_all)
        manifest['target_reproduction'] = check_saved_targets(ds, saved_target, genes, out)
        print('All 76,753 saved target rows match exactly.', flush=True)
        write_json(out / 'manifest.json', manifest)
        # Keep ds in its original 76,753-row order for identical inference batches.
        # Matching and metric denominators still use only the original 26,551 eligible receivers.
        neighbor_ids = ds.neighbors[meta.csv_row.to_numpy(np.int64)]
        types = ds.types[neighbor_ids]
        sender_ids = [ds.type_ids[x] for x in CONDITIONS[:2]]
        for k, column in enumerate(['n_tumor1_neighbors', 'n_tumor2_neighbors']):
            if not np.array_equal((types[:, 1:] == sender_ids[k]).sum(axis=1), meta[column]):
                raise ValueError('Neighbor composition differs from prior export')
        plan = distance_plan(types, ds.x[neighbor_ids], ds.y[neighbor_ids], sender_ids)
        active = plan['m'] > 0
        matching = meta.copy()
        matching['original_matched_count'] = plan['original_m']
        matching['distance_matched_count'] = plan['m']
        matching['distance_eligible'] = active
        matching['radius_in_saved_coordinate_units'] = plan['radius']
        matching['tumor1_removal_fraction'] = plan['m'] / meta.n_tumor1_neighbors
        matching['tumor2_removal_fraction'] = plan['m'] / meta.n_tumor2_neighbors
        matching.to_csv(out / 'receiver_matching.csv', index=False)
        coverage = []
        for receiver in CONDITIONS[:2]:
            in_type = meta.cell_type.to_numpy() == receiver
            use = in_type & active
            coverage.append(dict(receiver=receiver, original_eligible_receivers=int(in_type.sum()),
                distance_eligible_receivers=int(use.sum()), coverage_fraction=float(use.sum()/in_type.sum()),
                mean_original_matched_count=float(plan['original_m'][in_type].mean()),
                mean_control_removal_count=float(plan['m'][use].mean()) if use.any() else None,
                mean_tumor1_removal_fraction=float(matching.loc[use, 'tumor1_removal_fraction'].mean()),
                mean_tumor2_removal_fraction=float(matching.loc[use, 'tumor2_removal_fraction'].mean())))
        write_json(out / 'matching_coverage.json', coverage)
        manifest['matching_coverage'] = coverage
        if any(item['distance_eligible_receivers'] == 0 for item in coverage):
            raise ValueError('No overlapping distance support for one receiver population; do not widen bins after inspecting outcomes')
        print('Matching coverage:', coverage, flush=True)
        model = SpaGP(genes=genes, ligands_info=ligands, node_dim=256, edge_dim=48, num_heads=2,
            n_layers=1, att_dim=8, n_programs=4, program_aware=True, program_token_dim=32,
            message_decoder='free', routing_mode='program').cuda().eval()
        model.load_state_dict(cp['model'], strict=True)
        del cp; gc.collect()
        print('Verifying every original unmasked prediction BEFORE control analysis.', flush=True)
        manifest['prediction_reproduction'] = verify_original_export(model, ds, saved_pred, saved_target, genes, out)
        write_json(out / 'manifest.json', manifest)
        if not manifest['prediction_reproduction']['passed']:
            raise ValueError('Original-order prediction check failed; detailed numerical diagnostics are included, controls not started')
        print('All original predictions and per-population MSEs passed reproduction checks.', flush=True)
        replay = MessageReplay(model)
        chosen = np.concatenate([np.flatnonzero(active & (meta.cell_type.to_numpy() == r))[:8] for r in CONDITIONS[:2]])
        small = next(iter(DataLoader(torch.utils.data.Subset(ds, positions[chosen].tolist()), batch_size=len(chosen), shuffle=False)))
        cases = []
        for policy in POLICIES:
            masks = draw_masks(types[chosen], plan['bins'][chosen], plan['per_bin'][chosen], meta.dataset_index.to_numpy()[chosen], sender_ids, 0, policy)
            for ci, condition in enumerate(CONDITIONS):
                cases.append((policy + '_' + condition, torch.as_tensor(masks[ci], device='cuda')))
        manifest['forward_equivalence'] = validate_replay(model, replay, {k:v.cuda() for k,v in small.items()}, cases, frozen_attention, attention_module)
        print('Full-forward equivalence checks passed for both operators.', flush=True)
        write_json(out / 'manifest.json', manifest)
        acc = torch.zeros((2, 2, 2, 3, REPEATS, 5), dtype=torch.float64, device='cuda')
        audit = {}
        offset = 0; max_pred_error = 0.; max_target_error = 0.
        full_sse = np.zeros(2); full_counts = np.zeros(2, dtype=np.int64)
        evaluate_started = time.perf_counter()
        with torch.inference_mode():
            for bi, batch in enumerate(DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)):
                sl, local_positions = original_batch_selection(len(ds), positions, bi * BATCH_SIZE, BATCH_SIZE)
                size = len(sl)
                if not size:
                    continue
                reference = np.asarray(saved_pred[positions[sl]])
                target_reference = np.asarray(saved_target[positions[sl]])
                batch_gpu = {k:v.cuda() for k,v in batch.items()}
                original_cache = replay.capture(batch_gpu)
                local_gpu = torch.as_tensor(local_positions, device='cuda')
                cache = subset_cache(original_cache, local_gpu)
                batch_gpu = {k:v[local_gpu] for k,v in batch_gpu.items()}
                batch = {k:v[local_positions] for k,v in batch.items()}
                prediction = cache['prediction']
                target = batch_gpu['y']
                original = torch.as_tensor(reference, device='cuda')
                original_target = torch.as_tensor(target_reference, device='cuda')
                if not torch.allclose(prediction, original, atol=REPRO_ATOL, rtol=REPRO_RTOL) or not torch.equal(target, original_target):
                    np.savez_compressed(out / 'control_stage_reproduction_failure.npz',
                        current=prediction.cpu().numpy(), previous=reference,
                        current_target=target.cpu().numpy(), previous_target=target_reference,
                        original_export_positions=positions[sl])
                    raise ValueError('Control-stage reproduction failed after the full preflight; diagnostic arrays saved')
                max_pred_error = max(max_pred_error, float((prediction-original).abs().max()))
                max_target_error = max(max_target_error, float((target-original_target).abs().max()))
                receiver_types = meta.cell_type.to_numpy()[sl]
                for ri, receiver in enumerate(CONDITIONS[:2]):
                    take = receiver_types == receiver
                    take_gpu = torch.as_tensor(take, device='cuda')
                    full_sse[ri] += float(((prediction[take_gpu].double()-target[take_gpu].double())**2).sum())
                    full_counts[ri] += int(take.sum())
                use = active[sl]
                if use.any():
                    # Subset the captured factors; no resampling of receivers by condition.
                    use_gpu = torch.as_tensor(use, device='cuda')
                    small_cache = dict(heads=[{k:v[use_gpu] for k,v in h.items()} for h in cache['heads']],
                        gates=cache['gates'][use_gpu], offset=cache['offset'][use_gpu], prediction=prediction[use_gpu])
                    ids = sl[use]
                    yf = target[use_gpu].double(); pf = prediction[use_gpu].double()
                    groups = [receiver_types[use] == receiver for receiver in CONDITIONS[:2]]
                    gpu_groups = [torch.as_tensor(group, device='cuda') for group in groups]
                    full_errors = (pf-yf).square()
                    raw_x = batch['x'].numpy()[use]
                    baseline = batch['type_exp'].numpy()[use]
                    homo = types[ids] == types[ids, :1]
                    visible = np.where(homo[..., None], 0., raw_x)
                    input_rms = np.sqrt(np.mean((baseline.astype(np.float64) + visible)**2, axis=2))
                    residual_rms = np.sqrt(np.mean(visible.astype(np.float64)**2, axis=2))
                    for pi, policy in enumerate(POLICIES):
                        for repeat in range(REPEATS):
                            masks = draw_masks(types[ids], plan['bins'][ids], plan['per_bin'][ids], meta.dataset_index.to_numpy()[ids], sender_ids, repeat, policy)
                            if policy == POLICIES[1]:
                                for row in range(len(ids)):
                                    d0 = np.sort(plan['normalized'][ids[row]][masks[0,row,1:]])
                                    for ci in (1, 2):
                                        d1 = np.sort(plan['normalized'][ids[row]][masks[ci,row,1:]])
                                        if np.max(np.abs(d0-d1)) > BIN_WIDTH + 1e-12:
                                            raise AssertionError('Distance matching exceeds its declared tolerance')
                            for ci, condition in enumerate(CONDITIONS):
                                remove = torch.as_tensor(masks[ci], device='cuda')
                                for ri, receiver in enumerate(CONDITIONS[:2]):
                                    group = groups[ri]
                                    if not group.any():
                                        continue
                                    selected = masks[ci, group, 1:]
                                    raw_d = plan['distance'][ids[group]][selected]
                                    norm_d = plan['normalized'][ids[group]][selected]
                                    input_values = input_rms[group, 1:][selected]
                                    residual_values = residual_rms[group, 1:][selected]
                                    key = (receiver, policy, condition, repeat)
                                    if key not in audit:
                                        audit[key] = np.array([0.,0.,0.,0.,0.,np.inf,-np.inf])
                                    audit[key][:5] += [raw_d.sum(), norm_d.sum(), input_values.sum(), residual_values.sum(), len(raw_d)]
                                    audit[key][5] = min(audit[key][5], raw_d.min())
                                    audit[key][6] = max(audit[key][6], raw_d.max())
                                for oi, operator in enumerate(OPERATORS):
                                    perturbed = replay.predict(small_cache, remove, operator).double()
                                    change = perturbed - pf
                                    error = (perturbed-yf).square()
                                    for ri, group in enumerate(groups):
                                        n = int(group.sum())
                                        if n:
                                            gidx = gpu_groups[ri]
                                            vals = torch.stack([error[gidx].sum(), change[gidx].abs().sum(), change[gidx].sum(), full_errors[gidx].sum(), error.new_tensor(n)])
                                            acc[ri, pi, oi, ci, repeat] += vals
                offset += size
                if bi == 7 or (bi + 1) % 25 == 0:
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - evaluate_started
                    eta = elapsed / offset * (len(meta)-offset) / 60
                    print(f'  Evaluated {offset:,}/{len(meta):,} eligible receivers; estimated remaining inference {eta:.1f} min', flush=True)
                if (bi + 1) % 100 == 0:
                    write_json(out / 'progress.json', dict(receivers_done=offset, receivers_total=len(meta), elapsed_seconds=time.perf_counter()-evaluate_started))
                    np.save(cache_dir / 'running_metric_sums.npy', acc.cpu().numpy())
        verified = {}
        for ri, receiver in enumerate(CONDITIONS[:2]):
            value = full_sse[ri] / (full_counts[ri] * 1000)
            if full_counts[ri] != EXPECTED_COUNTS[receiver] or not np.isclose(value, EXPECTED_FULL_MSE[receiver], atol=1e-6, rtol=1e-5):
                raise ValueError('Full-cohort baseline MSE check failed')
            verified[receiver] = dict(n_receivers=int(full_counts[ri]), full_MSE=float(value))
        manifest['baseline_verification'] = dict(populations=verified, maximum_prediction_difference=max_pred_error, maximum_target_difference=max_target_error)
        repeats = summarize(acc, audit, out)
        for item in coverage:
            if not (repeats.loc[repeats.receiver == item['receiver'], 'n_receivers'] == item['distance_eligible_receivers']).all():
                raise AssertionError('A condition used a different receiver cohort')
        import io
        old = pd.read_csv(io.StringIO(historical_repeats_text))
        old.to_csv(out / 'existing_original_matched_count_repeats.csv', index=False)
        manifest['historical_repeat_table_sha256'] = hashlib.sha256(historical_repeats_text.encode()).hexdigest()
        try:
            plot_repeats(repeats, old, out)
            manifest['plot_status'] = 'complete'
        except Exception as exc:
            manifest['plot_status'] = 'tables_complete_plotting_failed'
            manifest['plot_error'] = str(exc)
            (out / 'plot_error_trace.txt').write_text(traceback.format_exc())
        manifest.update(status='complete', n_control_repeats=len(repeats), elapsed_seconds=time.perf_counter()-started)
        print('Completed. Original model, data, and previous results were not modified.', flush=True)
    except Exception as exc:
        manifest.update(status='stopped', error=str(exc), elapsed_seconds=time.perf_counter()-started)
        (out / 'error_trace.txt').write_text(traceback.format_exc())
        print('Stopped:', str(exc), 'Return the diagnostic ZIP; do not loosen verification thresholds.', flush=True)
    finally:
        if replay is not None:
            replay.close()
        source_dir = out / 'source'
        source_dir.mkdir(exist_ok=True)
        code_root = Path(__file__).parent
        for relative, expected in source_hashes.items():
            path = code_root / relative
            if path.is_file():
                target = source_dir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        write_json(out / 'manifest.json', manifest)
        (out / 'README.txt').write_text(
            'R1.5 sensitivity controls; existing fixed model, no training.\n'
            'Version 3 retains the original CSV parsing passes and original 76,753-row batch layout.\n'
            'Targets must be exactly equal. Historical prediction tolerance is atol=1e-5/rtol=2e-5 after auditing the V2 differences.\n'
            'Original tighter-tolerance discrepancy counts remain reported; population MSE agreement is checked before controls.\n'
            'Full-forward message equivalence retains its original atol=3e-6/rtol=2e-5; masks, cohorts, model and outcomes are unchanged.\n'
            'Both new selections use the same distance-eligible receivers and the same removal count per receiver.\n'
            'Distance bins have width 0.05 of each receiver neighborhood radius; bins are fixed before outcomes.\n'
            'The random distance control uses the same bin-wise counts. This is not a binned-label-permutation experiment.\n'
            'Both intervention operators use identical selected sender sets for each repeat.\n'
            'Fixed-weight messages preserves original softmax/gates and shared biases; it is not a biological cell-removal simulation.\n'
            'Expression magnitude and unequal removal fractions remain potential differences; balance and coverage are reported.\n'
            'No physical coordinate units are inferred; raw saved units and distances normalized by neighborhood radius are reported.\n'
            'Repeat SDs describe neighbor-selection variability, not training or biological replication.\n'
            'Existing-repeat plots use the original sender-origin repeat records and their original larger eligible cohort; do not treat them as the same cohort as new controls.\n'
            'Read status and forward/baseline verification in manifest.json before using the results.\n')
        archive = out.with_suffix('.zip')
        with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
            for path in sorted(out.rglob('*')):
                if path.is_file():
                    z.write(path, arcname=str(path.relative_to(out)))
        print('请上传结果 ZIP:', archive, flush=True)
    return archive
