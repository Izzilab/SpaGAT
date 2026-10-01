from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, re, shutil, time, traceback, uuid, zipfile
import numpy as np
import pandas as pd

BASE = Path('/content/drive/MyDrive')
INPUT = BASE/'SpaGAT_revision/R3_5_7_gene_signed_20260928_090940_UTC'
RECEIVERS = ['tumor_1', 'tumor_2']
MIN_PROGRAM_GENES = 5
BLOCK = 2048
GMT_URL = 'https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt'
GMT_SHA = 'eecaf6dad908334ae885406ec72bdc0646d8917588ed7c219fac92fc5363f596'


def save_json(p, data):
    Path(p).write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def stage(src, cache, manifest):
    src = Path(src)
    if not src.is_file():
        raise FileNotFoundError(f'缺少已保存数组：{src}；不会重新推理。')
    dest = cache/src.name
    h = hashlib.sha256()
    with src.open('rb') as a, dest.open('xb') as b:
        for chunk in iter(lambda: a.read(4*1024*1024), b''):
            b.write(chunk)
            h.update(chunk)
    manifest.append(dict(path=str(src), bytes=src.stat().st_size, sha256=h.hexdigest()))
    return np.load(dest, mmap_mode='r', allow_pickle=False)


def read_sets(text, genes):
    sets = {}
    for line in text.splitlines():
        row = line.split('\t')
        if len(row) < 3 or row[0] in sets:
            raise ValueError('Invalid or duplicate GMT row')
        sets[row[0]] = dict(url=row[1], genes=sorted(set(row[2:])))
    universe = set(g for s in sets.values() for g in s['genes'])
    # Only exact symbols or unambiguous hyphen-to-dot column-name transformations.
    # No fuzzy matching or inferred aliases.
    aliases = {}
    for symbol in universe:
        aliases.setdefault(symbol.replace('-', '.'), set()).add(symbol)
    mapping = []
    for gene in genes:
        candidates = aliases.get(gene, set())
        mapped = gene if gene in universe else next(iter(candidates)) if len(candidates) == 1 else None
        how = 'exact' if gene in universe else 'unique_hyphen_to_dot' if mapped else 'not_in_library_or_ambiguous'
        mapping.append(dict(panel_gene=gene, library_symbol=mapped, mapping=how))
    used = [x['library_symbol'] for x in mapping if x['library_symbol']]
    if len(used) != len(set(used)):
        raise ValueError('Multiple panel columns map to the same biological symbol')
    rows, members = [], {}
    for name, item in sorted(sets.items()):
        inds = [i for i, m in enumerate(mapping) if m['library_symbol'] in item['genes']]
        members[name] = inds
        rows.append(dict(program=name, full_set_size=len(item['genes']), panel_genes=len(inds),
                         panel_coverage=len(inds)/len(item['genes']), gene_set_url=item['url'],
                         panel_gene_list=';'.join(genes[i] for i in inds),
                         library_gene_list=';'.join(mapping[i]['library_symbol'] for i in inds)))
    return pd.DataFrame(mapping), pd.DataFrame(rows), members


def moments(pred, obs, indices):
    n, g = len(indices), pred.shape[1]
    if n < 2:
        raise ValueError('Need at least two cells')
    sx = np.zeros(g); sy = np.zeros(g); err = np.zeros(g); zero = np.zeros(g)
    neg = np.zeros(g, dtype=np.int64)
    for start in range(0, n, BLOCK):
        ids = indices[start:start+BLOCK]
        p, y = np.asarray(pred[ids], float), np.asarray(obs[ids], float)
        if not np.isfinite(p).all() or not np.isfinite(y).all():
            raise ValueError('Nonfinite predictions or targets')
        sx += p.sum(0); sy += y.sum(0); err += ((p-y)**2).sum(0); zero += (y*y).sum(0)
        neg += (y < 0).sum(0)
    mp, my = sx/n, sy/n
    vp = np.zeros(g); vy = np.zeros(g); cov = np.zeros(g)
    for start in range(0, n, BLOCK):
        ids = indices[start:start+BLOCK]
        p, y = np.asarray(pred[ids], float)-mp, np.asarray(obs[ids], float)-my
        vp += (p*p).sum(0); vy += (y*y).sum(0); cov += (p*y).sum(0)
    den = np.sqrt(vp*vy)
    pcc = np.divide(cov, den, out=np.full(g, np.nan), where=den>0)
    return dict(n=n, pred_mean=mp, obs_mean=my, pred_var=vp/n, obs_var=vy/n,
                MSE=err/n, MSE0=zero/n, PCC=pcc, eps_PCC=cov/(den+1e-8),
                negative_fraction=neg/n)


def weights_for_programs(m, members):
    names = sorted(members)
    raw = np.zeros((len(m['PCC']), len(names)))
    z = np.zeros_like(raw)
    sd = np.sqrt(m['obs_var'])
    valid_sd = np.isfinite(sd) & (sd > 0)
    details = []
    for k, name in enumerate(names):
        ids = np.asarray(members[name], dtype=int)
        usable = ids[valid_sd[ids]]
        raw_ok, z_ok = len(ids) >= MIN_PROGRAM_GENES, len(usable) >= MIN_PROGRAM_GENES
        if raw_ok:
            raw[ids, k] = 1/len(ids)
        if z_ok:
            z[usable, k] = 1/(len(usable)*sd[usable])
        details.append(dict(program=name, raw_evaluable=raw_ok, standardized_evaluable=z_ok,
                            standardized_genes=len(usable), excluded_zero_variance=len(ids)-len(usable),
                            passes_10_gene_sensitivity=len(usable)>=10))
    return names, raw, z, pd.DataFrame(details)


def project(arr, indices, weights):
    result = np.empty((len(indices), weights.shape[1]), dtype=np.float64)
    for start in range(0, len(indices), BLOCK):
        result[start:start+BLOCK] = np.asarray(arr[indices[start:start+BLOCK]], float) @ weights
    return result


def correlations(p, y):
    a, b = p-p.mean(), y-y.mean()
    den = np.sqrt(np.dot(a, a)*np.dot(b, b))
    return float(np.dot(a, b)/den) if den > 0 else np.nan


def score_metrics(p, y):
    mse = float(np.mean((p-y)**2)); mse0 = float(np.mean(y*y))
    # Inputs share observed-derived centering/scaling; predictions are not restandardized.
    nz = np.abs(y)>1e-12
    return dict(PCC=correlations(p,y),
                Spearman=correlations(pd.Series(p).rank().to_numpy(),pd.Series(y).rank().to_numpy()),
                MSE=mse, MSE0=mse0, EV_percent=100*(1-mse/mse0) if mse0>0 else np.nan,
                observed_mean=float(y.mean()), predicted_mean=float(p.mean()),
                observed_variance=float(y.var()), predicted_variance=float(p.var()),
                direction_agreement=float(np.mean(np.sign(p[nz])==np.sign(y[nz]))) if nz.any() else np.nan,
                direction_n=int(nz.sum()))


def program_recovery(pred, obs, meta, genes, members, coverage, out):
    metrics, audits, coverage_rows, scalings, receiver_models = [], [], [], [], {}
    for receiver in RECEIVERS:
        idx = np.flatnonzero(meta.cell_type.to_numpy()==receiver)
        m = moments(pred, obs, idx)
        old = pd.read_csv(INPUT/f'{receiver}_gene_recovery.csv').set_index('gene').loc[genes]
        for col, values in [('PCC',m['PCC']),('MSE',m['MSE']),('observed_mean',m['obs_mean'])]:
            if not np.allclose(old[col], values, rtol=1e-5, atol=1e-7, equal_nan=True):
                raise ValueError(f'{receiver} 数组与先前基因表不一致：{col}')
        pd.DataFrame(dict(gene=genes, observed_mean=m['obs_mean'], observed_sd=np.sqrt(m['obs_var']),
                          observed_negative_fraction=m['negative_fraction'])).to_csv(out/f'{receiver}_target_diagnostics.csv',index=False)
        names, raw, z, detail = weights_for_programs(m, members)
        info = coverage.merge(detail,on='program'); info.insert(0,'receiver',receiver); coverage_rows.append(info)
        scalings.append(pd.DataFrame(dict(receiver=receiver,gene=genes,observed_mean=m['obs_mean'],observed_sd=np.sqrt(m['obs_var']))))
        for label, W, center, valid_col in [('observed_z_mean',z,m['obs_mean']@z,'standardized_evaluable'),
                                          ('raw_mean',raw,np.zeros(len(names)),'raw_evaluable')]:
            yp = project(pred,idx,W)-center
            yt = project(obs,idx,W)-center
            for k,name in enumerate(names):
                valid = bool(detail.loc[k,valid_col])
                row = dict(receiver=receiver, program=name, score=label, n_cells=len(idx), evaluable=valid,
                           direction_reference='observed_population_mean' if label=='observed_z_mean' else 'original_target_zero')
                if valid:
                    row.update(score_metrics(yp[:,k],yt[:,k]))
                metrics.append(row)
            # Save modest score arrays in the new Drive folder, not the downloadable ZIP.
            np.savez_compressed(out/'score_arrays'/f'{receiver}_{label}.npz',
                                prediction=yp.astype('float32'),observed=yt.astype('float32'),
                                dataset_indices=meta.dataset_index.to_numpy()[idx],programs=np.asarray(names))
        receiver_models[receiver]=(names,z,detail)
        manuscript={'tumor_1':.267,'tumor_2':.277}[receiver]
        for subset, inds in [('all_validation',idx),('both_senders_present',idx[meta.masking_eligible.to_numpy()[idx]])]:
            mm=m if subset=='all_validation' else moments(pred,obs,inds)
            for rule,vals in [('Pearson_float64',mm['PCC']),('denominator_plus_1e-8',mm['eps_PCC'])]:
                vals=vals[np.isfinite(vals)]
                for median_rule,v in [('numpy_median',np.median(vals)),('lower_middle_like_torch',np.sort(vals)[(len(vals)-1)//2])]:
                    audits.append(dict(receiver=receiver,subset=subset,n_cells=len(inds),PCC_rule=rule,
                                       median_rule=median_rule,median_PCC=v,manuscript_Figure4=manuscript,
                                       difference_from_manuscript=v-manuscript))
        print(receiver, 'program recovery 完成；median gene PCC =', round(float(np.nanmedian(m['PCC'])),6),flush=True)
    result=pd.DataFrame(metrics)
    result.to_csv(out/'program_recovery.csv',index=False)
    pd.concat(coverage_rows).to_csv(out/'program_coverage.csv',index=False)
    pd.concat(scalings).to_csv(out/'program_score_scaling.csv',index=False)
    pd.DataFrame(audits).to_csv(out/'Figure4_metric_audit.csv',index=False)
    return receiver_models, result


def signed_programs(meta, genes, models, out, cache, manifest):
    repeats=pd.read_csv(INPUT/'signed_masking_gene_repeats.csv')
    if repeats.duplicated(['receiver','condition','repeat','gene']).any():
        raise ValueError('Duplicate signed masking records')
    program_repeats=[]; cellwise=[]
    for receiver in RECEIVERS:
        names,W,detail=models[receiver]
        wanted=meta[(meta.cell_type==receiver)&meta.masking_eligible].dataset_index.to_numpy()
        for condition in ['tumor_1','tumor_2','random']:
            prefix=f'{receiver}__{condition}'
            ids=np.load(INPUT/'arrays'/f'{prefix}__dataset_indices.npy',allow_pickle=False)
            if not np.array_equal(ids,wanted):
                raise ValueError('Signed array receiver order mismatch')
            arr=stage(INPUT/'arrays'/f'{prefix}__mean_delta.npy',cache,manifest)
            if arr.shape!=(len(wanted),len(genes)):
                raise ValueError('Signed array shape mismatch')
            scores=project(arr,np.arange(len(arr)),W)
            rt=repeats[(repeats.receiver==receiver)&(repeats.condition==condition)]
            table=rt.pivot(index='repeat',columns='gene',values='mean_signed_delta').reindex(columns=genes)
            if table.shape!=(10,len(genes)) or not np.isfinite(table.to_numpy()).all():
                raise ValueError('Missing repeat/gene combinations')
            projected=table.to_numpy()@W
            if not np.allclose(scores.mean(0),projected.mean(0),rtol=2e-5,atol=2e-6):
                raise ValueError('Program signed mean does not reconcile with gene-level repeat means')
            for k,name in enumerate(names):
                if not detail.loc[k,'standardized_evaluable']:
                    continue
                for rep,val in zip(table.index,projected[:,k]):
                    program_repeats.append(dict(receiver=receiver,condition=condition,repeat=int(rep),program=name,
                                                n_receivers=len(ids),mean_signed_score_delta=float(val)))
                cellwise.append(dict(receiver=receiver,condition=condition,program=name,n_receivers=len(ids),
                    mean_absolute_of_cellwise_repeat_mean_delta=float(np.abs(scores[:,k]).mean()),
                    fraction_cells_positive_repeat_mean_delta=float((scores[:,k]>0).mean()),
                    fraction_cells_negative_repeat_mean_delta=float((scores[:,k]<0).mean())))
            del arr,scores
        print(receiver,'signed program summary 完成',flush=True)
    rt=pd.DataFrame(program_repeats)
    rt.to_csv(out/'signed_program_repeats.csv',index=False)
    s=rt.groupby(['receiver','condition','program']).agg(
        n_receivers=('n_receivers','first'),mean_signed_score_delta=('mean_signed_score_delta','mean'),
        SD_across_masking_repeats=('mean_signed_score_delta','std'),
        min_repeat_mean=('mean_signed_score_delta','min'),max_repeat_mean=('mean_signed_score_delta','max')).reset_index()
    s=s.merge(pd.DataFrame(cellwise),on=['receiver','condition','program','n_receivers'])
    s['same_sign_in_all_repeat_means']=(s.min_repeat_mean>0)|(s.max_repeat_mean<0)
    s.to_csv(out/'signed_program_summary.csv',index=False)
    random=rt[rt.condition=='random'][['receiver','repeat','program','mean_signed_score_delta']].rename(columns={'mean_signed_score_delta':'random_delta'})
    control=rt[rt.condition!='random'].merge(random,on=['receiver','repeat','program'],validate='many_to_one')
    control['signed_delta_minus_random']=control.mean_signed_score_delta-control.random_delta
    control.to_csv(out/'signed_program_minus_random_repeats.csv',index=False)
    control.groupby(['receiver','condition','program']).signed_delta_minus_random.agg(['mean','std']).reset_index().to_csv(out/'signed_program_minus_random_summary.csv',index=False)
    # Identical-count random control is descriptive. Differences are not isolated causal effects.
    for receiver in RECEIVERS:
        rec=pd.read_csv(INPUT/f'{receiver}_gene_recovery.csv')
        a=pd.read_csv(INPUT/'signed_masking_gene_summary.csv')
        a=a[a.receiver==receiver].merge(rec[['gene','PCC','EV_percent']],on='gene',validate='many_to_one')
        rand=a[a.condition=='random'][['gene','mean_signed_delta','mean_absolute_delta']].rename(columns={
            'mean_signed_delta':'random_mean_signed_delta','mean_absolute_delta':'random_mean_absolute_delta'})
        a=a.merge(rand,on='gene',validate='many_to_one')
        a['signed_delta_minus_random']=a.mean_signed_delta-a.random_mean_signed_delta
        a['absolute_delta_minus_random']=a.mean_absolute_delta-a.random_mean_absolute_delta
        a['rank_mean_absolute_delta']=a.groupby('condition').mean_absolute_delta.rank(ascending=False,method='min').astype(int)
        a.sort_values(['condition','rank_mean_absolute_delta','gene']).to_csv(out/f'{receiver}_signed_gene_ranked.csv',index=False)
    return s


def evidence_audit(provenance, genes, out):
    """Read small project records/source only; do not execute discovered source."""
    start=time.monotonic(); visited=0; snippets=[]; records=[]; errors=[]; baselines=[]
    limits=dict(max_files=2500,max_seconds=90,max_source_bytes=1500000,max_csv_bytes=3000000,
                max_snippets=120,max_source_copies=15)
    source_rx=re.compile(r'typeexp|residual|groupby|subtract|log1p|log2|normalize_total|tumor_1|0\.267|0\.277',re.I)
    csv_rx=re.compile(r'liver|tumor|cell.?type|receiver|pcc|metric|figure.?4',re.I)
    source_dir=out/'source_evidence';source_dir.mkdir()
    candidates=[]
    for root in [BASE/'spagatv2',BASE/'SpaGAT_submission',BASE/'GITIII-main/liver_workdir']:
        if not root.exists():
            errors.append(dict(path=str(root),error='missing search root'));continue
        for d,dirs,files in os.walk(root):
            dirs[:]=sorted(x for x in dirs if x not in {'.git','__pycache__','.ipynb_checkpoints','node_modules','venv','.venv'})
            for name in sorted(files):
                visited+=1
                if visited>limits['max_files'] or time.monotonic()-start>limits['max_seconds']:
                    break
                p=Path(d)/name
                try:
                    size=p.stat().st_size
                    if p.suffix in {'.py','.ipynb'} and size<limits['max_source_bytes']:
                        text=p.read_text(errors='replace')
                        if p.suffix=='.ipynb':
                            nb=json.loads(text)
                            text='\n'.join(''.join(c.get('source',[])) for c in nb.get('cells',[]) if c.get('cell_type')=='code')
                        if not source_rx.search(text):continue
                        lines=text.splitlines()
                        hits=[i for i,l in enumerate(lines) if source_rx.search(l)]
                        strong=('TypeExp' in text and any(x in text for x in ['groupby','subtract',' - ','-=']))
                        candidates.append((strong,p,text))
                        for i in hits[:6]:
                            if len(snippets)<limits['max_snippets']:
                                snippets.append(dict(path=str(p),line=i+1,line_kind='flattened_code' if p.suffix=='.ipynb' else 'source',
                                    excerpt='\n'.join(lines[max(0,i-3):min(len(lines),i+4)])))
                    elif p.suffix=='.csv' and size<limits['max_csv_bytes'] and csv_rx.search(str(p.relative_to(root))):
                        df=pd.read_csv(p,nrows=30000)
                        cols=[c for c in df if re.search('pcc|correl',str(c),re.I)]
                        labelcols=[c for c in df if re.search('cell.?type|receiver|population|subclass',str(c),re.I)]
                        if not cols or not labelcols:continue
                        selected=np.zeros(len(df),dtype=bool)
                        for c in labelcols:selected|=df[c].astype(str).str.contains('tumor',case=False,regex=False).to_numpy()
                        for _,row in df[selected].head(100).iterrows():
                            records.append(dict(path=str(p),sha256=sha(p),values=row.to_dict()))
                except Exception as e:errors.append(dict(path=str(p),error=str(e)))
            if visited>limits['max_files'] or time.monotonic()-start>limits['max_seconds']:break
        if visited>limits['max_files'] or time.monotonic()-start>limits['max_seconds']:break
    copies=[]
    for i,(_,p,text) in enumerate(sorted(candidates,key=lambda x:(not x[0],str(x[1])))[:limits['max_source_copies']]):
        dest=source_dir/f'{i:02d}_{p.stem}.txt';dest.write_text(text)
        copies.append(dict(path=str(p),sha256=sha(p),copied_as=str(dest.relative_to(out))))
    baseline_path=Path(provenance['data_dir'])/'CancerousLiver_TypeExp.npz'
    if baseline_path.exists():
        try:
            with np.load(baseline_path,allow_pickle=False) as z:
                vals={t:np.asarray(z[t]) for t in RECEIVERS}
                if any(v.shape!=(len(genes),) or not np.isfinite(v).all() for v in vals.values()):
                    raise ValueError('Baseline shape or values invalid')
                frame=pd.DataFrame(dict(gene=genes,tumor_1_supplied_baseline=vals['tumor_1'],
                    tumor_2_supplied_baseline=vals['tumor_2'],baseline_tumor2_minus_tumor1=vals['tumor_2']-vals['tumor_1']))
                frame.to_csv(out/'supplied_tumor_baseline_profiles.csv',index=False)
            baselines.append(dict(path=str(baseline_path),sha256=sha(baseline_path),
                interpretation='Supplied baseline vectors; transformation and estimation population require source verification. Not a validated tumor subtype assignment.'))
        except Exception as e:errors.append(dict(path=str(baseline_path),error=str(e)))
    save_json(out/'source_snippets.json',snippets)
    save_json(out/'historical_tumor_metric_records.json',records)
    save_json(out/'preprocessing_and_history_audit.json',dict(limits=limits,visited_files=visited,
        incomplete=visited>limits['max_files'] or time.monotonic()-start>limits['max_seconds'],
        source_copies=copies,baseline_evidence=baselines,errors=errors,
        target_semantics='PENDING_MANUAL_SOURCE_VERIFICATION',
        Figure4_origin='PENDING_RECONCILIATION; numerical agreement alone does not prove identical checkpoint or evaluated cells'))
    print('源码和历史结果线索已收集；不会自动把线索标为预处理来源已验证。',flush=True)


def scalar_comparison(out):
    old=BASE/'spagatv2/analysis/liver_sender_origin_random_control/matched_sender_random_control_repeats.csv'
    if not old.exists():return
    keys=['receiver','condition']
    cols=['full_MSE','masked_MSE','mean_abs_prediction_change','relative_MSE_increase_percent']
    a=pd.read_csv(old).groupby(keys)[cols].mean()
    b=pd.read_csv(INPUT/'masking_scalar_repeats.csv').groupby(keys)[cols].mean()
    merged=a.join(b,lsuffix='_historical',rsuffix='_new',how='outer')
    for c in cols:merged[c+'_difference']=merged[c+'_new']-merged[c+'_historical']
    merged.reset_index().to_csv(out/'historical_vs_new_masking.csv',index=False)


METHODS = '''Analysis scope: existing validation predictions and fixed-model matched-count masking.
No model is loaded, trained, or run. No independent liver test or biological replication is added.
Gene sets: all 50 human MSigDB Hallmark sets, v2026.1.Hs, downloaded from the cited Broad Institute release.
Use exact symbol matching and only unambiguous hyphen-to-dot column-name mapping. Export all mappings and coverage.
Each program is evaluated if at least 5 panel genes are available; the standardized score also needs at least 5 positive-variance genes. Report all 50 sets, including non-evaluable sets. The >=10-gene indicator is a prespecified sensitivity subset, not a significance threshold. Sparse coverage does not validate a whole pathway.
Primary score within each annotated tumor population: mean of (gene value - observed validation mean)/observed validation SD over available program genes. Exactly the same observed means and SDs are applied to predictions. No prediction-specific rescaling. These descriptive validation-derived calibration constants are exported; this is not an independent-test estimate.
Sensitivity score: unscaled mean of measured target or predicted values for the same gene set. This secondary score includes constant genes, if any. No second baseline subtraction is performed.
Report PCC, Spearman, MSE, MSE0 and 100*(1-MSE/MSE0). For the standardized score, zero is the observed-mean program baseline. For the raw score, zero is the original target-zero baseline; its residual interpretation remains conditional on preprocessing provenance.
Direction agreement compares the signs of standardized predicted and observed scores, excluding only |observed score|<=1e-12. For the standardized score this is direction around the observed population mean; for the raw score the reference is original target zero, as explicitly labeled. Neither is differential expression between tumor populations or tumor versus normal.
Signed delta = masked prediction - full prediction. Program signed means for each masking repeat are exact linear combinations of existing gene-level repeat means. Repeat SD describes neighbor selection under a fixed model, not training variability or biological replication.
Cell-level program deltas are projected from arrays already averaged across 10 repeats. Mean absolute of these cell-level averages is |mean over repeats(delta)| averaged over cells. It is NOT mean over repeats and cells(|delta|). This distinction is reflected in column names.
Random-control subtraction is descriptive under equal removed counts; it does not remove all spatial, abundance, or graph-perturbation confounding.
The source audit gathers candidate preprocessing code and historical metric records only. Near-zero target means, numerical agreement, or a code keyword does not by itself verify historical provenance. Residual target semantics, old Figure 4 identity and tumor molecular subtype identity remain subject to manual evidence review.
Supplied tumor baseline vectors are exported for annotation follow-up, without inferring subtype identity or independent disease validation.
No gene-set significance tests, cell-bootstrap confidence intervals, clinical subtype assignments, or causal claims are generated.
'''


def run_followup():
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')+'_'+uuid.uuid4().hex[:6]
    out=BASE/'SpaGAT_revision'/('R3_5_7_program_followup_'+stamp)
    out.mkdir(parents=True,exist_ok=False)
    cache=Path('/content')/('R3_program_cache_'+stamp);cache.mkdir(exist_ok=False)
    manifest=[]; status=dict(status='running',new_training=False,new_inference=False,input=str(INPUT),
                            gene_set_source=GMT_URL,gene_set_sha256=GMT_SHA,min_program_genes=MIN_PROGRAM_GENES)
    try:
        if not INPUT.is_dir():raise FileNotFoundError(f'找不到原输出目录：{INPUT}。请检查 Drive 挂载和路径。')
        prov=json.loads((INPUT/'provenance.json').read_text())
        if prov['status']!='complete_gene_and_matched_count_masking':raise ValueError('Prior export did not complete')
        genes=json.loads((INPUT/'genes.json').read_text())
        meta=pd.read_csv(INPUT/'receiver_metadata.csv')
        if len(genes)!=1000 or len(set(genes))!=1000:raise ValueError('Unexpected gene panel')
        if meta.dataset_index.duplicated().any() or set(meta.cell_type)!=set(RECEIVERS):raise ValueError('Invalid receiver metadata')
        if not meta.masking_eligible.isin([True,False]).all():raise ValueError('Nonboolean masking eligibility')
        idx=np.load(INPUT/'arrays/dataset_indices.npy',allow_pickle=False)
        if not np.array_equal(idx,meta.dataset_index.to_numpy()):raise ValueError('Prediction row IDs do not match metadata')
        counts=meta.groupby('cell_type').agg(n=('dataset_index','size'),eligible=('masking_eligible','sum')).to_dict('index')
        if counts!={'tumor_1':{'n':69784,'eligible':19659},'tumor_2':{'n':6969,'eligible':6892}}:
            raise ValueError(f'Unexpected receiver counts: {counts}')
        status.update(receiver_counts=counts,selected_checkpoint=prov['selected_checkpoint'],prior_provenance=prov,
            input_metadata_hashes={name:sha(INPUT/name) for name in ['provenance.json','genes.json','receiver_metadata.csv',
                'tumor_1_gene_recovery.csv','tumor_2_gene_recovery.csv','signed_masking_gene_repeats.csv','signed_masking_gene_summary.csv']})
        if hashlib.sha256(GMT_TEXT.encode()).hexdigest()!=GMT_SHA:raise ValueError('Embedded gene-set checksum mismatch')
        (out/'h.all.v2026.1.Hs.symbols.gmt').write_text(GMT_TEXT)
        mapping,coverage,members=read_sets(GMT_TEXT,genes)
        if len(members)!=50:raise ValueError('Expected all 50 Hallmark gene sets')
        mapping.to_csv(out/'gene_symbol_mapping.csv',index=False)
        coverage.to_csv(out/'panel_program_coverage.csv',index=False)
        (out/'METHODS_AND_LIMITS.txt').write_text(METHODS+'\nGene-set source: '+GMT_URL+'\n')
        (out/'score_arrays').mkdir()
        sizes=sum(p.stat().st_size for p in (INPUT/'arrays').glob('*.npy'))
        if shutil.disk_usage(cache).free<sizes+512*1024**2:raise RuntimeError('Colab local disk has insufficient space')
        print('读取已保存的 prediction/target；不加载模型。',flush=True)
        pred=stage(INPUT/'arrays/prediction.npy',cache,manifest)
        obs=stage(INPUT/'arrays/target.npy',cache,manifest)
        if pred.shape!=obs.shape or pred.shape!=(len(meta),len(genes)):raise ValueError('Prediction/target shape mismatch')
        models,result=program_recovery(pred,obs,meta,genes,members,coverage,out)
        signed=signed_programs(meta,genes,models,out,cache,manifest)
        scalar_comparison(out)
        evidence_audit(prov,genes,out)
        status.update(status='complete_postprocessing_pending_provenance_review',
            target_semantics='PENDING_MANUAL_SOURCE_VERIFICATION',Figure4_origin='PENDING_RECONCILIATION',
            tumor_subtype_assignment='NOT_PERFORMED',independent_validation='NOT_PERFORMED')
    except Exception as e:
        status.update(status='stopped',error=str(e))
        (out/'error_trace.txt').write_text(traceback.format_exc())
        print('停止：',e,flush=True)
    finally:
        status['input_arrays']=manifest
        save_json(out/'provenance.json',status)
        archive=out.with_suffix('.zip')
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(out.rglob('*')):
                if p.is_file() and 'score_arrays' not in p.relative_to(out).parts:
                    z.write(p,out.name+'/'+str(p.relative_to(out)))
        print('运行状态：',status['status'])
        print('请下载并上传这个 ZIP：',archive,flush=True)
    return str(archive)
