"""Fixed, outcome-independent controls for an already evaluated benchmark."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

MATCH_RULES = {
    'statistics': 'mean and population variance of masked expression entering the edge branch over all 50 candidate nodes of each training receiver',
    'transforms': ['asinh(mean)', 'log1p(variance)'],
    'standardization': 'gene-panel mean and population SD, estimated from training inputs only',
    'maximum_pair_difference_per_feature_panel_SD': 1.0,
    'maximum_absolute_set_mean_difference_per_feature_panel_SD': 0.25,
    'cost': 'squared Euclidean feature distance plus Exponential(scale=0.25) random costs',
    'maximum_draws_per_set': 100,
    'candidate_pool': 'measured genes outside the original retained ligand-gene set',
    'replacement': 'one-to-one within a set; overlap between sets allowed; duplicate complete sets prohibited',
    'selection': 'first assignment passing the fixed expression-only criteria; no prediction metric is read',
    'set_seeds': [2026093001],
}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def digest_object(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))

def single_gene_slots(genes,ligands):
    slots=[]
    if len(ligands[0])!=len(ligands[1]):raise ValueError('Ligand group metadata mismatch')
    for names,groups in zip(*ligands):
        if len(names)!=1 or len(groups)!=1:raise ValueError('This control is defined for the audited single-gene ligand entries only')
        if names[0] not in genes:raise ValueError('Ligand missing from measured panel')
        slots.append(str(names[0]))
    if len(set(slots))!=len(slots):raise ValueError('Duplicate original ligand slots')
    return slots

def training_edge_statistics(data,split,chunk_size=128):
    """Replicate Embedding.forward's expression construction; never use y or validation/test rows."""
    import torch
    genes=list(map(str,data.genes));g=len(genes)
    train=np.asarray(split['train_indices'],dtype=np.int64)
    if train.size==0 or len(np.unique(train))!=len(train):raise ValueError('Invalid training indices')
    ends=np.cumsum(data.meta_counts);starts=ends-np.asarray(data.meta_counts)
    sid=np.searchsorted(ends,train,side='right')
    if np.any(sid>=len(data.samples)):raise ValueError('Training index out of bounds')
    count=0;mean=np.zeros(g);M2=np.zeros(g)
    h=hashlib.sha256()
    h.update(digest_object({'genes':genes,'train_indices':train.tolist(),'num_neighbors':50}).encode())
    samples=[]
    for j,name in enumerate(data.samples):
        selected=train[sid==j]-starts[j]
        if not len(selected):continue
        samples.append({'sample':name,'training_receivers':int(len(selected))})
        for start in range(0,len(selected),chunk_size):
            rows=data.arg_meta[j][torch.as_tensor(selected[start:start+chunk_size],dtype=torch.long)]
            ix=data.indexes[j][rows]
            if ix.ndim!=2 or ix.shape[1]!=50:raise ValueError('Expected original 50-cell graphs')
            baseline=data.type_exp[j][ix]
            residual=data.exps[j][ix]
            ct=data.cell_types[j][ix]
            # Identical floating-point addition and homotypic substitution to Embedding.forward.
            actual=torch.where((ct==ct[:,0:1]).unsqueeze(-1),baseline,baseline+residual)
            a=np.ascontiguousarray(actual.numpy(),dtype=np.float32)
            if not np.isfinite(a).all():raise ValueError('Nonfinite training edge expression')
            h.update(a.tobytes())
            x=a.reshape(-1,g).astype(np.float64)
            n=len(x);m=x.mean(0);q=((x-m)**2).sum(0)
            delta=m-mean;total=count+n
            M2 += q + delta**2*count*n/total
            mean += delta*n/total;count=total
    if count!=len(train)*50:raise ValueError('Training neighborhood count mismatch')
    var=np.maximum(M2/count,0)
    features=np.column_stack([np.arcsinh(mean),np.log1p(var)])
    center=features.mean(0);scale=features.std(0,ddof=0)
    if np.any(scale<1e-12):raise ValueError('Matching feature has no across-gene variation')
    z=(features-center)/scale
    table=pd.DataFrame({'gene':genes,'training_edge_mean':mean,'training_edge_variance':var,
        'asinh_mean_z':z[:,0],'log1p_variance_z':z[:,1]})
    info={'training_receivers':int(len(train)),'candidate_node_occurrences':int(count),
          'training_samples':samples,'training_edge_input_sha256':h.hexdigest(),
          'feature_center':center.tolist(),'feature_scale':scale.tolist(),
          'note':'Repeated neighbors are weighted as they occur in training graphs; these are feature-matching statistics, not biological replicates.'}
    return table,info

def make_matched_sets(stats,original_genes,rules=MATCH_RULES):
    from scipy.optimize import linear_sum_assignment
    genes=stats.gene.tolist();idx={g:i for i,g in enumerate(genes)}
    if len(idx)!=len(genes):raise ValueError('Duplicate gene names')
    if len(set(original_genes))!=len(original_genes):raise ValueError('Duplicate original genes')
    li=np.asarray([idx[g] for g in original_genes]);pool=np.asarray([i for i,g in enumerate(genes) if g not in set(original_genes)])
    if len(pool)<len(li):raise ValueError('Not enough distinct non-ligand panel genes')
    z=stats[['asinh_mean_z','log1p_variance_z']].to_numpy()
    dif=z[pool][None,:,:]-z[li][:,None,:]
    eligible=(np.abs(dif)<=rules['maximum_pair_difference_per_feature_panel_SD']).all(2)
    distances=(dif**2).sum(2)
    stats=stats.copy();stats['original_ligand']=stats.gene.isin(original_genes)
    coverage=[{'ligand_gene':g,'eligible_candidates':int(eligible[i].sum())} for i,g in enumerate(original_genes)]
    solutions=[];used=set();failure=None
    for number,seed in enumerate(rules['set_seeds'],1):
        rng=np.random.default_rng(seed);accepted=None;best_balance=None
        for attempt in range(rules['maximum_draws_per_set']):
            cost=distances+rng.exponential(0.25,size=distances.shape)
            cost=np.where(eligible,cost,np.inf)
            try:rr,cc=linear_sum_assignment(cost)
            except ValueError:
                failure='No complete one-to-one assignment satisfies the fixed pair caliper.';break
            if len(rr)!=len(li):raise ValueError('Incomplete Hungarian assignment')
            chosen=pool[cc];replacement=[genes[i] for i in chosen]
            delta=z[chosen]-z[li];balance=float(np.max(np.abs(delta.mean(0))))
            if best_balance is None or balance<best_balance:best_balance=balance
            key=tuple(sorted(replacement))
            if balance>rules['maximum_absolute_set_mean_difference_per_feature_panel_SD'] or key in used:continue
            used.add(key)
            accepted={'set_id':f'random_set_{number}','set_seed':seed,'accepted_draw':attempt+1,
                'original_genes':list(original_genes),'replacement_genes':replacement,
                'pair_feature_differences':delta.tolist(),
                'set_feature_mean_difference':delta.mean(0).tolist(),
                'maximum_pair_absolute_difference':float(np.abs(delta).max()),
                'mean_pair_squared_distance':float((delta**2).sum(1).mean()),'matching_passed':True}
            accepted['mapping_sha256']=digest_object({k:v for k,v in accepted.items() if k!='mapping_sha256'})
            break
        if accepted is None:
            failure=failure or f'No new gene set passed the prespecified balance criteria in {rules["maximum_draws_per_set"]} expression-only draws (set {number}; best balance={best_balance}).'
            break
        solutions.append(accepted)
    return {'passed':len(solutions)==len(rules['set_seeds']),'rules':rules,'candidate_pool_size':len(pool),
        'original_feature_count':len(li),'candidate_coverage':coverage,'sets':solutions,'failure':failure}

def validate_map(payload,genes,ligands):
    claimed=payload.get('mapping_sha256')
    if claimed!=digest_object({k:v for k,v in payload.items() if k!='mapping_sha256'}):raise ValueError('Gene-map integrity mismatch')
    original=single_gene_slots(genes,ligands)
    replacement=payload['replacement_genes']
    if payload['original_genes']!=original:raise ValueError('Original ligand set/order differs')
    if len(replacement)!=len(original) or len(set(replacement))!=len(original):raise ValueError('Replacement count/uniqueness mismatch')
    if set(replacement)&set(original) or not set(replacement)<=set(genes):raise ValueError('Invalid replacement gene pool')
    if payload.get('matching_passed') is not True:raise ValueError('Expression matching has not passed')
    delta=np.asarray(payload['pair_feature_differences'])
    if delta.shape!=(len(original),2) or not np.isfinite(delta).all():raise ValueError('Invalid matching diagnostics')
    if np.abs(delta).max()>MATCH_RULES['maximum_pair_difference_per_feature_panel_SD']+1e-12:raise ValueError('Pair caliper exceeded')
    if np.abs(delta.mean(0)).max()>MATCH_RULES['maximum_absolute_set_mean_difference_per_feature_panel_SD']+1e-12:raise ValueError('Set balance exceeded')

def apply_gene_map(model,genes,ligands,payload):
    import torch
    validate_map(payload,genes,ligands)
    model.embeddings.ligands_index=[[torch.tensor([genes.index(g)],dtype=torch.long)] for g in payload['replacement_genes']]

def prepare_matching(config):
    import ablation_support as support
    train=support.import_train(Path(config['source']))
    sp=Path(config['split_path']);split=json.loads(sp.read_text())
    data=train.SPAGAT_dataset(split['processed_dir'],num_neighbors=50)
    train.load_and_validate_shared_split(sp,len(data),split['processed_dir'])
    train.validate_sample_order(str(sp),data,split)
    stats,info=training_edge_statistics(data,split)
    genes=list(map(str,data.genes));original=single_gene_slots(genes,data.interactions)
    result=make_matched_sets(stats,original)
    result.update(info);result.update({'dataset':config['dataset'],'split_sha256':sha(sp),
         'genes_sha256':digest_object(genes),'original_genes':original,'genes':genes})
    dest=Path(config['output']);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'matching_report.json').exists():
        previous=json.loads((dest/'matching_report.json').read_text())
        if digest_object(previous)!=digest_object(result):raise ValueError('Training inputs or matching result changed on resume')
        print('MATCHING UNCHANGED:',config['dataset'],flush=True);return result
    stats.to_csv(dest/'training_gene_statistics.csv',index=False)
    for item in result['sets']:
        validate_map(item,genes,data.interactions)
        write_json(dest/(item['set_id']+'.json'),item)
        pairs=pd.DataFrame({'ligand_gene':original,'replacement_gene':item['replacement_genes'],
            'mean_feature_difference':np.asarray(item['pair_feature_differences'])[:,0],
            'variance_feature_difference':np.asarray(item['pair_feature_differences'])[:,1]})
        pairs.to_csv(dest/(item['set_id']+'_pairs.csv'),index=False)
    write_json(dest/'matching_report.json',result)
    print('MATCHING '+('PASSED: ' if result['passed'] else 'NOT PASSED: ')+config['dataset'],flush=True)
    return result

def record_prediction(job,dataset,variant,set_id,seed,reference):
    from ablation_support import array_sha,metrics
    job=Path(job);f=job/f'seed{seed}_test/test_predictions.npz'
    split=json.loads(Path(reference['split_path']).read_text())
    with np.load(reference['npz'],allow_pickle=False) as z:gene_order=list(z['genes'].astype(str))
    with np.load(f,allow_pickle=False) as z:
        if not np.array_equal(z['test_indices'],split['test_indices']):raise ValueError('Control test order changed')
        if list(z['genes'].astype(str))!=gene_order:raise ValueError('Control gene order changed')
        if array_sha(z['target'])!=reference['target_sha256_float32']:raise ValueError('Control target changed')
        values,pcc=metrics(z['prediction'],z['target'])
        row={'dataset':dataset,'variant':variant,'gene_set':set_id,'seed':seed,**values}
        meta={'prediction_path':str(f),'prediction_file_sha256':sha(f),
            'target_sha256_float32':array_sha(z['target']),'prediction_sha256_float32':array_sha(z['prediction']),
            'gene_order_sha256':digest_object(gene_order)}
    pd.DataFrame({'gene':gene_order,'PCC':pcc}).to_csv(job/'audited_per_gene_PCC.csv',index=False)
    write_json(job/'audited_metrics.json',row);write_json(job/'prediction_provenance.json',meta)
    return row

def aggregate_results(frame):
    measures=['median_gene_pcc','mse','mse0','ev_zero_percent']
    per_set=frame.groupby(['dataset','variant','gene_set'])[measures].agg(['count','mean','std'])
    per_set.columns=['_'.join(c) for c in per_set.columns];per_set=per_set.reset_index()
    diffs=[]
    for _,row in frame[~frame.variant.eq('original_ligand')].iterrows():
        base=frame[(frame.dataset==row.dataset)&frame.variant.eq('original_ligand')&(frame.seed==row.seed)]
        if len(base)!=1:raise ValueError('Exactly one original-model reference per seed required')
        diffs.append({'dataset':row.dataset,'variant':row.variant,'gene_set':row.gene_set,'seed':int(row.seed),
             **{'delta_'+m:float(row[m]-base.iloc[0][m]) for m in measures if m!='mse0'}})
    random=per_set[per_set.variant.eq('matched_random_edge') & per_set.median_gene_pcc_count.eq(3)]
    across=random.groupby('dataset')[[m+'_mean' for m in measures]].agg(['count','mean','std'])
    across.columns=['_'.join(c) for c in across.columns];across=across.reset_index()
    return per_set,pd.DataFrame(diffs),across

if __name__=='__main__':
    import sys
    prepare_matching(json.loads(Path(sys.argv[1]).read_text()))
