"""R1 Minor 6: saved-prediction analysis only; no model, inference or training."""
from pathlib import Path
from datetime import datetime, timezone
import gc, hashlib, json, platform, traceback, uuid, zipfile
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLOCK=2048
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})


def file_sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def array_sha(a):
    # Exact R3.3 provenance format; shape prefix is required.
    h=hashlib.sha256(str(a.shape).encode())
    for start in range(0,len(a),BLOCK):
        h.update(np.ascontiguousarray(a[start:start+BLOCK],dtype=np.float32).tobytes())
    return h.hexdigest()


def json_save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False))


def gene_metrics(pred,target,genes):
    """Float64 two-pass centered moments, bounded memory; every gene retained."""
    if pred.shape!=target.shape or target.ndim!=2 or target.shape[1]!=len(genes):
        raise ValueError('Prediction, target and gene dimensions disagree')
    n,g=target.shape
    if n<3 or len(set(genes))!=g:raise ValueError('Too few cells or duplicate genes')
    sy=np.zeros(g);sp=np.zeros(g);se=np.zeros(g);s0=np.zeros(g)
    for start in range(0,n,BLOCK):
        y=np.asarray(target[start:start+BLOCK],dtype=np.float64)
        p=np.asarray(pred[start:start+BLOCK],dtype=np.float64)
        if not np.isfinite(y).all() or not np.isfinite(p).all():raise ValueError('Nonfinite saved values')
        sy+=y.sum(0);sp+=p.sum(0);se+=((p-y)**2).sum(0);s0+=(y*y).sum(0)
    my=sy/n;mp=sp/n;vy=np.zeros(g);vp=np.zeros(g);cov=np.zeros(g)
    for start in range(0,n,BLOCK):
        y=np.asarray(target[start:start+BLOCK],dtype=np.float64)-my
        p=np.asarray(pred[start:start+BLOCK],dtype=np.float64)-mp
        vy+=(y*y).sum(0);vp+=(p*p).sum(0);cov+=(y*p).sum(0)
    product=vy*vp;defined=(vy>0)&(vp>0)
    pcc=np.full(g,np.nan)
    pcc[defined]=np.clip(cov[defined]/np.sqrt(product[defined]),-1,1)
    benchmark=np.clip(cov/np.sqrt(np.maximum(product,1e-12)),-1,1)
    reason=np.where(defined,'defined',np.where(vy==0,'constant_target','constant_prediction'))
    return pd.DataFrame({'gene':genes,'n_cells':n,'PCC':pcc,'PCC_defined':defined,'PCC_status':reason,
                         'benchmark_PCC_with_floor':benchmark,'MSE':se/n,'MSE0':s0/n,
                         'observed_mean':my,'predicted_mean':mp,'observed_variance':vy/n,
                         'predicted_variance':vp/n})


def summarize_gene_table(table):
    values=table.PCC.dropna().to_numpy()
    return {'n_genes':len(table),'n_defined':len(values),'n_undefined':int(table.PCC.isna().sum()),
            'median_defined_PCC':float(np.median(values)) if len(values) else None,
            'q25_defined_PCC':float(np.quantile(values,.25)) if len(values) else None,
            'q75_defined_PCC':float(np.quantile(values,.75)) if len(values) else None,
            'n_negative':int((values<0).sum()),'n_above_0_5':int((values>.5).sum()),
            'MSE':float(table.MSE.mean()),'MSE0':float(table.MSE0.mean())}


def save_figure(fig,folder,name):
    fig.savefig(folder/(name+'.png'),dpi=200,bbox_inches='tight')
    fig.savefig(folder/(name+'.pdf'),dpi=300,bbox_inches='tight')
    plt.close(fig)


def ecdf(ax,values,**kwargs):
    values=np.sort(np.asarray(values,dtype=float));values=values[np.isfinite(values)]
    if len(values):
        ax.step(np.r_[-1,values,1],np.r_[0,np.arange(1,len(values)+1)/len(values),1],where='post',**kwargs)
    ax.set_xlim(-1,1);ax.set_ylim(0,1);ax.axvline(0,color='#BBBBBB',lw=.6,zorder=0)
    ax.set_xlabel('Gene-wise PCC');ax.set_ylabel('Cumulative fraction of genes')
    ax.spines[['top','right']].set_visible(False)


def bool_flags(s):
    t=s.astype(str).str.lower().str.strip()
    if not t.isin(['true','false','1','0','1.0','0.0']).all():raise ValueError('Unrecognized receiver flags')
    return t.isin(['true','1','1.0']).to_numpy()


def processed_metadata(processed,sample,genes):
    sentinels=sorted(set(np.linspace(0,len(genes)-1,min(16,len(genes)),dtype=int).tolist()+
                         [i for i,g in enumerate(genes) if g in ['Gfap','APOE','GLUL']]))
    keep=['centerx','centery','section','subclass','index_0']+[genes[i] for i in sentinels]
    path=Path(processed)/(sample+'.csv');header=pd.read_csv(path,nrows=0).columns
    if not set(keep)<=set(header):raise ValueError('Required metadata columns missing: '+str(path))
    if 'flag' in header:keep+=['flag']
    with np.load(Path(processed)/(sample+'_TypeExp.npz'),allow_pickle=False) as z:valid=set(z.files)
    chunks=[]
    for c in pd.read_csv(path,usecols=keep,chunksize=20000):
        chunks.append(c[c.subclass.isin(valid)].copy())
    frame=pd.concat(chunks,ignore_index=True)
    if frame.empty or set(frame.section.astype(str))!={sample}:raise ValueError('Section metadata mismatch')
    if 'flag' not in frame:frame['flag']=True
    eligible=np.flatnonzero(bool_flags(frame.flag))
    return frame,eligible,sentinels


def center_metadata(frame,eligible,local):
    positions=eligible[np.asarray(local,dtype=np.int64)]
    indices=frame.index_0.to_numpy()[positions]
    if not np.isfinite(indices).all() or not np.equal(indices,np.floor(indices)).all():raise ValueError('Invalid index_0')
    indices=indices.astype(np.int64)
    if (indices<0).any() or (indices>=len(frame)).any():raise ValueError('index_0 out of range')
    return frame.iloc[indices].reset_index(drop=True)


def target_alignment(meta,genes,sentinels,target,allow_half=False):
    expected=meta[[genes[i] for i in sentinels]].to_numpy(dtype=np.float32)
    exact=True;half=True
    for start in range(0,len(meta),BLOCK):
        y=np.asarray(target[start:start+BLOCK])[:,sentinels]
        block=expected[start:start+BLOCK]
        exact &= bool(np.allclose(y,block,rtol=1e-6,atol=1e-6))
        half &= bool(np.allclose(y,block.astype(np.float16).astype(np.float32),rtol=1e-6,atol=1e-6))
    if not (exact or (allow_half and half)):raise ValueError('Saved target does not match documented metadata ordering')
    return {'sentinel_genes_checked':len(sentinels),'float32_matches':exact,
            'documented_half_roundtrip_matches':half,'full_gene_alignment_claimed':False}






def safe_saved_tensor(path):
    from importlib import import_module
    try:multiarray=import_module('numpy._core.multiarray')
    except ImportError:multiarray=import_module('numpy.core.multiarray')
    allowed=[np.ndarray,np.dtype]
    if tuple(map(int,torch.__version__.split('+')[0].split('.')[:2]))>=(2,6):
        allowed.extend([(multiarray._reconstruct,'numpy.core.multiarray._reconstruct'),
                        (multiarray._reconstruct,'numpy._core.multiarray._reconstruct')])
    else:allowed.append(multiarray._reconstruct)
    for name in ['Int32DType','Int64DType','Float16DType','Float32DType','Float64DType','StrDType','BoolDType']:
        if hasattr(np.dtypes,name):allowed.append(getattr(np.dtypes,name))
    with torch.serialization.safe_globals(allowed):
        return torch.load(path,map_location='cpu',mmap=True,weights_only=True)


def archived_coordinate_check(saved,meta,allow_half=False):
    """Compare the documented serialization, never fit a coordinate transform."""
    report={'allowed_half_roundtrip':bool(allow_half),'axes':{},'all_present_axes_verified':True}
    for key,column in [('position_x','centerx'),('position_y','centery')]:
        if key not in saved:continue
        v=saved[key].detach().cpu().numpy() if torch.is_tensor(saved[key]) else np.asarray(saved[key])
        if v.ndim==2:v=v[:,0]
        expected=meta[column].to_numpy(dtype=np.float32)
        if v.shape!=expected.shape or not np.isfinite(v).all() or not np.isfinite(expected).all():
            report['axes'][key]={'verified':False,'reason':'shape mismatch or nonfinite coordinates'}
            report['all_present_axes_verified']=False
            continue
        direct=bool(np.array_equal(v,expected))
        with np.errstate(over='ignore'):
            half=expected.astype(np.float16).astype(np.float32)
        half_match=bool(np.isfinite(half).all() and np.array_equal(v,half))
        ok=direct or (allow_half and half_match)
        entry={'verified':ok,'saved_dtype':str(v.dtype),'n_cells':len(v),
               'float32_exact':direct,'float32_half_float32_exact':half_match,
               'max_abs_difference_from_float32':float(np.max(np.abs(v.astype(np.float64)-expected))),
               'accepted_representation':'float32' if direct else ('documented_float16_roundtrip' if ok else None)}
        if not ok:
            reference=half if allow_half and np.isfinite(half).all() else expected
            mismatch=np.flatnonzero(v!=reference)[:10]
            entry['first_mismatches']=[{'row':int(i),'saved':float(v[i]),'float32':float(expected[i]),
                                       'half_roundtrip':float(half[i]) if np.isfinite(half[i]) else None} for i in mismatch]
        report['axes'][key]=entry
        report['all_present_axes_verified'] &= ok
    return report


def run_archive(spec,out):
    """Audit each candidate separately; a filename or PCC match is not model provenance."""
    tensor=Path(spec['tensor']);folder=out/('archive_'+spec['id']);folder.mkdir()
    if not tensor.is_file():return {'status':'missing','path':str(tensor)}
    genes=list(map(str,torch.load(Path(spec['processed']).parent/'genes.pth',map_location='cpu',weights_only=True)))
    print(f'{spec["id"]}: 读取归档预测；不加载模型。',flush=True)
    saved=safe_saved_tensor(tensor)
    if not isinstance(saved,dict) or not {'y','y_pred'}<=set(saved):raise ValueError('No documented y/y_pred arrays')
    pred=saved['y_pred'].detach().cpu().numpy();target=saved['y'].detach().cpu().numpy()
    if target.shape!=pred.shape or target.ndim!=2 or target.shape[1]!=len(genes):raise ValueError('Archived array dimensions differ')
    frame,eligible,sentinels=processed_metadata(spec['processed'],spec['sample'],genes)
    if len(target)==len(eligible):local=np.arange(len(eligible));ordering='All eligible cells in saved section order'
    elif spec['sample']=='CancerousLiver' and len(target)==30000 and len(eligible)==460429:
        local=np.random.RandomState(42).choice(len(eligible),30000,replace=False)
        ordering='Documented historical RandomState(42) 30000-cell sample'
    else:raise ValueError('No documented ordering for this saved receiver cohort')
    meta=center_metadata(frame,eligible,local);del frame
    alignment=target_alignment(meta,genes,sentinels,target,allow_half=True)
    if 'cell_type_name' in saved:
        labels=np.asarray([v if isinstance(v,str) else v[0] for v in saved['cell_type_name']]).astype(str)
        if not np.array_equal(labels,meta.subclass.astype(str)):raise ValueError('Saved labels disagree with recovered metadata')
    coordinate_check=archived_coordinate_check(saved,meta,allow_half=spec.get('documented_coordinate_half_roundtrip',False))
    json_save(folder/'coordinate_check.json',coordinate_check)
    if not coordinate_check['all_present_axes_verified']:
        raise ValueError('Saved coordinates disagree with documented serialization; see coordinate_check.json')
    # Reuse the prior Minor9 cohort counts and MSE/EV as an independent check.
    if spec.get('table'):
        if file_sha(spec['table'])!=spec['table_sha']:raise ValueError('Previously audited liver summary changed')
        original=pd.read_csv(spec['table'])
        if int(original.n_cells.sum())!=len(target):raise ValueError('Previously audited liver cohort count changed')
        # Bounded pooled check; per-type verification was already performed in Minor9.
        expected_mse=float(np.average(original.MSE,weights=original.n_cells))
    else:expected_mse=None
    table=gene_metrics(pred,target,genes)
    if expected_mse is not None and not np.isclose(table.MSE.mean(),expected_mse,rtol=1e-5,atol=1e-6):
        raise ValueError('Archived MSE does not reproduce the prior Minor9 audit')
    table.to_csv(folder/'all_genes_archived_residual_PCC.csv',index=False)
    summary=summarize_gene_table(table)
    with np.load(Path(spec['processed'])/(spec['sample']+'_TypeExp.npz'),allow_pickle=False) as z:
        baselines={k:z[k] for k in z.files}
    selected=[]
    for gene,reported in spec['figure3_genes'].items():
        if gene not in genes:raise ValueError('Named Figure 3 gene absent: '+gene)
        j=genes.index(gene);b=np.asarray([baselines[str(ct)][j] for ct in meta.subclass],dtype=np.float64)
        p=np.asarray(pred[:,j],dtype=np.float64);y=np.asarray(target[:,j],dtype=np.float64)
        residual=gene_metrics(p[:,None],y[:,None],[gene]).PCC.iloc[0]
        total=gene_metrics((p+b)[:,None],(y+b)[:,None],[gene]).PCC.iloc[0]
        rank=float(np.mean(table.PCC.dropna().to_numpy()<=residual)) if np.isfinite(residual) else None
        selected.append({'gene':gene,'manuscript_reported_PCC':reported,'saved_residual_PCC':residual,
                         'baseline_added_PCC':total,'residual_ecdf_percentile':rank,
                         'reported_matches_residual_to_3dp':bool(np.isfinite(residual) and round(float(residual),3)==reported),
                         'reported_matches_baseline_added_to_3dp':bool(np.isfinite(total) and round(float(total),3)==reported),
                         'match_is_not_proof_of_model_or_figure_identity':True})
    pd.DataFrame(selected).to_csv(folder/'Figure3_selected_gene_check.csv',index=False)
    fig,ax=plt.subplots(figsize=(6.4,4.2));ecdf(ax,table.PCC,color='#2D6BAD',lw=1.6,label=f'{summary["n_defined"]}/{len(genes)} defined genes')
    for row in selected:
        if np.isfinite(row['saved_residual_PCC']):
            ax.scatter(row['saved_residual_PCC'],row['residual_ecdf_percentile'],s=22,zorder=4)
            ax.annotate(row['gene'],(row['saved_residual_PCC'],row['residual_ecdf_percentile']),xytext=(-5,-14),textcoords='offset points',ha='right')
    ax.set_title(f'{spec["sample"]} | archived fitted-model predictions');ax.legend(frameon=False,loc='upper left')
    fig.tight_layout();save_figure(fig,folder,'all_gene_PCC_distribution')
    report={'status':'computed_requires_Figure3_source_confirmation','path':str(tensor),'file_bytes':tensor.stat().st_size,
            'target_sha256_shape_float32':array_sha(target),'prediction_sha256_shape_float32':array_sha(pred),
            'n_cells':len(target),'gene_order_source':str(Path(spec['processed']).parent/'genes.pth'),
            'ordering':ordering,'alignment':alignment,'coordinate_check':coordinate_check,'distribution':summary,
            'model_identity':'Not inferred from directory naming or PCC agreement',
            'evaluation_scope':'Archived within-dataset fitted-model receiver cohort; not an independent test',
            'historical_selection_rule':'Requires author confirmation; no prospective or random selection claimed'}
    json_save(folder/'verification.json',report)
    del saved,pred,target,meta;gc.collect()
    return report




def run_figure3_distributions(archives, output_parent):
    """Recompute all-gene distributions for the three reported Figure 3 cohorts."""
    out = Path(output_parent) / ('Figure3_distributions_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC') + '_' + uuid.uuid4().hex[:6])
    out.mkdir(parents=True, exist_ok=False)
    report = {'new_training': False, 'new_inference': False, 'archives': {},
              'scope': 'Figure 3 within-dataset cohorts and Supplementary Figure S2; this is a publication-scoped derivative.'}
    for spec in archives:
        try:
            report['archives'][spec['id']] = run_archive(spec, out)
        except Exception:
            report['archives'][spec['id']] = {'status': 'failed', 'traceback': traceback.format_exc()}
            print(report['archives'][spec['id']]['traceback'], flush=True)
        json_save(out / 'audit.json', report)
        gc.collect()
    report['computed'] = len(report['archives']) == len(archives) and all(r['status'].startswith('computed') for r in report['archives'].values())
    json_save(out / 'audit.json', report)
    (out / 'README.txt').write_text(
        'All measured genes are retained; undefined PCCs are reported explicitly. '
        'These are the within-dataset Figure 3 cohorts, separate from the Figure 2 test sets. '
        'Existing source, ordering, target and coordinate checks remain enforced. '
        'Numerical agreement is not independent proof of model identity. No training or inference is run.\n')
    bundle = out.with_suffix('.zip')
    with zipfile.ZipFile(bundle, 'x', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file(): z.write(p, p.relative_to(out))
    print('All requested cohort distributions computed:', report['computed'], flush=True)
    print('Result ZIP:', bundle, flush=True)
    return bundle
