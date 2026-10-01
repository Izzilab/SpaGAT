"""Fixed k=25/50/75 sensitivity; original expression and split stay read-only."""
from pathlib import Path
import json, sys, hashlib, re, time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from ablation_support import sha, array_sha, save_json, metrics, load_reference, import_train

KS = [25, 50, 75]

def extend_neighbors(xy, old, target=75, eligible_rows=None, audit_path=None):
    """Preserve every original neighbor, checking these are ordered nearest cells.

    Exact Euclidean neighbors for eligible receivers in this section. Stored
    rows for excluded receivers are never fetched by the original loader and
    do not define a graph used in training/validation/test. Their index rows
    remain available to preserve the original row layout, with findings logged.
    Boundary ties for new cells are resolved by row index. No expression used.
    """
    xy = np.asarray(xy, dtype=np.float64)
    old = np.asarray(old, dtype=np.int64)
    n = len(xy)
    if xy.shape != (n, 2) or old.shape != (n, 50) or n < target + 1:
        raise ValueError('Invalid coordinates/neighbor dimensions or section too small')
    if not np.isfinite(xy).all() or old.min() < 0 or old.max() >= n:
        raise ValueError('Nonfinite coordinates or invalid original neighbor indices')
    if not np.array_equal(old[:, 0], np.arange(n)):
        raise ValueError('Original index_0 is not the receiver; do not change this silently')
    if np.any(np.diff(np.sort(old, axis=1), axis=1) == 0):
        raise ValueError('Duplicate original neighbors')
    if eligible_rows is None:
        eligible_rows = np.arange(n)
    eligible_rows = np.asarray(eligible_rows, dtype=np.int64)
    if (eligible_rows.ndim != 1 or len(np.unique(eligible_rows)) != len(eligible_rows)
            or np.any(eligible_rows < 0) or np.any(eligible_rows >= n)):
        raise ValueError('Invalid eligible receiver row indices')
    eligible = np.zeros(n,dtype=bool);eligible[eligible_rows] = True
    findings={'n_section_rows':n,'eligible_receivers':int(eligible.sum()),
              'excluded_receivers':int((~eligible).sum()),'tolerance_rule':'1e-8 * max(1, exact 50th-neighbor distance), unchanged from v1',
              'eligible_order_failures':0,'excluded_order_failures':0,
              'eligible_membership_failures':0,'excluded_membership_failures':0,
              'order_failure_rows':[],'membership_failure_rows':[],'examples':[],
              'excluded_rows_are_not_fetched_by_loader':True,
              'all_original_neighbor_indices_retained':True,'checked_section_rows':0,'all_section_rows_checked':False}
    tree = cKDTree(xy)
    result = np.empty((n, target), dtype=np.int64)
    result[:, :50] = old
    for lo in range(0, n, 2048):
        hi = min(n, lo + 2048)
        distances, ids = tree.query(xy[lo:hi], k=target+1, workers=1)
        original_distances = np.linalg.norm(xy[old[lo:hi]] - xy[lo:hi, None, :], axis=2)
        tol = 1e-8 * np.maximum(1, distances[:, 49])
        bad_order = np.any(np.diff(original_distances, axis=1) < -tol[:, None], axis=1)
        # A set may be correct but internally unordered; use its maximum here.
        bad_membership = np.abs(original_distances.max(axis=1) - distances[:, 49]) > tol
        for kind,bad in [('order',bad_order),('membership',bad_membership)]:
            findings['eligible_'+kind+'_failures']+=int(np.sum(bad & eligible[lo:hi]))
            findings['excluded_'+kind+'_failures']+=int(np.sum(bad & ~eligible[lo:hi]))
            findings[kind+'_failure_rows'].extend((np.flatnonzero(bad)+lo).tolist())
        for local in np.flatnonzero(bad_order | bad_membership):
            if len(findings['examples'])>=20:break
            row=int(lo+local);raw=original_distances[local]
            delta=np.abs(xy-xy[row]); clipped=np.sqrt(np.square(np.minimum(delta,1e4)).sum(axis=1))
            saved_clipped=clipped[old[row]]; nearest_clipped=float(np.partition(clipped,49)[49])
            ct=1e-8*max(1,nearest_clipped)
            worst=int(np.argmin(np.diff(raw)))
            findings['examples'].append({'receiver_row':row,'eligible_receiver':bool(eligible[row]),
                'receiver_xy':xy[row].tolist(),'raw_order_failure':bool(bad_order[local]),
                'raw_membership_failure':bool(bad_membership[local]),
                'worst_adjacent_slots':[worst,worst+1],
                'worst_adjacent_neighbor_rows':old[row,[worst,worst+1]].tolist(),
                'worst_adjacent_distances':raw[[worst,worst+1]].tolist(),
                'max_reverse_step':max(0,float(-np.diff(raw).min())),
                'allowed_tolerance':float(tol[local]),
                'raw_max_saved_distance':float(raw.max()),'raw_50th_distance':float(distances[local,49]),
                'clipping_applies_to_saved_neighbors':bool(np.any(delta[old[row]]>1e4)),
                'legacy_clipped_order_passes':bool(np.all(np.diff(saved_clipped)>=-ct)),
                'legacy_clipped_k50_membership_passes':bool(abs(saved_clipped.max()-nearest_clipped)<=ct)})
        # Record diagnostics before raising, including on a stopped preparation.
        findings['checked_section_rows']=hi
        findings['all_section_rows_checked']=hi==n
        if audit_path is not None:save_json(audit_path,findings)
        if np.any((bad_order | bad_membership) & eligible[lo:hi]):
            raise ValueError('Eligible receiver neighborhoods differ from ordered Euclidean nearest neighbors. '
                             'No eligibility, coordinate, tolerance, or neighbor change was applied. '
                             'See receiver_order_audit JSON in the summary ZIP.')
        for j, row in enumerate(range(lo, hi)):
            candidates = ids[j]
            # Include all cells tied at the new boundary, not an arbitrary tree subset.
            radius = distances[j, target - 1]
            if np.isclose(radius, distances[j, target], rtol=1e-12, atol=1e-12):
                candidates = np.asarray(tree.query_ball_point(xy[row], radius + max(1., radius)*1e-12))
            candidates = candidates[~np.isin(candidates, old[row])]
            d = np.linalg.norm(xy[candidates] - xy[row], axis=1)
            candidates = candidates[np.lexsort((candidates, d))]
            result[row, 50:] = candidates[:target-50]
    if audit_path is not None:save_json(audit_path,findings)
    return result

def quantiles(values):
    return {k: float(v) for k, v in zip(['min','p05','median','p95','max'],
                                      np.quantile(values, [0,.05,.5,.95,1]))}

def verify_cache(cache, processed_dir):
    cache = Path(cache)
    m = json.loads((cache/'manifest.json').read_text())
    if Path(m['processed_dir']).resolve() != Path(processed_dir).resolve():
        raise ValueError('Cache belongs to a different input directory')
    for path, digest in m['input_hashes'].items():
        if sha(path) != digest:
            raise ValueError('Source input changed: '+path)
    for rec in m['sections']:
        if sha(cache/rec['index_file']) != rec['index_sha256']:
            raise ValueError('Cached neighbor index changed: '+rec['sample'])
    return m

def dataset_for_k(processed_dir, num_neighbors, cache_dir):
    import torch
    from spagat.dataloader import SPAGAT_dataset
    if num_neighbors not in KS:
        raise ValueError('The planned comparison is fixed to k=25,50,75')
    manifest = verify_cache(cache_dir, processed_dir)
    data = SPAGAT_dataset(str(processed_dir), num_neighbors=50)
    if data.samples != [r['sample'] for r in manifest['sections']]:
        raise ValueError('Sample order changed')
    for i, rec in enumerate(manifest['sections']):
        a = np.load(Path(cache_dir)/rec['index_file'], allow_pickle=False)
        if not np.array_equal(a[:, :50], data.indexes[i].numpy()):
            raise ValueError('Original neighbor ordering changed')
        if len(a) != len(data.exps[i]):
            raise ValueError('Loader removed/reordered section rows')
        data.indexes[i] = torch.from_numpy(a[:, :num_neighbors].copy())
    data.index_index = ['index_'+str(i) for i in range(num_neighbors)]
    return data

def prepare_cache(config):
    import torch
    rec=config['reference']; split=json.loads(Path(rec['split_path']).read_text())
    processed=Path(split['processed_dir']); cache=Path(config['cache']); cache.mkdir(parents=True, exist_ok=True)
    if (cache/'manifest.json').exists():
        verify_cache(cache, processed)
        print('PREFLIGHT cached neighbors verified:',rec['dataset'],flush=True)
        return
    train=import_train(config['source'])
    data=train.SPAGAT_dataset(str(processed), num_neighbors=50)
    train.load_and_validate_shared_split(rec['split_path'],len(data),str(processed))
    train.validate_sample_order(rec['split_path'],data,split)
    audit=pd.read_csv(rec['samples_path']).set_index('sample')
    source_files=[processed.parent/'genes.pth',processed.parent/'ligands.pth']
    source_files += [processed/(s+suffix) for s in data.samples for suffix in ['.csv','_TypeExp.npz']]
    before={str(p):sha(p) for p in source_files}
    manifest={'dataset':rec['dataset'],'processed_dir':str(processed), 'input_hashes':before,
              'k_includes_receiver':True,'units':'stored coordinate units; physical conversion not inferred',
              'coordinate_rescaling_in_this_experiment':False,'original_first_50_retained':True,
              'same_sample_candidates_only':True,'sections':[]}
    rows=[]
    for i,sample in enumerate(data.samples):
        print('PREFLIGHT neighborhood:',rec['dataset'],sample,flush=True)
        frame=pd.read_csv(processed/(sample+'.csv'),usecols=['centerx','centery','subclass','flag'])
        if len(frame)!=len(data.exps[i]):
            raise ValueError('Baseline filtering changes index interpretation: '+sample)
        xy=frame[['centerx','centery']].to_numpy(dtype=np.float64)
        old=data.indexes[i].numpy()
        receivers=data.arg_meta[i].numpy()
        if not np.array_equal(receivers,np.flatnonzero(frame['flag'].to_numpy()!=0)):
            raise ValueError('Saved receiver flag differs from loader eligibility: '+sample)
        audit_path=cache/(sample+'_receiver_order_audit.json')
        extended=extend_neighbors(xy,old,eligible_rows=receivers,audit_path=audit_path)
        order_audit=json.loads(audit_path.read_text())
        if order_audit['excluded_order_failures'] or order_audit['excluded_membership_failures']:
            print('PREFLIGHT excluded receiver rows logged (never used as focal receivers):',
                  sample,order_audit['excluded_order_failures'],order_audit['excluded_membership_failures'],flush=True)
        dest=cache/(sample+'_k75.npy'); np.save(dest,extended,allow_pickle=False)
        manifest['sections'].append({'sample':sample,'rows':len(frame),'eligible_receivers':int(data.meta_counts[i]),
            'index_file':dest.name,'index_sha256':sha(dest),'coordinate_array_sha256':array_sha(xy),
            'receiver_order_audit':audit_path.name})
        for k in KS:
            radii=np.linalg.norm(xy[extended[receivers,k-1]]-xy[receivers],axis=1)
            rows.append({'dataset':rec['dataset'],'sample':sample,'partition':audit.loc[sample,'split'],
                         'k_including_receiver':k,'n_receivers':len(receivers),
                         'unit':'stored_coordinate_units',**quantiles(radii)})
    # Detect edits during construction. Only small index files were written.
    if before!={str(p):sha(p) for p in source_files}:
        raise ValueError('Source files changed during neighborhood preparation')
    pd.DataFrame(rows).to_csv(cache/'neighborhood_radii_by_section.csv',index=False)
    save_json(cache/'manifest.json',manifest)
    # This loader keeps all non-index tensors and receiver eligibility untouched.
    print('PREFLIGHT k=25/50/75 prepared:',rec['dataset'],flush=True)

def summarize(root, references):
    root=Path(root); rows=[]; genes_out=[]; provenance=[]
    for rec in references:
        split=json.loads(Path(rec['split_path']).read_text())
        full,y,genes=load_reference(rec,split)
        base,_=metrics(full,y)
        for k in KS:
            path=Path(rec['npz']); p=full
            if k!=50:
                path=root/'runs'/rec['dataset']/f'k{k}_seed123/seed123_test/test_predictions.npz'
                if not path.exists():continue
                with np.load(path,allow_pickle=False) as z:
                    if not np.array_equal(z['test_indices'],split['test_indices']):raise ValueError('Test receivers changed')
                    if list(z['genes'].astype(str))!=list(genes):raise ValueError('Gene order changed')
                    if array_sha(z['target'])!=rec['target_sha256_float32']:raise ValueError('Targets changed')
                    p=z['prediction']
            m,corr=metrics(p,y)
            rows.append({'dataset':rec['dataset'],'k_including_receiver':k,'seed':123,
                         'n_test':len(y),'n_genes':len(genes),**m,
                         **{'difference_vs_k50_'+key:m[key]-base[key] for key in ['median_gene_pcc','mse','ev_zero_percent']}})
            genes_out.extend({'dataset':rec['dataset'],'k_including_receiver':k,'seed':123,'gene':str(g),'PCC':float(c)} for g,c in zip(genes,corr))
            provenance.append({'dataset':rec['dataset'],'k':k,'path':str(path),
                               'prediction_sha256':array_sha(p),'target_sha256':array_sha(y)})
    frame=pd.DataFrame(rows);frame.to_csv(root/'neighborhood_test_metrics.csv',index=False)
    pd.DataFrame(genes_out).to_csv(root/'neighborhood_per_gene_PCC.csv',index=False)
    save_json(root/'prediction_provenance.json',provenance)
    success=sum((root/'runs'/r['dataset']/f'k{k}_seed123/SUCCESS.json').exists() for r in references for k in [25,75])
    save_json(root/'summary_status.json',{'complete':len(rows)==len(references)*3 and success==len(references)*2,
         'metric_rows':len(rows),'expected_rows':len(references)*3,'completed_new_trainings':success,
         'expected_new_trainings':len(references)*2,'training_seed':123,'k_includes_receiver':True,
         'scope':'Single-seed retrained sensitivity on fixed independent-test partitions. No across-seed SD or optimal-k claim.',
         'selection':'Validation PCC selects the checkpoint separately within each fixed k; test does not select k.',
         'physical_coordinate_units_verified':False})
    return frame

def coordinate_audit(root):
    """Collect concrete metadata and bounded source excerpts; do not assign units."""
    root=Path(root); out=root/'coordinate_audit';out.mkdir(exist_ok=True)
    liver=Path('/content/drive/MyDrive/GITIII-main/liver_workdir/gitiii/data/processed')
    info={'distance_from_input_coordinates':'Euclidean distance in stored centerx/centery units',
          'model_stage_extra_unit_rescaling':False,
          'distance_features':['1/(d+1)','1/(sqrt(d)+1)','1/(1+d*d)','exp(-d)','exp(-d*d)'],
          'physical_units':'Unresolved: raw-to-CSV conversion/source documentation must be traced.',
          'notes':'Coordinate ranges and comments alone do not prove micrometer units.', 'liver_metadata':[]}
    for sample in ['CancerousLiver','NormalLiver']:
        p=liver/(sample+'.csv')
        if not p.exists():continue
        f=pd.read_csv(p,usecols=['centerx','centery'])
        xy=f.to_numpy(dtype=np.float64)
        info['liver_metadata'].append({'sample':sample,'source':str(p),'n_cells':len(f),
               'centerx':quantiles(xy[:,0]),'centery':quantiles(xy[:,1]),'unit':'unverified'})
    snippets=[]; candidates=[]
    # Limited project locations, at most two directory levels; no recursive Drive scan.
    for base in [Path('/content/drive/MyDrive/spagatv2'),Path('/content/drive/MyDrive/GITIII-main')]:
        if not base.exists():continue
        for pat in ['*.py','*.ipynb','*/*.py','*/*.ipynb']:
            candidates.extend(base.glob(pat))
    candidates+=list(Path('/content/drive/MyDrive').glob('*.ipynb'))
    pattern=re.compile(r'centerx|centery|pixel.?size|micromet|x_centroid|y_centroid|global_px|0\.108',re.I)
    for p in sorted(set(candidates))[:300]:
        if not p.is_file() or p.stat().st_size>12_000_000:continue
        try:
            raw=p.read_text()
            if p.suffix=='.ipynb':
                raw='\n'.join(''.join(c.get('source',[])) for c in json.loads(raw).get('cells',[]))
            lines=raw.splitlines(); hits=[i for i,l in enumerate(lines) if pattern.search(l)]
            if not hits:continue
            selected=sorted({j for i in hits[:25] for j in range(max(0,i-2),min(len(lines),i+3))})
            snippets.append({'source':str(p),'sha256':sha(p),'excerpts':[{'line':j+1,'text':lines[j][:600]} for j in selected]})
        except (OSError,ValueError,UnicodeError) as e:
            snippets.append({'source':str(p),'read_error':str(e)})
    save_json(out/'audit.json',info);save_json(out/'source_excerpts.json',snippets)
    print('Coordinate audit saved; physical units await source verification.',flush=True)

if __name__=='__main__':
    config=json.loads(Path(sys.argv[2]).read_text())
    if sys.argv[1]=='prepare':prepare_cache(config)
    else:raise ValueError('Unknown operation')
