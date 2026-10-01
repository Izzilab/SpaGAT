"""Read stored metadata/flags; never fit a model or invent upstream QC counts."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,uuid,zipfile,traceback
import numpy as np
import pandas as pd

META=['centerx','centery','section','subclass']

def boolean_flags(s):
    if s.isna().any():raise ValueError('Missing eligibility flags')
    t=s.astype(str).str.lower().str.strip()
    if not t.isin(['true','false','1','0','1.0','0.0']).all():raise ValueError('Unrecognized flag values')
    return t.isin(['true','1','1.0']).to_numpy()

def meta_digest(df):
    x=df[META].copy()
    x['centerx']=pd.to_numeric(x.centerx).astype('float64');x['centery']=pd.to_numeric(x.centery).astype('float64')
    x['section']=x.section.astype(str);x['subclass']=x.subclass.astype(str)
    return hashlib.sha256(pd.util.hash_pandas_object(x,index=False).values.tobytes()).hexdigest()

def read_meta(path):
    header=pd.read_csv(path,nrows=0).columns
    if not set(META).issubset(header):raise ValueError('Missing metadata columns '+str(path))
    return pd.read_csv(path,usecols=META+(['flag'] if 'flag' in header else []))

def source_inputs(dataset,candidates,events):
    found=[]
    for s in candidates:
        p=Path(s)
        if p.is_file():
            try:
                d=read_meta(p);found.append((p,d,meta_digest(d)))
                events.append({'dataset':dataset,'input_candidate':str(p),'rows':len(d),'metadata_sha256':found[-1][2]})
            except Exception as e:events.append({'dataset':dataset,'input_candidate':str(p),'error':str(e)})
    if not found:
        events.append({'dataset':dataset,'gap':'No original model-input CSV located; upstream count unknown.'});return None
    if len({f[2] for f in found})>1:
        events.append({'dataset':dataset,'gap':'Different candidate metadata; no automatic source selection.'});return None
    return found[0][1]

def audit_section(dataset,sample,path,input_df,partition=None,expected_eligible=None):
    d=read_meta(path)
    if d['section'].nunique()!=1 or str(d['section'].iloc[0])!=sample:raise ValueError('Section mismatch '+sample)
    if d[META].isna().any().any():raise ValueError('Missing required metadata '+sample)
    if 'flag' not in d:raise ValueError('Stored flag missing; do not assume all cells eligible')
    flag=boolean_flags(d.flag)
    type_file=Path(path).with_name(sample+'_TypeExp.npz')
    with np.load(type_file,allow_pickle=False) as z:types=set(z.files)
    valid=d.subclass.isin(types).to_numpy()
    eligible=valid & flag
    if expected_eligible is not None and int(eligible.sum())!=expected_eligible:raise ValueError('Saved split count mismatch '+sample)
    same_input=None;input_n=None
    if input_df is not None:
        raw=input_df[input_df.section.astype(str).eq(sample)]
        input_n=len(raw);same_input=bool(len(raw)==len(d) and meta_digest(raw)==meta_digest(d))
    labeled=d.subclass.ne('Unlabeled').to_numpy()
    from scipy.spatial import cKDTree
    xy=d[['centerx','centery']].to_numpy(dtype=float)
    if not np.isfinite(xy).all():raise ValueError('Nonfinite coordinates '+sample)
    distance=cKDTree(xy).query(xy,k=2)[0][:,1]
    reconstructed=labeled & (distance<80)
    # Reconstructed counts are explicitly diagnostic, not asserted historical stages.
    if dataset=='SEA_AD':bio='.'.join(sample.split('.')[:3])
    elif dataset=='Mouse':bio=sample.split('_')[0]
    else:bio=sample
    return {'dataset':dataset,'sample':bio,'section':sample,'partition':partition or 'within_specimen',
      'input_csv_cells':input_n,'stored_processed_cells':len(d),
      'input_processed_metadata_identical':same_input,
      'baseline_compatible_cells':int(valid.sum()),'eligible_receivers':int(eligible.sum()),
      'excluded_by_stored_flag_after_baseline_check':int((valid & ~flag).sum()),
      'baseline_incompatible_cells':int((~valid).sum()),
      'stored_flag_equals_annotation_and_nearest_distance_lt80':bool(np.array_equal(flag,reconstructed)),
      'diagnostic_unlabeled_cells':int((~labeled).sum()),
      'diagnostic_labeled_without_neighbor_lt80':int((labeled & (distance>=80)).sum()),
      'diagnostic_after_label_check':int(labeled.sum()),
      'diagnostic_after_label_and_distance_check':int(reconstructed.sum()),
      'processed_path':str(path),'type_exp_path':str(type_file),
      'metadata_sha256':meta_digest(d),'eligibility_flag_sha256':hashlib.sha256(flag.tobytes()).hexdigest()}

def run_count_audit():
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')
    out=Path('/content')/f'R1_Minor3_flow_{stamp}_{uuid.uuid4().hex[:6]}';out.mkdir(exist_ok=False)
    rows=[];events=[];errors=[]
    for dataset in ['SEA_AD','Mouse','Liver']:
        print('读取 metadata 和 flag：',dataset,flush=True)
        input_df=source_inputs(dataset,RAW_CANDIDATES[dataset],events)
        try:
            if dataset!='Liver':
                rec=next(r for r in REFERENCES if r['dataset']==dataset)
                for pathkey,hashkey in [('split_path','split_sha256'),('samples_path','samples_sha256')]:
                    if hashlib.sha256(Path(rec[pathkey]).read_bytes()).hexdigest()!=rec[hashkey]:raise ValueError('Fixed split audit hash changed')
                split=json.loads(Path(rec['split_path']).read_text());audit=pd.read_csv(rec['samples_path'])
                specs=[(str(r['sample']),Path(split['processed_dir'])/(str(r['sample'])+'.csv'),r['split'],int(r['eligible_cells'])) for _,r in audit.iterrows()]
                events.append({'dataset':dataset,'split':rec['split_path'],'sample_audit':rec['samples_path']})
            else:
                folder=Path(LIVER_PROCESSED)
                specs=[(s,folder/(s+'.csv'),None,n) for s,n in [('CancerousLiver',460429),('NormalLiver',332828)]]
            for i,(sample,path,partition,expected) in enumerate(specs):
                try:rows.append(audit_section(dataset,sample,path,input_df,partition,expected))
                except Exception as e:errors.append({'dataset':dataset,'section':sample,'error':str(e)})
                if (i+1)%10==0:print(dataset,i+1,'/',len(specs),flush=True)
            if input_df is not None:
                extra=sorted(set(input_df.section.astype(str))-set(s[0] for s in specs))
                if extra:events.append({'dataset':dataset,'input_sections_not_in_processed_set':extra,
                                        'n_cells_in_these_sections':int(input_df.section.astype(str).isin(extra).sum())})
        except Exception as e:errors.append({'dataset':dataset,'error':str(e)})
    df=pd.DataFrame(rows);df.to_csv(out/'counts_by_section.csv',index=False)
    if len(df):
        grouped=[]
        for (ds,sample),g in df.groupby(['dataset','sample'],sort=True):
            grouped.append({'dataset':ds,'sample':sample,'input_sections':len(g) if g.input_csv_cells.notna().all() else None,
              'input_csv_cells':int(g.input_csv_cells.sum()) if g.input_csv_cells.notna().all() else None,
              'processed_sections':len(g),'processed_cells':int(g.stored_processed_cells.sum()),
              'baseline_compatible_cells':int(g.baseline_compatible_cells.sum()),
              'eligible_sections':int((g.eligible_receivers>0).sum()),'eligible_receivers':int(g.eligible_receivers.sum()),
              'excluded_flag_cells':int(g.excluded_by_stored_flag_after_baseline_check.sum()),
              'all_input_metadata_identical':bool(g.input_processed_metadata_identical.eq(True).all()),
              'flag80_reconstruction_matches_all_sections':bool(g.stored_flag_equals_annotation_and_nearest_distance_lt80.all()),
              'train_receivers':int(g.loc[g.partition.eq('train'),'eligible_receivers'].sum()) if ds!='Liver' else None,
              'validation_receivers':int(g.loc[g.partition.eq('validation'),'eligible_receivers'].sum()) if ds!='Liver' else None,
              'test_receivers':int(g.loc[g.partition.eq('test'),'eligible_receivers'].sum()) if ds!='Liver' else None})
        pd.DataFrame(grouped).to_csv(out/'counts_by_sample.csv',index=False)
    # Record provenance/filters from small source files only. Do not scan the entire Drive.
    srcdir=out/'preprocessing_source';srcdir.mkdir()
    for p in PREPROCESSING_SOURCES:
        p=Path(p)
        if p.is_file() and p.stat().st_size<2_000_000:
            dest=srcdir/(hashlib.sha256(str(p).encode()).hexdigest()[:8]+'_'+p.name)
            dest.write_bytes(p.read_bytes());events.append({'source_copied':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    report={'scope':'Input CSV to saved processed rows to loader-compatible rows to stored receiver eligibility. Upstream provider QC and raw-to-input conversion not inferred.',
      'errors':errors,'source_events':events,'expected_sections':{'SEA_AD':69,'Mouse':64,'Liver':2},
      'actual_sections':df.groupby('dataset').size().to_dict() if len(df) else {},
      'remaining_requirement':'Verify source release/raw-to-input filters. A matching reconstructed 80-unit flag is evidence about the stored flag, not evidence that no upstream filtering occurred.',
      'training_performed':False,'inference_performed':False,'original_files_modified':False}
    (out/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (out/'README.txt').write_text('此包用于核对 S1 数据流程表，尚非最终发表表。\n逐样本数量在 counts_by_sample.csv，逐切片在 counts_by_section.csv。\n未知值留空；distance80 为诊断重建，必须结合源码核对。\n源数据提供者的 QC、从原始文件到输入 CSV 的步骤仍需原始转换记录。\n')
    bundle=out.with_suffix('.zip')
    with zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out))
    print('计数 ZIP：',bundle,'；未完成的读取：',len(errors),flush=True)
    return bundle
