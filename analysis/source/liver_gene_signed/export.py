from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, shutil, sys, gc, zipfile, traceback, platform
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

BASE = Path('/content/drive/MyDrive')
DATA = BASE/'GITIII-main/liver_workdir/gitiii/data/processed'
CANDIDATES = [BASE/'spagatv2/results/liver_free_k4_test/free_seed123_best.pth',
              BASE/'spagatv2/results/liver_free_k4_3epochs/free_seed123_best.pth']
REFERENCE = BASE/'spagatv2/analysis/liver_sender_origin_random_control/matched_sender_random_control_repeats.csv'
ACTIVATIONS = BASE/'spagatv2/analysis/programs/Liver_validation_program_activations.npz'
BATCH_SIZE = 32
REPEATS = 10
MASK_SEED = 123
RECEIVERS = ['tumor_1','tumor_2']

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''): h.update(x)
    return h.hexdigest()

def jsonsave(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str))

class DiskDataset:
    """Same sample/flag order and batch fields as the supplied loader; disk-backed expression."""
    def __init__(self,data,cache,genes):
        self.genes=list(genes);self.samples=sorted(p.name[:-12] for p in data.glob('*_TypeExp.npz'))
        if self.samples!=['CancerousLiver','NormalLiver']:
            raise ValueError(f'Unexpected samples: {self.samples}')
        self.meta=[]; alltypes=set()
        for sample in self.samples:
            p=data/(sample+'.csv');header=pd.read_csv(p,nrows=0).columns
            cols=['subclass','centerx','centery']+(['flag'] if 'flag' in header else [])
            m=pd.read_csv(p,usecols=cols)
            if m['subclass'].isna().any():raise ValueError('Missing cell annotations')
            if 'flag' not in m:m['flag']=True
            if not m['flag'].isin([0,1,False,True]).all():raise ValueError('Nonbinary flags')
            alltypes.update(m['subclass']);self.meta.append(m)
        self.names=sorted(alltypes);self.type_ids={x:i for i,x in enumerate(self.names)}
        self.parts=[];self.refs=[];self.global_types=[];self.global_sample=[];self.global_rows=[]
        for k,(sample,m) in enumerate(zip(self.samples,self.meta)):
            n=len(m);directory=cache/sample;directory.mkdir(parents=True)
            exp=np.lib.format.open_memmap(directory/'expression.npy',mode='w+',dtype='float32',shape=(n,len(genes)))
            ids=np.lib.format.open_memmap(directory/'neighbors.npy',mode='w+',dtype='int64',shape=(n,50))
            types=m['subclass'].map(self.type_ids).to_numpy(dtype=np.int64)
            with np.load(data/(sample+'_TypeExp.npz'),allow_pickle=False) as z:
                if set(m['subclass'])-set(z.files):raise ValueError('Missing baseline types; refusing silent row filtering')
                baselines=np.zeros((len(self.names),len(genes)),np.float32)
                for name in set(m['subclass']):
                    v=z[name]
                    if v.shape!=(len(genes),) or not np.isfinite(v).all():raise ValueError('Invalid baseline')
                    baselines[self.type_ids[name]]=v
            offset=0;nc=[f'index_{i}' for i in range(50)]
            for chunk in pd.read_csv(data/(sample+'.csv'),usecols=list(genes)+nc,chunksize=4000):
                a=chunk.loc[:,genes].to_numpy(dtype=np.float32)
                nb=chunk.loc[:,nc].to_numpy()
                if not np.isfinite(a).all() or not np.isfinite(nb).all():raise ValueError('Nonfinite input')
                if not np.equal(nb,np.floor(nb)).all():raise ValueError('Noninteger neighbor indices')
                nb=nb.astype(np.int64)
                if nb.min()<0 or nb.max()>=n:raise ValueError('Neighbor index out of range')
                if not np.array_equal(nb[:,0],np.arange(offset,offset+len(a))):raise ValueError('index_0 is not the receiver row')
                exp[offset:offset+len(a)]=a;ids[offset:offset+len(a)]=nb;offset+=len(a)
            if offset!=n:raise ValueError('Row count mismatch')
            exp.flush();ids.flush()
            rows=np.flatnonzero(m['flag'].to_numpy()!=0)
            self.parts.append(dict(exp=exp,ids=ids,types=types,baseline=baselines,
                                   x=m.centerx.to_numpy(np.float32),y=m.centery.to_numpy(np.float32)))
            self.refs.extend((k,int(r)) for r in rows)
            self.global_types.extend(types[rows]);self.global_sample.extend([sample]*len(rows));self.global_rows.extend(rows)
            print('Cached',sample,n,'rows;',len(rows),'eligible dataset entries',flush=True)
        self.global_types=np.asarray(self.global_types);self.global_rows=np.asarray(self.global_rows)
        del self.meta;gc.collect()
    def __len__(self):return len(self.refs)
    def __getitem__(self,i):
        k,row=self.refs[int(i)];p=self.parts[k];nb=p['ids'][row];types=p['types'][nb]
        exp=np.asarray(p['exp'][nb],dtype=np.float32)
        return dict(x=exp,type_exp=p['baseline'][types],y=exp[0],cell_types=types.copy(),
                    position_x=p['x'][nb],position_y=p['y'][nb],neighbor_mask=np.ones(50,np.float32))
    def counts(self,indices):
        result=[]
        t1,t2=[self.type_ids[x] for x in RECEIVERS]
        for i in indices:
            k,r=self.refs[int(i)];p=self.parts[k];v=p['types'][p['ids'][r,1:]]
            result.append((int((v==t1).sum()),int((v==t2).sum())))
        return np.asarray(result,dtype=np.int64)

def instantiate(checkpoint,genes,ligands):
    from spagat.gene_program_model import SpaGP
    state=checkpoint['model'];cfg=checkpoint.get('model_config',checkpoint.get('config',{}))
    if not isinstance(cfg,dict):cfg=vars(cfg)
    if 'program_tokens' not in state:raise ValueError('Not the expected program-aware model')
    args={k:cfg.get(k,v) for k,v in dict(node_dim=256,edge_dim=48,num_heads=2,n_layers=1,att_dim=8).items()}
    args.update(n_programs=state['program_tokens'].shape[0],program_token_dim=state['program_tokens'].shape[1],
                program_aware=True,message_decoder='free',routing_mode='program')
    if cfg.get('message_decoder','free')!='free' or cfg.get('routing_mode','program')!='program':
        raise ValueError('Checkpoint configuration differs from the supplied evaluation script')
    model=SpaGP(genes=genes,ligands_info=ligands,**args)
    model.load_state_dict(state,strict=True)
    return model.cuda().eval(),args

def predict(model,dataset,indices,directory):
    directory.mkdir(parents=True,exist_ok=True);shape=(len(indices),len(dataset.genes))
    pred=np.lib.format.open_memmap(directory/'prediction.npy',mode='w+',dtype='float32',shape=shape)
    target=np.lib.format.open_memmap(directory/'target.npy',mode='w+',dtype='float32',shape=shape)
    offset=0
    with torch.inference_mode():
        for batch in DataLoader(Subset(dataset,indices.tolist()),batch_size=BATCH_SIZE,shuffle=False,num_workers=0):
            batch={k:v.cuda() for k,v in batch.items()};p,_=model(batch);n=len(p)
            if not torch.isfinite(p).all():raise ValueError('Nonfinite model predictions')
            pred[offset:offset+n]=p.cpu().numpy();target[offset:offset+n]=batch['y'].cpu().numpy();offset+=n
            if offset%3200==0:print('Predicted',offset,'/',len(indices),flush=True)
    pred.flush();target.flush();return pred,target

def mse_rows(pred,target,selection):
    total=0.;n=0
    idx=np.flatnonzero(selection)
    for block in np.array_split(idx,max(1,len(idx)//1000+1)):
        e=pred[block].astype(np.float64)-target[block];total+=np.square(e).sum();n+=e.size
    return total/n

def gene_metrics(pred,target,selection,genes):
    # Two-pass centered moments; report undefined PCC explicitly for constant genes.
    idx=np.flatnonzero(selection);g=len(genes);sx=np.zeros(g);sy=np.zeros(g);ss=np.zeros(g);zero=np.zeros(g)
    blocks=np.array_split(idx,max(1,len(idx)//1000+1))
    for b in blocks:
        x=np.asarray(pred[b],np.float64);y=np.asarray(target[b],np.float64)
        sx+=x.sum(0);sy+=y.sum(0);ss+=((x-y)**2).sum(0);zero+=(y*y).sum(0)
    mx=sx/len(idx);my=sy/len(idx);xx=np.zeros(g);yy=np.zeros(g);xy=np.zeros(g)
    for b in blocks:
        x=np.asarray(pred[b],np.float64)-mx;y=np.asarray(target[b],np.float64)-my
        xx+=(x*x).sum(0);yy+=(y*y).sum(0);xy+=(x*y).sum(0)
    denom=np.sqrt(xx*yy);pcc=np.full(g,np.nan);np.divide(xy,denom,out=pcc,where=denom>0)
    ev=np.full(g,np.nan);np.divide(ss,zero,out=ev,where=zero>0);ev=100*(1-ev)
    return pd.DataFrame(dict(gene=genes,n_cells=len(idx),PCC=pcc,MSE=ss/len(idx),MSE0=zero/len(idx),
                             EV_percent=ev,observed_mean=my,predicted_mean=mx,
                             observed_variance=yy/len(idx),predicted_variance=xx/len(idx),
                             PCC_defined=denom>0))

def mask_positions(types,ids,condition,rng):
    result=[]
    for row in types:
        a=np.flatnonzero(row[1:]==ids[0]);b=np.flatnonzero(row[1:]==ids[1]);m=min(len(a),len(b))
        if m==0:raise ValueError('Ineligible receiver reached masking')
        pool=a if condition=='tumor_1' else b if condition=='tumor_2' else np.arange(49)
        result.append(rng.choice(pool,size=m,replace=False)+1)
    return result

def export_masking(model,ds,all_indices,full,truth,types,counts,out,cache):
    rows=[];scalar=[];g=len(ds.genes);conditions=['tumor_1','tumor_2','random']
    ids=[ds.type_ids[x] for x in RECEIVERS];both=(counts>0).all(1)
    for receiver_no,receiver in enumerate(RECEIVERS):
        positions=np.flatnonzero((types==ds.type_ids[receiver])&both)
        indices=all_indices[positions];n=len(indices)
        for ci,condition in enumerate(conditions):
            avgpath=cache/f'{receiver}__{condition}__mean_delta.npy'
            avg=np.lib.format.open_memmap(avgpath,mode='w+',dtype='float32',shape=(n,g));avg[:]=0
            for repeat in range(REPEATS):
                rng=np.random.default_rng(MASK_SEED+repeat*1000+ci*100+receiver_no)
                signed=np.zeros(g);absolute=np.zeros(g);sq=np.zeros(g);offset=0
                with torch.inference_mode():
                    for batch in DataLoader(Subset(ds,indices.tolist()),batch_size=BATCH_SIZE,shuffle=False,num_workers=0):
                        choices=mask_positions(batch['cell_types'].numpy(),ids,condition,rng)
                        size=len(choices);mask=torch.ones((size,50,50,model.n_programs),dtype=torch.bool,device='cuda')
                        for r,chosen in enumerate(choices):mask[r,0,torch.as_tensor(chosen,device='cuda'),:]=False
                        b={k:v.cuda() for k,v in batch.items()};p,_=model(b,program_edge_mask=mask)
                        p=p.cpu().numpy();pos=positions[offset:offset+size]
                        if not np.isfinite(p).all():raise ValueError('Nonfinite masked predictions')
                        delta=p.astype(np.float64)-full[pos]
                        avg[offset:offset+size]+=np.asarray(delta/REPEATS,np.float32)
                        signed+=delta.sum(0);absolute+=np.abs(delta).sum(0)
                        sq+=((p.astype(np.float64)-truth[pos])**2).sum(0);offset+=size
                for k,gene in enumerate(ds.genes):
                    rows.append(dict(receiver=receiver,condition=condition,repeat=repeat,gene=gene,n_receivers=n,
                                     mean_signed_delta=signed[k]/n,mean_absolute_delta=absolute[k]/n,masked_MSE=sq[k]/n))
                fm=mse_rows(full,truth,(types==ds.type_ids[receiver])&both)
                scalar.append(dict(receiver=receiver,condition=condition,repeat=repeat,n_receivers=n,
                                   mean_edges_removed=float(np.min(counts[positions],axis=1).mean()),
                                   full_MSE=fm,masked_MSE=float(sq.sum()/(n*g)),
                                   mean_abs_prediction_change=float(absolute.sum()/(n*g)),
                                   relative_MSE_increase_percent=float(100*(sq.sum()/(n*g)/fm-1))))
                pd.DataFrame(scalar).to_csv(out/'masking_scalar_repeats.csv',index=False)
                print(receiver,condition,'repeat',repeat+1,'/',REPEATS,flush=True)
            avg.flush();del avg
            shutil.copy2(avgpath,out/'arrays'/avgpath.name)
            np.save(out/'arrays'/f'{receiver}__{condition}__dataset_indices.npy',indices)
    table=pd.DataFrame(rows);table.to_csv(out/'signed_masking_gene_repeats.csv',index=False)
    summary=table.groupby(['receiver','condition','gene'],sort=False).agg(
        n_receivers=('n_receivers','first'),mean_signed_delta=('mean_signed_delta','mean'),
        SD_signed_delta_across_repeats=('mean_signed_delta','std'),
        mean_absolute_delta=('mean_absolute_delta','mean'),
        SD_absolute_delta_across_repeats=('mean_absolute_delta','std'),
        mean_masked_MSE=('masked_MSE','mean')).reset_index()
    summary.to_csv(out/'signed_masking_gene_summary.csv',index=False)

def run_export():
    if not torch.cuda.is_available():raise RuntimeError('请选择 Colab GPU 后再运行；本程序不会改用 CPU 长时间推理。')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')
    out=BASE/'SpaGAT_revision'/('R3_5_7_gene_signed_'+stamp);out.mkdir(parents=True,exist_ok=False)
    cache=Path('/content')/('R3_5_7_cache_'+stamp);cache.mkdir(exist_ok=False)
    provenance=dict(status='running',new_training=False,new_inference=True,
                    scope='Existing-model validation gene recovery and matched-count signed masking; no independent liver test',
                    data_dir=str(DATA),batch_size=BATCH_SIZE,repeats=REPEATS,mask_seed=MASK_SEED,
                    torch_version=torch.__version__,numpy_version=np.__version__,python_version=platform.python_version(),
                    candidate_checks=[],source_hashes=SOURCE_HASHES,
                    program_recovery_status='Pending external biological gene sets; latent channels are not used as gene sets')
    try:
        torch.manual_seed(123);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        genes=list(torch.load(DATA.parent/'genes.pth',map_location='cpu',weights_only=False))
        ligands=torch.load(DATA.parent/'ligands.pth',map_location='cpu',weights_only=False)
        if len(genes)!=1000 or len(set(genes))!=len(genes):raise ValueError('Unexpected or duplicated Liver genes')
        if shutil.disk_usage('/content').free<8*1024**3:raise RuntimeError('Local runtime needs at least 8 GB free disk space')
        ds=DiskDataset(DATA,cache/'data',genes)
        provenance['dataset_size']=len(ds);provenance['cell_type_names']=ds.names
        provenance['input_files']=[dict(path=str(p),size=p.stat().st_size,mtime=p.stat().st_mtime)
                                   for p in DATA.iterdir() if p.suffix in {'.csv','.npz'}]
        reference=pd.read_csv(REFERENCE);provenance['reference_sha256']=sha(REFERENCE)
        expected={}
        for name in RECEIVERS:
            x=reference[reference.receiver==name]
            if x.empty or x.n_receivers.nunique()!=1 or x.full_MSE.max()-x.full_MSE.min()>1e-7:
                raise ValueError('Inconsistent reference records')
            expected[name]=(int(x.n_receivers.iloc[0]),float(x.full_MSE.iloc[0]))
        selected=[]
        for ci,path in enumerate(CANDIDATES):
            rec=dict(checkpoint=str(path),matched=False)
            try:
                cp=torch.load(path,map_location='cpu',weights_only=False)
                val=np.asarray(cp['shared_split']['val_indices'],dtype=np.int64)
                train=np.asarray(cp['shared_split']['train_indices'],dtype=np.int64)
                split=json.loads((path.parent/'shared_split.json').read_text())
                if not np.array_equal(val,np.asarray(split['val_indices'])):raise ValueError('Saved split mismatch')
                if len(np.unique(val))!=len(val) or len(np.intersect1d(val,train)) or val.min()<0 or val.max()>=len(ds):
                    raise ValueError('Invalid validation indices')
                with np.load(ACTIVATIONS,allow_pickle=False) as old:
                    if not np.array_equal(val,old['val_indices']):raise ValueError('Validation order differs from saved activation export')
                    actual=np.asarray(ds.names)[ds.global_types[val]]
                    if not np.array_equal(actual,old['cell_type']):raise ValueError('Cell annotation mapping differs from saved export')
                    for coord,key in [('x','position_x'),('y','position_y')]:
                        coords=np.array([ds.parts[ds.refs[int(i)][0]][coord][ds.refs[int(i)][1]] for i in val])
                        if not np.allclose(coords,old[key],rtol=0,atol=1e-4):raise ValueError('Receiver coordinates differ from saved export')
                use=val[np.isin(ds.global_types[val],[ds.type_ids[x] for x in RECEIVERS])]
                types=ds.global_types[use];counts=ds.counts(use);both=(counts>0).all(1)
                rec['counts']={n:int(((types==ds.type_ids[n])&both).sum()) for n in RECEIVERS}
                if any(rec['counts'][n]!=expected[n][0] for n in RECEIVERS):raise ValueError('Eligible receiver count mismatch')
                model,args=instantiate(cp,genes,ligands);del cp;gc.collect()
                pred,target=predict(model,ds,use,cache/f'candidate_{ci}')
                values={n:mse_rows(pred,target,(types==ds.type_ids[n])&both) for n in RECEIVERS}
                rec.update(model_config=args,full_MSE=values,checkpoint_sha256=sha(path),
                           split_sha256=sha(path.parent/'shared_split.json'),n_validation=len(val))
                # Tolerance allows float32/reduction differences, not broad performance similarity.
                rec['matched']=all(np.isclose(values[n],expected[n][1],rtol=1e-5,atol=1e-6) for n in RECEIVERS)
                if rec['matched']:selected.append((path,ci,use,types,counts,args))
                del model,pred,target;torch.cuda.empty_cache();gc.collect()
            except Exception as e:rec['error']=str(e)
            provenance['candidate_checks'].append(rec);jsonsave(out/'provenance.json',provenance)
            print('Checkpoint check:',rec,flush=True)
        if len(selected)!=1:
            raise RuntimeError(f'Need exactly one matching checkpoint; found {len(selected)}. Export stopped; upload the diagnostic ZIP.')
        path,ci,use,types,counts,args=selected[0]
        provenance['selected_checkpoint']=str(path)
        provenance['identity_evidence']='Unique agreement among two candidates on saved validation order, cell labels, coordinates, eligible counts and two full MSE values; not a recovered historical command.'
        full=np.load(cache/f'candidate_{ci}/prediction.npy',mmap_mode='r');truth=np.load(cache/f'candidate_{ci}/target.npy',mmap_mode='r')
        # The supplied loader returns processed expression as y, without extra centering.
        # Record sign/centering diagnostics instead of silently subtracting a second baseline.
        provenance['target_semantics']='Target is the processed CSV gene vector used by the supplied loader; no additional baseline subtraction. Residual interpretation still requires preprocessing provenance.'
        (out/'arrays').mkdir();shutil.copy2(cache/f'candidate_{ci}/prediction.npy',out/'arrays/prediction.npy')
        shutil.copy2(cache/f'candidate_{ci}/target.npy',out/'arrays/target.npy')
        np.save(out/'arrays/dataset_indices.npy',use)
        (out/'genes.json').write_text(json.dumps(genes))
        pd.DataFrame(dict(dataset_index=use,sample=[ds.global_sample[i] for i in use],
                          csv_row=ds.global_rows[use],cell_type=np.asarray(ds.names)[types],
                          n_tumor1_neighbors=counts[:,0],n_tumor2_neighbors=counts[:,1],
                          masking_eligible=(counts>0).all(1))).to_csv(out/'receiver_metadata.csv',index=False)
        for n in RECEIVERS:
            gene_metrics(full,truth,types==ds.type_ids[n],genes).to_csv(out/f'{n}_gene_recovery.csv',index=False)
        model,_=instantiate(torch.load(path,map_location='cpu',weights_only=False),genes,ligands)
        export_masking(model,ds,use,full,truth,types,counts,out,cache)
        provenance['status']='complete_gene_and_matched_count_masking'
        print('基因预测和等数量 signed masking 导出完成；biological program analysis 尚未完成。')
    except Exception as e:
        provenance['status']='stopped';provenance['error']=str(e)
        (out/'error_trace.txt').write_text(traceback.format_exc());print('停止：',str(e))
    finally:
        jsonsave(out/'provenance.json',provenance)
        (out/'README.txt').write_text('Small ZIP contains tables and provenance, not large arrays.\n'
          'Arrays remain in the Drive output directory. prediction/target rows align with receiver_metadata.csv; columns align with genes.json.\n'
          'Signed delta is masked prediction minus full prediction. Saved mean_delta arrays average cellwise signed delta over 10 repeats; they are not individual-repeat predictions.\n'
          'Repeat SD describes fixed-model neighbor sampling only. Gene PCC is NaN for zero-variance observed/predicted genes; no genes are removed.\n'
          'Matched-count masking only; this is not a rerun of the fraction50 control. No biological gene sets or disease identities have been inferred.\n'
          'Do not interpret outputs as residual recovery until processed-target semantics are verified; no second residualization was applied.\n')
        archive=out.with_suffix('.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in out.iterdir():
                if p.is_file():z.write(p,out.name+'/'+p.name)
        print('请上传：',archive,flush=True)
    return str(archive)
