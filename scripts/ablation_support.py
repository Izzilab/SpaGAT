"""Audit and summarize a fixed, predeclared component comparison."""
from pathlib import Path
import hashlib, json, importlib.util, sys
import numpy as np
import pandas as pd

VARIANTS = {'no_distance': {'n_programs': 4, 'routing_mode': 'program', 'definition': 'All five distance input features are fixed to zero throughout training and evaluation; the original 50-cell neighborhoods, ordering, ligand inputs and all layers are retained. This tests explicit distance features conditional on the spatial graph.'}}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def array_sha(a):
    # Match the existing Figure 2/R3.3 provenance format exactly.
    a=np.ascontiguousarray(a,dtype=np.float32)
    h=hashlib.sha256(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()

def save_json(p,x):
    Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))

def check_reference_files(records):
    """Check explicit training-record paths; never infer weights from NPZ location."""
    missing=[]
    for rec in records:
        label=f"{rec['dataset']} seed={rec['seed']}"
        if not rec.get('checkpoint_path'):
            raise ValueError(f'{label}: explicit checkpoint_path is required; reload the v3 notebook.')
        ckpt=Path(rec['checkpoint_path'])
        if ckpt.name!=f"spagat_seed{rec['seed']}_best.pth":
            raise ValueError(f'{label}: checkpoint filename does not match seed: {ckpt}')
        for field in ['checkpoint_path','npz','split_path','samples_path']:
            p=Path(rec[field])
            if not p.is_file():missing.append(f'{label} {field}: {p}')
    if missing:
        raise FileNotFoundError('Missing reference files; no training started and no substitute selected:\n'+'\n'.join(missing))
    return [{'dataset':r['dataset'],'seed':r['seed'],
             'checkpoint_path':r['checkpoint_path'],'prediction_path':r['npz']} for r in records]

def metrics(p,y):
    p=np.asarray(p,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    if p.shape!=y.shape or p.ndim!=2 or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Invalid prediction/target arrays')
    pc=p-p.mean(0);yc=y-y.mean(0)
    denom=np.sqrt(np.maximum(np.sum(pc*pc,0)*np.sum(yc*yc,0),1e-12))
    corr=np.clip(np.sum(pc*yc,0)/denom,-1,1)
    mse=float(np.mean((p-y)**2));mse0=float(np.mean(y*y))
    if mse0<=0:raise ValueError('Zero MSE0')
    return {'median_gene_pcc':float(np.sort(corr)[(len(corr)-1)//2]),
            'mse':mse,'mse0':mse0,'ev_zero_percent':100*(1-mse/mse0)},corr

def load_reference(rec,split):
    with np.load(rec['npz'],allow_pickle=False) as z:
        p=z['prediction'];y=z['target'];genes=z['genes'].astype(str)
        if not np.array_equal(z['test_indices'],split['test_indices']):raise ValueError('Reference test order changed')
        if array_sha(y)!=rec['target_sha256_float32']:raise ValueError('Reference target hash changed')
        if array_sha(p)!=rec['prediction_sha256_float32']:raise ValueError('Reference prediction hash changed')
        return p,y,genes

def import_train(source):
    sys.path.insert(0,str(source))
    spec=importlib.util.spec_from_file_location('fixed_train',Path(source)/'scripts/train.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m

def build_model(genes,ligands,variant='full'):
    from spagat.gene_program_model import SpaGP
    v=VARIANTS.get(variant,{'n_programs':4,'routing_mode':'program'})
    m=SpaGP(genes,ligands,node_dim=256,edge_dim=48,num_heads=2,n_layers=1,att_dim=8,
            n_programs=v['n_programs'],program_aware=True,program_token_dim=32,
            message_decoder='free',routing_mode=v['routing_mode'])
    m.embeddings.component_ablation=variant
    return m

def synthetic_checks():
    import torch
    from spagat.gene_program_model import SpaGP_Loss
    torch.manual_seed(23)
    genes=['g'+str(i) for i in range(8)];ligands=([['g0'],['g1']],[[0],[0]])
    x={'x':torch.rand(2,50,8)*.1,'type_exp':torch.ones(2,50,8),
       'cell_types':torch.randint(0,3,(2,50)), 'position_x':torch.rand(2,50)*40,
       'position_y':torch.rand(2,50)*40,'y':torch.rand(2,8),'neighbor_mask':torch.ones(2,50)}
    report={}
    for variant in ['full',*VARIANTS]:
        model=build_model(genes,ligands,variant)
        seen=[]
        hook=model.embeddings.ligands_encoder.register_forward_pre_hook(lambda m,a:seen.append(a[0].detach().clone()))
        result=model(x);hook.remove()
        loss=SpaGP_Loss(genes,ligands,lambda_orth=.1,lambda_sparse=.01)(result,x['y'])[0]
        if not torch.isfinite(loss):raise ValueError('Nonfinite synthetic loss '+variant)
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Nonfinite gradient')
        if variant=='no_distance':
            altered=dict(x,position_x=x['position_x']*37+10,position_y=-x['position_y']*13)
            with torch.no_grad():assert torch.equal(model(x)[0],model(altered)[0]),'Distance dependence remains'
        report[variant]={'parameters':sum(p.numel() for p in model.parameters()),'finite_loss_and_gradients':True}
    assert report['full']['parameters']==report['no_distance']['parameters']
    return report

def full_preflight(config):
    import torch
    from torch.utils.data import DataLoader,Subset
    source=Path(config['source']);train=import_train(source)
    from spagat.dataloader import SPAGAT_dataset
    ds=config['dataset'];records=config['references']
    check_reference_files(records)
    split_path=Path(records[0]['split_path']);split=json.loads(split_path.read_text())
    data=train.SPAGAT_dataset(split['processed_dir'],num_neighbors=50)
    train.load_and_validate_shared_split(split_path,len(data),split['processed_dir'])
    train.validate_sample_order(str(split_path),data,split)
    # Collect actual test targets directly from the frozen loader's tensors.
    # This avoids unnecessary graph construction, and checks every test target.
    ix=np.asarray(split['test_indices'],dtype=np.int64)
    ends=np.cumsum(data.meta_counts);starts=ends-np.asarray(data.meta_counts)
    sample_id=np.searchsorted(ends,ix,side='right')
    target=np.empty((len(ix),len(data.genes)),dtype=np.float32)
    for j in range(len(data.samples)):
        which=np.where(sample_id==j)[0]
        if len(which):
            row=data.arg_meta[j][torch.as_tensor(ix[which]-starts[j],dtype=torch.long)]
            target[which]=data.exps[j][row].numpy()
    if array_sha(target)!=records[0]['target_sha256_float32']:raise ValueError('Current loader targets differ from Figure 2')
    report={'dataset':ds,'all_test_targets_match':True,'full_runs':[],'synthetic_checks':synthetic_checks()}
    device=torch.device('cuda')
    batch=next(iter(DataLoader(Subset(data,split['test_indices'][:32]),batch_size=32,shuffle=False)))
    batch={k:v.to(device) for k,v in batch.items()}
    for rec in records:
        p,y,genes=load_reference(rec,split)
        if list(genes)!=list(map(str,data.genes)):raise ValueError('Gene order mismatch')
        ckpt=Path(rec['checkpoint_path'])
        saved=torch.load(ckpt,map_location='cpu',weights_only=False)
        expected={'max_epochs':10,'patience':5,'batch_size':32,'lr':1e-4,'num_neighbors':50,
                  'node_dim':256,'edge_dim':48,'num_heads':2,'n_layers':1,'att_dim':8,
                  'n_programs':4,'program_token_dim':32,'lambda_orth':.1,'lambda_sparse':.01}
        for k,v in expected.items():
            if saved['config'].get(k)!=v:raise ValueError(f'Full config mismatch {ds}/{rec["seed"]}/{k}')
        if saved['seed']!=rec['seed']:raise ValueError('Checkpoint seed differs')
        oldsplit=saved['shared_split']
        for key in ['train_indices','val_indices','test_indices']:
            if oldsplit[key]!=split[key]:raise ValueError('Checkpoint split differs')
        model=build_model(data.genes,data.interactions).to(device)
        model.load_state_dict(saved['model'],strict=True);model.eval()
        with torch.no_grad():test=model(batch)[0].cpu().numpy()
        if not np.allclose(test,p[:len(test)],rtol=1e-4,atol=1e-4):raise ValueError('Frozen full model does not reproduce saved prediction batch')
        m,_=metrics(p,y)
        report['full_runs'].append({'seed':rec['seed'],'checkpoint_path':str(ckpt),
          'reference_prediction_path':rec['npz'],'checkpoint_sha256':sha(ckpt),
          'prediction_sha256_float32':array_sha(p),'target_sha256_float32':array_sha(y),
          'first_32_predictions_match':True,'parameter_count':sum(q.numel() for q in model.parameters()),**m})
        del saved,model,p,y
        torch.cuda.empty_cache()
    save_json(config['output'],report)
    print('PREFLIGHT PASSED:',ds,flush=True)

def summarize(root,references,datasets,seeds):
    root=Path(root);rows=[];per_gene=[];provenance=[]
    for ds in datasets:
        for rec in [r for r in references if r['dataset']==ds and r['seed'] in seeds]:
            split=json.loads(Path(rec['split_path']).read_text());full,y,genes=load_reference(rec,split)
            for variant in ['full',*VARIANTS]:
                p=full;path=Path(rec['npz'])
                if variant!='full':
                    path=root/'runs'/ds/f"{variant}_seed{rec['seed']}"/f"seed{rec['seed']}_test/test_predictions.npz"
                    if not path.exists():continue
                    with np.load(path,allow_pickle=False) as z:
                        if not np.array_equal(z['test_indices'],split['test_indices']):raise ValueError('Ablation test indices changed')
                        if list(z['genes'].astype(str))!=list(genes):raise ValueError('Ablation gene order changed')
                        if array_sha(z['target'])!=rec['target_sha256_float32']:raise ValueError('Ablation target changed')
                        p=z['prediction']
                met,corr=metrics(p,y)
                rows.append({'dataset':ds,'variant':variant,'seed':rec['seed'],**met})
                per_gene.extend({'dataset':ds,'variant':variant,'seed':rec['seed'],'gene':g,'PCC':float(c)} for g,c in zip(genes,corr))
                provenance.append({'dataset':ds,'variant':variant,'seed':rec['seed'],'path':str(path),
                                   'target_sha256_float32':array_sha(y),'prediction_sha256_float32':array_sha(p)})
    frame=pd.DataFrame(rows);frame.to_csv(root/'per_seed_test_metrics.csv',index=False)
    pd.DataFrame(per_gene).to_csv(root/'per_gene_test_PCC.csv',index=False)
    aggregate=frame.groupby(['dataset','variant'])[['median_gene_pcc','mse','mse0','ev_zero_percent']].agg(['count','mean','std'])
    aggregate.columns=['_'.join(c) for c in aggregate.columns]
    aggregate.reset_index().to_csv(root/'mean_sample_sd.csv',index=False)
    expected=len(datasets)*len(seeds)*(1+len(VARIANTS))
    save_json(root/'summary_status.json',{'complete':len(rows)==expected,'actual_rows':len(rows),'expected_rows':expected,
        'definitions':VARIANTS,'sd_ddof':1,'test_used_for_selection':False,
        'scope':'This helper checks the full model and distance control on fixed partitions. Current Table 1 aggregation is provided by scripts/reproduce_tables.py.'})
    save_json(root/'prediction_provenance.json',provenance)
    return frame

if __name__=='__main__':
    full_preflight(json.loads(Path(sys.argv[1]).read_text()))
