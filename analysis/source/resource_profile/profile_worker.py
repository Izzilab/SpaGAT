"""Isolated, bounded resource profiling of the frozen SpaGAT benchmark.

All writes go to a new local output directory. Existing models are read only.
This script is not used to replace the published prediction/accuracy results.
"""
from pathlib import Path
import argparse
import contextlib
import gc
import hashlib
import importlib.util
from importlib import metadata as package_metadata
import json
import os
import platform
import random
import resource
import sys
import threading
import time
import traceback

import numpy as np
import pandas as pd
import psutil
import torch
from torch.utils.data import DataLoader, Subset

GIB = 1024 ** 3

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))

class Measure:
    """20 ms process-tree RSS samples and PyTorch allocator high-water marks."""
    def __init__(self, gpu=False):
        self.gpu = gpu
        self.proc = psutil.Process()
        self.stop = threading.Event()
        self.peak_rss = 0

    def sample(self):
        try:
            rss = self.proc.memory_info().rss
            for child in self.proc.children(recursive=True):
                with contextlib.suppress(psutil.Error):
                    rss += child.memory_info().rss
            self.peak_rss = max(self.peak_rss, rss)
        except psutil.Error:
            pass

    def poll(self):
        while not self.stop.wait(0.02):
            self.sample()

    def __enter__(self):
        if self.gpu:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.sample()
        self.initial_rss = self.peak_rss
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self.poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        try:
            if self.gpu:
                torch.cuda.synchronize()
        finally:
            self.seconds = time.perf_counter() - self.started
            self.sample()
            self.stop.set()
            self.thread.join()
            self.result = {
                'seconds': self.seconds,
                'peak_process_tree_rss_gib_sampled': self.peak_rss / GIB,
                'initial_process_tree_rss_gib': self.initial_rss / GIB,
                'rss_sampling_interval_seconds': 0.02,
                'peak_cuda_allocated_gib': torch.cuda.max_memory_allocated() / GIB if self.gpu else None,
                'peak_cuda_reserved_gib': torch.cuda.max_memory_reserved() / GIB if self.gpu else None,
            }

def module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def initialize_dataset(config):
    from spagat.dataloader import SPAGAT_dataset
    with Measure() as measure:
        dataset = SPAGAT_dataset(config['processed_local'], num_neighbors=50)
    split = json.loads(Path(config['split_local']).read_text())
    counts = tuple(len(split[key]) for key in ('train_indices', 'val_indices', 'test_indices'))
    if counts != tuple(config['expected_counts']) or len(dataset) != sum(counts):
        raise ValueError(f'Dataset or partition count mismatch: {counts}, dataset={len(dataset)}')
    if len(dataset.genes) != config['n_genes']:
        raise ValueError('Gene count mismatch')
    # Validate every eligible row is assigned once without regenerating the split.
    joined = np.concatenate([np.asarray(split[k], dtype=np.int64) for k in ('train_indices','val_indices','test_indices')])
    if not np.array_equal(np.sort(joined), np.arange(len(dataset))):
        raise ValueError('Split indices are not a partition of the dataset')
    audit = pd.read_csv(config['sample_audit_local'])
    if 'sample' not in audit or list(audit['sample'].astype(str)) != list(map(str,dataset.samples)):
        raise ValueError('Saved sample order differs from the dataset order')
    eligible=np.asarray(dataset.meta_counts,dtype=np.int64)
    ends=np.cumsum(eligible);starts=ends-eligible
    for col,expected in [('eligible_cells',eligible),('global_start',starts),('global_end_exclusive',ends)]:
        if col not in audit or not np.array_equal(audit[col].to_numpy(),expected):
            raise ValueError('Sample audit count/offset mismatch: '+col)
    membership=np.empty(len(dataset),dtype=np.int8)
    for label,key in enumerate(('train_indices','val_indices','test_indices')):membership[split[key]]=label
    labels=('train','validation','test')
    for row,start,end in zip(audit.itertuples(index=False),starts,ends):
        if row.split not in labels or not np.all(membership[start:end]==labels.index(row.split)):
            raise ValueError('Sample partition mismatch: '+row.sample)
    return dataset, split, measure.result

def neural_model(name, dataset, config, device):
    runner = None
    criterion = None
    checkpoint_path = config['models_local'][name]
    if name == 'SpaGAT':
        from spagat.gene_program_model import SpaGP, SpaGP_Loss
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        old = saved.get('config', {})
        if old.get('num_neighbors',50)!=50 or old.get('batch_size',32)!=32:
            raise ValueError('Unexpected SpaGAT neighborhood/batch configuration')
        model = SpaGP(genes=dataset.genes, ligands_info=dataset.interactions,
                      node_dim=256, edge_dim=48, num_heads=2, n_layers=1,
                      att_dim=8, n_programs=4, program_aware=True,
                      program_token_dim=32, message_decoder='free', routing_mode='program')
        model.load_state_dict(saved['model'], strict=True)
        criterion = SpaGP_Loss(dataset.genes,dataset.interactions,lambda_orth=0.1,lambda_sparse=0.01).to(device)
        del saved
    else:
        runner = module_from_path('profile_runner_'+name, Path(config['source_local'])/'runners'/f'{name}.py')
        if name == 'GAT':
            model = runner.LocalGATMouseNoLog(gene_dim=len(dataset.genes),num_celltypes=len(dataset.cell_types_dict))
        elif name == 'SPICE_adapted':
            model = runner.SpiceAdaptedMouseNoLog(gene_dim=len(dataset.genes),num_celltypes=len(dataset.cell_types_dict))
        elif name == 'GITIII_official':
            cls = runner.load_official_model_class()
            model = cls(dataset.genes,dataset.interactions,node_dim=256,edge_dim=48,
                        num_heads=2,n_layers=1,node_dim_small=16,att_dim=8,use_cell_type_embedding=True)
        else:
            raise ValueError(name)
        saved = torch.load(checkpoint_path,map_location='cpu',weights_only=True)
        model.load_state_dict(saved['model_state_dict'],strict=True)
        del saved
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if parameter_count != config['expected_parameters'][name]:
        raise ValueError(f'{name} parameter count mismatch: {parameter_count}')
    model = model.to(device)
    return model, runner, criterion, parameter_count

def forward(name, model, runner, criterion, batch, device, training=False):
    if name == 'SpaGAT':
        values={k:v.to(device) for k,v in batch.items()}
        output=model(values)
        pred=output[0]
        loss=criterion(output,values['y'])[0] if training else None
    elif name == 'GITIII_official':
        values,target=runner.prepare_batch(batch,device)
        pred=model(values)[0]
        loss=torch.nn.functional.mse_loss(pred,target) if training else None
    else:
        x,ct,target=runner.prepare_batch(batch,device)
        pred=model(x,ct)
        loss=torch.nn.functional.mse_loss(pred,target) if training else None
    return pred,loss

def neural_train(name,dataset,split,config,device):
    with Measure(gpu=True) as setup:
        model,runner,criterion,count=neural_model(name,dataset,config,device)
    betas=(0.99,0.999) if name in ('SpaGAT','GITIII_official') else (0.9,0.999)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.0001,betas=betas)
    # Same fixed shuffled receiver sequence across all neural methods.
    rng=np.random.default_rng(config['seed'])
    indices=rng.permutation(np.asarray(split['train_indices']))
    total=(config['warmup_batches']+config['train_batches'])*config['batch_size']
    if len(indices)<total:
        raise ValueError('Insufficient training receivers for the requested profile')
    loader=iter(DataLoader(Subset(dataset,indices[:total].tolist()),batch_size=config['batch_size'],num_workers=0))
    model.train()
    def step(batch):
        optimizer.zero_grad(set_to_none=True)
        _,loss=forward(name,model,runner,criterion,batch,device,training=True)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite training loss')
        loss.backward()
        optimizer.step()
    with Measure(gpu=True) as warmup:
        for _ in range(config['warmup_batches']):
            step(next(loader))
    with Measure(gpu=True) as measure:
        for i in range(config['train_batches']):
            step(next(loader))
            if (i+1)%25==0:
                print(f'{name}: measured training batch {i+1}/{config["train_batches"]}',flush=True)
    return {'parameter_count':count,'warmup_batches':config['warmup_batches'],
            'measured_batches':config['train_batches'],'batch_size':config['batch_size'],
            'model_setup':setup.result,'warmup_profile':warmup.result,'training_profile':measure.result,
            'scope':'Fixed-batch training profile from a saved checkpoint with a fresh optimizer; excludes validation and checkpoint serialization. No trained weights saved.'}

def validate_predictions(pred,config,name):
    path=Path(config['references_original'][name])
    with np.load(path,allow_pickle=False) as z:
        key='prediction' if 'prediction' in z.files else ('predictions' if 'predictions' in z.files else None)
        if key is None:
            raise ValueError('Saved reference prediction key not recognized')
        reference=np.asarray(z[key],dtype=np.float32)
        if reference.shape!=pred.shape:
            raise ValueError('Reference prediction shape mismatch')
        if 'genes' not in z.files or list(z['genes'].astype(str))!=list(config['genes']):
            raise ValueError('Reference gene order mismatch')
        if 'test_indices' not in z.files or not np.array_equal(z['test_indices'],config['test_indices']):
            raise ValueError('Reference test receiver order mismatch')
        error=np.abs(pred-reference)
        valid=bool(np.allclose(pred,reference,rtol=1e-4,atol=1e-4))
        result={'matches_saved_predictions_within_tolerance':valid,'max_abs_error':float(error.max()),
                'mean_abs_error':float(error.mean()),'rtol':1e-4,'atol':1e-4}
    return result

def inference(name,dataset,split,config,device):
    # Start from cached processed data and local checkpoint files. Include model
    # loading, test input preparation, forward passes, and CPU output assembly.
    with Measure(gpu=name!='LightGBM') as measure:
        predictions=np.empty((len(split['test_indices']),len(dataset.genes)),dtype=np.float32)
        if name=='LightGBM':
            import lightgbm as lgb
            runner=module_from_path('profile_lgb',Path(config['source_local'])/'runners/LightGBM.py')
            with Measure() as features:
                x_test,y_test=runner.build_features(dataset,split['test_indices'],dataset.genes,'test',128)
            for g in range(len(dataset.genes)):
                booster=lgb.Booster(model_file=str(Path(config['models_local'][name])/f'gene_{g:04d}.txt'))
                predictions[:,g]=booster.predict(x_test)
                del booster
                if (g+1)%20==0: print(f'LightGBM inference: {g+1}/{len(dataset.genes)} genes',flush=True)
            extra={'test_feature_construction':features.result,'parameter_count':None}
        else:
            model,runner,criterion,count=neural_model(name,dataset,config,device)
            model.eval()
            cursor=0
            loader=DataLoader(Subset(dataset,split['test_indices']),batch_size=config['batch_size'],num_workers=0)
            with torch.no_grad():
                for i,batch in enumerate(loader):
                    pred,_=forward(name,model,runner,criterion,batch,device)
                    n=len(pred)
                    predictions[cursor:cursor+n]=pred.detach().cpu().numpy()
                    cursor+=n
                    if (i+1)%400==0:print(f'{name} inference: {cursor}/{len(predictions)} receivers',flush=True)
            if cursor!=len(predictions):raise ValueError('Incomplete inference')
            extra={'parameter_count':count}
    if not np.isfinite(predictions).all():raise ValueError('Nonfinite predictions')
    comparison_config={**config,'genes':list(map(str,dataset.genes)),'test_indices':split['test_indices']}
    agreement=validate_predictions(predictions,comparison_config,name)
    return {**extra,'inference':measure.result,'prediction_agreement':agreement,
            'n_test_receivers':len(predictions),'n_genes':len(dataset.genes),
            'scope':'Checkpoint-to-prediction wall time; includes local model loading, test input preparation, and CPU prediction collection; excludes processed dataset loading, initial Drive copy, accuracy scoring and reference comparison.'}

def lightgbm_train(dataset,split,config,job_dir):
    import lightgbm as lgb
    runner=module_from_path('profile_lgb_train',Path(config['source_local'])/'runners/LightGBM.py')
    params=dict(runner.PARAMETERS);params['random_state']=config['seed']
    with Measure() as measure:
        with Measure() as feature_measure:
            x_train,y_train=runner.build_features(dataset,split['train_indices'],dataset.genes,'train',128)
            x_val,y_val=runner.build_features(dataset,split['val_indices'],dataset.genes,'validation',128)
        val_pred=np.empty_like(y_val)
        progress=[]
        for g,gene in enumerate(dataset.genes):
            started=time.perf_counter()
            model=lgb.LGBMRegressor(**params)
            model.fit(x_train,y_train[:,g])
            val_pred[:,g]=model.predict(x_val)
            progress.append({'gene_index':g,'gene':str(gene),'fit_and_validation_seconds':time.perf_counter()-started})
            pd.DataFrame(progress).to_csv(job_dir/'per_gene_profile.csv',index=False)
            del model
            if (g+1)%10==0:print(f'LightGBM fitting: {g+1}/{len(dataset.genes)} genes',flush=True)
    return {'training_profile':measure.result,'feature_construction':feature_measure.result,
            'n_genes_fitted':len(progress),'trees_per_gene':params['n_estimators'],
            'cpu_model':platform.processor(),'logical_cpu_count':os.cpu_count(),
            'scope':'All genes, full training and validation partitions, original 100-tree settings; includes feature construction, fitting and validation prediction; excludes checkpoint serialization. Fresh fitted models discarded.'}

def nearest_row(x,y,i,k=50):
    # Same clipped-distance argpartition/sort operations as the supplied
    # process_dataset.py get_index/argsort_topk implementation.
    dx=np.abs(x-x[i]);dy=np.abs(y-y[i])
    dx[dx>1e4]=1e4;dy[dy>1e4]=1e4
    distance=np.sqrt(dx*dx+dy*dy)
    indices=np.argpartition(distance,k)[:k]
    return indices[np.argsort(distance[indices])]

def distance_equivalent(x,y,i,a,b):
    def distances(indices):
        dx=np.minimum(np.abs(x[indices]-x[i]),1e4)
        dy=np.minimum(np.abs(y[indices]-y[i]),1e4)
        return np.sort(np.sqrt(dx*dx+dy*dy))
    return bool(np.allclose(distances(a),distances(b),rtol=1e-7,atol=1e-7))

def graph_profile(config,job_dir):
    folder=Path(config['processed_local']);reports=[]
    with Measure() as measure:
        for metadata in sorted(folder.glob('*_TypeExp.npz')):
            name=metadata.name[:-len('_TypeExp.npz')]
            cols=['centerx','centery']+[f'index_{j}' for j in range(50)]
            frame=pd.read_csv(folder/(name+'.csv'),usecols=cols)
            x=frame.centerx.to_numpy(dtype=np.float64);y=frame.centery.to_numpy(dtype=np.float64)
            saved=frame[[f'index_{j}' for j in range(50)]].to_numpy(dtype=np.int64)
            n=len(x)
            if n<=50 or saved.min()<0 or saved.max()>=n or not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError('Invalid section geometry/index table: '+name)
            checks=np.unique(np.linspace(0,n-1,min(n,32),dtype=int))
            compatible=all(distance_equivalent(x,y,int(i),nearest_row(x,y,int(i)),saved[i]) for i in checks)
            if not compatible:
                reports.append({'section':name,'n_cells':n,'status':'incompatible_neighbor_rule','search_seconds':None})
                print(f'{name}: neighbor rule differs; no timing extrapolated',flush=True)
                continue
            rebuilt=np.empty_like(saved)
            started=time.perf_counter()
            for i in range(n):rebuilt[i]=nearest_row(x,y,i)
            seconds=time.perf_counter()-started
            mismatch=np.flatnonzero(np.any(rebuilt!=saved,axis=1))
            unresolved=sum(not distance_equivalent(x,y,int(i),rebuilt[i],saved[i]) for i in mismatch)
            reports.append({'section':name,'n_cells':n,'status':'verified' if not unresolved else 'unresolved_difference',
                            'search_seconds':seconds,'index_order_mismatch_rows':len(mismatch),'unresolved_rows':unresolved})
            pd.DataFrame(reports).to_csv(job_dir/'graph_sections.csv',index=False)
            print(f'Neighborhood reconstruction {len(reports)}: {name}, {n} cells, {seconds:.1f}s',flush=True)
            del frame,x,y,saved,rebuilt
    complete=bool(reports) and all(r['status']=='verified' for r in reports)
    return {'shared_neighbor_construction':measure.result,'all_sections_verified':complete,
            'sections':reports,'total_neighbor_search_seconds':sum(r['search_seconds'] or 0 for r in reports) if complete else None,
            'scope':'Shared 50-index neighborhood reconstruction from saved float64 coordinates, including all cells per section. Search timer excludes CSV loading, index allocation, eligibility filtering, expression preprocessing and verification. Whole-stage RSS includes CSV loading and verification. Existing input graph files are not changed. Model-specific edge tensor creation remains part of model runtime.'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--model',required=True)
    parser.add_argument('--phase',required=True,choices=['graph','train','inference']);parser.add_argument('--output',required=True)
    args=parser.parse_args();config=json.loads(Path(args.config).read_text());job_dir=Path(args.output);job_dir.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,config['source_local'])
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    gpu=args.model not in ('LightGBM','shared_graph')
    if gpu and not torch.cuda.is_available():raise RuntimeError('Select a Colab GPU runtime for neural profiling.')
    if gpu:
        torch.cuda.manual_seed_all(config['seed']);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device=torch.device('cuda' if gpu else 'cpu')
    result={'dataset':config['dataset'],'model':args.model,'phase':args.phase,'seed':config['seed'],'status':'started',
            'versions':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__},
            'hardware':{'gpu':torch.cuda.get_device_name() if gpu else None,'logical_cpus':os.cpu_count(),'torch_cpu_threads':torch.get_num_threads(),
                        'cpu_affinity':psutil.Process().cpu_affinity(),
                        'host_total_ram_gib':psutil.virtual_memory().total/GIB},
            'original_files_modified':False,'source_hashes':config['source_hashes']}
    for package in ['torch-geometric','lightgbm','psutil']:
        with contextlib.suppress(package_metadata.PackageNotFoundError):result['versions'][package]=package_metadata.version(package)
    with contextlib.suppress(OSError):
        result['hardware']['cpu_model']=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),platform.processor())
    started=time.perf_counter()
    try:
        if args.phase=='graph':result.update(graph_profile(config,job_dir))
        else:
            dataset,split,loading=initialize_dataset(config);result['dataset_loading']=loading
            if args.phase=='inference':result.update(inference(args.model,dataset,split,config,device))
            elif args.model=='LightGBM':result.update(lightgbm_train(dataset,split,config,job_dir))
            else:result.update(neural_train(args.model,dataset,split,config,device))
        result['status']='complete'
    except Exception as exc:
        result.update(status='error',error=str(exc),traceback=traceback.format_exc());traceback.print_exc()
    finally:
        result['job_wall_seconds']=time.perf_counter()-started
        result['whole_process_peak_rss_gib_including_postcheck']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/GIB
        write_json(job_dir/'profile.json',result)
    if result['status']!='complete':raise SystemExit(1)
    print('Completed:',args.model,args.phase,flush=True)

if __name__=='__main__':main()
