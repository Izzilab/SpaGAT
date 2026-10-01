"""Recompute numeric summaries from included records. No model inference."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def aggregate(frame, keys, metrics):
    g = frame.groupby(keys, sort=True)
    assert g.seed.apply(lambda x: sorted(x.tolist()) == [123,456,789]).all()
    out = g[metrics].agg(['mean','std'])
    out.columns = ['_'.join(c) for c in out.columns]
    return out.reset_index()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    a = p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    metrics = ['median_gene_pcc','mse','ev_zero_percent']
    brain = pd.read_csv(ROOT/'results/brain_benchmarks/per_seed_metrics.csv')
    abl = pd.read_csv(ROOT/'results/table1/per_seed_test_metrics.csv')
    assert len(brain)==30 and len(abl)==24
    assert np.allclose(brain.ev_zero_percent, 100*(1-brain.mse/brain.mse_zero),atol=1e-10)
    assert np.allclose(abl.ev_zero_percent, 100*(1-abl.mse/abl.mse0),atol=1e-10)
    b = aggregate(brain,['dataset','model'],metrics)
    c = aggregate(abl,['dataset','variant'],metrics)
    fig = pd.read_csv(ROOT/'figures/Figure2_source_values.csv')
    for row in fig.itertuples(index=False):
        selected = b[(b.dataset==row.dataset)&(b.model==row.method)].iloc[0]
        assert np.isclose(selected[row.metric+'_mean'],row.mean,rtol=1e-10,atol=1e-12)
        assert np.isclose(selected[row.metric+'_std'],row.sample_SD,rtol=1e-10,atol=1e-12)
    k = pd.read_csv(ROOT/'results/neighborhood_size/neighborhood_test_metrics.csv')
    assert len(k)==6 and k.seed.eq(123).all()
    assert k.groupby('dataset').k_including_receiver.apply(lambda x: sorted(x)==[25,50,75]).all()
    records = pd.concat([pd.read_csv(p) for p in sorted((ROOT/'results/liver/sender_masking').glob('*_repeats.csv'))])
    reference = pd.read_csv(ROOT/'results/liver/sender_masking/Figure5_mean_SD.csv')
    rebuilt = []
    for r in reference.to_dict('records'):
        group = records[(records.receiver==r['receiver'])&(records.condition==r['condition'])]
        assert sorted(group['repeat'].tolist())==list(range(10))
        value = group[r['metric']]
        assert np.isclose(value.mean(),r['mean'],rtol=1e-11,atol=1e-12)
        assert np.isclose(value.std(ddof=1),r['sample_SD'],rtol=1e-11,atol=1e-12)
        rebuilt.append({**r,'mean':float(value.mean()),'sample_SD':float(value.std(ddof=1))})
    a.output_dir.mkdir(parents=True)
    b.to_csv(a.output_dir/'brain_mean_sample_SD.csv',index=False)
    c.to_csv(a.output_dir/'Table1_mean_sample_SD.csv',index=False)
    k.to_csv(a.output_dir/'neighborhood_size.csv',index=False)
    pd.DataFrame(rebuilt).to_csv(a.output_dir/'Figure5_mean_sample_SD.csv',index=False)
    report = {'brain_runs':30,'ablation_rows':24,'neighborhood_rows':6,
              'Figure5_numeric_rows':len(rebuilt),'agreement_with_saved_Figure2_and_Figure5':'passed',
              'training_or_inference':False,'donor_bootstrap_recomputed':False}
    (a.output_dir/'checks.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
