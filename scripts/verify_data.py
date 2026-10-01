"""Verify the actual loader, fixed split and every test target without training."""
import argparse,json
from pathlib import Path
from data_assets import validate_inputs

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,choices=['Mouse','SEA_AD'])
    p.add_argument('--data-dir',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    _,_,_,report=validate_inputs(a.data_dir,a.dataset)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(report,f,indent=2)
    print('Verified sample order, partitions, neighbor indices and all test targets:',a.dataset)
