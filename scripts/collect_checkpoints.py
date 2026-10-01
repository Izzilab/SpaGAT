"""Optionally copy six manifest-listed checkpoints from already-mounted author Drive."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]

def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=False)
    records = json.loads((ROOT/'assets/checkpoint_manifest.json').read_text())
    report = []
    for row in records:
        src = Path(row['author_drive_path'])
        if not src.is_file():
            report.append({**row,'collection_status':'missing'})
            continue
        if file_sha(src) != row['sha256']:
            report.append({**row,'collection_status':'hash_mismatch_not_copied'})
            continue
        target = a.output_dir/f"{row['dataset']}_SpaGAT_seed{row['seed']}.pth"
        shutil.copyfile(src,target)
        if file_sha(target) != row['sha256']:
            raise RuntimeError(f'Copied file integrity mismatch: {target}')
        report.append({**row,'collection_status':'copied_and_verified','local_copy':str(target)})
    (a.output_dir/'collection_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'copied':sum(r['collection_status']=='copied_and_verified' for r in report),
                      'missing_or_mismatched':sum(r['collection_status']!='copied_and_verified' for r in report),
                      'report':str(a.output_dir/'collection_report.json'),'uploaded':False},indent=2))

if __name__ == '__main__':
    main()
