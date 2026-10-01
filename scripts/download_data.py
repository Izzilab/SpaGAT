"""Download and verify the public model-ready brain inputs (no training)."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['Mouse', 'SEA_AD', 'all'], default='all')
    p.add_argument('--output-dir', type=Path, default=Path('EXTERNAL_DATA'))
    p.add_argument('--cache-dir', type=Path, default=Path('asset_downloads'))
    a = p.parse_args()
    m = json.loads((ROOT/'assets/processed_data_manifest.json').read_text())
    wanted = ['Mouse', 'SEA_AD'] if a.dataset == 'all' else [a.dataset]
    download_manifest(m, wanted, a.output_dir, a.cache_dir)


def download_manifest(m, wanted, output_dir, cache_dir):
    """Verify release archives and members without overwriting different files."""
    expected = {r['path']: r for r in m['files'] if r['group'] in wanted}
    archives = [r for r in m['archives'] if any(name in expected for name in r['members'])]
    if not archives or any(not r['public_url'].startswith('https://github.com/WuBoFu/SpaGAT/releases/download/') for r in archives):
        raise RuntimeError('This checkout has no published download URLs; update to the published release first.')
    cache_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for asset in archives:
        path = cache_dir/asset['file']
        if not (path.is_file() and sha(path) == asset['sha256']):
            if path.exists():
                raise ValueError(f'Existing archive checksum mismatch: {path}; retain it for inspection and use a new cache directory.')
            partial = path.with_suffix(path.suffix+'.partial')
            print('Downloading', asset['file'], flush=True)
            with urllib.request.urlopen(asset['public_url'], timeout=120) as response, partial.open('wb') as out:
                shutil.copyfileobj(response, out, length=8*1024*1024)
            if partial.stat().st_size != asset['bytes'] or sha(partial) != asset['sha256']:
                raise ValueError('Download integrity check failed; incomplete file retained as .partial')
            partial.rename(path)
        with zipfile.ZipFile(path) as archive:
            if sorted(archive.namelist()) != sorted(asset['members']):
                raise ValueError('Unexpected archive member list')
            for member in archive.infolist():
                rel = PurePosixPath(member.filename)
                if rel.is_absolute() or '..' in rel.parts or member.filename not in expected:
                    raise ValueError('Unsafe or unexpected ZIP path')
                record = expected[member.filename]
                target = (root/member.filename).resolve()
                if not target.is_relative_to(root) or member.file_size != record['bytes']:
                    raise ValueError('Archive path/size mismatch')
                if target.exists():
                    if target.is_file() and sha(target) == record['sha256']:
                        continue
                    raise FileExistsError(f'Refusing to overwrite different existing input: {target}')
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open('xb') as dest:
                    shutil.copyfileobj(source, dest, length=8*1024*1024)
                if sha(target) != record['sha256']:
                    raise ValueError(f'Extracted file hash mismatch: {target}')
        print('Verified', asset['file'], flush=True)
    for rel, record in expected.items():
        if sha(root/rel) != record['sha256']:
            raise ValueError(f'Missing/corrupt input: {rel}')
    print('All requested inputs verified:', root, flush=True)


if __name__ == '__main__':
    main()
