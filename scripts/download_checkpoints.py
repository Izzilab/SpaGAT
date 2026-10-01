"""Download and verify the six public full-model SpaGAT brain checkpoints."""
import argparse
import json
from pathlib import Path

from download_data import ROOT, download_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['Mouse', 'SEA_AD', 'all'], default='all')
    parser.add_argument('--output-dir', type=Path, default=Path('EXTERNAL_DATA'))
    parser.add_argument('--cache-dir', type=Path, default=Path('asset_downloads'))
    args = parser.parse_args()
    manifest = json.loads((ROOT/'assets/checkpoint_download_manifest.json').read_text())
    wanted = ['Mouse', 'SEA_AD'] if args.dataset == 'all' else [args.dataset]
    download_manifest(manifest, wanted, args.output_dir, args.cache_dir)


if __name__ == '__main__':
    main()
