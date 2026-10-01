# External assets

Large inputs and weights are distributed through GitHub Releases, separately
from this source checkout:

- `processed_data_manifest.json`: verified SEA-AD and Mouse model-ready inputs.
- `checkpoint_download_manifest.json`: two archives containing six original
  SpaGAT full-model checkpoints, with archive and individual-file hashes.
- `checkpoint_manifest.json`: original author paths and checkpoint provenance.
  Each `public_url` points to an archive; `archive_member` identifies the weight
  inside it. `included` refers to bytes in the source checkout, not the release.

Use `scripts/download_data.py` and `scripts/download_checkpoints.py` as described
in README.md. Both downloaders verify checksums and refuse to overwrite different
existing files. `scripts/evaluate_checkpoint.py` evaluates the full held-out
test sets without training; its validation report is in `environment/`.

Baseline, ablation and liver checkpoints are not part of this release. Public
raw-data entry points are listed in `docs/DATA_AND_PREPROCESSING.md`; they do
not replace the model-ready inputs required by the saved partition indices.

The optional author utility `scripts/collect_checkpoints.py` copies only the six
manifest-listed files from already mounted Drive and checks their hashes. It
does not train, download or upload anything.
