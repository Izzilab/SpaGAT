# External assets
The six full-model checkpoint paths and hashes in checkpoint_manifest.json are provenance records, not public download URLs. No checkpoint or model-ready expression/neighborhood matrix is included in this bundle. No inference from a checkpoint was performed during assembly.

Training requires an existing dataset folder containing genes.pth, ligands.pth, and processed/<section>.csv plus <section>_TypeExp.npz, in the exact stored order and scale. Public raw-data entry points are listed in docs/DATA_AND_PREPROCESSING.md; they are not substitutes for a verified model-ready download.

The optional author utility scripts/collect_checkpoints.py copies only the six manifest-listed files from already mounted Drive and checks their hashes. It neither downloads nor uploads anything and never trains a model. An author may elect to deposit these files separately and then fill public_url. Missing files stay explicitly missing.
