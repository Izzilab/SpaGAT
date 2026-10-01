# Revision code update — October 2026

- Replace the root random train/validation workflow with the fixed three-way
  brain protocol and a common launcher for the five methods.
- Include the final distance, uniform-routing and matched-gene controls with
  per-seed values, sample SDs, fixed mappings and supporting analysis records.
- Add supplementary-analysis notebooks, recovered source and resource results.
- Stage all matching-control dependencies and the frozen map with each run;
  `--prepare-only` creates a run without starting model fitting.
- Retain original sender analysis under analysis/source/original_sender/;
  remove old root validation-only evaluation and sender CLI files.
- Track published results and analysis folders; ignore generated weights,
  external matrices and local run outputs.

This replaces the working files of WuBoFu/SpaGAT at inspected commit
38c27adc029715800e735999469cd6c15e98c076. Existing Git history is preserved.
The public component package covers the configurations reported in revised Table 1.

- Publish verified model-ready SEA-AD/Mouse inputs and six original full-model
  SpaGAT checkpoints in separate GitHub Releases. Add checksum-verifying
  downloaders and full held-out inference without retraining. Record all
  six checkpoint runs and their numerical agreement with Figure 2.
