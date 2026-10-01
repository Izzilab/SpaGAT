# Supplementary analysis workflows

These notebooks and source snapshots support the analyses reported in the
manuscript and supplement. Execution outputs are cleared. The neighborhood-size
and gene-distribution copies are publication-scoped derivatives; other code cells
retain their saved author workflows. The gene-distribution entry points cover
only the three Figure 3 cohorts.
Most workflows expect the author's original Colab/Drive layout, prediction
files, checkpoints or model-ready data. Review their paths and checks before
execution. They must not be described as runnable from the small repository
alone. No full-data rerun was performed when assembling this release.

| Analysis | Notebook | Included results |
|---|---|---|
| k = 25, 50, 75; seed 123 | `notebooks/neighborhood_size.ipynb` | `results/neighborhood_size/` |
| Liver gene recovery and signed masking | `notebooks/liver_gene_signed.ipynb` | `results/liver/signed_gene/` |
| Liver program recovery and signed changes | `notebooks/liver_programs.ipynb` | `results/liver/programs/` |
| Distance matching and fixed-attention message controls | `notebooks/liver_distance_message_controls.ipynb` | `results/liver/distance_message_controls/` |
| Measured resource profile | `notebooks/resource_profile.ipynb` | `results/resource_profile/` |
| Model-input-to-receiver cohort flow | `notebooks/cohort_flow.ipynb` | `data_records/` |
| Full-panel gene PCC distributions | `notebooks/gene_distributions.ipynb`, `notebooks/normal_liver_gene_distributions.ipynb` | `results/figure3_cohorts/` |
| Normal-liver cell-type EV audit | `notebooks/cell_type_EV_audit.ipynb` | `results/liver/cell_types/` |

Paths in the results column are relative to the repository root. Historical
author paths are provenance, not public download locations. The program
notebook records the gene-set library release and checksum. The cohort-flow
notebook starts from model-input tables, not every provider filtering stage.

`source/neighborhood_size/` and `source/liver_distance_controls/` derive from the completed returned runs. The neighborhood-size copy removes
unused component-test switches and retains the full-model k-size computation;
its notebook has a new workflow fingerprint. Other source folders expose recovered
analysis functions for inspection; the associated notebooks supply workflow
configuration. `source/original_sender/` retains the original within-specimen
sender routine with compatible model source, independently of the revised
brain training entry point.

The k-size notebook records the experiment definition, including its original
resume behavior. The completed SEA-AD k=75 export was recovered separately
after an output-directory collision; included final metrics come from the
verified recovery record. Changing paths does not waive source, target, split
or checkpoint checks. Saved donor-bootstrap intervals are supplied as output;
their original complete execution workflow is not included here.
