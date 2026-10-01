# Materials not included or not revalidated

1. This local package has not itself been pushed to GitHub or published as a release.
2. Actual checkpoint files and full expression/neighborhood model inputs are absent. Six full-model checkpoint paths/hashes are records, not downloads.
3. Complete raw-to-model-input conversion and exact source versions were not recovered for every dataset. Fixed indices require the original input ordering, and upstream provider filtering remains incomplete.
4. Mouse LightGBM uses a documented recovered adapter. A fresh full-data rerun of that adapter was not performed.
5. Figures 1/3/4 lack complete recovered original plotting/inference workflows. Figure 5 numeric repeats are included, but its original composite artwork is not rebuilt. Original donor-bootstrap execution code is not included.
6. Supplementary notebooks and additional source are now included for k sensitivity, liver gene/program/signed masking, distance/message controls, resources, cohort flow and gene/cell-type audits. They require external inputs and often retain the original Colab configuration. Inclusion is not a claim of portable end-to-end execution.
7. The original 50% proportional-masking execution workflow is not recovered here. Its manuscript results must not be represented as a newly reproduced analysis from this package.
8. The final one-set random-gene control and uniform-routing control are complete. Current Table 1 has 24 rows. three training seeds for one set do not estimate variability across random gene sets.
9. The recorded environments are not tested installation lockfiles. No fresh full-data GPU training, raw-data reconstruction or checkpoint inference was performed during assembly. See validation_report.json for the actual local checks.
