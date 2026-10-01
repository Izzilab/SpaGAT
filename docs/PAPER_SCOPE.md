# Publication scope

This is the code and recorded-result subset supporting the revised manuscript
and supplement, not an archive of every exploratory experiment. Original author
records are retained outside this publication package. Numerical records for the
reported analyses are unchanged.

| Manuscript location | Included files |
|---|---|
| Figure 2; Tables S2–S5; Supplementary Data S1 | `baselines/`, `scripts/`, `splits/`, `results/brain_benchmarks/`, Figure 2 plot/data |
| Table 1; Section S7 | Full model plus distance removal, uniform routing, and matched random edge-gene replacement; `results/table1/`, frozen maps |
| Figure 3; Section S9; Figure S2 | Three original cohorts, all-gene CSVs, cohort checks and distribution workflows |
| Figure 4; Section S6; Table S9 | Liver cell-type MSE/EV records and their audit |
| Figure 5 | Original matched-count sender routine and compatible model source, repeat-level numeric records |
| Section S4; Tables S6–S7 | Liver gene/program recovery and signed masking source, coverage and full-panel records |
| Section S5; Table S8 | Bounded measured resource-profile source and measurements |
| Section S1; Table S1 | Model-input cohort counts and recorded fixed partitions |
| Section S8; Table S11; Figure S1 | Distance-matched and fixed-attention sender controls |
| Section S10; Table S12 | k=25/50/75 results and the source for that analysis |

Per-gene, per-seed and per-repeat records support the displayed summaries even
where every value is not printed in a paper table. Integrity checks, source
provenance, licenses, dependencies and checkpoint-compatible internal names are
necessary supporting material. Original within-specimen liver source is retained
because those analyses remain in the manuscript. This does not make the liver
analysis an independent-patient benchmark.

Current benchmark entry points use the fixed train/validation/test protocol.
Figure 3/4/5 cohorts retain their separately documented original protocols.
Required external inputs and unrecovered workflows are listed in KNOWN_GAPS.md.
