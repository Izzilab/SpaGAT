# Data provenance and preprocessing

## SEA-AD MERFISH
Source: Seattle Alzheimer's Disease Brain Cell Atlas, middle temporal gyrus, spatial transcriptomics (not the snRNA-seq bucket).
- Provider registry: https://registry.opendata.aws/allen-sea-ad-atlas/
- Spatial bucket: s3://sea-ad-spatial-transcriptomics/
- Coordinate comparison used the public object middle-temporal-gyrus/all_donors-h5ad/SEAAD_MTG_MERFISH.2024-12-11.h5ad. This establishes coordinate correspondence; it is not proof that this exact release supplied every historical expression value/filter.
- Model input AD.csv: 366,272 cells, 69 sections, 27 donors, 140 genes. Receiver-eligible: 365,940 cells. Partitions: 257,152 train / 54,026 validation / 54,762 test receivers; 18/5/4 donors and 47/13/9 sections.
- Input expression was transformed with natural log ln(1+x), not log2. Local baseline comparison and reconstruction of test MSE0=0.345257702239 support this scale. The earlier scale audit did not recover a complete historical conversion script. The subsequently recovered original processed inputs reproduce the full Figure 2 test-target hash exactly (see assets/processed_data_manifest.json).
- Baselines come from training sections and are applied unchanged to validation/test. The numerical reconstruction agrees when all input cells in training sections contribute to the baseline, including cells not eligible as prediction receivers. No validation/test cells are used.
- All 366,272 local coordinates were matched to provider tiled coordinates by section-specific translations; raw coordinates additionally involve a y-axis reflection. Neither operation changes within-section Euclidean distance; no scale multiplier was required. Units are micrometers.

## Mouse MOp MERFISH
Source: Zhang et al. (2021), Spatially resolved cell atlas of the mouse primary motor cortex by MERFISH, Nature, https://doi.org/10.1038/s41586-021-03705-x.
- Data DOI: https://doi.org/10.35077/g.21 (provider dataset citation dated 2020).
- Provider files: https://download.brainimagelibrary.org/cf/1c/cf1c1a431ef8d021/
- Model input mouse.csv: 276,385 cells in 64 sections, two animals, 254 genes. Receiver-eligible: 276,336 cells. Fixed partitions: 122,886 train / 36,193 validation / 117,257 test receivers. Mouse 1 supplies 26 train and 7 validation sections; Mouse 2 supplies 31 test sections.
- No additional logarithmic transform was applied to the input matrix. This does not assert that the provider's matrix was unprocessed raw transcript counts.
- Baselines were estimated from the training partition. Coordinates are micrometers. Original boundary-to-model correspondence was checked for a source cell, not by reprocessing the entire raw dataset.
- A complete original raw-to-mouse.csv conversion and all upstream filtering decisions were not recovered.

## Liver CosMx
Provider documentation: https://www.brukerspatialbiology.com/wp-content/uploads/2023/01/LiverPublicDataRelease.html
- Historical source object: LiverDataReleaseSeurat_newUMAP.RDS. The exact original download date/release checksum was not recovered.
- Source conversion evidence excludes cellType == "NotDet", uses Run_Tissue_name as section and cellType as annotation, and converts x_slide_mm and y_slide_mm to micrometers by multiplying by 1000. The complete expression-export cell was not recovered, so no replacement raw-RDS converter is claimed.
- Input liver_cosmx.csv has 793,309 cells: 460,436 carcinoma and 332,873 normal. Receiver-eligible counts are 460,429 and 332,828. The gene panel contains 1,000 genes.
- The input expression matrix was transformed using ln(1+x). Historical code printed "log2" while executing numpy.log(x+1); the executable operation and saved numerical evidence determine the scale.
- Baselines were estimated over pooled input specimens before the cell-level 80/20 training/validation split. These are within-specimen analyses, not independent-patient test results. Different downstream analyses used different receiver cohorts; their counts must be taken from the individual result files.

## Common model-ready input contract
Each section CSV contains ordered gene columns, centerx/centery, subclass, flag, and index_0 ... index_49. It stores expression residuals, not total expression; index_0 is the receiver. Each section's _TypeExp.npz stores the corresponding baseline vectors. genes.pth and ligands.pth sit one directory above processed/. Gene order, cell order, baselines and neighbor indices are part of the experiment and must not be reconstructed by an unrecorded sort.

The recorded spatial eligibility check is a labeled receiver with nearest-other-cell distance <80 micrometers. Noneligible cells can remain in the neighborhood context. This model-input-to-receiver filtering history is provided in data_records/counts_by_section.csv and counts_by_sample.csv; it is not an assertion that no earlier provider filtering occurred.

Neighborhoods stay within sections and partitions. The focal receiver residual and all homotypic-neighbor residuals are zeroed in inputs; heterotypic residuals remain. k includes the receiver. Distance features are [1/(d+1), 1/(sqrt(d)+1), 1/(1+d^2), exp(-d), exp(-d^2)] without an additional coordinate rescaling. Original graph indices should be retained, including their tie order.

This bundle intentionally does not provide a speculative raw-data converter that could silently change targets. Use recorded model-ready inputs for exact-run reconstruction from the verified brain input release; this does not imply that all earlier raw-data filtering scripts have been recovered.

## Download the verified brain inputs

Run `python scripts/download_data.py --dataset all --output-dir EXTERNAL_DATA`.
The public release retains the original processed CSV/NPZ/PTH bytes and preprocessing
audit records. The supplied splits use that exact cell/gene/sample order.
SHA-256 checks for every archive and member are in `assets/processed_data_manifest.json`.
Both full test-target matrices match the archived Figure 2 hashes.
Use `scripts/verify_data.py` to repeat the loader/target check.
Liver inputs and trained checkpoints are not included in this data release.
See DATA_LICENSES.md for provider terms and attribution.
