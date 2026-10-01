# Processed benchmark data: sources and use

These files are the SpaGAT author's processed research inputs, not an official
provider release. Processing includes cell-type centering, saved neighborhood
indices and fixed section/donor partitions. CSVs and baseline arrays are retained
without reordering or rewriting. Provider attribution and terms continue to apply;
this deposit does not assign a new license to the underlying datasets.

## SEA-AD

Source: Seattle Alzheimer's Disease Brain Cell Atlas (SEA-AD), Allen Institute
and the SEA-AD consortium, middle temporal gyrus MERFISH data.
Source registry: https://registry.opendata.aws/allen-sea-ad-atlas/
Spatial data: s3://sea-ad-spatial-transcriptomics/
Provider terms: https://alleninstitute.org/legal/terms-of-use
Citation policy: https://alleninstitute.org/legal/citation-policy/

The provider permits research/noncommercial use and distribution subject to its
terms and attribution requirements. This deposit is for research reproducibility.
It does not grant commercial-use permission or claim ownership of provider data.
The exact historical raw release checksum was not recovered; see
DATA_AND_PREPROCESSING.md for the evidence and remaining provenance limits.

## Mouse MOp MERFISH

Data creators: Xiaowei Zhuang and Meng Zhang (2020), *A molecularly defined and
spatially resolved cell atlas of the mouse primary motor cortex*, Brain Image
Library, https://doi.org/10.35077/g.21.
Publication: Zhang et al. (2021), *Spatially resolved cell atlas of the mouse
primary motor cortex by MERFISH*, Nature, https://doi.org/10.1038/s41586-021-03705-x.
Original files: https://download.brainimagelibrary.org/cf/1c/cf1c1a431ef8d021/

Cite the original dataset and publication as well as the SpaGAT method when
using these processed research inputs. The provider's data-use terms apply.

## Scope

Only the revised SEA-AD and Mouse brain benchmark inputs are included. Liver
inputs and model checkpoints are separate assets and are not implied by the
presence of these archives. Checksums identify the original processed file bytes;
matching held-out target hashes verifies correspondence to Figure 2 but does not
reconstruct all upstream raw-data filtering decisions.
