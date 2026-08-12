# SpaGAT

SpaGAT is a receiver-conditioned graph attention framework for predicting residual cell-state variation from spatial transcriptomics data.

The model decomposes observed expression into a cell-type baseline and a residual cell-state component, and predicts the residual component using spatial neighborhood information, ligand-associated features, distance-aware edge representations, and receiver-conditioned latent graph routing.

## Repository structure

SpaGAT_submission/
- README.md
- requirements.txt
- spagat/
  - __init__.py
  - preprocess.py
  - dataloader.py
  - embedding.py
  - attention.py
  - gene_program_model.py
- scripts/
  - train.py
  - evaluate.py
  - sender_attribution.py

## Requirements

Install dependencies with:

pip install -r requirements.txt

## Data preparation

SpaGAT expects processed spatial transcriptomics data containing gene expression, cell-type annotations, spatial coordinates, cell-type baseline expression, residual-expression targets, and spatial-neighborhood information.

The processed data directory should contain the metadata files:

- genes.pth
- ligands.pth
- cell_types.pth

By default, each receiver cell is represented together with 49 neighboring cells, resulting in a neighborhood size of 50 cells.

The receiver residual expression is used only as the prediction target and is excluded from the model input. For neighbors sharing the same cell type as the receiver, SpaGAT uses the corresponding cell-type baseline expression. Heterotypic neighbors retain their cell-specific residual variation.

## Training

Run:

python scripts/train.py --data-dir /path/to/processed_data --output-dir /path/to/output

The default configuration uses a neighborhood size of 50, node dimension of 256, edge dimension of 48, two attention heads, one graph layer, four latent routing channels, and a program-token dimension of 32.

The model is optimized with AdamW using a learning rate of 1e-4. The orthogonality and sparsity regularization weights are 0.1 and 0.01, respectively.

The best checkpoint is selected according to the highest validation median gene-wise Pearson correlation coefficient (PCC).

## Evaluation

Run:

python scripts/evaluate.py --data-dir /path/to/processed_data --checkpoint /path/to/best.pth --output-dir /path/to/evaluation_results

The evaluation script reports median gene-wise PCC, mean gene-wise PCC, mean squared error (MSE), and explained variance (EV).

## Sender-origin attribution

SpaGAT includes an inference-time matched sender-masking analysis for estimating model-derived dependence on neighboring cell populations.

For two sender populations, the matched removal count for receiver i is:

m_i = min(n_i,1, n_i,2)

For each eligible receiver, three conditions are compared:

1. random removal of m_i neighbors;
2. removal of m_i neighbors from sender population 1;
3. removal of m_i neighbors from sender population 2.

The receiver itself remains unchanged. Selected sender-to-receiver edges are removed across all latent routing channels during inference.

Example:

python scripts/sender_attribution.py --data-dir /path/to/processed_data --checkpoint /path/to/best.pth --output-dir /path/to/sender_results --receiver-types tumor_1 tumor_2 --sender-types tumor_1 tumor_2 --repeats 10

The primary sender-attribution metrics are mean absolute prediction change and relative increase in prediction MSE.

These quantities measure model-derived predictive dependence and should not be interpreted as evidence of causal cell-cell signaling.

## Preprocessing note

Expression preprocessing may differ across datasets and should follow the processed scale used for each dataset.

For the Mouse MERFISH dataset used in the study, no additional logarithmic transformation was applied before residualization.

## Citation

If you use SpaGAT, please cite the corresponding manuscript.
