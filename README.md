# SpaGAT
**SpaGAT: Multi-Head Graph Attention for Spatial Cell-Cell Communication Inference**

SpaGAT is a deep learning framework for inferring cell-cell communication (CCC) from spatial transcriptomics data at single-cell resolution. It extends graph attention architectures with a multi-head prediction layer that captures diverse spatial signaling patterns.

## Features

- **Multi-head prediction**: K independent attention heads capture distinct spatial communication modes
- **Influence tensor extraction**: Fully decomposable CCC signals for downstream interpretation
- **Cross-platform support**: Validated on MERFISH and CosMx spatial transcriptomics data
- **Downstream analysis suite**: Distance decay, CCI network inference, CCC-informed subclustering, and information flow visualization
 
## Installation
 
```bash
git clone https://github.com/WuBoFu/SpaGAT.git
cd SpaGAT
pip install -r requirements.txt
```
 
### Requirements
 
- Python >= 3.9
- PyTorch >= 2.0
- CUDA-compatible GPU (recommended)

## Quick Start
 
### Input Data Format
 
Prepare a CSV file with the following columns:
- Gene expression columns (one per gene)
- `centerx`: x spatial coordinate
- `centery`: y spatial coordinate
- `section`: sample/section identifier
- `subclass`: cell type annotation
 
### Running SpaGAT
 
```bash
python run_spagat.py \
    --data_csv your_data.csv \
    --genes GeneA GeneB GeneC \
    --species human \
    --sample YourSampleName \
    --epochs 50 \
    --output_dir output
```

### Key Parameters
 
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_csv` | required | Path to input CSV file |
| `--genes` | required | List of gene names (columns in CSV) |
| `--species` | required | `human` or `mouse` |
| `--sample` | required | Sample name for downstream analysis |
| `--epochs` | 50 | Number of training epochs |
| `--num_neighbors` | 50 | Number of spatial neighbors per cell |
| `--batch_size` | 256 | Training batch size |
| `--lr` | 1e-4 | Learning rate |
| `--device` | cuda | `cuda` or `cpu` |
| `--use_log_normalize` | False | Log-normalize expression values |

### Pipeline Steps
 
The pipeline runs four steps automatically:
 
1. **Preprocessing**: Constructs spatial neighbor graph and extracts features
2. **Training**: Trains multi-head graph attention model
3. **Influence tensor**: Computes per-cell-pair, per-gene communication scores
4. **Downstream analysis**: Generates visualizations and network statistics
 
## Output
 
Results are saved to the specified `--output_dir`:
 
- Distance decay plots
- Gene expression prediction maps
- CCI UMAP embeddings
- CCC network z-score heatmaps
- Cell subtype analysis (UMAP, spatial clusters, marker genes, sender profiles)

## Acknowledgments
 
SpaGAT builds upon the [GITIII](https://github.com/lugia-xiao/GITIII) framework. We thank the authors for making their code publicly available.
 
## License
 
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
