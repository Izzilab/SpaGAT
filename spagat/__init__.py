"""SpaGAT public API; historical aliases preserve checkpoint compatibility."""
from .gene_program_model import SpaGP, SpaGP_Loss
from .dataloader import SPAGAT_dataset
SpaGAT = SpaGP
SpaGATLoss = SpaGP_Loss
__all__ = ['SpaGAT', 'SpaGATLoss', 'SPAGAT_dataset', 'SpaGP', 'SpaGP_Loss']
