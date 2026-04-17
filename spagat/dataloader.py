import torch
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd

class SPAGAT_dataset(Dataset):
    def __init__(self, processed_dir, num_neighbors=50, col_centerx="centerx",
                 col_centery="centery", col_cell_type="subclass"):
        if processed_dir[-1] != "/":
            processed_dir = processed_dir + "/"

        samples = []
        for filei in os.listdir(processed_dir):
            if filei.find("_TypeExp.npz") >= 0:
                samples.append(filei.split("_TypeExp.npz")[0])
        self.samples = list(sorted(list(set(samples))))
        print("Have samples:", len(self.samples), self.samples)

        self.index_index = ["index_" + str(i) for i in range(num_neighbors)]
        self.interactions = torch.load("/".join(processed_dir.split("/")[:-2]) + "/ligands.pth", weights_only=False)
        self.genes = torch.load("/".join(processed_dir.split("/")[:-2]) + "/genes.pth", weights_only=False)

        # Collect all cell types from all data files
        all_cell_types = set()
        for samplei in self.samples:
            try:
                dfi = pd.read_csv(processed_dir + samplei + ".csv")
                all_cell_types.update(dfi[col_cell_type].unique())
            except Exception as e:
                print(f"Warning: Cannot read sample {samplei}: {e}")
                continue
        # Build cell type dictionary, sorted for consistency
        self.cell_types_dict = {ct: i for i, ct in enumerate(sorted(all_cell_types))}
        print(f"Built cell type dictionary with {len(self.cell_types_dict)} cell types: {sorted(all_cell_types)}")

        self.indexes = []
        self.flags = []
        self.centerx = []
        self.centery = []
        self.exps = []
        self.cell_types = []
        self.type_exp = []
        for samplei in self.samples:
            try:
                dfi = pd.read_csv(processed_dir + samplei + ".csv")
                # Assign cell type numbers for each cell in the sample
                dfi["cell_type_number"] = np.array([self.cell_types_dict.get(cell_type, -1)
                                                     for cell_type in dfi[col_cell_type]])
                if "flag" not in dfi.columns.tolist():
                    dfi["flag"] = [True for _ in range(dfi.shape[0])]

                type_exp_dicti = np.load(processed_dir + samplei + "_TypeExp.npz", allow_pickle=True)
                type_exp_dicti = {k: torch.Tensor(np.array(v).astype(np.float64)) for k, v in type_exp_dicti.items()}
                # Only keep cell types present in both the dict and the global cell type dict
                valid_cell_types = [ct for ct in dfi[col_cell_type].unique()
                                   if ct in type_exp_dicti and ct in self.cell_types_dict]
                type_exp = torch.stack([type_exp_dicti[cell_type] for cell_type in dfi[col_cell_type]
                                       if cell_type in valid_cell_types], dim=0)
                # Skip sample if no valid cell types
                if len(valid_cell_types) == 0:
                    print(f"Skip sample {samplei}: no valid cell types")
                    continue
                # Only keep rows with valid cell types
                valid_mask = dfi[col_cell_type].isin(valid_cell_types)
                dfi_valid = dfi[valid_mask].copy()
                if len(dfi_valid) == 0:
                    print(f"Skip sample {samplei}: no valid rows after filtering")
                    continue
                self.indexes.append(torch.LongTensor(dfi_valid.loc[:, self.index_index].values))
                self.flags.append(dfi_valid.loc[:, "flag"].values)
                self.centerx.append(torch.Tensor(dfi_valid.loc[:, col_centerx].values))
                self.centery.append(torch.Tensor(dfi_valid.loc[:, col_centery].values))
                self.exps.append(torch.Tensor(dfi_valid.loc[:, self.genes].values))
                self.cell_types.append(torch.LongTensor(dfi_valid.loc[:, "cell_type_number"].values))
                # Rebuild type_exp for valid cell types
                type_exp_valid = torch.stack([type_exp_dicti[cell_type] for cell_type in dfi_valid[col_cell_type]], dim=0)
                self.type_exp.append(type_exp_valid)
            except Exception as e:
                print(f"Error: Cannot process sample {samplei}: {e}")
                import traceback
                traceback.print_exc()
                continue

        self.meta_counts = []
        self.arg_meta = []
        for i in range(len(self.samples)):
            if i < len(self.flags):
                self.meta_counts.append(np.sum(self.flags[i]))
                self.arg_meta.append(torch.LongTensor(np.where(self.flags[i] != 0)[0]))
        self.meta_counts = np.array(self.meta_counts) if self.meta_counts else np.array([])
        self.cumsum_meta = np.cumsum(self.meta_counts) if len(self.meta_counts) > 0 else np.array([])
        print("There are totally", self.cumsum_meta[-1] if len(self.cumsum_meta) > 0 else 0, "cells in this dataset")

    def __len__(self):
        return self.cumsum_meta[-1] if len(self.cumsum_meta) > 0 else 0

    def __getitem__(self, idx):
        if len(self.cumsum_meta) == 0:
            raise RuntimeError("No valid data in dataset")
        # Find which sample this idx comes from
        sample_id = np.searchsorted(self.cumsum_meta, idx, side='right')
        idx = idx - self.cumsum_meta[sample_id]
        idx = self.arg_meta[sample_id][idx]
        indices = self.indexes[sample_id][idx]
        centerx = self.centerx[sample_id][indices]
        centery = self.centery[sample_id][indices]
        exp = self.exps[sample_id][indices]
        type_exp = self.type_exp[sample_id][indices]
        y = exp[0]
        cell_types = torch.LongTensor(self.cell_types[sample_id][indices])
        num_real_neighbors = len(indices)
        neighbor_mask = torch.ones(num_real_neighbors)
        return {
            "x": exp,
            "type_exp": type_exp,
            "y": y,
            "cell_types": cell_types,
            "position_x": centerx,
            "position_y": centery,
            "neighbor_mask": neighbor_mask
        }

class SPAGAT_evaluate_dataset(Dataset):
    def __init__(self, processed_dir, sample, num_neighbors=50, col_centerx="centerx",
                 col_centery="centery", col_cell_type="subclass"):
        if processed_dir[-1] != "/":
            processed_dir = processed_dir + "/"
        self.samples = [sample]
        print("Have samples:", self.samples)
        self.index_index = ["index_" + str(i) for i in range(num_neighbors)]
        self.interactions = torch.load("/".join(processed_dir.split("/")[:-2]) + "/ligands.pth", weights_only=False)
        self.genes = torch.load("/".join(processed_dir.split("/")[:-2]) + "/genes.pth", weights_only=False)
        # Collect all cell types from all data files
        all_cell_types = set()
        for samplei in self.samples:
            try:
                dfi = pd.read_csv(processed_dir + samplei + ".csv")
                all_cell_types.update(dfi[col_cell_type].unique())
            except Exception as e:
                print(f"Warning: Cannot read sample {samplei}: {e}")
                continue
        # Build cell type dictionary, sorted for consistency
        self.cell_types_dict = {ct: i for i, ct in enumerate(sorted(all_cell_types))}
        print(f"Built cell type dictionary with {len(self.cell_types_dict)} cell types: {sorted(all_cell_types)}")
        self.indexes = []
        self.flags = []
        self.centerx = []
        self.centery = []
        self.exps = []
        self.cell_types = []
        self.type_exp = []
        for samplei in self.samples:
            try:
                dfi = pd.read_csv(processed_dir + samplei + ".csv")
                dfi["cell_type_number"] = np.array([self.cell_types_dict.get(cell_type, -1)
                                                     for cell_type in dfi[col_cell_type]])
                if "flag" not in dfi.columns.tolist():
                    dfi["flag"] = [True for _ in range(dfi.shape[0])]
                type_exp_dicti = np.load(processed_dir + samplei + "_TypeExp.npz", allow_pickle=True)
                type_exp_dicti = {k: torch.Tensor(np.array(v).astype(np.float64)) for k, v in type_exp_dicti.items()}
                type_exp = torch.stack([type_exp_dicti[cell_type] for cell_type in dfi[col_cell_type]], dim=0)
                self.type_exp.append(type_exp)
                self.indexes.append(torch.LongTensor(dfi.loc[:, self.index_index].values))
                self.flags.append(dfi.loc[:, "flag"].values)
                self.centerx.append(torch.Tensor(dfi.loc[:, col_centerx].values))
                self.centery.append(torch.Tensor(dfi.loc[:, col_centery].values))
                self.exps.append(torch.Tensor(dfi.loc[:, self.genes].values))
                self.cell_types.append(torch.LongTensor(dfi.loc[:, "cell_type_number"].values))
            except Exception as e:
                print(f"Error: Cannot process sample {samplei}: {e}")
                import traceback
                traceback.print_exc()
                continue
        self.meta_counts = []
        self.arg_meta = []
        for i in range(len(self.samples)):
            if i < len(self.flags):
                self.meta_counts.append(np.sum(self.flags[i]))
                self.arg_meta.append(torch.LongTensor(np.where(self.flags[i] != 0)[0]))
        self.meta_counts = np.array(self.meta_counts) if self.meta_counts else np.array([])
        self.cumsum_meta = np.cumsum(self.meta_counts) if len(self.meta_counts) > 0 else np.array([])
        print("There are totally", self.cumsum_meta[-1] if len(self.cumsum_meta) > 0 else 0, "cells in this dataset")

    def __len__(self):
        return self.cumsum_meta[-1] if len(self.cumsum_meta) > 0 else 0

    def __getitem__(self, idx):
        if len(self.cumsum_meta) == 0:
            raise RuntimeError("No valid data in dataset")
        sample_id = np.searchsorted(self.cumsum_meta, idx, side='right')
        idx = idx - self.cumsum_meta[sample_id]
        idx = self.arg_meta[sample_id][idx]
        indices = self.indexes[sample_id][idx]
        centerx = self.centerx[sample_id][indices]
        centery = self.centery[sample_id][indices]
        exp = self.exps[sample_id][indices]
        type_exp = self.type_exp[sample_id][indices]
        y = exp[0]
        cell_types = torch.LongTensor(self.cell_types[sample_id][indices])
        indices = torch.LongTensor(indices)
        return {
            "x": exp,
            "type_exp": type_exp,
            "y": y,
            "cell_types": cell_types,
            "position_x": centerx,
            "position_y": centery,
            "NN": indices
        }
