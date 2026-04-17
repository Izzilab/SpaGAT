import os
import torch
import numpy as np
import pandas as pd

import scanpy as sc
from anndata import AnnData
import anndata as ad
import matplotlib.pyplot as plt


def extract_genes_and_pvals_by_group(adata, group_index, cutoff=0.05, up=True):
    '''
    Select significant differential expressed genes within an adata object
    :param adata:
    :param group_index: leiden index
    :param cutoff: adjusted p-value cutoff
    :param up: only to select up-regulated genes if True otherwise only select down-regulated genes
    :return:
    '''
    # Extracting gene names and p-values from the adata object
    gene_names = adata.uns['rank_genes_groups']['names']
    p_values = adata.uns['rank_genes_groups']['pvals']
    logfoldchanges = adata.uns['rank_genes_groups']['logfoldchanges']
    p_adj = adata.uns['rank_genes_groups']['pvals_adj']

    # Lists to hold filtered gene names and their corresponding p-values
    filtered_genes = []
    filtered_pvals = []

    # Iterate through each group in the gene names and p-values
    for gene_group, pval_group, fold_group in zip(gene_names, p_adj, logfoldchanges):
        gene = gene_group[group_index]
        pval = pval_group[group_index]
        foldchange = fold_group[group_index]

        # Check if p-value is below the cutoff and add to the lists if it is
        # print(pval,foldchange)
        if pval < cutoff and ((foldchange > 0) == up):
            filtered_genes.append(gene)
            filtered_pvals.append(pval)

class Subtyping_anlayzer():
    def __init__(self,sample,normalize_to_1=True,use_abs=False,noise_threshold=1e-5):
        '''

        :param sample: which tissue slide to analyze
        :param normalize_to_1: whether normalize the aggregated influence tensor so that their abs value sum
        up to one on the second dimension
        :param use_abs: whehter to use the absolute value for the aggregated influence tensor for downstream
        analysis
        :param noise_threshold: For values in the influence tensor, if its abs value is less than this threshold,
        we would treat it as noise, ignore it by setting it to 0
        '''
        # load data
        print("Start loading data")
        result_dir=os.path.join(os.getcwd(),"influence_tensor")
        if result_dir[-1]!="/":
            result_dir=result_dir+"/"
        data_dir=os.path.join(os.getcwd(),"data","processed")
        if data_dir[-1]!="/":
            data_dir=data_dir+"/"
        cell_types=torch.load(os.path.join(os.getcwd(),"data","processed","cell_types.pth"),weights_only=False)
        self.cell_types=cell_types
        genes = torch.load(os.path.join(os.getcwd(),"data","genes.pth"),weights_only=False)
        self.genes=genes
        type_exp_dict = np.load(os.path.join(data_dir, sample + "_TypeExp.npz"), allow_pickle=True)
        results = torch.load(result_dir + "edges_" + sample + ".pth", map_location=torch.device('cpu'),weights_only=False)
        print("Finish loading data")

        feature_names = []
        for i in range(len(cell_types)):
            for j in range(len(genes)):
                feature_names.append(cell_types[i] + "--" + genes[j])

        position_x = results["position_x"][:, 0]
        position_y = results["position_y"][:, 0]
        cell_type_name = np.array(results["cell_type_name"])
        cell_type_target = cell_type_name[:, 0]

        type_exps = torch.stack(
            [torch.Tensor(type_exp_dict[cell_type_targeti]) for cell_type_targeti in cell_type_target], dim=0)
        results["y"] = results["y"] + type_exps

        attention_scores = results["attention_score"]
        cell_type_names = np.array(results["cell_type_name"])

        proportion = torch.abs(attention_scores)
        proportion = proportion / torch.sum(proportion, dim=1, keepdim=True)
        attention_scores[proportion < noise_threshold] = 0

        # Initialize a tensor to hold aggregated interaction strengths
        B, _, C = attention_scores.shape
        t = len(cell_types)
        aggregated_interactions = torch.zeros((B, t, C))

        # Map cell type names to indices
        cell_type_to_index = {ct: idx for idx, ct in enumerate(cell_types)}

        # Aggregate interaction strengths by cell type
        print("Start aggregating")
        for b in range(B):
            if b%500==0:
                print(b,"/",B)
            for n in range(1, 50):  # Skip the first element, which is the target cell type
                neighbor_type = cell_type_names[b][n]
                if neighbor_type in cell_type_to_index:
                    idx = cell_type_to_index[neighbor_type]
                    aggregated_interactions[b, idx] += attention_scores[b, n - 1]
        print("Finish aggregating")

        if normalize_to_1:
            aggregated_interactions1 = aggregated_interactions / torch.sum(torch.abs(aggregated_interactions), dim=1,
                                                                           keepdim=True)
            aggregated_interactions = torch.where(
                torch.sum(torch.abs(aggregated_interactions), dim=1, keepdim=True) == 0,
                torch.zeros_like(aggregated_interactions), aggregated_interactions1)

        if use_abs:
            aggregated_interactions=torch.abs(aggregated_interactions)

        adata = AnnData(aggregated_interactions.reshape(B, -1).numpy())
        adata.obs['cell_type'] = cell_type_target
        adata.obs['position_x'] = position_x
        adata.obs['position_y'] = position_y
        adata.var_names = feature_names
        adata.obsm["y"] = results["y"].numpy()
        self.adata=adata

        with plt.rc_context({'figure.figsize': (6, 6)}):
            sc.pl.scatter(
                adata,
                x='position_x',  # 'position_x',
                y='position_y',  # 'position_y',
                color="cell_type"
            )

        self.adata_type=None
        self.adata_type_y=None
        self.leiden_labels=None  # 用于存储聚类标签

    def subtyping(self, COI, resolution=0.2, cluster_labels=None):
        '''
        Do cell subtyping analysis, plot UMAP for subgroup and plot their spatial distribution
        :param COI: Cell Of Interest
        :param resolution: resolution in leiden clustering
        :param cluster_labels: dict, optional. Map leiden cluster numbers to custom names.
                              Example: {0: "Subtype A", 1: "Subtype B", ...}
        :return:
        '''
        self.adata_type = self.adata[self.adata.obs["cell_type"] == COI]

        sc.tl.pca(self.adata_type, n_comps=50)
        sc.pp.neighbors(self.adata_type)  # Compute the neighborhood graph

        # Clustering
        sc.tl.leiden(self.adata_type, resolution=resolution)  # or sc.tl.louvain(adata)

        # Plot UMAP
        sc.tl.umap(self.adata_type)  # Compute UMAP
        n_clusters = len(self.adata_type.obs['leiden'].unique())
        print(f"📊 Leiden聚类完成。发现 {n_clusters} 个聚类")
        
        # 添加聚类标签（如果提供了）
        if cluster_labels is not None:
            self.leiden_labels = cluster_labels
            # 创建新的标签列
            self.adata_type.obs['leiden_label'] = self.adata_type.obs['leiden'].map(
                lambda x: cluster_labels.get(int(x), f"Cluster {x}")
            )
            print(f"✅ 已应用自定义聚类标签")
            sc.pl.umap(self.adata_type, color='leiden_label', legend_loc='on data')
        else:
            sc.pl.umap(self.adata_type, color='leiden')

        sc.pl.scatter(
            self.adata_type,
            x='position_x',
            y='position_y',
            color="leiden_label" if cluster_labels is not None else "leiden",
            title=f"Spatial cluster of {COI}"
        )

    def visualize_spatial_gene_expression(self,gene):
        sc.pl.scatter(
            self.adata_type,
            x='position_x',  # 'position_x',
            y='position_y',  # 'position_y',
            color=gene,
            title=f"Spatial gene expression of {gene}"
        )

    def subtyping_filter_groups(self, group_to_remain, replot_umap=True):
        '''
        Filter and keep only specified leiden groups. Optionally redraw UMAP with filtered groups.
        
        :param group_to_remain: list of str, leiden group indices to keep.
                              For example: ["0", "1", "2", "3"]
                              Note: items are strings, not integers
        :param replot_umap: If True (default), redraw UMAP after filtering to show only selected groups.
                           If False, skip redrawing (faster).
        :return:
        '''
        if self.adata_type is None:
            raise ValueError("Please do subtyping analysis first")
        
        # Keep track of original cluster count
        original_clusters = len(self.adata_type.obs['leiden'].unique())
        
        # Filter data
        self.adata_type = self.adata_type[self.adata_type.obs["leiden"].isin(group_to_remain)].copy()
        
        print(f"✅ 聚类过滤完成: {original_clusters} 个 → {len(group_to_remain)} 个")
        print(f"   已保留聚类: {group_to_remain}")
        print(f"   包含细胞数: {self.adata_type.shape[0]}")
        
        # Optionally redraw UMAP with only filtered groups
        if replot_umap:
            print(f"�� 重新绘制 UMAP (只显示已过滤的聚类)...")
            color_col = 'leiden_label' if self.leiden_labels is not None else 'leiden'
            if color_col == 'leiden_label' and 'leiden_label' in self.adata_type.obs.columns:
                sc.pl.umap(self.adata_type, color=color_col, legend_loc='on data')
            else:
                sc.pl.umap(self.adata_type, color='leiden')

    def subtyping_DE(self, method='wilcoxon', n_gene_show=5):
        '''
        Do and visualize the differential expression analysis
        :param method: statistical method to make comparison using scanpy, default to 'wilcoxon' (rank-sum test),
            other available methods are: 'logreg', 't-test', 'wilcoxon', 't-test_overestim_var'
        :param n_gene_show: how many DE gene to plot for one subgroup
        :return:
        '''
        if self.adata_type is None:
            raise ValueError("Please do subtyping analysis first before perform DEG analysis on subtypes that do not exist")
        self.adata_type_y = ad.AnnData(X=np.abs(self.adata_type.obsm["y"]), obs=self.adata_type.obs)
        # adata_y=adata_y[adata_y.obs['leiden'].isin(["0","1"])]#,"2","2","3"
        self.adata_type_y.var_names = self.genes
        
        # 使用标签列如果存在
        group_by = 'leiden_label' if self.leiden_labels is not None and 'leiden_label' in self.adata_type_y.obs.columns else 'leiden'
        
        sc.tl.rank_genes_groups(self.adata_type_y, group_by, method=method)
        sc.pl.rank_genes_groups_heatmap(self.adata_type_y, n_genes=n_gene_show, show_gene_labels=True,
                                        standard_scale='var', cmap='viridis')

    def subtyping_get_aggregated_influence(self, show_rank_genes=False):
        '''
        Analyze, proportionally, how each cell of the COI influenced by other cell types.
        Shows aggregated influence by cell type (cells x sender_cell_types).
        
        :param show_rank_genes: If False (default), only show aggregated influence heatmap (cell subtype view).
                               If True, also show rank genes heatmap (gene view).
        :return:
        '''
        if self.adata_type is None:
            raise ValueError("Please do subtyping analysis first")

        x = np.zeros((self.adata_type.shape[0], len(self.cell_types)))
        for i in range(len(self.cell_types)):
            offset = i * len(self.genes)
            x[:, i] = np.mean(np.abs(self.adata_type.X[:, offset:offset + len(self.genes)]), axis=1)

        self.adata_type_aggregated = ad.AnnData(X=x, obs=self.adata_type.obs)
        self.adata_type_aggregated.var_names = self.cell_types

        # Main plot: Aggregated influence heatmap (cell subtype view)
        print("  📊 绘制聚合影响热力图 (细胞亚型视角)...")
        
        # 使用标签列如果存在
        group_by = 'leiden_label' if self.leiden_labels is not None and 'leiden_label' in self.adata_type_aggregated.obs.columns else 'leiden'
        
        sc.pl.heatmap(self.adata_type_aggregated, var_names=self.adata_type_aggregated.var_names, 
                     groupby=group_by, cmap='bwr', show_gene_labels=True)
        
        if show_rank_genes:
            # Optional: Rank genes heatmap (gene view)
            print("  📊 绘制秩基因热力图 (基因视角)...")
            sc.tl.rank_genes_groups(self.adata_type_aggregated, group_by, method='wilcoxon')
            sc.pl.rank_genes_groups_heatmap(self.adata_type_aggregated, n_genes=10, show_gene_labels=True, cmap='bwr')

    def subtyping_get_aggregated_influence_target_gene(self, target_gene, show_rank_genes=False):
        '''
        Analyze, proportionally, how each cell of the COI's one target gene influenced by other cell types.
        Shows influence on specific target gene by sender cell types (cells x sender_cell_types).
        
        :param target_gene: Target gene to analyze
        :param show_rank_genes: If False (default), only show aggregated influence heatmap (cell subtype view).
                               If True, also show rank genes heatmap (gene view).
        :return:
        '''
        if self.adata_type is None:
            raise ValueError("Please do subtyping analysis first")

        x = np.zeros((self.adata_type.shape[0], len(self.cell_types)))

        offset = self.genes.index(target_gene)
        for i in range(len(self.cell_types)):
            x[:, i] = x[:, i] + self.adata_type.X[:, i * len(self.genes) + offset]

        self.adata_type_aggregated_target_gene = ad.AnnData(X=x, obs=self.adata_type.obs)
        self.adata_type_aggregated_target_gene.var_names = self.cell_types

        # Main plot: Aggregated influence heatmap for target gene (cell subtype view)
        print(f"  📊 绘制 {target_gene} 聚合影响热力图 (细胞亚型视角)...")
        
        # 使用标签列如果存在
        group_by = 'leiden_label' if self.leiden_labels is not None and 'leiden_label' in self.adata_type_aggregated_target_gene.obs.columns else 'leiden'
        
        sc.pl.heatmap(self.adata_type_aggregated_target_gene, 
                     var_names=self.adata_type_aggregated_target_gene.var_names, 
                     groupby=group_by, cmap='bwr', show_gene_labels=True)
        
        if show_rank_genes:
            # Optional: Rank genes heatmap (gene view)
            print(f"  📊 绘制 {target_gene} 秩基因热力图 (基因视角)...")
            sc.tl.rank_genes_groups(self.adata_type_aggregated_target_gene, group_by, method='wilcoxon')
            sc.pl.rank_genes_groups_heatmap(self.adata_type_aggregated_target_gene, n_genes=10, show_gene_labels=True, cmap='bwr')
