import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from statannot import add_stat_annotation
# Needed for Illustrator
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = ['Arial','Helvetica']

#%%
##### check the effects of different 3D data and FDRs #####

data_path = '/media/labuser/STORAGE/GraphReg'   # data path
df = pd.DataFrame(columns=['cell', 'Method', 'Set', 'valid_chr', 'test_chr', 'n_gene_test', '3D_data', 'FDR', 'R','NLL'])

cell_line = 'K562'
for assay_type in ['HiChIP', 'HiC']:
    for fdr in ['1', '01', '001']:
        df1 = pd.read_csv(data_path+'/results/csv/cage_prediction/epi_models/R_NLL_epi_models_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.csv', sep='\t')
        df = df.append(df1, ignore_index=True)

cell_line = 'GM12878'
for assay_type in ['HiChIP', 'HiC']:
    for fdr in ['1', '01', '001']:
        df1 = pd.read_csv(data_path+'/results/csv/cage_prediction/epi_models/R_NLL_epi_models_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.csv', sep='\t')
        df = df.append(df1, ignore_index=True)

cell_line = 'hESC'
for assay_type in ['MicroC', 'HiCAR']:
    for fdr in ['1', '01', '001']:
        df1 = pd.read_csv(data_path+'/results/csv/cage_prediction/epi_models/R_NLL_epi_models_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.csv', sep='\t')
        df = df.append(df1, ignore_index=True)

cell_line = 'mESC'
for assay_type in ['HiChIP']:
    for fdr in ['1', '01', '001']:
        df1 = pd.read_csv(data_path+'/results/csv/cage_prediction/epi_models/R_NLL_epi_models_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.csv', sep='\t')
        df = df.append(df1, ignore_index=True)

df = df.rename(columns={"cell": "Cell", "3D_data": "3D data", "n_gene_test": "Number of Genes", "Set": "Genes"})

## K562
df_sub = df[df['Cell']=='K562']

# R
g = sns.catplot(x="Genes", y="R",
                hue="Method", row='3D data', col="FDR",
                data=df_sub, kind="box", palette={"Epi-GraphReg": "orange", "Epi-CNN": "deepskyblue"},
                height=4, aspect=1, sharey=True)

n_row, n_col = g.axes.shape
for i in range(n_row):
    for j in range(n_col):
        df_annt = df_sub[(df_sub['3D data'] == g.row_names[i]) & (df_sub['FDR'] == g.col_names[j])]
        add_stat_annotation(g.axes[i,j], data=df_annt, x='Genes', y='R', hue='Method',
                            box_pairs=[(("All", "Epi-GraphReg"), ("All", "Epi-CNN")),
                                        (("Expressed", "Epi-GraphReg"), ("Expressed", "Epi-CNN")),
                                        (("Interacting", "Epi-GraphReg"), ("Interacting", "Epi-CNN"))],
                            test='Wilcoxon', text_format='star', loc='inside', verbose=0, order=['All', 'Expressed', 'Interacting'], fontsize='x-large', comparisons_correction=None)
plt.savefig('../figs/Epi-models/final/boxplot_R_K562_check_3D_and_fdr.pdf')

# NLL
g = sns.catplot(x="Genes", y="NLL",
                hue="Method", row='3D data', col="FDR",
                data=df_sub, kind="box", palette={"Epi-GraphReg": "orange", "Epi-CNN": "deepskyblue"},
                height=4, aspect=1, sharey=True)

n_row, n_col = g.axes.shape
for i in range(n_row):
    for j in range(n_col):
        df_annt = df_sub[(df_sub['3D data'] == g.row_names[i]) & (df_sub['FDR'] == g.col_names[j])]
        add_stat_annotation(g.axes[i,j], data=df_annt, x='Genes', y='NLL', hue='Method',
                            box_pairs=[(("All", "Epi-GraphReg"), ("All", "Epi-CNN")),
                                        (("Expressed", "Epi-GraphReg"), ("Expressed", "Epi-CNN")),
                                        (("Interacting", "Epi-GraphReg"), ("Interacting", "Epi-CNN"))],
                            test='Wilcoxon', text_format='star', loc='inside', verbose=0, order=['All', 'Expressed', 'Interacting'], fontsize='x-large', comparisons_correction=None)
plt.savefig('../figs/Epi-models/final/boxplot_NLL_K562_check_3D_and_fdr.pdf')

# Number of genes
g = sns.catplot(x="Genes", y="Number of Genes",
                hue="Method", row='3D data', col="FDR",
                data=df_sub, kind="swarm", palette={"Epi-GraphReg": "orange", "Epi-CNN": "deepskyblue"},
                height=4, aspect=1, sharey=True)
plt.savefig('../figs/Epi-models/final/boxplot_n_genes_K562.pdf')

## GM12878
df_sub = df[df['Cell']=='GM12878']

# R
g = sns.catplot(x="Genes", y="R",
                hue="Method", row='3D data', col="FDR",
                data=df_sub, kind="box", palette={"Epi-GraphReg": "orange", "Epi-CNN": "deepskyblue"},
                height=4, aspect=1, sharey=True)

n_row, n_col = g.axes.shape
for i in range(n_row):
    for j in range(n_col):
        df_annt = df_sub[(df_sub['3D data'] == g.row_names[i]) & (df_sub['FDR'] == g.col_names[j])]
        add_stat_annotation(g.axes[i,j], data=df_annt, x='Genes', y='R', hue='Method',
                            box_pairs=[(("All", "Epi-GraphReg"), ("All", "Epi-CNN")),
                                        (("Expressed", "Epi-GraphReg"), ("Expressed", "Epi-CNN")),
                                        (("Interacting", "Epi-GraphReg"), ("Interacting", "Epi-CNN"))],
                            test='Wilcoxon', text_format='star', loc='inside', verbose=0, order=['All', 'Expressed', 'Interacting'], fontsize='x-large', comparisons_correction=None)
plt.savefig('../figs/Epi-models/final/boxplot_R_GM12878_check_3D_and_fdr.pdf')

# NLL
g = sns.catplot(x="Genes", y="NLL",
                hue="Method", row='3D data', col="FDR",
                data=df_sub, kind="box", palette={"Epi-GraphReg": "orange", "Epi-CNN": "deepskyblue"},
                height=4, aspect=1, sharey=True)

n_row, n_col = g.axes.shape
for i in range(n_row):
    for j in range(n_col):
        df_annt = df_sub[(df_sub['3D data'] == g.row_names[i]) & (df_sub['FDR'] == g.col_names[j])]
        add_stat_annotation(g.axes[i,j], data=df_annt, x='Genes', y='NLL', hue='Method',
                            box_pairs=[(("All", "Epi-GraphReg"), ("All", "Epi-CNN")),
                                        (("Expressed", "Epi-GraphReg"), ("Expressed", "Epi-CNN")),
                                        (("Interacting", "Epi-GraphReg"), ("Interacting", "Epi-CNN"))],
                            test='Wilcoxon', text_format='star', loc='inside', verbose=0, order=['All', 'Expressed', 'Interacting'], fontsize='x-large', comparisons_correction=None)
plt.savefig('../figs/Epi-models/final/boxplot_NLL_GM12878_check_3D_and_fdr.pdf')

# Number of genes
g = sns.catplot(x="Genes", y="Number of Genes",
                hue="Method", row='3D data', col="FDR",
                data=df_sub, kind="swarm", palette={"Epi-GraphReg": "orange", "Epi-CNN": "deepskyblue"},
                height=4, aspect=1, sharey=True)
plt.savefig('../figs/Epi-models/final/boxplot_n_genes_GM12878.pdf')

## hESC
df_sub = df[df['Cell']=='hESC']

# R
g = sns.catplot(x="Genes", y="R",
                hue="Method", row='3D data', col="FDR",
                data=df_sub, kind="box", palette={"Epi-GraphReg": "orange", "Epi-CNN": "deepskyblue"},
                height=4, aspect=1, sharey=True)

n_row, n_col = g.axes.shape
for i in range(n_row):
    for j in range(n_col):
        df_annt = df_sub[(df_sub['3D data'] == g.row_names[i]) & (df_sub['FDR'] == g.col_names[j])]
        add_stat_annotation(g.axes[i,j], data=df_annt, x='Genes', y='R', hue='Method',
                            box_pairs=[(("All", "Epi-GraphReg"), ("All", "Epi-CNN")),
                                        (("Expressed", "Epi-GraphReg"), ("Expressed", "Epi-CNN")),
                                        (("Interacting", "Epi-GraphReg"), ("Interacting", "Epi-CNN"))],
                            test='Wilcoxon', text_format='star', loc='inside', verbose=0, order=['All', 'Expressed', 'Interacting'], fontsize='x-large', comparisons_correction=None)
plt.savefig('../figs/Epi-models/final/boxplot_R_hESC_check_3D_and_fdr.pdf')

# NLL
g = sns.catplot(x="Genes", y="NLL",
                hue="Method", row='3D data', col="FDR",
                data=df_sub, kind="box", palette={"Epi-GraphReg": "orange", "Epi-CNN": "deepskyblue"},
                height=4, aspect=1, sharey=True)

n_row, n_col = g.axes.shape
for i in range(n_row):
    for j in range(n_col):
        df_annt = df_sub[(df_sub['3D data'] == g.row_names[i]) & (df_sub['FDR'] == g.col_names[j])]
        add_stat_annotation(g.axes[i,j], data=df_annt, x='Genes', y='NLL', hue='Method',
                            box_pairs=[(("All", "Epi-GraphReg"), ("All", "Epi-CNN")),
                                        (("Expressed", "Epi-GraphReg"), ("Expressed", "Epi-CNN")),
                                        (("Interacting", "Epi-GraphReg"), ("Interacting", "Epi-CNN"))],
                            test='Wilcoxon', text_format='star', loc='inside', verbose=0, order=['All', 'Expressed', 'Interacting'], fontsize='x-large', comparisons_correction=None)
plt.savefig('../figs/Epi-models/final/boxplot_NLL_hESC_check_3D_and_fdr.pdf')

# Number of genes
g = sns.catplot(x="Genes", y="Number of Genes",
                hue="Method", row='3D data', col="FDR",
                data=df_sub, kind="swarm", palette={"Epi-GraphReg": "orange", "Epi-CNN": "deepskyblue"},
                height=4, aspect=1, sharey=True)
plt.savefig('../figs/Epi-models/final/boxplot_n_genes_hESC.pdf')

## mESC
df_sub = df[df['Cell']=='mESC']

# R
g = sns.catplot(x="Genes", y="R",
                hue="Method", row='3D data', col="FDR",
                data=df_sub, kind="box", palette={"Epi-GraphReg": "orange", "Epi-CNN": "deepskyblue"},
                height=4, aspect=1, sharey=True)

n_row, n_col = g.axes.shape
for i in range(n_row):
    for j in range(n_col):
        df_annt = df_sub[(df_sub['3D data'] == g.row_names[i]) & (df_sub['FDR'] == g.col_names[j])]
        add_stat_annotation(g.axes[i,j], data=df_annt, x='Genes', y='R', hue='Method',
                            box_pairs=[(("All", "Epi-GraphReg"), ("All", "Epi-CNN")),
                                        (("Expressed", "Epi-GraphReg"), ("Expressed", "Epi-CNN")),
                                        (("Interacting", "Epi-GraphReg"), ("Interacting", "Epi-CNN"))],
                            test='Wilcoxon', text_format='star', loc='inside', verbose=0, order=['All', 'Expressed', 'Interacting'], fontsize='x-large', comparisons_correction=None)
plt.savefig('../figs/Epi-models/final/boxplot_R_mESC_check_3D_and_fdr.pdf')

# NLL
g = sns.catplot(x="Genes", y="NLL",
                hue="Method", row='3D data', col="FDR",
                data=df_sub, kind="box", palette={"Epi-GraphReg": "orange", "Epi-CNN": "deepskyblue"},
                height=4, aspect=1, sharey=True)

n_row, n_col = g.axes.shape
for i in range(n_row):
    for j in range(n_col):
        df_annt = df_sub[(df_sub['3D data'] == g.row_names[i]) & (df_sub['FDR'] == g.col_names[j])]
        add_stat_annotation(g.axes[i,j], data=df_annt, x='Genes', y='NLL', hue='Method',
                            box_pairs=[(("All", "Epi-GraphReg"), ("All", "Epi-CNN")),
                                        (("Expressed", "Epi-GraphReg"), ("Expressed", "Epi-CNN")),
                                        (("Interacting", "Epi-GraphReg"), ("Interacting", "Epi-CNN"))],
                            test='Wilcoxon', text_format='star', loc='inside', verbose=0, order=['All', 'Expressed', 'Interacting'], fontsize='x-large', comparisons_correction=None)
plt.savefig('../figs/Epi-models/final/boxplot_NLL_mESC_check_3D_and_fdr.pdf')

# Number of genes
g = sns.catplot(x="Genes", y="Number of Genes",
                hue="Method", row='3D data', col="FDR",
                data=df_sub, kind="swarm", palette={"Epi-GraphReg": "orange", "Epi-CNN": "deepskyblue"},
                height=4, aspect=1, sharey=True)
plt.savefig('../figs/Epi-models/final/boxplot_n_genes_mESC.pdf')


## Only GraphReg

## K562
df_sub = df[(df['Method'] == 'Epi-GraphReg') & (df['Genes'] != 'Interacting')]

# R
g = sns.catplot(x="FDR", y="R",
                hue="3D data", row='Genes', col="Cell",
                data=df_sub, kind="box",
                height=4, aspect=1, sharey=False)

n_row, n_col = g.axes.shape
for i in range(n_row):
    for j in range(n_col):
        df_annt = df_sub[(df_sub['Genes'] == g.row_names[i]) & (df_sub['Cell'] == g.col_names[j])]
        if g.col_names[j] in ['K562', 'GM12878']:
            assay1 = 'HiChIP'
            assay2 = 'HiC'

            add_stat_annotation(g.axes[i,j], data=df_annt, x='FDR', y='R', hue='3D data',
                            box_pairs=[((0.001, assay1), (0.001, assay2)),
                                        ((0.01, assay1), (0.01, assay2)),
                                        ((0.1, assay1), (0.1, assay2))],
                            test='Wilcoxon', text_format='star', loc='inside', verbose=0, order=[0.001, 0.01, 0.1], fontsize='x-large', comparisons_correction=None)

        elif g.col_names[j] == 'hESC':
            assay1 = 'MicroC'
            assay2 = 'HiCAR'

            add_stat_annotation(g.axes[i,j], data=df_annt, x='FDR', y='R', hue='3D data',
                            box_pairs=[((0.001, assay1), (0.001, assay2)),
                                        ((0.01, assay1), (0.01, assay2)),
                                        ((0.1, assay1), (0.1, assay2))],
                            test='Wilcoxon', text_format='star', loc='inside', verbose=0, order=[0.001, 0.01, 0.1], fontsize='x-large', comparisons_correction=None)

        else:
            add_stat_annotation(g.axes[i,j], data=df_annt, x='FDR', y='R',
                            box_pairs=[(0.001, 0.01),
                                        (0.001, 0.1),
                                        (0.01, 0.1)],
                            test='Wilcoxon', text_format='star', loc='inside', verbose=0, order=[0.001, 0.01, 0.1], fontsize='x-large', comparisons_correction=None)
plt.savefig('../figs/Epi-models/final/boxplot_R_only_graphreg_check_3D_and_fdr.pdf')


# NLL
g = sns.catplot(x="FDR", y="NLL",
                hue="3D data", row='Genes', col="Cell",
                data=df_sub, kind="box",
                height=4, aspect=1, sharey=False)

n_row, n_col = g.axes.shape
for i in range(n_row):
    for j in range(n_col):
        df_annt = df_sub[(df_sub['Genes'] == g.row_names[i]) & (df_sub['Cell'] == g.col_names[j])]
        if g.col_names[j] in ['K562', 'GM12878']:
            assay1 = 'HiChIP'
            assay2 = 'HiC'

            add_stat_annotation(g.axes[i,j], data=df_annt, x='FDR', y='NLL', hue='3D data',
                            box_pairs=[((0.001, assay1), (0.001, assay2)),
                                        ((0.01, assay1), (0.01, assay2)),
                                        ((0.1, assay1), (0.1, assay2))],
                            test='Wilcoxon', text_format='star', loc='inside', verbose=0, order=[0.001, 0.01, 0.1], fontsize='x-large', comparisons_correction=None)

        elif g.col_names[j] == 'hESC':
            assay1 = 'MicroC'
            assay2 = 'HiCAR'

            add_stat_annotation(g.axes[i,j], data=df_annt, x='FDR', y='NLL', hue='3D data',
                            box_pairs=[((0.001, assay1), (0.001, assay2)),
                                        ((0.01, assay1), (0.01, assay2)),
                                        ((0.1, assay1), (0.1, assay2))],
                            test='Wilcoxon', text_format='star', loc='inside', verbose=0, order=[0.001, 0.01, 0.1], fontsize='x-large', comparisons_correction=None)

        else:
            add_stat_annotation(g.axes[i,j], data=df_annt, x='FDR', y='NLL',
                            box_pairs=[(0.001, 0.01),
                                        (0.001, 0.1),
                                        (0.01, 0.1)],
                            test='Wilcoxon', text_format='star', loc='inside', verbose=0, order=[0.001, 0.01, 0.1], fontsize='x-large', comparisons_correction=None)
plt.savefig('../figs/Epi-models/final/boxplot_NLL_only_graphreg_check_3D_and_fdr.pdf')


#%%
##### universal analysis of CAGE predictions (subsample predictions 50 times) #####

data_path = '/media/labuser/STORAGE/GraphReg'   # data path
cell_line = 'K562'
assay_type = 'HiChIP'
fdr = '001'              # 1/01/001

df = pd.read_csv(data_path+'/results/csv/cage_prediction/epi_models/cage_predictions_epi_models_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.csv', sep='\t')

if cell_line == 'mESC':
    df_chr2 = df[df['chr'] == 'chr2']
    len_chr2 = len(df_chr2)//2
    idx_repeated = df_chr2.index[-len_chr2:]
    df = df.drop(idx_repeated).reset_index()
df['pred_cage_epi_graphreg_log2'] = np.log2(df['pred_cage_epi_graphreg']+1)
df['pred_cage_epi_cnn_log2'] = np.log2(df['pred_cage_epi_cnn']+1)
df['true_cage_log2'] = np.log2(df['true_cage']+1)
df_expressed = df[df['true_cage']>=5].reset_index()
df_expressed_interacting = df[((df['true_cage']>=5) & (df['n_contact']>=1))].reset_index()

qval_dict = {'1': 0.1, '01': 0.01, '001': 0.001}
df_R_NLL = pd.DataFrame(columns=['Cell', 'Method', 'Genes', 'Number of Genes', '3D Data', 'FDR', 'R', 'NLL'])

for i in range(50):
    print('i: ', i)
    N = len(df)
    ramdom_idx = np.random.choice(N, 2000, replace=False)
    df_sub = df.iloc[ramdom_idx]

    N_expressed = len(df_expressed)
    ramdom_idx = np.random.choice(N_expressed, 2000, replace=False)
    df_sub_expressed = df_expressed.iloc[ramdom_idx]

    N_interacting = len(df_expressed_interacting)
    ramdom_idx = np.random.choice(N_interacting, 2000, replace=False)
    df_sub_interacting = df_expressed_interacting.iloc[ramdom_idx]

    # NLL
    nll_graphreg_all = df_sub['nll_epi_graphreg'].values
    nll_graphreg_all_mean = np.mean(nll_graphreg_all)
    nll_graphreg_ex = df_sub_expressed['nll_epi_graphreg'].values
    nll_graphreg_ex_mean = np.mean(nll_graphreg_ex)
    nll_graphreg_ex_int = df_sub_interacting['nll_epi_graphreg'].values
    nll_graphreg_ex_int_mean = np.mean(nll_graphreg_ex_int)

    nll_cnn_all = df_sub['nll_epi_cnn'].values
    nll_cnn_all_mean = np.mean(nll_cnn_all)
    nll_cnn_ex = df_sub_expressed['nll_epi_cnn'].values
    nll_cnn_ex_mean = np.mean(nll_cnn_ex)
    nll_cnn_ex_int = df_sub_interacting['nll_epi_cnn'].values
    nll_cnn_ex_int_mean = np.mean(nll_cnn_ex_int)

    # R
    r_graphreg_all = np.corrcoef(df_sub['true_cage_log2'].values, df_sub['pred_cage_epi_graphreg_log2'])[0,1]
    r_graphreg_ex = np.corrcoef(df_sub_expressed['true_cage_log2'].values, df_sub_expressed['pred_cage_epi_graphreg_log2'])[0,1]
    r_graphreg_ex_int = np.corrcoef(df_sub_interacting['true_cage_log2'].values, df_sub_interacting['pred_cage_epi_graphreg_log2'])[0,1]

    r_cnn_all = np.corrcoef(df_sub['true_cage_log2'].values, df_sub['pred_cage_epi_cnn_log2'])[0,1]
    r_cnn_ex = np.corrcoef(df_sub_expressed['true_cage_log2'].values, df_sub_expressed['pred_cage_epi_cnn_log2'])[0,1]
    r_cnn_ex_int = np.corrcoef(df_sub_interacting['true_cage_log2'].values, df_sub_interacting['pred_cage_epi_cnn_log2'])[0,1]

    df_R_NLL = df_R_NLL.append({'Cell': cell_line, 'Method': 'Epi-GraphReg', 'Genes': 'All',
                    'Number of Genes': len(df_sub), '3D Data': assay_type, 'FDR': qval_dict[fdr], 'R': r_graphreg_all, 'NLL': nll_graphreg_all_mean}, ignore_index=True)
    df_R_NLL = df_R_NLL.append({'Cell': cell_line, 'Method': 'Epi-GraphReg', 'Genes': 'Expressed', 
                    'Number of Genes': len(df_sub_expressed), '3D Data': assay_type, 'FDR': qval_dict[fdr], 'R': r_graphreg_ex, 'NLL': nll_graphreg_ex_mean}, ignore_index=True)
    df_R_NLL = df_R_NLL.append({'Cell': cell_line, 'Method': 'Epi-GraphReg', 'Genes': 'Interacting',
                    'Number of Genes': len(df_sub_interacting), '3D Data': assay_type, 'FDR': qval_dict[fdr], 'R': r_graphreg_ex_int, 'NLL': nll_graphreg_ex_int_mean}, ignore_index=True)

    df_R_NLL = df_R_NLL.append({'Cell': cell_line, 'Method': 'Epi-CNN', 'Genes': 'All',
                'Number of Genes': len(df_sub), '3D Data': assay_type, 'FDR': qval_dict[fdr], 'R': r_cnn_all, 'NLL': nll_cnn_all_mean}, ignore_index=True)
    df_R_NLL = df_R_NLL.append({'Cell': cell_line, 'Method': 'Epi-CNN', 'Genes': 'Expressed',
                'Number of Genes': len(df_sub_expressed), '3D Data': assay_type, 'FDR': qval_dict[fdr], 'R': r_cnn_ex, 'NLL': nll_cnn_ex_mean}, ignore_index=True)
    df_R_NLL = df_R_NLL.append({'Cell': cell_line, 'Method': 'Epi-CNN', 'Genes': 'Interacting',
                'Number of Genes': len(df_sub_interacting), '3D Data': assay_type, 'FDR': qval_dict[fdr], 'R': r_cnn_ex_int, 'NLL': nll_cnn_ex_int_mean}, ignore_index=True)

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
g = sns.boxplot(data=df_R_NLL, x='Genes', y='R', hue='Method', palette={"Epi-GraphReg": "orange", "Epi-CNN": "deepskyblue"}, ax=ax1)
add_stat_annotation(ax1, data=df_R_NLL, x='Genes', y='R', hue='Method',
                            box_pairs=[(("All", "Epi-GraphReg"), ("All", "Epi-CNN")),
                                        (("Expressed", "Epi-GraphReg"), ("Expressed", "Epi-CNN")),
                                        (("Interacting", "Epi-GraphReg"), ("Interacting", "Epi-CNN"))],
                            test='Wilcoxon', text_format='star', loc='inside', verbose=0, order=['All', 'Expressed', 'Interacting'], fontsize='x-large', comparisons_correction=None)
ax1.yaxis.set_tick_params(labelsize=20)
ax1.xaxis.set_tick_params(labelsize=20)
#ax1.set_title('title', fontsize=20)
g.set_xlabel("Genes",fontsize=20)
g.set_ylabel("R",fontsize=20)
plt.setp(ax1.get_legend().get_texts(), fontsize='20')
plt.setp(ax1.get_legend().get_title(), fontsize='20')
plt.tight_layout()
plt.savefig('../figs/Epi-models/final/boxplot_R_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.pdf')

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
g = sns.boxplot(data=df_R_NLL, x='Genes', y='NLL', hue='Method', palette={"Epi-GraphReg": "orange", "Epi-CNN": "deepskyblue"}, ax=ax1)
add_stat_annotation(ax1, data=df_R_NLL, x='Genes', y='NLL', hue='Method',
                            box_pairs=[(("All", "Epi-GraphReg"), ("All", "Epi-CNN")),
                                        (("Expressed", "Epi-GraphReg"), ("Expressed", "Epi-CNN")),
                                        (("Interacting", "Epi-GraphReg"), ("Interacting", "Epi-CNN"))],
                            test='Wilcoxon', text_format='star', loc='inside', verbose=0, order=['All', 'Expressed', 'Interacting'], fontsize='x-large', comparisons_correction=None)
ax1.yaxis.set_tick_params(labelsize=20)
ax1.xaxis.set_tick_params(labelsize=20)
#ax1.set_title('title', fontsize=20)
g.set_xlabel("Genes",fontsize=20)
g.set_ylabel("NLL",fontsize=20)
plt.setp(ax1.get_legend().get_texts(), fontsize='20')
plt.setp(ax1.get_legend().get_title(), fontsize='20')
plt.tight_layout()
plt.savefig('../figs/Epi-models/final/boxplot_NLL_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.pdf')

#%%
##### logfold change (subsample predictions 50 times) #####

data_path = '/media/labuser/STORAGE/GraphReg'   # data path
qval_dict = {'1': 0.1, '01': 0.01, '001': 0.001}
cell_line_1 = 'GM12878'
cell_line_2 = 'K562'
assay_type_1 = 'HiC'
assay_type_2 = 'HiChIP'
fdr_1 = '001'
fdr_2 = '01'

df_1 = pd.read_csv(data_path+'/results/csv/cage_prediction/epi_models/cage_predictions_epi_models_'+cell_line_1+'_'+assay_type_1+'_FDR_'+fdr_1+'.csv', sep='\t')
df_2 = pd.read_csv(data_path+'/results/csv/cage_prediction/epi_models/cage_predictions_epi_models_'+cell_line_2+'_'+assay_type_2+'_FDR_'+fdr_2+'.csv', sep='\t')
assert (df_1['genes'] == df_2['genes']).all()
df = df_1[['chr', 'genes', 'n_tss', 'tss', 'tss_distance_from_center']]

df['n_contact_1'] = df_1['n_contact']
df['n_contact_2'] = df_2['n_contact']
df['n_contact_min'] = np.minimum(df_1['n_contact'].values, df_2['n_contact'].values)
df['n_contact_min_log2'] = np.log2(df['n_contact_min']+1)

df['true_cage_1'] = df_1['true_cage']
df['true_cage_2'] = df_2['true_cage']
df['true_cage_min'] = np.minimum(df_1['true_cage'].values, df_2['true_cage'].values)
df['true_cage_logfc'] = np.log2((df_1['true_cage']+1)/(df_2['true_cage']+1))

df['pred_cage_epi_graphreg_1'] = df_1['pred_cage_epi_graphreg']
df['pred_cage_epi_graphreg_2'] = df_2['pred_cage_epi_graphreg']
df['pred_cage_epi_graphreg_logfc'] = np.log2((df_1['pred_cage_epi_graphreg']+1)/(df_2['pred_cage_epi_graphreg']+1))

df['pred_cage_epi_cnn_1'] = df_1['pred_cage_epi_cnn']
df['pred_cage_epi_cnn_2'] = df_2['pred_cage_epi_cnn']
df['pred_cage_epi_cnn_logfc'] = np.log2((df_1['pred_cage_epi_cnn']+1)/(df_2['pred_cage_epi_cnn']+1))

df_expressed = df[df['true_cage_min']>=5].reset_index(drop=True)
df_expressed_interacting = df[((df['true_cage_min']>=5) & (df['n_contact_min']>=1))].reset_index(drop=True)


df_R_MSE = pd.DataFrame(columns=['Cell', 'Method', 'Genes', 'Number of Genes', '3D Data', 'FDR', 'R', 'MSE'])

for i in range(50):
    print('i: ', i)
    N = len(df)
    ramdom_idx = np.random.choice(N, 2000, replace=False)
    df_sub = df.iloc[ramdom_idx]

    N_expressed = len(df_expressed)
    ramdom_idx = np.random.choice(N_expressed, 2000, replace=False)
    df_sub_expressed = df_expressed.iloc[ramdom_idx]

    N_interacting = len(df_expressed_interacting)
    ramdom_idx = np.random.choice(N_interacting, 2000, replace=False)
    df_sub_interacting = df_expressed_interacting.iloc[ramdom_idx]

    # MSE
    mse_graphreg_all = np.mean((df_sub['pred_cage_epi_graphreg_logfc'].values - df_sub['true_cage_logfc'].values)**2)
    mse_graphreg_ex = np.mean((df_sub_expressed['pred_cage_epi_graphreg_logfc'].values - df_sub_expressed['true_cage_logfc'].values)**2)
    mse_graphreg_ex_int = np.mean((df_sub_interacting['pred_cage_epi_graphreg_logfc'].values - df_sub_interacting['true_cage_logfc'].values)**2)

    mse_cnn_all = np.mean((df_sub['pred_cage_epi_cnn_logfc'].values - df_sub['true_cage_logfc'].values)**2)
    mse_cnn_ex = np.mean((df_sub_expressed['pred_cage_epi_cnn_logfc'].values - df_sub_expressed['true_cage_logfc'].values)**2)
    mse_cnn_ex_int = np.mean((df_sub_interacting['pred_cage_epi_cnn_logfc'].values - df_sub_interacting['true_cage_logfc'].values)**2)

    # R
    r_graphreg_all = np.corrcoef(df_sub['true_cage_logfc'].values, df_sub['pred_cage_epi_graphreg_logfc'])[0,1]
    r_graphreg_ex = np.corrcoef(df_sub_expressed['true_cage_logfc'].values, df_sub_expressed['pred_cage_epi_graphreg_logfc'])[0,1]
    r_graphreg_ex_int = np.corrcoef(df_sub_interacting['true_cage_logfc'].values, df_sub_interacting['pred_cage_epi_graphreg_logfc'])[0,1]

    r_cnn_all = np.corrcoef(df_sub['true_cage_logfc'].values, df_sub['pred_cage_epi_cnn_logfc'])[0,1]
    r_cnn_ex = np.corrcoef(df_sub_expressed['true_cage_logfc'].values, df_sub_expressed['pred_cage_epi_cnn_logfc'])[0,1]
    r_cnn_ex_int = np.corrcoef(df_sub_interacting['true_cage_logfc'].values, df_sub_interacting['pred_cage_epi_cnn_logfc'])[0,1]

    df_R_MSE = df_R_MSE.append({'Cell': cell_line_1+' / '+cell_line_2, 'Method': 'Epi-GraphReg', 'Genes': 'All', 
                    'Number of Genes': len(df_sub), '3D Data': assay_type_1+' / '+assay_type_2, 'FDR': str(qval_dict[fdr_1])+' / '+str(qval_dict[fdr_2]), 'R': r_graphreg_all, 'MSE': mse_graphreg_all}, ignore_index=True)
    df_R_MSE = df_R_MSE.append({'Cell': cell_line_1+' / '+cell_line_2, 'Method': 'Epi-GraphReg', 'Genes': 'Expressed', 
                    'Number of Genes': len(df_sub_expressed), '3D Data': assay_type_1+' / '+assay_type_2, 'FDR': str(qval_dict[fdr_1])+' / '+str(qval_dict[fdr_2]), 'R': r_graphreg_ex, 'MSE': mse_graphreg_ex}, ignore_index=True)
    df_R_MSE = df_R_MSE.append({'Cell': cell_line_1+' / '+cell_line_2, 'Method': 'Epi-GraphReg', 'Genes': 'Interacting',
                    'Number of Genes': len(df_sub_interacting), '3D Data': assay_type_1+' / '+assay_type_2, 'FDR': str(qval_dict[fdr_1])+' / '+str(qval_dict[fdr_2]), 'R': r_graphreg_ex_int, 'MSE': mse_graphreg_ex_int}, ignore_index=True)

    df_R_MSE = df_R_MSE.append({'Cell': cell_line_1+' / '+cell_line_2, 'Method': 'Epi-CNN', 'Genes': 'All', 
                    'Number of Genes': len(df_sub), '3D Data': assay_type_1+' / '+assay_type_2, 'FDR': str(qval_dict[fdr_1])+' / '+str(qval_dict[fdr_2]), 'R': r_cnn_all, 'MSE': mse_cnn_all}, ignore_index=True)
    df_R_MSE = df_R_MSE.append({'Cell': cell_line_1+' / '+cell_line_2, 'Method': 'Epi-CNN', 'Genes': 'Expressed',
                'Number of Genes': len(df_sub_expressed), '3D Data': assay_type_1+' / '+assay_type_2, 'FDR': str(qval_dict[fdr_1])+' / '+str(qval_dict[fdr_2]), 'R': r_cnn_ex, 'MSE': mse_cnn_ex}, ignore_index=True)
    df_R_MSE = df_R_MSE.append({'Cell': cell_line_1+' / '+cell_line_2, 'Method': 'Epi-CNN', 'Genes': 'Interacting',
                'Number of Genes': len(df_sub_interacting), '3D Data': assay_type_1+' / '+assay_type_2, 'FDR': str(qval_dict[fdr_1])+' / '+str(qval_dict[fdr_2]), 'R': r_cnn_ex_int, 'MSE': mse_cnn_ex_int}, ignore_index=True)

sns.set_style("ticks")
fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
g = sns.boxplot(data=df_R_MSE, x='Genes', y='R', hue='Method', palette={"Epi-GraphReg": "orange", "Epi-CNN": "deepskyblue"}, ax=ax1)
add_stat_annotation(ax1, data=df_R_MSE, x='Genes', y='R', hue='Method',
                            box_pairs=[(("All", "Epi-GraphReg"), ("All", "Epi-CNN")),
                                        (("Expressed", "Epi-GraphReg"), ("Expressed", "Epi-CNN")),
                                        (("Interacting", "Epi-GraphReg"), ("Interacting", "Epi-CNN"))],
                            test='Wilcoxon', text_format='star', loc='inside', verbose=0, order=['All', 'Expressed', 'Interacting'], fontsize='x-large', comparisons_correction=None)
ax1.yaxis.set_tick_params(labelsize=20)
ax1.xaxis.set_tick_params(labelsize=20)
#ax1.set_title('title', fontsize=20)
g.set_xlabel("Genes",fontsize=20)
g.set_ylabel("R",fontsize=20)
plt.setp(ax1.get_legend().get_texts(), fontsize='20')
plt.setp(ax1.get_legend().get_title(), fontsize='20')
plt.tight_layout()
plt.savefig('../figs/Epi-models/final/boxplot_R_logfc_'+cell_line_1+'_'+assay_type_1+'_FDR_'+fdr_1+'_to_'+cell_line_2+'_'+assay_type_2+'_FDR_'+fdr_2+'.pdf')

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
g = sns.boxplot(data=df_R_MSE, x='Genes', y='MSE', hue='Method', palette={"Epi-GraphReg": "orange", "Epi-CNN": "deepskyblue"}, ax=ax1)
add_stat_annotation(ax1, data=df_R_MSE, x='Genes', y='MSE', hue='Method',
                            box_pairs=[(("All", "Epi-GraphReg"), ("All", "Epi-CNN")),
                                        (("Expressed", "Epi-GraphReg"), ("Expressed", "Epi-CNN")),
                                        (("Interacting", "Epi-GraphReg"), ("Interacting", "Epi-CNN"))],
                            test='Wilcoxon', text_format='star', loc='inside', verbose=0, order=['All', 'Expressed', 'Interacting'], fontsize='x-large', comparisons_correction=None)
ax1.yaxis.set_tick_params(labelsize=20)
ax1.xaxis.set_tick_params(labelsize=20)
#ax1.set_title('title', fontsize=20)
g.set_xlabel("Genes",fontsize=20)
g.set_ylabel("MSE",fontsize=20)
plt.setp(ax1.get_legend().get_texts(), fontsize='20')
plt.setp(ax1.get_legend().get_title(), fontsize='20')
plt.tight_layout()
plt.savefig('../figs/Epi-models/final/boxplot_MSE_logfc_'+cell_line_1+'_'+assay_type_1+'_FDR_'+fdr_1+'_to_'+cell_line_2+'_'+assay_type_2+'_FDR_'+fdr_2+'.pdf')


#%%
###############################################################################################################################################
##########################################################            Swarm Plots             #################################################
###############################################################################################################################################

data_path = '/media/labuser/STORAGE/GraphReg'   # data path
qval_dict = {'1': 0.1, '01': 0.01, '001': 0.001}
df_R_NLL_all = pd.DataFrame(columns=['Cell', 'Method', 'Genes', 'Number of Genes', '3D Data', 'FDR', 'R', 'NLL'])

for cell_line in ['K562', 'GM12878', 'hESC', 'mESC']:
    if cell_line in ['K562', 'GM12878']:
        assay_type_list = ['HiChIP', 'HiC']
    elif cell_line == 'hESC':
        assay_type_list = ['MicroC', 'HiCAR']
    elif cell_line == 'mESC':
        assay_type_list = ['HiChIP']

    df_R_NLL = pd.DataFrame(columns=['Cell', 'Method', 'Genes', 'Number of Genes', '3D Data', 'FDR', 'R', 'NLL'])

    for assay_type in assay_type_list:
        for fdr in ['1', '01', '001']:

            df = pd.read_csv(data_path+'/results/csv/cage_prediction/epi_models/cage_predictions_epi_models_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.csv', sep='\t')
            if cell_line == 'mESC':
                df_chr2 = df[df['chr'] == 'chr2']
                len_chr2 = len(df_chr2)//2
                idx_repeated = df_chr2.index[-len_chr2:]
                df = df.drop(idx_repeated).reset_index()
            
            df['pred_cage_epi_graphreg_log2'] = np.log2(df['pred_cage_epi_graphreg']+1)
            df['pred_cage_epi_cnn_log2'] = np.log2(df['pred_cage_epi_cnn']+1)
            df['true_cage_log2'] = np.log2(df['true_cage']+1)
            df['n_contact_log2'] = np.log2(df['n_contact']+1)

            df_expressed = df[df['true_cage']>=5].reset_index()
            df_expressed_interacting = df[((df['true_cage']>=5) & (df['n_contact']>=1))].reset_index()

            # NLL
            nll_graphreg_all = df['nll_epi_graphreg'].values
            nll_graphreg_all_mean = np.mean(nll_graphreg_all)
            nll_graphreg_ex = df_expressed['nll_epi_graphreg'].values
            nll_graphreg_ex_mean = np.mean(nll_graphreg_ex)
            nll_graphreg_ex_int = df_expressed_interacting['nll_epi_graphreg'].values
            nll_graphreg_ex_int_mean = np.mean(nll_graphreg_ex_int)

            nll_cnn_all = df['nll_epi_cnn'].values
            nll_cnn_all_mean = np.mean(nll_cnn_all)
            nll_cnn_ex = df_expressed['nll_epi_cnn'].values
            nll_cnn_ex_mean = np.mean(nll_cnn_ex)
            nll_cnn_ex_int = df_expressed_interacting['nll_epi_cnn'].values
            nll_cnn_ex_int_mean = np.mean(nll_cnn_ex_int)

            # R
            r_graphreg_all = np.corrcoef(df['true_cage_log2'].values, df['pred_cage_epi_graphreg_log2'])[0,1]
            r_graphreg_ex = np.corrcoef(df_expressed['true_cage_log2'].values, df_expressed['pred_cage_epi_graphreg_log2'])[0,1]
            r_graphreg_ex_int = np.corrcoef(df_expressed_interacting['true_cage_log2'].values, df_expressed_interacting['pred_cage_epi_graphreg_log2'])[0,1]

            r_cnn_all = np.corrcoef(df['true_cage_log2'].values, df['pred_cage_epi_cnn_log2'])[0,1]
            r_cnn_ex = np.corrcoef(df_expressed['true_cage_log2'].values, df_expressed['pred_cage_epi_cnn_log2'])[0,1]
            r_cnn_ex_int = np.corrcoef(df_expressed_interacting['true_cage_log2'].values, df_expressed_interacting['pred_cage_epi_cnn_log2'])[0,1]

            # plot scatterplots
            sns.set_style("whitegrid")
            fig, ax = plt.subplots()
            g = sns.scatterplot(data=df_expressed, x="n_contact", y="delta_nll", hue='true_cage_log2', alpha=.5, palette=plt.cm.get_cmap('viridis_r'), ax=ax)
            norm = plt.Normalize(df_expressed['true_cage_log2'].min(), df_expressed['true_cage_log2'].max())
            sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=norm)
            sm.set_array([])
            ax.get_legend().remove()
            ax.figure.colorbar(sm, label="log2 (cage + 1)")
            ax.set_title('{} | Mean Delta NLL = {:6.2f}'.format(cell_line, df_expressed['delta_nll'].mean()))
            ax.set_xlabel('Number of Contacts')
            ax.set_ylabel('Delta NLL')
            plt.tight_layout()
            plt.savefig('../figs/Epi-models/final/scatterplot/scatterplot_delta_nll_hue_cage_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.pdf')
            plt.close()

            fig, ax = plt.subplots()
            g = sns.scatterplot(data=df_expressed, x="true_cage_log2", y="pred_cage_epi_cnn_log2", hue='n_contact_log2', alpha=.5, palette=plt.cm.get_cmap('viridis_r'), ax=ax)
            norm = plt.Normalize(df_expressed['n_contact_log2'].min(), df_expressed['n_contact_log2'].max())
            sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=norm)
            sm.set_array([])
            ax.get_legend().remove()
            ax.figure.colorbar(sm, label="log2 (N + 1)")
            ax.set_title('Epi-CNN | {} | R = {:5.3f} | NLL = {:6.2f}'.format(cell_line, r_cnn_ex, nll_cnn_ex_mean))
            ax.set_xlabel('log2 (true + 1)')
            ax.set_ylabel('log2 (pred + 1)')
            plt.tight_layout()
            plt.savefig('../figs/Epi-models/final/scatterplot/scatterplot_cage_vs_pred_epi_cnn_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.pdf')
            plt.close()

            fig, ax = plt.subplots()
            g = sns.scatterplot(data=df_expressed, x="true_cage_log2", y="pred_cage_epi_graphreg_log2", hue='n_contact_log2', alpha=.5, palette=plt.cm.get_cmap('viridis_r'), ax=ax)
            norm = plt.Normalize(df_expressed['n_contact_log2'].min(), df_expressed['n_contact_log2'].max())
            sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=norm)
            sm.set_array([])
            ax.get_legend().remove()
            ax.figure.colorbar(sm, label="log2 (N + 1)")
            ax.set_title('Epi-GraphReg | {} | R = {:5.3f} | NLL = {:6.2f}'.format(cell_line, r_graphreg_ex, nll_graphreg_ex_mean))
            ax.set_xlabel('log2 (true + 1)')
            ax.set_ylabel('log2 (pred + 1)')
            plt.tight_layout()
            plt.savefig('../figs/Epi-models/final/scatterplot/scatterplot_cage_vs_pred_epi_graphreg_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.pdf')
            plt.close()

            # append dataframe

            df_R_NLL = df_R_NLL.append({'Cell': cell_line, 'Method': 'Epi-GraphReg', 'Genes': 'All',
                            'Number of Genes': len(df), '3D Data': assay_type, 'FDR': qval_dict[fdr], 'R': r_graphreg_all, 'NLL': nll_graphreg_all_mean}, ignore_index=True)
            df_R_NLL = df_R_NLL.append({'Cell': cell_line, 'Method': 'Epi-GraphReg', 'Genes': 'Expressed', 
                            'Number of Genes': len(df_expressed), '3D Data': assay_type, 'FDR': qval_dict[fdr], 'R': r_graphreg_ex, 'NLL': nll_graphreg_ex_mean}, ignore_index=True)
            df_R_NLL = df_R_NLL.append({'Cell': cell_line, 'Method': 'Epi-GraphReg', 'Genes': 'Interacting',
                            'Number of Genes': len(df_expressed_interacting), '3D Data': assay_type, 'FDR': qval_dict[fdr], 'R': r_graphreg_ex_int, 'NLL': nll_graphreg_ex_int_mean}, ignore_index=True)

            df_R_NLL = df_R_NLL.append({'Cell': cell_line, 'Method': 'Epi-CNN', 'Genes': 'All',
                        'Number of Genes': len(df), '3D Data': assay_type, 'FDR': qval_dict[fdr], 'R': r_cnn_all, 'NLL': nll_cnn_all_mean}, ignore_index=True)
            df_R_NLL = df_R_NLL.append({'Cell': cell_line, 'Method': 'Epi-CNN', 'Genes': 'Expressed',
                        'Number of Genes': len(df_expressed), '3D Data': assay_type, 'FDR': qval_dict[fdr], 'R': r_cnn_ex, 'NLL': nll_cnn_ex_mean}, ignore_index=True)
            df_R_NLL = df_R_NLL.append({'Cell': cell_line, 'Method': 'Epi-CNN', 'Genes': 'Interacting',
                        'Number of Genes': len(df_expressed_interacting), '3D Data': assay_type, 'FDR': qval_dict[fdr], 'R': r_cnn_ex_int, 'NLL': nll_cnn_ex_int_mean}, ignore_index=True)

    df_R_NLL_all = df_R_NLL_all.append(df_R_NLL).reset_index(drop=True)

    # R
    sns.set_style("darkgrid")
    g = sns.catplot(x="FDR", y="R",
                hue="Method", row='3D Data', col="Genes",
                data=df_R_NLL, kind="swarm", palette={"Epi-GraphReg": "orange", "Epi-CNN": "deepskyblue"},
                height=3, aspect=1, sharey=False)
    g.fig.savefig('../figs/Epi-models/final/swarmplot_R_'+cell_line+'_check_3D_and_fdr.pdf')
    g.fig.clf()

    # NLL
    g = sns.catplot(x="FDR", y="NLL",
                hue="Method", row='3D Data', col="Genes",
                data=df_R_NLL, kind="swarm", palette={"Epi-GraphReg": "orange", "Epi-CNN": "deepskyblue"},
                height=3, aspect=1, sharey=False)
    g.fig.savefig('../figs/Epi-models/final/swarmplot_NLL_'+cell_line+'_check_3D_and_fdr.pdf')
    g.fig.clf()

    # Number of genes
    df_R_NLL_remove_cnn = df_R_NLL[df_R_NLL['Method']=='Epi-GraphReg']
    g = sns.catplot(x="FDR", y="Number of Genes",
                row='3D Data', col="Genes",
                data=df_R_NLL_remove_cnn, kind="bar", palette={0.1: "blue", 0.01: "blue", 0.001: "blue"},
                height=3, aspect=1, sharey=True)
    g.fig.savefig('../figs/Epi-models/final/swarmplot_n_genes_'+cell_line+'.pdf')
    g.fig.clf()



# Only GraphReg
df_only_GR = df_R_NLL_all[(df_R_NLL_all['Method'] == 'Epi-GraphReg') & (df_R_NLL_all['Genes'] != 'Interacting')]
g = sns.catplot(x="FDR", y="R", hue="3D Data",
            row='Genes', col="Cell",
            data=df_only_GR, kind="swarm",
            height=3, aspect=1, sharey=False)
g.fig.savefig('../figs/Epi-models/final/swarmplot_R_only_graphreg_check_3D_and_fdr.pdf')
g.fig.clf()

g = sns.catplot(x="FDR", y="NLL", hue="3D Data",
            row='Genes', col="Cell",
            data=df_only_GR, kind="swarm",
            height=3, aspect=1, sharey=False)
g.fig.savefig('../figs/Epi-models/final/swarmplot_NLL_only_graphreg_check_3D_and_fdr.pdf')
g.fig.clf()

#%%
##### Log-fold change #####

data_path = '/media/labuser/STORAGE/GraphReg'   # data path
qval_dict = {'1': 0.1, '01': 0.01, '001': 0.001}
cell_line_1 = 'GM12878'
cell_line_2 = 'K562'

if cell_line_1 in ['K562', 'GM12878']:
    assay_type_list_1 = ['HiChIP', 'HiC']
elif cell_line_1 == 'hESC':
    assay_type_list_1 = ['MicroC', 'HiCAR']

if cell_line_2 in ['K562', 'GM12878']:
    assay_type_list_2 = ['HiChIP', 'HiC']
elif cell_line_2 == 'hESC':
    assay_type_list_2 = ['MicroC', 'HiCAR']


df_R_MSE = pd.DataFrame(columns=['Cell', 'Method', 'Genes', 'Number of Genes', '3D Data', 'FDR', 'R', 'MSE'])

for assay_type_1 in assay_type_list_1:
    for assay_type_2 in assay_type_list_2:
        for fdr_1 in ['1', '01', '001']:
            for fdr_2 in ['1', '01', '001']:

                df_1 = pd.read_csv(data_path+'/results/csv/cage_prediction/epi_models/cage_predictions_epi_models_'+cell_line_1+'_'+assay_type_1+'_FDR_'+fdr_1+'.csv', sep='\t')
                df_2 = pd.read_csv(data_path+'/results/csv/cage_prediction/epi_models/cage_predictions_epi_models_'+cell_line_2+'_'+assay_type_2+'_FDR_'+fdr_2+'.csv', sep='\t')
                assert (df_1['genes'] == df_2['genes']).all()
                df = df_1[['chr', 'genes', 'n_tss', 'tss', 'tss_distance_from_center']]

                df['n_contact_1'] = df_1['n_contact']
                df['n_contact_2'] = df_2['n_contact']
                df['n_contact_min'] = np.minimum(df_1['n_contact'].values, df_2['n_contact'].values)
                df['n_contact_min_log2'] = np.log2(df['n_contact_min']+1)

                df['true_cage_1'] = df_1['true_cage']
                df['true_cage_2'] = df_2['true_cage']
                df['true_cage_min'] = np.minimum(df_1['true_cage'].values, df_2['true_cage'].values)
                df['true_cage_logfc'] = np.log2((df_1['true_cage']+1)/(df_2['true_cage']+1))

                df['pred_cage_epi_graphreg_1'] = df_1['pred_cage_epi_graphreg']
                df['pred_cage_epi_graphreg_2'] = df_2['pred_cage_epi_graphreg']
                df['pred_cage_epi_graphreg_logfc'] = np.log2((df_1['pred_cage_epi_graphreg']+1)/(df_2['pred_cage_epi_graphreg']+1))

                df['pred_cage_epi_cnn_1'] = df_1['pred_cage_epi_cnn']
                df['pred_cage_epi_cnn_2'] = df_2['pred_cage_epi_cnn']
                df['pred_cage_epi_cnn_logfc'] = np.log2((df_1['pred_cage_epi_cnn']+1)/(df_2['pred_cage_epi_cnn']+1))

                df_expressed = df[df['true_cage_min']>=5].reset_index(drop=True)
                df_expressed_interacting = df[((df['true_cage_min']>=5) & (df['n_contact_min']>=1))].reset_index(drop=True)


                # MSE
                mse_graphreg_all = np.mean((df['pred_cage_epi_graphreg_logfc'].values - df['true_cage_logfc'].values)**2)
                mse_graphreg_ex = np.mean((df_expressed['pred_cage_epi_graphreg_logfc'].values - df_expressed['true_cage_logfc'].values)**2)
                mse_graphreg_ex_int = np.mean((df_expressed_interacting['pred_cage_epi_graphreg_logfc'].values - df_expressed_interacting['true_cage_logfc'].values)**2)

                mse_cnn_all = np.mean((df['pred_cage_epi_cnn_logfc'].values - df['true_cage_logfc'].values)**2)
                mse_cnn_ex = np.mean((df_expressed['pred_cage_epi_cnn_logfc'].values - df_expressed['true_cage_logfc'].values)**2)
                mse_cnn_ex_int = np.mean((df_expressed_interacting['pred_cage_epi_cnn_logfc'].values - df_expressed_interacting['true_cage_logfc'].values)**2)

                # R
                r_graphreg_all = np.corrcoef(df['true_cage_logfc'].values, df['pred_cage_epi_graphreg_logfc'])[0,1]
                r_graphreg_ex = np.corrcoef(df_expressed['true_cage_logfc'].values, df_expressed['pred_cage_epi_graphreg_logfc'])[0,1]
                r_graphreg_ex_int = np.corrcoef(df_expressed_interacting['true_cage_logfc'].values, df_expressed_interacting['pred_cage_epi_graphreg_logfc'])[0,1]

                r_cnn_all = np.corrcoef(df['true_cage_logfc'].values, df['pred_cage_epi_cnn_logfc'])[0,1]
                r_cnn_ex = np.corrcoef(df_expressed['true_cage_logfc'].values, df_expressed['pred_cage_epi_cnn_logfc'])[0,1]
                r_cnn_ex_int = np.corrcoef(df_expressed_interacting['true_cage_logfc'].values, df_expressed_interacting['pred_cage_epi_cnn_logfc'])[0,1]

                # plot scatterplots
                sns.set_style("whitegrid")
                fig, ax = plt.subplots(nrows=1, ncols=1)
                g = sns.scatterplot(data=df_expressed, x="true_cage_logfc", y="pred_cage_epi_cnn_logfc", hue='n_contact_min_log2', alpha=.5, palette=plt.cm.get_cmap('viridis_r'), ax=ax)
                norm = plt.Normalize(df_expressed['n_contact_min_log2'].min(), df_expressed['n_contact_min_log2'].max())
                sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=norm)
                sm.set_array([])
                ax.get_legend().remove()
                ax.figure.colorbar(sm, label="log2 (M + 1)")
                ax.set_title('Epi-CNN | {}/{} | R = {:5.3f} | MSE = {:6.2f}'.format(cell_line_1, cell_line_2, r_cnn_ex, mse_cnn_ex))
                ax.set_xlabel('log2 (FC true)', fontsize=20)
                ax.set_ylabel('log2 (FC pred)', fontsize=20)
                #ax.yaxis.set_tick_params(labelsize=20)
                #ax.xaxis.set_tick_params(labelsize=20)
                plt.tight_layout()
                plt.savefig('../figs/Epi-models/final/scatterplot/scatterplot_logfc_epi_cnn_'+cell_line_1+'_'+assay_type_1+'_FDR_'+fdr_1+'_to_'+cell_line_2+'_'+assay_type_2+'_FDR_'+fdr_2+'.pdf')
                plt.close()

                fig, ax = plt.subplots(nrows=1, ncols=1)
                g = sns.scatterplot(data=df_expressed, x="true_cage_logfc", y="pred_cage_epi_graphreg_logfc", hue='n_contact_min_log2', alpha=.5, palette=plt.cm.get_cmap('viridis_r'), ax=ax)
                norm = plt.Normalize(df_expressed['n_contact_min_log2'].min(), df_expressed['n_contact_min_log2'].max())
                sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=norm)
                sm.set_array([])
                ax.get_legend().remove()
                ax.figure.colorbar(sm, label="log2 (M + 1)")
                ax.set_title('Epi-GraphReg | {}/{} | R = {:5.3f} | MSE = {:6.2f}'.format(cell_line_1, cell_line_2, r_graphreg_ex, mse_graphreg_ex))
                ax.set_xlabel('log2 (FC true)', fontsize=20)
                ax.set_ylabel('log2 (FC pred)', fontsize=20)
                #ax.yaxis.set_tick_params(labelsize=20)
                #ax.xaxis.set_tick_params(labelsize=20)
                plt.tight_layout()
                plt.savefig('../figs/Epi-models/final/scatterplot/scatterplot_logfc_epi_graphreg_'+cell_line_1+'_'+assay_type_1+'_FDR_'+fdr_1+'_to_'+cell_line_2+'_'+assay_type_2+'_FDR_'+fdr_2+'.pdf')
                plt.close()

                # append dataframe

                df_R_MSE = df_R_MSE.append({'Cell': cell_line_1+' / '+cell_line_2, 'Method': 'Epi-GraphReg', 'Genes': 'Expressed', 
                                'Number of Genes': len(df_expressed), '3D Data': assay_type_1+' / '+assay_type_2, 'FDR': str(qval_dict[fdr_1])+' / '+str(qval_dict[fdr_2]), 'R': r_graphreg_ex, 'MSE': mse_graphreg_ex}, ignore_index=True)
                df_R_MSE = df_R_MSE.append({'Cell': cell_line_1+' / '+cell_line_2, 'Method': 'Epi-GraphReg', 'Genes': 'Interacting',
                                'Number of Genes': len(df_expressed_interacting), '3D Data': assay_type_1+' / '+assay_type_2, 'FDR': str(qval_dict[fdr_1])+' / '+str(qval_dict[fdr_2]), 'R': r_graphreg_ex_int, 'MSE': mse_graphreg_ex_int}, ignore_index=True)

                df_R_MSE = df_R_MSE.append({'Cell': cell_line_1+' / '+cell_line_2, 'Method': 'Epi-CNN', 'Genes': 'Expressed',
                            'Number of Genes': len(df_expressed), '3D Data': assay_type_1+' / '+assay_type_2, 'FDR': str(qval_dict[fdr_1])+' / '+str(qval_dict[fdr_2]), 'R': r_cnn_ex, 'MSE': mse_cnn_ex}, ignore_index=True)
                df_R_MSE = df_R_MSE.append({'Cell': cell_line_1+' / '+cell_line_2, 'Method': 'Epi-CNN', 'Genes': 'Interacting',
                            'Number of Genes': len(df_expressed_interacting), '3D Data': assay_type_1+' / '+assay_type_2, 'FDR': str(qval_dict[fdr_1])+' / '+str(qval_dict[fdr_2]), 'R': r_cnn_ex_int, 'MSE': mse_cnn_ex_int}, ignore_index=True)

                del df
# R
sns.set_style("darkgrid")
g = sns.catplot(x="FDR", y="R",
            hue="Method", row='3D Data', col="Genes",
            data=df_R_MSE, kind="swarm", palette={"Epi-GraphReg": "orange", "Epi-CNN": "deepskyblue"},
            height=3, aspect=2, sharey=False)
loc, labels = plt.xticks()
g.set_xticklabels(labels, rotation=45)
g.fig.savefig('../figs/Epi-models/final/swarmplot_logfc_R_'+cell_line_1+'_to_'+cell_line_2+'_check_3D_and_fdr.pdf', bbox_inches='tight')
g.fig.clf()

# MSE
g = sns.catplot(x="FDR", y="MSE",
            hue="Method", row='3D Data', col="Genes",
            data=df_R_MSE, kind="swarm", palette={"Epi-GraphReg": "orange", "Epi-CNN": "deepskyblue"},
            height=3, aspect=2, sharey=False)
loc, labels = plt.xticks()
g.set_xticklabels(labels, rotation=45)
g.fig.savefig('../figs/Epi-models/final/swarmplot_logfc_MSE_'+cell_line_1+'_to_'+cell_line_2+'_check_3D_and_fdr.pdf', bbox_inches='tight')
g.fig.clf()

#%%
##### plot heatmaps for the best genes predicted by GraphReg

data_path = '/media/labuser/STORAGE/GraphReg'   # data path

cell_line = 'K562'
assay_type = 'HiChIP'
fdr = '1'
df = pd.read_csv(data_path+'/results/csv/cage_prediction/epi_models/cage_predictions_epi_models_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.csv', sep='\t')
df['true_cage_log2'] = np.log2(df['true_cage']+1)
df['n_contact_log2'] = np.log2(df['n_contact']+1)

df_expressed = df[df['true_cage']>=5].reset_index(drop=True)
df_expressed_sorted = df_expressed.sort_values(by=['delta_nll']).reset_index(drop=True)
df_expressed_sorted['Gene Index'] = np.arange(len(df_expressed_sorted))

sns.set_style("whitegrid")
fig, ax = plt.subplots()
g = sns.scatterplot(data=df_expressed_sorted, x="Gene Index", y="delta_nll", hue='n_contact', alpha=.5, palette=plt.cm.get_cmap('viridis_r'), size='n_contact', ax=ax)
norm = plt.Normalize(df_expressed_sorted['n_contact'].min(), df_expressed_sorted['n_contact'].max())
sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=norm)
sm.set_array([])
ax.get_legend().remove()
ax.figure.colorbar(sm, label="Number of Contacts")
ax.set_title('{}'.format(cell_line))
ax.set_xlabel('Gene Index')
ax.set_ylabel('Delta NLL')
plt.tight_layout()
plt.savefig('../figs/Epi-models/final/scatterplot/scatterplot_delta_nll_sorted_hue_n_contact_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.pdf')
plt.close()

nll_threshod = 800
delta_nll_threshold = 1000
df_expressed_sorted = df_expressed.sort_values(by=['delta_nll'], ascending=False).reset_index(drop=True)
df_expressed_sorted['Gene Index'] = np.arange(len(df_expressed_sorted))
df_expressed_sorted = df_expressed_sorted[((np.abs(df_expressed_sorted['delta_nll']) > delta_nll_threshold) & ((df_expressed_sorted['nll_epi_graphreg'] < nll_threshod) | (df_expressed_sorted['nll_epi_cnn'] < nll_threshod)))]
df_expressed_sorted.index = df_expressed_sorted['genes']

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(ncols=5, sharey=True, figsize=(15, 15))
max_nll = np.max(df_expressed_sorted['delta_nll'].values)
max_cage = np.max(df_expressed_sorted['true_cage'].values)
min_cage = np.min(df_expressed_sorted['true_cage'].values)
sns.heatmap(df_expressed_sorted[['delta_nll']], cmap="coolwarm", yticklabels=1, ax=ax1, vmax=max_nll, vmin=-max_nll,
            cbar_kws={"orientation": "vertical"})
ax1.set_xticklabels(['Delta NLL'])

sns.heatmap(df_expressed_sorted[['n_contact_log2']], cmap="viridis_r", yticklabels=1, ax=ax2,
            cbar_kws={"orientation": "vertical"})
ax2.set_xticklabels(['log2(N+1)'])

sns.heatmap(df_expressed_sorted[['true_cage']], cmap="viridis_r", yticklabels=1, ax=ax3,
            cbar_kws={"orientation": "vertical"})
ax3.set_xticklabels(['CAGE'])

sns.heatmap(df_expressed_sorted[['pred_cage_epi_graphreg']], cmap="viridis_r", yticklabels=1, ax=ax4, vmax=max_cage, vmin=min_cage,
            cbar_kws={"orientation": "vertical"})
ax4.set_xticklabels(['Epi-GraphReg'])

sns.heatmap(df_expressed_sorted[['pred_cage_epi_cnn']], cmap="viridis_r", yticklabels=1, ax=ax5, vmax=max_cage, vmin=min_cage,
            cbar_kws={"orientation": "vertical"})
ax5.set_xticklabels(['Epi-CNN'])

plt.tight_layout()
fig.savefig('../figs/Epi-models/final/scatterplot/heatmap_best_genes_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.pdf', bbox_inches='tight')

