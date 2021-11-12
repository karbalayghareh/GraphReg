import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statannot import add_stat_annotation


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
                                        (("Interacted", "Epi-GraphReg"), ("Interacted", "Epi-CNN"))],
                            test='Wilcoxon', text_format='star', loc='inside', verbose=0, order=['All', 'Expressed', 'Interacted'], fontsize='x-large', comparisons_correction=None)
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
                                        (("Interacted", "Epi-GraphReg"), ("Interacted", "Epi-CNN"))],
                            test='Wilcoxon', text_format='star', loc='inside', verbose=0, order=['All', 'Expressed', 'Interacted'], fontsize='x-large', comparisons_correction=None)
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
                                        (("Interacted", "Epi-GraphReg"), ("Interacted", "Epi-CNN"))],
                            test='Wilcoxon', text_format='star', loc='inside', verbose=0, order=['All', 'Expressed', 'Interacted'], fontsize='x-large', comparisons_correction=None)
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
                                        (("Interacted", "Epi-GraphReg"), ("Interacted", "Epi-CNN"))],
                            test='Wilcoxon', text_format='star', loc='inside', verbose=0, order=['All', 'Expressed', 'Interacted'], fontsize='x-large', comparisons_correction=None)
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
                                        (("Interacted", "Epi-GraphReg"), ("Interacted", "Epi-CNN"))],
                            test='Wilcoxon', text_format='star', loc='inside', verbose=0, order=['All', 'Expressed', 'Interacted'], fontsize='x-large', comparisons_correction=None)
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
                                        (("Interacted", "Epi-GraphReg"), ("Interacted", "Epi-CNN"))],
                            test='Wilcoxon', text_format='star', loc='inside', verbose=0, order=['All', 'Expressed', 'Interacted'], fontsize='x-large', comparisons_correction=None)
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
                                        (("Interacted", "Epi-GraphReg"), ("Interacted", "Epi-CNN"))],
                            test='Wilcoxon', text_format='star', loc='inside', verbose=0, order=['All', 'Expressed', 'Interacted'], fontsize='x-large', comparisons_correction=None)
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
                                        (("Interacted", "Epi-GraphReg"), ("Interacted", "Epi-CNN"))],
                            test='Wilcoxon', text_format='star', loc='inside', verbose=0, order=['All', 'Expressed', 'Interacted'], fontsize='x-large', comparisons_correction=None)
plt.savefig('../figs/Epi-models/final/boxplot_NLL_mESC_check_3D_and_fdr.pdf')

# Number of genes
g = sns.catplot(x="Genes", y="Number of Genes",
                hue="Method", row='3D data', col="FDR",
                data=df_sub, kind="swarm", palette={"Epi-GraphReg": "orange", "Epi-CNN": "deepskyblue"},
                height=4, aspect=1, sharey=True)
plt.savefig('../figs/Epi-models/final/boxplot_n_genes_mESC.pdf')


## Only GraphReg

## K562
df_sub = df[(df['Method'] == 'Epi-GraphReg') & (df['Genes'] != 'Interacted')]

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



##### universal analysis of CAGE predictions #####

data_path = '/media/labuser/STORAGE/GraphReg'   # data path
cell_line = 'mESC'
assay_type = 'HiChIP'
fdr = '1'              # 1/01/001

df = pd.read_csv(data_path+'/results/csv/cage_prediction/epi_models/cage_predictions_epi_models_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.csv', sep='\t')
df['pred_cage_epi_graphreg_log2'] = np.log2(df['pred_cage_epi_graphreg']+1)
df['pred_cage_epi_cnn_log2'] = np.log2(df['pred_cage_epi_cnn']+1)
df['true_cage_log2'] = np.log2(df['true_cage']+1)

df_expressed = df[df['true_cage']>=5].reset_index()
df_expressed_interacted = df[((df['true_cage']>=5) & (df['n_contact']>=1))].reset_index()

#g = sns.scatterplot(data=df, x="n_contact", y="delta_nll", hue='true_cage', alpha=.7, palette=plt.cm.get_cmap('viridis_r'))
#plt.savefig('../figs/Epi-models/final/scatterplot/scatterplot_delta_nll_hue_cage_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.pdf')

# NLL
nll_graphreg_all = df['nll_epi_graphreg'].values
nll_graphreg_all_mean = np.mean(nll_graphreg_all)
nll_graphreg_ex = df_expressed['nll_epi_graphreg'].values
nll_graphreg_ex_mean = np.mean(nll_graphreg_ex)
nll_graphreg_ex_int = df_expressed_interacted['nll_epi_graphreg'].values
nll_graphreg_ex_int_mean = np.mean(nll_graphreg_ex_int)

nll_cnn_all = df['nll_epi_cnn'].values
nll_cnn_all_mean = np.mean(nll_cnn_all)
nll_cnn_ex = df_expressed['nll_epi_cnn'].values
nll_cnn_ex_mean = np.mean(nll_cnn_ex)
nll_cnn_ex_int = df_expressed_interacted['nll_epi_cnn'].values
nll_cnn_ex_int_mean = np.mean(nll_cnn_ex_int)

# R
r_graphreg_all = np.corrcoef(df['true_cage_log2'].values, df['pred_cage_epi_graphreg_log2'])[0,1]
r_graphreg_ex = np.corrcoef(df_expressed['true_cage_log2'].values, df_expressed['pred_cage_epi_graphreg_log2'])[0,1]
r_graphreg_ex_int = np.corrcoef(df_expressed_interacted['true_cage_log2'].values, df_expressed_interacted['pred_cage_epi_graphreg_log2'])[0,1]

r_cnn_all = np.corrcoef(df['true_cage_log2'].values, df['pred_cage_epi_cnn_log2'])[0,1]
r_cnn_ex = np.corrcoef(df_expressed['true_cage_log2'].values, df_expressed['pred_cage_epi_cnn_log2'])[0,1]
r_cnn_ex_int = np.corrcoef(df_expressed_interacted['true_cage_log2'].values, df_expressed_interacted['pred_cage_epi_cnn_log2'])[0,1]

fig, ax = plt.subplots()
g = sns.scatterplot(data=df_expressed_interacted, x="true_cage_log2", y="pred_cage_epi_graphreg_log2", hue='n_contact', alpha=.1, palette=plt.cm.get_cmap('viridis_r'), ax=ax)
ax.set_title('Epi-GraphReg/{}: R = {:5.3f}, NLL = {:6.2f}'.format(cell_line, r_graphreg_ex_int, nll_graphreg_ex_int_mean))
ax.set_xlabel('log2 (true + 1)')
ax.set_ylabel('log2 (pred + 1)')
plt.savefig('../figs/Epi-models/final/scatterplot/scatterplot_cage_vs_pred_epi_graphreg_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.pdf')

fig, ax = plt.subplots()
g = sns.scatterplot(data=df_expressed_interacted, x="true_cage_log2", y="pred_cage_epi_cnn_log2", hue='n_contact', alpha=.1, palette=plt.cm.get_cmap('viridis_r'), ax=ax)
ax.set_title('Epi-CNN/{}: R = {:5.3f}, NLL = {:6.2f}'.format(cell_line, r_cnn_ex_int, nll_cnn_ex_int_mean))
ax.set_xlabel('log2 (true + 1)')
ax.set_ylabel('log2 (pred + 1)')
plt.savefig('../figs/Epi-models/final/scatterplot/scatterplot_cage_vs_pred_epi_cnn_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.pdf')


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
            df['pred_cage_epi_graphreg_log2'] = np.log2(df['pred_cage_epi_graphreg']+1)
            df['pred_cage_epi_cnn_log2'] = np.log2(df['pred_cage_epi_cnn']+1)
            df['true_cage_log2'] = np.log2(df['true_cage']+1)
            df['n_contact_log2'] = np.log2(df['n_contact']+1)

            df_expressed = df[df['true_cage']>=5].reset_index()
            df_expressed_interacted = df[((df['true_cage']>=5) & (df['n_contact']>=1))].reset_index()

            # NLL
            nll_graphreg_all = df['nll_epi_graphreg'].values
            nll_graphreg_all_mean = np.mean(nll_graphreg_all)
            nll_graphreg_ex = df_expressed['nll_epi_graphreg'].values
            nll_graphreg_ex_mean = np.mean(nll_graphreg_ex)
            nll_graphreg_ex_int = df_expressed_interacted['nll_epi_graphreg'].values
            nll_graphreg_ex_int_mean = np.mean(nll_graphreg_ex_int)

            nll_cnn_all = df['nll_epi_cnn'].values
            nll_cnn_all_mean = np.mean(nll_cnn_all)
            nll_cnn_ex = df_expressed['nll_epi_cnn'].values
            nll_cnn_ex_mean = np.mean(nll_cnn_ex)
            nll_cnn_ex_int = df_expressed_interacted['nll_epi_cnn'].values
            nll_cnn_ex_int_mean = np.mean(nll_cnn_ex_int)

            # R
            r_graphreg_all = np.corrcoef(df['true_cage_log2'].values, df['pred_cage_epi_graphreg_log2'])[0,1]
            r_graphreg_ex = np.corrcoef(df_expressed['true_cage_log2'].values, df_expressed['pred_cage_epi_graphreg_log2'])[0,1]
            r_graphreg_ex_int = np.corrcoef(df_expressed_interacted['true_cage_log2'].values, df_expressed_interacted['pred_cage_epi_graphreg_log2'])[0,1]

            r_cnn_all = np.corrcoef(df['true_cage_log2'].values, df['pred_cage_epi_cnn_log2'])[0,1]
            r_cnn_ex = np.corrcoef(df_expressed['true_cage_log2'].values, df_expressed['pred_cage_epi_cnn_log2'])[0,1]
            r_cnn_ex_int = np.corrcoef(df_expressed_interacted['true_cage_log2'].values, df_expressed_interacted['pred_cage_epi_cnn_log2'])[0,1]

            # plot scatterplots
            sns.set_style("whitegrid")
            fig, ax = plt.subplots()
            g = sns.scatterplot(data=df_expressed, x="n_contact", y="delta_nll", hue='true_cage_log2', alpha=.2, palette=plt.cm.get_cmap('viridis_r'), ax=ax)
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
            g = sns.scatterplot(data=df_expressed, x="true_cage_log2", y="pred_cage_epi_cnn_log2", hue='n_contact_log2', alpha=.2, palette=plt.cm.get_cmap('viridis_r'), ax=ax)
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
            g = sns.scatterplot(data=df_expressed, x="true_cage_log2", y="pred_cage_epi_graphreg_log2", hue='n_contact_log2', alpha=.2, palette=plt.cm.get_cmap('viridis_r'), ax=ax)
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
            df_R_NLL = df_R_NLL.append({'Cell': cell_line, 'Method': 'Epi-GraphReg', 'Genes': 'Interacted',
                            'Number of Genes': len(df_expressed_interacted), '3D Data': assay_type, 'FDR': qval_dict[fdr], 'R': r_graphreg_ex_int, 'NLL': nll_graphreg_ex_int_mean}, ignore_index=True)

            df_R_NLL = df_R_NLL.append({'Cell': cell_line, 'Method': 'Epi-CNN', 'Genes': 'All',
                        'Number of Genes': len(df), '3D Data': assay_type, 'FDR': qval_dict[fdr], 'R': r_cnn_all, 'NLL': nll_cnn_all_mean}, ignore_index=True)
            df_R_NLL = df_R_NLL.append({'Cell': cell_line, 'Method': 'Epi-CNN', 'Genes': 'Expressed',
                        'Number of Genes': len(df_expressed), '3D Data': assay_type, 'FDR': qval_dict[fdr], 'R': r_cnn_ex, 'NLL': nll_cnn_ex_mean}, ignore_index=True)
            df_R_NLL = df_R_NLL.append({'Cell': cell_line, 'Method': 'Epi-CNN', 'Genes': 'Interacted',
                        'Number of Genes': len(df_expressed_interacted), '3D Data': assay_type, 'FDR': qval_dict[fdr], 'R': r_cnn_ex_int, 'NLL': nll_cnn_ex_int_mean}, ignore_index=True)

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
df_only_GR = df_R_NLL_all[(df_R_NLL_all['Method'] == 'Epi-GraphReg') & (df_R_NLL_all['Genes'] != 'Interacted')]
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








