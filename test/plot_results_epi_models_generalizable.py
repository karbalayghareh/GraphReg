import numpy as np
from numpy.core.numeric import False_
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statannot import add_stat_annotation
from scipy.stats import spearmanr



##### check the effects of different 3D data and FDRs #####

data_path = '/media/labuser/STORAGE/GraphReg'   # data path
df = pd.DataFrame(columns=['Cell_train', 'Cell_test', 'Method', 'Set', 'valid_chr', 'test_chr', 'n_gene_test', '3D_data_train', '3D_data_test', 'FDR_train', 'FDR_test', 'R','NLL'])

cell_line_train_list = ['K562', 'GM12878']
cell_line_test_list = ['GM12878', 'K562']
fdr_dict = {'1': 0.1, '01': 0.01, '001': 0.001}

for c in range(2):
    cell_line_train = cell_line_train_list[c]
    cell_line_test = cell_line_test_list[c]

    if cell_line_test == 'GM12878' or cell_line_test == 'K562':
        genome='hg19'
        assay_type_test_list = ['HiC', 'HiChIP']
    elif cell_line_test == 'hESC':
        genome='hg38'
        assay_type_test_list = ['MicroC', 'HiCAR']

    if cell_line_train == 'GM12878' or cell_line_train == 'K562':
        assay_type_train_list = ['HiC', 'HiChIP']
    elif cell_line_train == 'hESC':
        assay_type_train_list = ['MicroC', 'HiCAR']

    for assay_type_train in assay_type_train_list:
        for assay_type_test in assay_type_test_list:
            for fdr_train in ['1', '01', '001']:
                for fdr_test in ['1', '01', '001']:

                    df1 = pd.read_csv(data_path+'/results/csv/cage_prediction/cell_to_cell/R_NLL_epi_models_'+cell_line_train+'_'+assay_type_train+'_FDR_'+fdr_train+'_to_'+cell_line_test+'_'+assay_type_test+'_FDR_'+fdr_test+'.csv', sep='\t')
                    df = df.append(df1, ignore_index=True)


            df_sub = df[(df['Cell_train']==cell_line_train) & (df['3D_data_train']==assay_type_train) & (df['3D_data_test']==assay_type_test)]

            g = sns.catplot(x="Set", y="R",
                            hue="Method", row="FDR_train", col="FDR_test",
                            data=df_sub, kind="box",
                            height=4, aspect=1, palette={"Epi-GraphReg": "orange", "Epi-CNN": "deepskyblue"}, sharey=False)

            n_row, n_col = g.axes.shape
            for i in range(n_row):
                for j in range(n_col):
                    df_annt = df_sub[(df_sub['FDR_train'] == g.row_names[i]) & (df_sub['FDR_test'] == g.col_names[j])]
                    add_stat_annotation(g.axes[i,j], data=df_annt, x='Set', y='R', hue='Method',
                                        box_pairs=[(("All", "Epi-GraphReg"), ("All", "Epi-CNN")),
                                                    (("Expressed", "Epi-GraphReg"), ("Expressed", "Epi-CNN")),
                                                    (("Interacted", "Epi-GraphReg"), ("Interacted", "Epi-CNN"))],
                                        test='Wilcoxon', text_format='star', loc='inside', verbose=0, order=['All', 'Expressed', 'Interacted'], fontsize='x-large', comparisons_correction=None)
            g.fig.savefig('../figs/Epi-models/final/cell_to_cell/boxplot_R_check_3D_and_fdr_'+cell_line_train+'_'+assay_type_train+'_to_'+cell_line_test+'_'+assay_type_test+'.pdf')
            g.fig.clf()

            g = sns.catplot(x="Set", y="NLL",
                            hue="Method", row="FDR_train", col="FDR_test",
                            data=df_sub, kind="box",
                            height=4, aspect=1, palette={"Epi-GraphReg": "orange", "Epi-CNN": "deepskyblue"}, sharey=False)

            n_row, n_col = g.axes.shape
            for i in range(n_row):
                for j in range(n_col):
                    df_annt = df_sub[(df_sub['FDR_train'] == g.row_names[i]) & (df_sub['FDR_test'] == g.col_names[j])]
                    add_stat_annotation(g.axes[i,j], data=df_annt, x='Set', y='NLL', hue='Method',
                                        box_pairs=[(("All", "Epi-GraphReg"), ("All", "Epi-CNN")),
                                                    (("Expressed", "Epi-GraphReg"), ("Expressed", "Epi-CNN")),
                                                    (("Interacted", "Epi-GraphReg"), ("Interacted", "Epi-CNN"))],
                                        test='Wilcoxon', text_format='star', loc='inside', verbose=0, order=['All', 'Expressed', 'Interacted'], fontsize='x-large', comparisons_correction=None)
            g.fig.savefig('../figs/Epi-models/final/cell_to_cell/boxplot_NLL_check_3D_and_fdr_'+cell_line_train+'_'+assay_type_train+'_to_'+cell_line_test+'_'+assay_type_test+'.pdf')
            g.fig.clf()


##### universal analysis of CAGE predictions #####

data_path = '/media/labuser/STORAGE/GraphReg'   # data path
cell_line_train = 'K562'
cell_line_test = 'GM12878'
assay_type_train = 'HiChIP'
assay_type_test = 'HiChIP'
fdr_train = '1'              # 1/01/001
fdr_test = '1'              # 1/01/001

df = pd.read_csv(data_path+'/results/csv/cage_prediction/cell_to_cell/cage_predictions_epi_models_'+cell_line_train+'_'+assay_type_train+'_FDR_'+fdr_train+'_to_'+cell_line_test+'_'+assay_type_test+'_FDR_'+fdr_test+'.csv', sep='\t')
df['pred_cage_epi_graphreg_log2'] = np.log2(df['pred_cage_epi_graphreg']+1)
df['pred_cage_epi_cnn_log2'] = np.log2(df['pred_cage_epi_cnn']+1)
df['true_cage_log2'] = np.log2(df['true_cage']+1)

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

fig, ax = plt.subplots()
g = sns.scatterplot(data=df_expressed_interacted, x="true_cage_log2", y="pred_cage_epi_graphreg_log2", hue='n_contact', alpha=.1, palette=plt.cm.get_cmap('viridis_r'), ax=ax)
ax.set_title('Epi-GraphReg/{} to {}: R = {:5.3f}, NLL = {:6.2f}'.format(cell_line_train, cell_line_test, r_graphreg_ex_int, nll_graphreg_ex_int_mean))
ax.set_xlabel('log2 (true + 1)')
ax.set_ylabel('log2 (pred + 1)')
plt.savefig('../figs/Epi-models/final/cell_to_cell/scatterplot/scatterplot_cage_vs_pred_epi_graphreg_'+cell_line_train+'_'+assay_type_train+'_FDR_'+fdr_train+'_to_'+cell_line_test+'_'+assay_type_test+'_FDR_'+fdr_test+'.pdf')

fig, ax = plt.subplots()
g = sns.scatterplot(data=df_expressed_interacted, x="true_cage_log2", y="pred_cage_epi_cnn_log2", hue='n_contact', alpha=.1, palette=plt.cm.get_cmap('viridis_r'), ax=ax)
ax.set_title('Epi-CNN/{} to {}: R = {:5.3f}, NLL = {:6.2f}'.format(cell_line_train, cell_line_test, r_cnn_ex_int, nll_cnn_ex_int_mean))
ax.set_xlabel('log2 (true + 1)')
ax.set_ylabel('log2 (pred + 1)')
plt.savefig('../figs/Epi-models/final/cell_to_cell/scatterplot/scatterplot_cage_vs_pred_epi_cnn_'+cell_line_train+'_'+assay_type_train+'_FDR_'+fdr_train+'_to_'+cell_line_test+'_'+assay_type_test+'_FDR_'+fdr_test+'.pdf')


###############################################################################################################################################
##########################################################            Swarm Plots             #################################################
###############################################################################################################################################


data_path = '/media/labuser/STORAGE/GraphReg'   # data path
qval_dict = {'1': 0.1, '01': 0.01, '001': 0.001}
df_R_NLL_all = pd.DataFrame(columns=['Cell', 'Method', 'Genes', 'Number of Genes', '3D Data', 'FDR', 'R', 'NLL'])

cell_line_train_list = ['K562', 'GM12878']
cell_line_test_list = ['GM12878', 'K562']

for c in range(2):
    cell_line_train = cell_line_train_list[c]
    cell_line_test = cell_line_test_list[c]

    df_R_NLL = pd.DataFrame(columns=['Cell', 'Method', 'Genes', 'Number of Genes', '3D Data', 'FDR', 'R', 'NLL'])

    if cell_line_test == 'GM12878' or cell_line_test == 'K562':
        genome='hg19'
        assay_type_test_list = ['HiC', 'HiChIP']
    elif cell_line_test == 'hESC':
        genome='hg38'
        assay_type_test_list = ['MicroC', 'HiCAR']

    if cell_line_train == 'GM12878' or cell_line_train == 'K562':
        assay_type_train_list = ['HiC', 'HiChIP']
    elif cell_line_train == 'hESC':
        assay_type_train_list = ['MicroC', 'HiCAR']

    for assay_type_train in assay_type_train_list:
        for assay_type_test in assay_type_test_list:
            for fdr_train in ['1', '01', '001']:
                for fdr_test in ['1', '01', '001']:

                    df = pd.read_csv(data_path+'/results/csv/cage_prediction/cell_to_cell/cage_predictions_epi_models_'+cell_line_train+'_'+assay_type_train+'_FDR_'+fdr_train+'_to_'+cell_line_test+'_'+assay_type_test+'_FDR_'+fdr_test+'.csv', sep='\t')
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
                    ax.set_title('{} to {} | Mean Delta NLL = {:6.2f}'.format(cell_line_train, cell_line_test, df_expressed['delta_nll'].mean()))
                    ax.set_xlabel('Number of Contacts')
                    ax.set_ylabel('Delta NLL')
                    plt.tight_layout()
                    plt.savefig('../figs/Epi-models/final/cell_to_cell/scatterplot/scatterplot_delta_nll_hue_cage_'+cell_line_train+'_'+assay_type_train+'_FDR_'+fdr_train+'_to_'+cell_line_test+'_'+assay_type_test+'_FDR_'+fdr_test+'.pdf')
                    plt.close()

                    fig, ax = plt.subplots()
                    g = sns.scatterplot(data=df_expressed, x="true_cage_log2", y="pred_cage_epi_cnn_log2", hue='n_contact_log2', alpha=.2, palette=plt.cm.get_cmap('viridis_r'), ax=ax)
                    norm = plt.Normalize(df_expressed['n_contact_log2'].min(), df_expressed['n_contact_log2'].max())
                    sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=norm)
                    sm.set_array([])
                    ax.get_legend().remove()
                    ax.figure.colorbar(sm, label="log2 (N + 1)")
                    ax.set_title('Epi-CNN | {} to {} | R = {:5.3f} | NLL = {:6.2f}'.format(cell_line_train, cell_line_test, r_cnn_ex, nll_cnn_ex_mean))
                    ax.set_xlabel('log2 (true + 1)')
                    ax.set_ylabel('log2 (pred + 1)')
                    plt.tight_layout()
                    plt.savefig('../figs/Epi-models/final/cell_to_cell/scatterplot/scatterplot_cage_vs_pred_epi_cnn_'+cell_line_train+'_'+assay_type_train+'_FDR_'+fdr_train+'_to_'+cell_line_test+'_'+assay_type_test+'_FDR_'+fdr_test+'.pdf')
                    plt.close()

                    fig, ax = plt.subplots()
                    g = sns.scatterplot(data=df_expressed, x="true_cage_log2", y="pred_cage_epi_graphreg_log2", hue='n_contact_log2', alpha=.2, palette=plt.cm.get_cmap('viridis_r'), ax=ax)
                    norm = plt.Normalize(df_expressed['n_contact_log2'].min(), df_expressed['n_contact_log2'].max())
                    sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=norm)
                    sm.set_array([])
                    ax.get_legend().remove()
                    ax.figure.colorbar(sm, label="log2 (N + 1)")
                    ax.set_title('Epi-GraphReg | {} to {}| R = {:5.3f} | NLL = {:6.2f}'.format(cell_line_train, cell_line_test, r_graphreg_ex, nll_graphreg_ex_mean))
                    ax.set_xlabel('log2 (true + 1)')
                    ax.set_ylabel('log2 (pred + 1)')
                    plt.tight_layout()
                    plt.savefig('../figs/Epi-models/final/cell_to_cell/scatterplot/scatterplot_cage_vs_pred_epi_graphreg_'+cell_line_train+'_'+assay_type_train+'_FDR_'+fdr_train+'_to_'+cell_line_test+'_'+assay_type_test+'_FDR_'+fdr_test+'.pdf')
                    plt.close()

                    # append dataframe

                    df_R_NLL = df_R_NLL.append({'Cell': cell_line_train+' -> '+cell_line_test, 'Method': 'Epi-GraphReg', 'Genes': 'All',
                                    'Number of Genes': len(df), '3D Data': assay_type_train+' -> '+assay_type_test, 'FDR': str(qval_dict[fdr_train])+' -> '+str(qval_dict[fdr_test]), 'R': r_graphreg_all, 'NLL': nll_graphreg_all_mean}, ignore_index=True)
                    df_R_NLL = df_R_NLL.append({'Cell': cell_line_train+' -> '+cell_line_test, 'Method': 'Epi-GraphReg', 'Genes': 'Expressed', 
                                    'Number of Genes': len(df_expressed), '3D Data': assay_type_train+' -> '+assay_type_test, 'FDR': str(qval_dict[fdr_train])+' -> '+str(qval_dict[fdr_test]), 'R': r_graphreg_ex, 'NLL': nll_graphreg_ex_mean}, ignore_index=True)
                    df_R_NLL = df_R_NLL.append({'Cell': cell_line_train+' -> '+cell_line_test, 'Method': 'Epi-GraphReg', 'Genes': 'Interacted',
                                    'Number of Genes': len(df_expressed_interacted), '3D Data': assay_type_train+' -> '+assay_type_test, 'FDR': str(qval_dict[fdr_train])+' -> '+str(qval_dict[fdr_test]), 'R': r_graphreg_ex_int, 'NLL': nll_graphreg_ex_int_mean}, ignore_index=True)

                    df_R_NLL = df_R_NLL.append({'Cell': cell_line_train+' -> '+cell_line_test, 'Method': 'Epi-CNN', 'Genes': 'All',
                                'Number of Genes': len(df), '3D Data': assay_type_train+' -> '+assay_type_test, 'FDR': str(qval_dict[fdr_train])+' -> '+str(qval_dict[fdr_test]), 'R': r_cnn_all, 'NLL': nll_cnn_all_mean}, ignore_index=True)
                    df_R_NLL = df_R_NLL.append({'Cell': cell_line_train+' -> '+cell_line_test, 'Method': 'Epi-CNN', 'Genes': 'Expressed',
                                'Number of Genes': len(df_expressed), '3D Data': assay_type_train+' -> '+assay_type_test, 'FDR': str(qval_dict[fdr_train])+' -> '+str(qval_dict[fdr_test]), 'R': r_cnn_ex, 'NLL': nll_cnn_ex_mean}, ignore_index=True)
                    df_R_NLL = df_R_NLL.append({'Cell': cell_line_train+' -> '+cell_line_test, 'Method': 'Epi-CNN', 'Genes': 'Interacted',
                                'Number of Genes': len(df_expressed_interacted), '3D Data': assay_type_train+' -> '+assay_type_test, 'FDR': str(qval_dict[fdr_train])+' -> '+str(qval_dict[fdr_test]), 'R': r_cnn_ex_int, 'NLL': nll_cnn_ex_int_mean}, ignore_index=True)

    df_R_NLL_all = df_R_NLL_all.append(df_R_NLL).reset_index(drop=True)

    # R
    sns.set_style("darkgrid")
    g = sns.catplot(x="FDR", y="R",
                hue="Method", row='3D Data', col="Genes",
                data=df_R_NLL, kind="swarm", palette={"Epi-GraphReg": "orange", "Epi-CNN": "deepskyblue"},
                height=3, aspect=2, sharey=False)
    loc, labels = plt.xticks()
    g.set_xticklabels(labels, rotation=45)
    g.fig.savefig('../figs/Epi-models/final/cell_to_cell/swarmplot_R_'+cell_line_train+'_to_'+cell_line_test+'_check_3D_and_fdr.pdf', bbox_inches='tight')
    g.fig.clf()

    # NLL
    g = sns.catplot(x="FDR", y="NLL",
                hue="Method", row='3D Data', col="Genes",
                data=df_R_NLL, kind="swarm", palette={"Epi-GraphReg": "orange", "Epi-CNN": "deepskyblue"},
                height=3, aspect=2, sharey=False)
    loc, labels = plt.xticks()
    g.set_xticklabels(labels, rotation=45)
    g.fig.savefig('../figs/Epi-models/final/cell_to_cell/swarmplot_NLL_'+cell_line_train+'_to_'+cell_line_test+'_check_3D_and_fdr.pdf', bbox_inches='tight')
    g.fig.clf()

    # Number of genes
    df_R_NLL_remove_cnn = df_R_NLL[df_R_NLL['Method']=='Epi-GraphReg']
    g = sns.catplot(x="FDR", y="Number of Genes",
                row='3D Data', col="Genes",
                data=df_R_NLL_remove_cnn, kind="bar", #palette={0.1: "blue", 0.01: "blue", 0.001: "blue"},
                height=3, aspect=2, sharey=True)
    loc, labels = plt.xticks()
    g.set_xticklabels(labels, rotation=45)
    g.fig.savefig('../figs/Epi-models/final/cell_to_cell/swarmplot_n_genes_'+cell_line_train+'_to_'+cell_line_test+'.pdf', bbox_inches='tight')
    g.fig.clf()



# Only GraphReg
df_only_GR = df_R_NLL_all[(df_R_NLL_all['Method'] == 'Epi-GraphReg') & (df_R_NLL_all['Genes'] != 'Interacted')]
g = sns.catplot(x="FDR", y="R", hue="3D Data",
            row='Genes', col="Cell",
            data=df_only_GR, kind="swarm",
            height=3, aspect=2, sharey=False)
loc, labels = plt.xticks()
g.set_xticklabels(labels, rotation=45)
g.fig.savefig('../figs/Epi-models/final/cell_to_cell/swarmplot_R_only_graphreg_check_3D_and_fdr.pdf', bbox_inches='tight')
g.fig.clf()

g = sns.catplot(x="FDR", y="NLL", hue="3D Data",
            row='Genes', col="Cell",
            data=df_only_GR, kind="swarm",
            height=3, aspect=2, sharey=False)
loc, labels = plt.xticks()
g.set_xticklabels(labels, rotation=45)
g.fig.savefig('../figs/Epi-models/final/cell_to_cell/swarmplot_NLL_only_graphreg_check_3D_and_fdr.pdf', bbox_inches='tight')
g.fig.clf()


