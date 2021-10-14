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
        df1 = pd.read_csv(data_path+'/results/csv/cage_prediction/epi_models_R_NLL_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.csv', sep='\t')
        df = df.append(df1, ignore_index=True)

cell_line = 'GM12878'
for assay_type in ['HiChIP', 'HiC']:
    for fdr in ['1', '01', '001']:
        df1 = pd.read_csv(data_path+'/results/csv/cage_prediction/epi_models_R_NLL_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.csv', sep='\t')
        df = df.append(df1, ignore_index=True)

cell_line = 'hESC'
for assay_type in ['MicroC', 'HiCAR']:
    for fdr in ['1', '01', '001']:
        df1 = pd.read_csv(data_path+'/results/csv/cage_prediction/epi_models_R_NLL_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.csv', sep='\t')
        df = df.append(df1, ignore_index=True)

cell_line = 'mESC'
for assay_type in ['HiChIP']:
    for fdr in ['1', '01', '001']:
        df1 = pd.read_csv(data_path+'/results/csv/cage_prediction/epi_models_R_NLL_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.csv', sep='\t')
        df = df.append(df1, ignore_index=True)


df = df.dropna().reset_index()
df = df.drop(['index'], axis=1)

df = df.rename(columns={"cell": "Cell", "3D_data": "3D data", "n_gene_test": "Number of Genes", "Set": "Genes"})


g = sns.catplot(x="FDR", y="R",
                hue="3D data", row='Genes', col="Cell",
                data=df, kind="box",
                height=4, aspect=1, sharey=False)
plt.savefig('../figs/Epi-models/boxplot_R_check_3D_and_fdr.pdf')

g = sns.catplot(x="FDR", y="NLL",
                hue="3D data", row='Genes', col="Cell",
                data=df, kind="box",
                height=4, aspect=1, sharey=False)
plt.savefig('../figs/Epi-models/boxplot_NLL_check_3D_and_fdr.pdf')

g = sns.catplot(x="FDR", y="Number of Genes",
                hue="3D data", row='Genes', col="Cell",
                data=df, kind="swarm",
                height=4, aspect=1, sharey=False)
plt.savefig('../figs/Epi-models/boxplot_n_genes.pdf')


##### universal analysis of CAGE predictions #####

data_path = '/media/labuser/STORAGE/GraphReg'   # data path
cell_line = 'K562'
assay_type = 'HiChIP'
fdr = '1'              # 1/01/001

df = pd.read_csv(data_path+'/results/csv/cage_prediction/'+cell_line+'_cage_predictions_epi_models_'+assay_type+'_FDR_'+fdr+'.csv', sep='\t')
df['pred_cage_epi_graphreg_log2'] = np.log2(df['pred_cage_epi_graphreg']+1)
df['pred_cage_epi_cnn_log2'] = np.log2(df['pred_cage_epi_cnn']+1)
df['true_cage_log2'] = np.log2(df['true_cage']+1)

df_expressed = df[df['true_cage']>=5].reset_index()
df_expressed_interacted = df[((df['true_cage']>=5) & (df['n_contact']>=1))].reset_index()


g = sns.scatterplot(data=df_expressed_interacted, x="n_contact", y="delta_nll", hue='true_cage', alpha=.7)
plt.savefig('../figs/Epi-models/boxplot_K562_delta_nll_hue_cage.pdf')

g = sns.scatterplot(data=df_expressed_interacted, x="n_contact", y="delta_nll", hue='average_h3k4me3', alpha=.7)
plt.savefig('../figs/Epi-models/boxplot_K562_delta_nll_hue_h3k4me3.pdf')

g = sns.scatterplot(data=df_expressed_interacted, x="n_contact", y="delta_nll", hue='tss_distance_from_center', alpha=.7)
plt.savefig('../figs/Epi-models/boxplot_K562_delta_nll_hue_dist.pdf')

g = sns.scatterplot(data=df_expressed_interacted, x="tss_distance_from_center", y="delta_nll", hue='true_cage', alpha=.7)
plt.savefig('../figs/Epi-models/boxplot_K562_y_delta_nll_x_dist_hue_dist.pdf')

g = sns.scatterplot(data=df_expressed_interacted, x="tss_distance_from_center", y="delta_nll", hue='n_tss', alpha=.7, legend='full')
plt.savefig('../figs/Epi-models/boxplot_K562_y_delta_nll_x_dist_hue_ntss.pdf')

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

g = sns.scatterplot(data=df_expressed_interacted, x="true_cage_log2", y="pred_cage_epi_graphreg_log2", hue='n_contact', alpha=.7)
g = sns.scatterplot(data=df_expressed_interacted, x="true_cage_log2", y="pred_cage_epi_cnn_log2", hue='n_contact', alpha=.7)



g = sns.catplot(y="nll_epi_cnn",
                data=df, kind="violin",
                height=4, aspect=1, sharey=False)





