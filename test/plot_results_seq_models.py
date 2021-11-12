import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statannot import add_stat_annotation


##### check the effects of FFT and dilated layers in K562 #####

data_path = '/media/labuser/STORAGE/GraphReg'   # data path
cell_line = 'K562'
assay_type = 'HiChIP'
fdr = '1'

df = pd.DataFrame(columns=['cell', 'Method', 'Train_mode', 'Set', 'valid_chr', 'test_chr', 'n_gene_test', '3D_data', 'FDR', 'R','NLL'])

df1 = pd.read_csv(data_path+'/results/csv/cage_prediction/seq_models/R_NLL_seq_models_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.csv', sep='\t')
df = df.append(df1, ignore_index=True)

df1 = pd.read_csv(data_path+'/results/csv/cage_prediction/seq_models/R_NLL_seq_models_nodilation_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.csv', sep='\t')
df = df.append(df1, ignore_index=True)

df1 = pd.read_csv(data_path+'/results/csv/cage_prediction/seq_models/R_NLL_seq_models_fft_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.csv', sep='\t')
df = df.append(df1, ignore_index=True)

df1 = pd.read_csv(data_path+'/results/csv/cage_prediction/seq_models/R_NLL_seq_models_nodilation_fft_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.csv', sep='\t')
df = df.append(df1, ignore_index=True)

df1 = pd.read_csv(data_path+'/results/csv/cage_prediction/seq_models/R_NLL_seq_e2e_models_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.csv', sep='\t')
df = df.append(df1, ignore_index=True)

# R
g = sns.catplot(x="Train_mode", y="R",
                hue="Method", col='Set',
                data=df, kind="box",
                height=5, aspect=1, sharey=False, palette={"Seq-GraphReg": "orange", "Seq-CNN": "deepskyblue"})
g.set_xticklabels(rotation=15, horizontalalignment='right')

n_row, n_col = g.axes.shape
for j in range(n_col):
    df_annt = df[df['Set'] == g.col_names[j]]
    add_stat_annotation(g.axes[0,j], data=df_annt, x='Train_mode', y='R', hue='Method',
                        box_pairs=[(("fft:no/dilated:yes", "Seq-GraphReg"), ("fft:no/dilated:yes", "Seq-CNN")),
                                    (("fft:no/dilated:no", "Seq-GraphReg"), ("fft:no/dilated:no", "Seq-CNN")),
                                    (("fft:yes/dilated:yes", "Seq-GraphReg"), ("fft:yes/dilated:yes", "Seq-CNN")),
                                    (("fft:yes/dilated:no", "Seq-GraphReg"), ("fft:yes/dilated:no", "Seq-CNN")),
                                    (("e2e", "Seq-GraphReg"), ("e2e", "Seq-CNN"))],
                        test='Wilcoxon', text_format='star', loc='inside', verbose=0, fontsize='x-large', comparisons_correction=None)

plt.savefig('../figs/Seq-models/final/boxplot_R_K562_check_fft_and_dilation.pdf')

# NLL
g = sns.catplot(x="Train_mode", y="NLL",
                hue="Method", col='Set',
                data=df, kind="box",
                height=5, aspect=1, sharey=False, palette={"Seq-GraphReg": "orange", "Seq-CNN": "deepskyblue"})
g.set_xticklabels(rotation=15, horizontalalignment='right')

n_row, n_col = g.axes.shape
for j in range(n_col):
    df_annt = df[df['Set'] == g.col_names[j]]
    add_stat_annotation(g.axes[0,j], data=df_annt, x='Train_mode', y='NLL', hue='Method',
                        box_pairs=[(("fft:no/dilated:yes", "Seq-GraphReg"), ("fft:no/dilated:yes", "Seq-CNN")),
                                    (("fft:no/dilated:no", "Seq-GraphReg"), ("fft:no/dilated:no", "Seq-CNN")),
                                    (("fft:yes/dilated:yes", "Seq-GraphReg"), ("fft:yes/dilated:yes", "Seq-CNN")),
                                    (("fft:yes/dilated:no", "Seq-GraphReg"), ("fft:yes/dilated:no", "Seq-CNN")),
                                    (("e2e", "Seq-GraphReg"), ("e2e", "Seq-CNN"))],
                        test='Wilcoxon', text_format='star', loc='inside', verbose=0, fontsize='x-large', comparisons_correction=None)

plt.savefig('../figs/Seq-models/final/boxplot_NLL_K562_check_fft_and_dilation.pdf')

# add Basenji

df1 = pd.read_csv(data_path+'/results/csv/cage_prediction/seq_models/R_NLL_basenji_'+cell_line+'.csv', sep='\t')
df = df.append(df1, ignore_index=True)

g = sns.catplot(x="Train_mode", y="R",
                hue="Method", col='Set',
                data=df, kind="box",
                height=5, aspect=1, sharey=False, palette={"Seq-GraphReg": "orange", "Seq-CNN": "deepskyblue", "Basenji": "green"})
g.set_xticklabels(rotation=15, horizontalalignment='right')
plt.savefig('../figs/Seq-models/final/boxplot_R_K562_check_fft_and_dilation_with_basenji.pdf')


##### End-to-end Seq-models #####

data_path = '/media/labuser/STORAGE/GraphReg'   # data path
cell_line = 'K562'
assay_type = 'HiChIP'
fdr = '1'

df = pd.DataFrame(columns=['cell', 'Method', 'Train_mode', 'Set', 'valid_chr', 'test_chr', 'n_gene_test', '3D_data', 'FDR', 'R','NLL'])

df1 = pd.read_csv(data_path+'/results/csv/cage_prediction/seq_models/R_NLL_seq_e2e_models_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.csv', sep='\t')
df = df.append(df1, ignore_index=True)

cell_line = 'GM12878'
assay_type = 'HiChIP'
fdr = '1'

df1 = pd.read_csv(data_path+'/results/csv/cage_prediction/seq_models/R_NLL_seq_e2e_models_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.csv', sep='\t')
df = df.append(df1, ignore_index=True)

cell_line = 'hESC'
assay_type = 'MicroC'
fdr = '1'

df1 = pd.read_csv(data_path+'/results/csv/cage_prediction/seq_models/R_NLL_seq_e2e_models_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.csv', sep='\t')
df = df.append(df1, ignore_index=True)

cell_line = 'mESC'
assay_type = 'HiChIP'
fdr = '1'

df1 = pd.read_csv(data_path+'/results/csv/cage_prediction/seq_models/R_NLL_seq_e2e_models_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.csv', sep='\t')
df = df.append(df1, ignore_index=True)

# R
g = sns.catplot(x="Set", y="R",
                hue="Method", col='cell',
                data=df, kind="box",
                height=4, aspect=1, sharey=False, palette={"Seq-GraphReg": "orange", "Seq-CNN": "deepskyblue"})

n_row, n_col = g.axes.shape
for j in range(n_col):
    df_annt = df[df['cell'] == g.col_names[j]]
    add_stat_annotation(g.axes[0,j], data=df_annt, x='Set', y='R', hue='Method',
                        box_pairs=[(("All", "Seq-GraphReg"), ("All", "Seq-CNN")),
                                    (("Expressed", "Seq-GraphReg"), ("Expressed", "Seq-CNN")),
                                    (("Interacted", "Seq-GraphReg"), ("Interacted", "Seq-CNN"))],
                        test='Wilcoxon', text_format='star', loc='inside', verbose=0, order=['All', 'Expressed', 'Interacted'], fontsize='x-large', comparisons_correction=None)
plt.savefig('../figs/Seq-models/final/boxplot_R_e2e_models.pdf')

# NLL
g = sns.catplot(x="Set", y="NLL",
                hue="Method", col='cell',
                data=df, kind="box",
                height=4, aspect=1, sharey=False, palette={"Seq-GraphReg": "orange", "Seq-CNN": "deepskyblue"})

n_row, n_col = g.axes.shape
for j in range(n_col):
    df_annt = df[df['cell'] == g.col_names[j]]
    add_stat_annotation(g.axes[0,j], data=df_annt, x='Set', y='NLL', hue='Method',
                        box_pairs=[(("All", "Seq-GraphReg"), ("All", "Seq-CNN")),
                                    (("Expressed", "Seq-GraphReg"), ("Expressed", "Seq-CNN")),
                                    (("Interacted", "Seq-GraphReg"), ("Interacted", "Seq-CNN"))],
                        test='Wilcoxon', text_format='star', loc='inside', verbose=0, order=['All', 'Expressed', 'Interacted'], fontsize='x-large', comparisons_correction=None)
plt.savefig('../figs/Seq-models/final/boxplot_NLL_e2e_models.pdf')



##### universal analysis of CAGE predictions #####

data_path = '/media/labuser/STORAGE/GraphReg'   # data path
cell_line = 'K562'
assay_type = 'HiChIP'
fdr = '1'              # 1/01/001

df = pd.read_csv(data_path+'/results/csv/cage_prediction/seq_models/cage_predictions_seq_e2e_models_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.csv', sep='\t')
df['pred_cage_seq_graphreg_log2'] = np.log2(df['pred_cage_seq_graphreg']+1)
df['pred_cage_seq_cnn_log2'] = np.log2(df['pred_cage_seq_cnn']+1)
df['true_cage_log2'] = np.log2(df['true_cage']+1)

df_expressed = df[df['true_cage']>=5].reset_index()
df_expressed_interacted = df[((df['true_cage']>=5) & (df['n_contact']>=1))].reset_index()

g = sns.scatterplot(data=df, x="n_contact", y="delta_nll", hue='true_cage', alpha=.7, palette=plt.cm.get_cmap('viridis_r'))
plt.savefig('../figs/Seq-models/final/scatterplot/scatterplot_delta_nll_hue_cage_e2e_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.pdf')

# NLL
nll_graphreg_all = df['nll_seq_graphreg'].values
nll_graphreg_all_mean = np.mean(nll_graphreg_all)
nll_graphreg_ex = df_expressed['nll_seq_graphreg'].values
nll_graphreg_ex_mean = np.mean(nll_graphreg_ex)
nll_graphreg_ex_int = df_expressed_interacted['nll_seq_graphreg'].values
nll_graphreg_ex_int_mean = np.mean(nll_graphreg_ex_int)

nll_cnn_all = df['nll_seq_cnn'].values
nll_cnn_all_mean = np.mean(nll_cnn_all)
nll_cnn_ex = df_expressed['nll_seq_cnn'].values
nll_cnn_ex_mean = np.mean(nll_cnn_ex)
nll_cnn_ex_int = df_expressed_interacted['nll_seq_cnn'].values
nll_cnn_ex_int_mean = np.mean(nll_cnn_ex_int)

# R
r_graphreg_all = np.corrcoef(df['true_cage_log2'].values, df['pred_cage_seq_graphreg_log2'])[0,1]
r_graphreg_ex = np.corrcoef(df_expressed['true_cage_log2'].values, df_expressed['pred_cage_seq_graphreg_log2'])[0,1]
r_graphreg_ex_int = np.corrcoef(df_expressed_interacted['true_cage_log2'].values, df_expressed_interacted['pred_cage_seq_graphreg_log2'])[0,1]

r_cnn_all = np.corrcoef(df['true_cage_log2'].values, df['pred_cage_seq_cnn_log2'])[0,1]
r_cnn_ex = np.corrcoef(df_expressed['true_cage_log2'].values, df_expressed['pred_cage_seq_cnn_log2'])[0,1]
r_cnn_ex_int = np.corrcoef(df_expressed_interacted['true_cage_log2'].values, df_expressed_interacted['pred_cage_seq_cnn_log2'])[0,1]

fig, ax = plt.subplots()
g = sns.scatterplot(data=df_expressed_interacted, x="true_cage_log2", y="pred_cage_seq_graphreg_log2", hue='n_contact', alpha=.1, palette=plt.cm.get_cmap('viridis_r'), ax=ax)
ax.set_title('Seq-GraphReg/{}: R = {:5.3f}, NLL = {:6.2f}'.format(cell_line, r_graphreg_ex_int, nll_graphreg_ex_int_mean))
ax.set_xlabel('log2 (true + 1)')
ax.set_ylabel('log2 (pred + 1)')
plt.savefig('../figs/Seq-models/final/scatterplot/scatterplot_cage_vs_pred_seq_graphreg_e2e_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.pdf')

fig, ax = plt.subplots()
g = sns.scatterplot(data=df_expressed_interacted, x="true_cage_log2", y="pred_cage_seq_cnn_log2", hue='n_contact', alpha=.1, palette=plt.cm.get_cmap('viridis_r'), ax=ax)
ax.set_title('Seq-CNN/{}: R = {:5.3f}, NLL = {:6.2f}'.format(cell_line, r_cnn_ex_int, nll_cnn_ex_int_mean))
ax.set_xlabel('log2 (true + 1)')
ax.set_ylabel('log2 (pred + 1)')
plt.savefig('../figs/Seq-models/final/scatterplot/scatterplot_cage_vs_pred_seq_cnn_e2e_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.pdf')



###############################################################################################################################################
##########################################################            Swarm Plots             #################################################
###############################################################################################################################################

data_path = '/media/labuser/STORAGE/GraphReg'   # data path
qval_dict = {'1': 0.1, '01': 0.01, '001': 0.001}
train_mode = ['dilated', 'dilated+fft', 'nodilated', 'nodilated+fft', 'end to end', 'end to end', 'end to end', 'end to end']
fdr = '1'
df_R_NLL = pd.DataFrame(columns=['Cell', 'Method', 'Train Mode', 'Genes', 'Number of Genes', '3D Data', 'FDR', 'R', 'NLL'])

k = -1
for cell_line in ['K562', 'K562', 'K562', 'K562', 'K562', 'GM12878', 'hESC', 'mESC']:

    k += 1
    if cell_line == 'hESC':
        assay_type = 'MicroC'
    else:
        assay_type = 'HiChIP'
    
    if k == 0:
        df = pd.read_csv(data_path+'/results/csv/cage_prediction/seq_models/cage_predictions_seq_models_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.csv', sep='\t')
    elif k == 1:
        df = pd.read_csv(data_path+'/results/csv/cage_prediction/seq_models/cage_predictions_seq_models_fft_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.csv', sep='\t')
    elif k == 2:
        df = pd.read_csv(data_path+'/results/csv/cage_prediction/seq_models/cage_predictions_seq_models_nodilation_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.csv', sep='\t')
    elif k == 3:
        df = pd.read_csv(data_path+'/results/csv/cage_prediction/seq_models/cage_predictions_seq_models_nodilation_fft_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.csv', sep='\t')        
    else:
        df = pd.read_csv(data_path+'/results/csv/cage_prediction/seq_models/cage_predictions_seq_e2e_models_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.csv', sep='\t')
        
    df['pred_cage_seq_graphreg_log2'] = np.log2(df['pred_cage_seq_graphreg']+1)
    df['pred_cage_seq_cnn_log2'] = np.log2(df['pred_cage_seq_cnn']+1)
    df['true_cage_log2'] = np.log2(df['true_cage']+1)
    df['n_contact_log2'] = np.log2(df['n_contact']+1)

    df_expressed = df[df['true_cage']>=5].reset_index()
    df_expressed_interacted = df[((df['true_cage']>=5) & (df['n_contact']>=1))].reset_index()

    # NLL
    nll_graphreg_all = df['nll_seq_graphreg'].values
    nll_graphreg_all_mean = np.mean(nll_graphreg_all)
    nll_graphreg_ex = df_expressed['nll_seq_graphreg'].values
    nll_graphreg_ex_mean = np.mean(nll_graphreg_ex)
    nll_graphreg_ex_int = df_expressed_interacted['nll_seq_graphreg'].values
    nll_graphreg_ex_int_mean = np.mean(nll_graphreg_ex_int)

    nll_cnn_all = df['nll_seq_cnn'].values
    nll_cnn_all_mean = np.mean(nll_cnn_all)
    nll_cnn_ex = df_expressed['nll_seq_cnn'].values
    nll_cnn_ex_mean = np.mean(nll_cnn_ex)
    nll_cnn_ex_int = df_expressed_interacted['nll_seq_cnn'].values
    nll_cnn_ex_int_mean = np.mean(nll_cnn_ex_int)

    # R
    r_graphreg_all = np.corrcoef(df['true_cage_log2'].values, df['pred_cage_seq_graphreg_log2'])[0,1]
    r_graphreg_ex = np.corrcoef(df_expressed['true_cage_log2'].values, df_expressed['pred_cage_seq_graphreg_log2'])[0,1]
    r_graphreg_ex_int = np.corrcoef(df_expressed_interacted['true_cage_log2'].values, df_expressed_interacted['pred_cage_seq_graphreg_log2'])[0,1]

    r_cnn_all = np.corrcoef(df['true_cage_log2'].values, df['pred_cage_seq_cnn_log2'])[0,1]
    r_cnn_ex = np.corrcoef(df_expressed['true_cage_log2'].values, df_expressed['pred_cage_seq_cnn_log2'])[0,1]
    r_cnn_ex_int = np.corrcoef(df_expressed_interacted['true_cage_log2'].values, df_expressed_interacted['pred_cage_seq_cnn_log2'])[0,1]

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
    if k == 0:
        plt.savefig('../figs/Seq-models/final/scatterplot/scatterplot_delta_nll_hue_cage_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.pdf')
    elif k == 1:
        plt.savefig('../figs/Seq-models/final/scatterplot/scatterplot_delta_nll_hue_cage_fft_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.pdf')
    elif k == 2:
        plt.savefig('../figs/Seq-models/final/scatterplot/scatterplot_delta_nll_hue_cage_nodilation_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.pdf')
    elif k == 3:
        plt.savefig('../figs/Seq-models/final/scatterplot/scatterplot_delta_nll_hue_cage_nodilation_fft_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.pdf')
    else:
        plt.savefig('../figs/Seq-models/final/scatterplot/scatterplot_delta_nll_hue_cage_e2e_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.pdf')
    plt.close()

    fig, ax = plt.subplots()
    g = sns.scatterplot(data=df_expressed, x="true_cage_log2", y="pred_cage_seq_cnn_log2", hue='n_contact_log2', alpha=.2, palette=plt.cm.get_cmap('viridis_r'), ax=ax)
    norm = plt.Normalize(df_expressed['n_contact_log2'].min(), df_expressed['n_contact_log2'].max())
    sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=norm)
    sm.set_array([])
    ax.get_legend().remove()
    ax.figure.colorbar(sm, label="log2 (N + 1)")
    ax.set_title('Seq-CNN | {} | R = {:5.3f} | NLL = {:6.2f}'.format(cell_line, r_cnn_ex, nll_cnn_ex_mean))
    ax.set_xlabel('log2 (true + 1)')
    ax.set_ylabel('log2 (pred + 1)')
    plt.tight_layout()
    if k == 0:
        plt.savefig('../figs/Seq-models/final/scatterplot/scatterplot_cage_vs_pred_seq_cnn_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.pdf')
    elif k == 1:
        plt.savefig('../figs/Seq-models/final/scatterplot/scatterplot_cage_vs_pred_seq_cnn_fft_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.pdf')
    elif k == 2:
        plt.savefig('../figs/Seq-models/final/scatterplot/scatterplot_cage_vs_pred_seq_cnn_nodilation_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.pdf')
    elif k == 3:
        plt.savefig('../figs/Seq-models/final/scatterplot/scatterplot_cage_vs_pred_seq_cnn_nodilation_fft_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.pdf')
    else:
        plt.savefig('../figs/Seq-models/final/scatterplot/scatterplot_cage_vs_pred_seq_cnn_e2e_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.pdf')
    plt.close()

    fig, ax = plt.subplots()
    g = sns.scatterplot(data=df_expressed, x="true_cage_log2", y="pred_cage_seq_graphreg_log2", hue='n_contact_log2', alpha=.2, palette=plt.cm.get_cmap('viridis_r'), ax=ax)
    norm = plt.Normalize(df_expressed['n_contact_log2'].min(), df_expressed['n_contact_log2'].max())
    sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=norm)
    sm.set_array([])
    ax.get_legend().remove()
    ax.figure.colorbar(sm, label="log2 (N + 1)")
    ax.set_title('Seq-GraphReg | {} | R = {:5.3f} | NLL = {:6.2f}'.format(cell_line, r_graphreg_ex, nll_graphreg_ex_mean))
    ax.set_xlabel('log2 (true + 1)')
    ax.set_ylabel('log2 (pred + 1)')
    plt.tight_layout()
    if k == 0:
        plt.savefig('../figs/Seq-models/final/scatterplot/scatterplot_cage_vs_pred_seq_graphreg_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.pdf')
    elif k == 1:
        plt.savefig('../figs/Seq-models/final/scatterplot/scatterplot_cage_vs_pred_seq_graphreg_fft_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.pdf')
    elif k == 2:
        plt.savefig('../figs/Seq-models/final/scatterplot/scatterplot_cage_vs_pred_seq_graphreg_nodilation_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.pdf')
    elif k == 3:
        plt.savefig('../figs/Seq-models/final/scatterplot/scatterplot_cage_vs_pred_seq_graphreg_nodilation_fft_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.pdf')
    else:
        plt.savefig('../figs/Seq-models/final/scatterplot/scatterplot_cage_vs_pred_seq_graphreg_e2e_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.pdf')
    plt.close()

    # append dataframe

    df_R_NLL = df_R_NLL.append({'Cell': cell_line, 'Method': 'Seq-GraphReg', 'Train Mode': train_mode[k], 'Genes': 'All',
                    'Number of Genes': len(df), '3D Data': assay_type, 'FDR': qval_dict[fdr], 'R': r_graphreg_all, 'NLL': nll_graphreg_all_mean}, ignore_index=True)
    df_R_NLL = df_R_NLL.append({'Cell': cell_line, 'Method': 'Seq-GraphReg', 'Train Mode': train_mode[k], 'Genes': 'Expressed', 
                    'Number of Genes': len(df_expressed), '3D Data': assay_type, 'FDR': qval_dict[fdr], 'R': r_graphreg_ex, 'NLL': nll_graphreg_ex_mean}, ignore_index=True)
    df_R_NLL = df_R_NLL.append({'Cell': cell_line, 'Method': 'Seq-GraphReg', 'Train Mode': train_mode[k], 'Genes': 'Interacted',
                    'Number of Genes': len(df_expressed_interacted), '3D Data': assay_type, 'FDR': qval_dict[fdr], 'R': r_graphreg_ex_int, 'NLL': nll_graphreg_ex_int_mean}, ignore_index=True)

    df_R_NLL = df_R_NLL.append({'Cell': cell_line, 'Method': 'Seq-CNN', 'Train Mode': train_mode[k], 'Genes': 'All',
                'Number of Genes': len(df), '3D Data': assay_type, 'FDR': qval_dict[fdr], 'R': r_cnn_all, 'NLL': nll_cnn_all_mean}, ignore_index=True)
    df_R_NLL = df_R_NLL.append({'Cell': cell_line, 'Method': 'Seq-CNN', 'Train Mode': train_mode[k], 'Genes': 'Expressed',
                'Number of Genes': len(df_expressed), '3D Data': assay_type, 'FDR': qval_dict[fdr], 'R': r_cnn_ex, 'NLL': nll_cnn_ex_mean}, ignore_index=True)
    df_R_NLL = df_R_NLL.append({'Cell': cell_line, 'Method': 'Seq-CNN', 'Train Mode': train_mode[k], 'Genes': 'Interacted',
                'Number of Genes': len(df_expressed_interacted), '3D Data': assay_type, 'FDR': qval_dict[fdr], 'R': r_cnn_ex_int, 'NLL': nll_cnn_ex_int_mean}, ignore_index=True)


# Add Basenji


# End to End models
df_e2e = df_R_NLL[(df_R_NLL['Train Mode'] == 'end to end') & (df_R_NLL['Method'] != 'Basenji')]

sns.set_style("darkgrid")
g = sns.catplot(x="Genes", y="R",
            hue="Method", col="Cell",
            data=df_e2e, kind="swarm", palette={"Seq-GraphReg": "orange", "Seq-CNN": "deepskyblue"},
            height=3, aspect=1, sharey=False)
g.fig.savefig('../figs/Seq-models/final/swarmplot_R_e2e_models.pdf')
g.fig.clf()

g = sns.catplot(x="Genes", y="NLL",
            hue="Method", col="Cell",
            data=df_e2e, kind="swarm", palette={"Seq-GraphReg": "orange", "Seq-CNN": "deepskyblue"},
            height=3, aspect=1, sharey=False)
g.fig.savefig('../figs/Seq-models/final/swarmplot_NLL_e2e_models.pdf')
g.fig.clf()


# Only K562
df_k562 = df_R_NLL[(df_R_NLL['Cell'] == 'K562') & (df_R_NLL['Method'] != 'Basenji')]
g = sns.catplot(x="Train Mode", y="R", hue="Method", col="Genes",
            data=df_k562, kind="swarm", palette={"Seq-GraphReg": "orange", "Seq-CNN": "deepskyblue"},
            height=3, aspect=1.5, sharey=False)
loc, labels = plt.xticks()
g.set_xticklabels(labels, rotation=45)
g.fig.savefig('../figs/Seq-models/final/swarmplot_R_K562_check_fft_and_dilation.pdf', bbox_inches='tight')
g.fig.clf()

g = sns.catplot(x="Train Mode", y="NLL", hue="Method", col="Genes",
            data=df_k562, kind="swarm", palette={"Seq-GraphReg": "orange", "Seq-CNN": "deepskyblue"},
            height=3, aspect=1.5, sharey=False)
loc, labels = plt.xticks()
g.set_xticklabels(labels, rotation=45)
g.fig.savefig('../figs/Seq-models/final/swarmplot_NLL_K562_check_fft_and_dilation.pdf', bbox_inches='tight')
g.fig.clf()









