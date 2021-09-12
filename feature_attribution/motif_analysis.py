import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import seaborn as sns
import scipy as sp
#import statsmodels as sm
#import statsmodels.api as sm
import statsmodels.stats.multitest as fdr

p_adj_thr = 0.05
p_val_thr = 1e-4
gene_thr = 50
TF_list_CRISPRi = ['NFYA','NR2C2','HSF1','MITF','MAF1','NR4A1','STAT5A','MXD3',
                    'ZBTB33','TFDP1','HMBOX1','SIX5','SMAD5','GATA1','NRF1','TRIM28',
                    'NFATC1','MEIS2','NR2F2','HOXB9','STAT1','HMGB2','NFYB','SRF',
                    'USF2','JUND','GATA2','STAT2','UBTF','ATF3','RNF2','ZNF384','KLF2',
                    'HOXB4','DLX1','USF1','STAT6','FOXK2','ZNF395','RFX5','BACH1','ZNF143',
                    'ARID3A','SP1','ILF2','TEAD4','HINFP','RELA','SP2','TEAD2','THAP1']
TF_list_CRISPRi.sort()
df_K_GM = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CAGE_GM12878_K562.csv')

##### GM12878
fimo_GM_GraphReg = pd.read_csv('/media/labuser/STORAGE/GraphReg/results/fimo/fimo_out_GM12878_GraphReg_distal_cisbp/fimo.tsv', sep='\t')
fimo_GM_CNN = pd.read_csv('/media/labuser/STORAGE/GraphReg/results/fimo/fimo_out_GM12878_CNN_distal_cisbp/fimo.tsv', sep='\t')

fimo_GM_GraphReg = fimo_GM_GraphReg[fimo_GM_GraphReg['p-value']<=p_val_thr].reset_index(drop=True)
fimo_GM_CNN = fimo_GM_CNN[fimo_GM_CNN['p-value']<=p_val_thr].reset_index(drop=True)

motifs_GM_GraphReg = np.unique(fimo_GM_GraphReg['motif_alt_id'].values)
df_GM = df_K_GM[df_K_GM['CAGE_GM12878']>=gene_thr].reset_index(drop=True)
Expressed_genes_GM = np.unique(df_GM['gene'])
motifs_GM_GraphReg = np.intersect1d(motifs_GM_GraphReg,Expressed_genes_GM)
fimo_GM_GraphReg = fimo_GM_GraphReg[fimo_GM_GraphReg['motif_alt_id'].isin(motifs_GM_GraphReg)].reset_index(drop=True)
print(len(motifs_GM_GraphReg))
print(motifs_GM_GraphReg)

motifs_GM_CNN = np.unique(fimo_GM_CNN['motif_alt_id'].values)
motifs_GM_CNN = np.intersect1d(motifs_GM_CNN,Expressed_genes_GM)
fimo_GM_CNN = fimo_GM_CNN[fimo_GM_CNN['motif_alt_id'].isin(motifs_GM_CNN)].reset_index(drop=True)
print(len(motifs_GM_CNN))
print(motifs_GM_CNN)

motifs_GM_shared = np.intersect1d(motifs_GM_GraphReg, motifs_GM_CNN)
motifs_GM_only_GraphReg = np.setdiff1d(motifs_GM_GraphReg, motifs_GM_shared)
motifs_GM_only_CNN = np.setdiff1d(motifs_GM_CNN, motifs_GM_shared)

##### K562
fimo_K562_GraphReg = pd.read_csv('/media/labuser/STORAGE/GraphReg/results/fimo/fimo_out_K562_GraphReg_distal_cisbp/fimo.tsv', sep='\t')
fimo_K562_CNN = pd.read_csv('/media/labuser/STORAGE/GraphReg/results/fimo/fimo_out_K562_CNN_distal_cisbp/fimo.tsv', sep='\t')

fimo_K562_GraphReg = fimo_K562_GraphReg[fimo_K562_GraphReg['p-value']<=p_val_thr].reset_index(drop=True)
fimo_K562_CNN = fimo_K562_CNN[fimo_K562_CNN['p-value']<=p_val_thr].reset_index(drop=True)

motifs_K562_GraphReg = np.unique(fimo_K562_GraphReg['motif_alt_id'].values)
df_K = df_K_GM[df_K_GM['CAGE_K562']>=gene_thr].reset_index(drop=True)
Expressed_genes_K562 = np.unique(df_K['gene'])
motifs_K562_GraphReg = np.intersect1d(motifs_K562_GraphReg,Expressed_genes_K562)
fimo_K562_GraphReg = fimo_K562_GraphReg[fimo_K562_GraphReg['motif_alt_id'].isin(motifs_K562_GraphReg)].reset_index(drop=True)
print(len(motifs_K562_GraphReg))
print(motifs_K562_GraphReg)

motifs_K562_CNN = np.unique(fimo_K562_CNN['motif_alt_id'].values)
motifs_K562_CNN = np.intersect1d(motifs_K562_CNN,Expressed_genes_K562)
fimo_K562_CNN = fimo_K562_CNN[fimo_K562_CNN['motif_alt_id'].isin(motifs_K562_CNN)].reset_index(drop=True)
print(len(motifs_K562_CNN))
print(motifs_K562_CNN)

motifs_K562_shared = np.intersect1d(motifs_K562_GraphReg, motifs_K562_CNN)
motifs_K562_only_GraphReg = np.setdiff1d(motifs_K562_GraphReg, motifs_K562_shared)
motifs_K562_only_CNN = np.setdiff1d(motifs_K562_CNN, motifs_K562_shared)

##### GM12878 and K562
motifs_shared_GraphReg = np.intersect1d(motifs_GM_GraphReg, motifs_K562_GraphReg)
motifs_only_K562_GraphReg = np.setdiff1d(motifs_K562_GraphReg, motifs_shared_GraphReg)
motifs_only_GM_GraphReg = np.setdiff1d(motifs_GM_GraphReg, motifs_shared_GraphReg)

motifs_shared_CNN = np.intersect1d(motifs_GM_CNN, motifs_K562_CNN)
motifs_only_K562_CNN = np.setdiff1d(motifs_K562_CNN, motifs_shared_CNN)
motifs_only_GM_CNN = np.setdiff1d(motifs_GM_CNN, motifs_shared_CNN)

##### venn diagram
set1 = set(motifs_GM_GraphReg)
set2 = set(motifs_K562_GraphReg)
venn2([set1, set2], ('GM12878 TF motifs (GR)', 'K562 TF motifs (GR)'))
plt.show()

set1 = set(motifs_GM_CNN)
set2 = set(motifs_K562_CNN)
venn2([set1, set2], ('GM12878 TF motifs (CNN)', 'K562 TF motifs (CNN)'))
plt.show()

##### heatmap plot
N = len(fimo_GM_GraphReg)
GM_GraphReg_df = fimo_GM_GraphReg.copy()
score = np.zeros(N)
for i in range(N):
    motif = fimo_GM_GraphReg['motif_alt_id'].values[i]
    strand = fimo_GM_GraphReg['strand'].values[i]
    seq_name = fimo_GM_GraphReg['sequence_name'].values[i]
    start = int(seq_name.split(':')[1].split('-')[0])
    end = int(seq_name.split(':')[1].split('-')[1])
    gene_name = seq_name.split(':')[0].split('_')[1]
    chr = seq_name.split(':')[0].split('_')[3]
    score[i] = seq_name.split(':')[0].split('_')[2]
    GM_GraphReg_df['sequence_name'].values[i] = gene_name

GM_GraphReg_df = GM_GraphReg_df.rename(columns={'sequence_name': 'gene_name'})
GM_GraphReg_df['score'] = score
motifs_GM_GraphReg = np.unique(GM_GraphReg_df['motif_alt_id'].values)
genes_GM = np.unique(GM_GraphReg_df['gene_name'].values)

df_num_motifs_GM_GraphReg = pd.DataFrame(columns=['motif', 'n_positives', 'n_negatives', 'p-value'])
for i in range(len(motifs_GM_GraphReg)):
    df_num_motifs_GM_GraphReg.loc[i,'motif'] = motifs_GM_GraphReg[i]
    n_positives = len(GM_GraphReg_df[((GM_GraphReg_df['motif_alt_id']==motifs_GM_GraphReg[i]) & (GM_GraphReg_df['score']>=.001))])
    n_negatives = len(GM_GraphReg_df[((GM_GraphReg_df['motif_alt_id']==motifs_GM_GraphReg[i]) & (GM_GraphReg_df['score']<.001))])
    df_num_motifs_GM_GraphReg.loc[i,'n_positives'] = n_positives
    df_num_motifs_GM_GraphReg.loc[i,'n_negatives'] = n_negatives
    n_all = n_positives + n_negatives
    _, df_num_motifs_GM_GraphReg.loc[i,'p-value'] = sp.stats.fisher_exact([[n_positives, n_negatives],[5000-n_positives,5000-n_negatives]], alternative='greater')

df_num_motifs_GM_GraphReg = df_num_motifs_GM_GraphReg.sort_values(by=['p-value']).reset_index(drop=True)
p_values = df_num_motifs_GM_GraphReg['p-value'].values
_, pvals_adjusted, _, _ = fdr.multipletests(p_values, alpha=p_adj_thr, method='fdr_bh', is_sorted=True)
df_num_motifs_GM_GraphReg['p-adj'] = pvals_adjusted
print('df_num_motifs_GM_GraphReg: ', df_num_motifs_GM_GraphReg[df_num_motifs_GM_GraphReg['p-adj']<=p_adj_thr])


N = len(fimo_GM_CNN)
GM_CNN_df = fimo_GM_CNN.copy()
score = np.zeros(N)
for i in range(N):
    motif = fimo_GM_CNN['motif_alt_id'].values[i]
    strand = fimo_GM_CNN['strand'].values[i]
    seq_name = fimo_GM_CNN['sequence_name'].values[i]
    start = int(seq_name.split(':')[1].split('-')[0])
    end = int(seq_name.split(':')[1].split('-')[1])
    gene_name = seq_name.split(':')[0].split('_')[1]
    chr = seq_name.split(':')[0].split('_')[3]
    score[i] = seq_name.split(':')[0].split('_')[2]
    GM_CNN_df['sequence_name'].values[i] = gene_name

GM_CNN_df = GM_CNN_df.rename(columns={'sequence_name': 'gene_name'})
GM_CNN_df['score'] = score
motifs_GM_CNN = np.unique(GM_CNN_df['motif_alt_id'].values)
genes_GM = np.unique(GM_CNN_df['gene_name'].values)

df_num_motifs_GM_CNN = pd.DataFrame(columns=['motif', 'n_positives', 'n_negatives', 'p-value'])
for i in range(len(motifs_GM_CNN)):
    df_num_motifs_GM_CNN.loc[i,'motif'] = motifs_GM_CNN[i]
    n_positives = len(GM_CNN_df[((GM_CNN_df['motif_alt_id']==motifs_GM_CNN[i]) & (GM_CNN_df['score']>=.001))])
    n_negatives = len(GM_CNN_df[((GM_CNN_df['motif_alt_id']==motifs_GM_CNN[i]) & (GM_CNN_df['score']<.001))])
    df_num_motifs_GM_CNN.loc[i,'n_positives'] = n_positives
    df_num_motifs_GM_CNN.loc[i,'n_negatives'] = n_negatives
    n_all = n_positives + n_negatives
    _, df_num_motifs_GM_CNN.loc[i,'p-value'] = sp.stats.fisher_exact([[n_positives, n_negatives],[5000-n_positives,5000-n_negatives]], alternative='greater')

df_num_motifs_GM_CNN = df_num_motifs_GM_CNN.sort_values(by=['p-value']).reset_index(drop=True)
p_values = df_num_motifs_GM_CNN['p-value'].values
_, pvals_adjusted, _, _ = fdr.multipletests(p_values, alpha=p_adj_thr, method='fdr_bh', is_sorted=True)
df_num_motifs_GM_CNN['p-adj'] = pvals_adjusted
print('df_num_motifs_GM_CNN: ', df_num_motifs_GM_CNN[df_num_motifs_GM_CNN['p-adj']<=p_adj_thr])

N = len(fimo_K562_GraphReg)
K562_GraphReg_df = fimo_K562_GraphReg.copy()
score = np.zeros(N)
for i in range(N):
    motif = fimo_K562_GraphReg['motif_alt_id'].values[i]
    strand = fimo_K562_GraphReg['strand'].values[i]
    seq_name = fimo_K562_GraphReg['sequence_name'].values[i]
    start = int(seq_name.split(':')[1].split('-')[0])
    end = int(seq_name.split(':')[1].split('-')[1])
    gene_name = seq_name.split(':')[0].split('_')[1]
    chr = seq_name.split(':')[0].split('_')[3]
    score[i] = seq_name.split(':')[0].split('_')[2]
    K562_GraphReg_df['sequence_name'].values[i] = gene_name

K562_GraphReg_df = K562_GraphReg_df.rename(columns={'sequence_name': 'gene_name'})
K562_GraphReg_df['score'] = score
motifs_K562_GraphReg = np.unique(K562_GraphReg_df['motif_alt_id'].values)
genes_K562 = np.unique(K562_GraphReg_df['gene_name'].values)

df_num_motifs_K562_GraphReg = pd.DataFrame(columns=['motif', 'n_positives', 'n_negatives', 'p-value'])
for i in range(len(motifs_K562_GraphReg)):
    df_num_motifs_K562_GraphReg.loc[i,'motif'] = motifs_K562_GraphReg[i]
    n_positives = len(K562_GraphReg_df[((K562_GraphReg_df['motif_alt_id']==motifs_K562_GraphReg[i]) & (K562_GraphReg_df['score']>=.001))])
    n_negatives = len(K562_GraphReg_df[((K562_GraphReg_df['motif_alt_id']==motifs_K562_GraphReg[i]) & (K562_GraphReg_df['score']<.001))])
    df_num_motifs_K562_GraphReg.loc[i,'n_positives'] = n_positives
    df_num_motifs_K562_GraphReg.loc[i,'n_negatives'] = n_negatives
    n_all = n_positives + n_negatives
    _, df_num_motifs_K562_GraphReg.loc[i,'p-value'] = sp.stats.fisher_exact([[n_positives, n_negatives],[5000-n_positives,5000-n_negatives]], alternative='greater')

df_num_motifs_K562_GraphReg = df_num_motifs_K562_GraphReg.sort_values(by=['p-value']).reset_index(drop=True)
p_values = df_num_motifs_K562_GraphReg['p-value'].values
_, pvals_adjusted, _, _ = fdr.multipletests(p_values, alpha=p_adj_thr, method='fdr_bh', is_sorted=True)
df_num_motifs_K562_GraphReg['p-adj'] = pvals_adjusted
print('df_num_motifs_K562_GraphReg: ', df_num_motifs_K562_GraphReg[df_num_motifs_K562_GraphReg['p-adj']<=p_adj_thr])

N = len(fimo_K562_CNN)
K562_CNN_df = fimo_K562_CNN.copy()
score = np.zeros(N)
for i in range(N):
    motif = fimo_K562_CNN['motif_alt_id'].values[i]
    strand = fimo_K562_CNN['strand'].values[i]
    seq_name = fimo_K562_CNN['sequence_name'].values[i]
    start = int(seq_name.split(':')[1].split('-')[0])
    end = int(seq_name.split(':')[1].split('-')[1])
    gene_name = seq_name.split(':')[0].split('_')[1]
    chr = seq_name.split(':')[0].split('_')[3]
    score[i] = seq_name.split(':')[0].split('_')[2]
    K562_CNN_df['sequence_name'].values[i] = gene_name

K562_CNN_df = K562_CNN_df.rename(columns={'sequence_name': 'gene_name'})
K562_CNN_df['score'] = score
motifs_K562_CNN = np.unique(K562_CNN_df['motif_alt_id'].values)
genes_K562 = np.unique(K562_CNN_df['gene_name'].values)

df_num_motifs_K562_CNN = pd.DataFrame(columns=['motif', 'n_positives', 'n_negatives', 'p-value'])
for i in range(len(motifs_K562_CNN)):
    df_num_motifs_K562_CNN.loc[i,'motif'] = motifs_K562_CNN[i]
    n_positives = len(K562_CNN_df[((K562_CNN_df['motif_alt_id']==motifs_K562_CNN[i]) & (K562_CNN_df['score']>=.001))])
    n_negatives = len(K562_CNN_df[((K562_CNN_df['motif_alt_id']==motifs_K562_CNN[i]) & (K562_CNN_df['score']<.001))])
    df_num_motifs_K562_CNN.loc[i,'n_positives'] = n_positives
    df_num_motifs_K562_CNN.loc[i,'n_negatives'] = n_negatives
    n_all = n_positives + n_negatives
    _, df_num_motifs_K562_CNN.loc[i,'p-value'] = sp.stats.fisher_exact([[n_positives, n_negatives],[5000-n_positives,5000-n_negatives]], alternative='greater')

df_num_motifs_K562_CNN = df_num_motifs_K562_CNN.sort_values(by=['p-value']).reset_index(drop=True)
p_values = df_num_motifs_K562_CNN['p-value'].values
_, pvals_adjusted, _, _ = fdr.multipletests(p_values, alpha=p_adj_thr, method='fdr_bh', is_sorted=True)
df_num_motifs_K562_CNN['p-adj'] = pvals_adjusted
print('df_num_motifs_K562_CNN: ', df_num_motifs_K562_CNN[df_num_motifs_K562_CNN['p-adj']<=p_adj_thr])


df_num_motifs_GM_GraphReg_sig = df_num_motifs_GM_GraphReg[df_num_motifs_GM_GraphReg['p-adj']<=p_adj_thr]
motifs_GM_GraphReg = df_num_motifs_GM_GraphReg_sig['motif'].values
pos = df_num_motifs_GM_GraphReg_sig['n_positives'].values.astype(np.int64)
neg = df_num_motifs_GM_GraphReg_sig['n_negatives'].values.astype(np.int64)
p_adj = -np.log10(df_num_motifs_GM_GraphReg_sig['p-adj'].values.astype(np.float64))
df = pd.DataFrame(data=np.vstack((pos, neg)), index=['High saliency', 'Low saliency'], columns=motifs_GM_GraphReg)
plt.figure(figsize = (30,5))
ax = sns.heatmap(np.log2(df+1), xticklabels=1, yticklabels=1, cmap="YlGnBu", annot=df, annot_kws={'rotation': 90}, fmt="d")
ax.set_title('GM12878/GraphReg/Significant distal (>50kb from TSS) motifs (p_adj<='+str(p_adj_thr)+')', fontsize=30)
ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 20, rotation=0)
ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 15, rotation=90)
cbar = ax.collections[0].colorbar
cbar.set_label(label='log2 (n + 1)', size=20)
cbar.ax.tick_params(labelsize=20)
#ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('../figs/motif_analysis/heatmap_GM_GraphReg_distal_cisbp.png')

df_num_motifs_GM_CNN_sig = df_num_motifs_GM_CNN[df_num_motifs_GM_CNN['p-adj']<=p_adj_thr]
motifs_GM_CNN = df_num_motifs_GM_CNN_sig['motif'].values
pos = df_num_motifs_GM_CNN_sig['n_positives'].values.astype(np.int64)
neg = df_num_motifs_GM_CNN_sig['n_negatives'].values.astype(np.int64)
p_adj = -np.log10(df_num_motifs_GM_CNN_sig['p-adj'].values.astype(np.float64))
df = pd.DataFrame(data=np.vstack((pos, neg)), index=['High saliency', 'Low saliency'], columns=motifs_GM_CNN)
plt.figure(figsize = (30,5))
ax = sns.heatmap(np.log2(df+1), xticklabels=1, yticklabels=1, cmap="YlGnBu", annot=df, annot_kws={'rotation': 90}, fmt="d")
ax.set_title('GM12878/CNN/Significant distal (>50kb from TSS) motifs (p_adj<='+str(p_adj_thr)+')', fontsize=30)
ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 20, rotation=0)
ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 15, rotation=90)
cbar = ax.collections[0].colorbar
cbar.set_label(label='log2 (n + 1)', size=20)
cbar.ax.tick_params(labelsize=20)
#ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('../figs/motif_analysis/heatmap_GM_CNN_distal_cisbp.png')

df_num_motifs_K562_GraphReg_sig = df_num_motifs_K562_GraphReg[df_num_motifs_K562_GraphReg['p-adj']<=p_adj_thr]
motifs_K562_GraphReg = df_num_motifs_K562_GraphReg_sig['motif'].values
pos = df_num_motifs_K562_GraphReg_sig['n_positives'].values.astype(np.int64)
neg = df_num_motifs_K562_GraphReg_sig['n_negatives'].values.astype(np.int64)
p_adj = -np.log10(df_num_motifs_K562_GraphReg_sig['p-adj'].values.astype(np.float64))
df = pd.DataFrame(data=np.vstack((pos, neg)), index=['High saliency', 'Low saliency'], columns=motifs_K562_GraphReg)
plt.figure(figsize = (30,5))
ax = sns.heatmap(np.log2(df+1), xticklabels=1, yticklabels=1, cmap="YlGnBu", annot=df, annot_kws={'rotation': 90}, fmt="d")
ax.set_title('K562/GraphReg/Significant distal (>50kb from TSS) motifs (p_adj<='+str(p_adj_thr)+')', fontsize=30)
ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 20, rotation=0)
ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 15, rotation=90)
cbar = ax.collections[0].colorbar
cbar.set_label(label='log2 (n + 1)', size=20)
cbar.ax.tick_params(labelsize=20)
#ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('../figs/motif_analysis/heatmap_K562_GraphReg_distal_cisbp.png')

df_num_motifs_K562_CNN_sig = df_num_motifs_K562_CNN[df_num_motifs_K562_CNN['p-adj']<=p_adj_thr]
motifs_K562_CNN = df_num_motifs_K562_CNN_sig['motif'].values
pos = df_num_motifs_K562_CNN_sig['n_positives'].values.astype(np.int64)
neg = df_num_motifs_K562_CNN_sig['n_negatives'].values.astype(np.int64)
p_adj = -np.log10(df_num_motifs_K562_CNN_sig['p-adj'].values.astype(np.float64))
df = pd.DataFrame(data=np.vstack((pos, neg)), index=['High saliency', 'Low saliency'], columns=motifs_K562_CNN)
plt.figure(figsize = (30,5))
ax = sns.heatmap(np.log2(df+1), xticklabels=1, yticklabels=1, cmap="YlGnBu", annot=df, annot_kws={'rotation': 90}, fmt="d")
ax.set_title('K562/CNN/Significant distal (>50kb from TSS) motifs (p_adj<='+str(p_adj_thr)+')', fontsize=30)
ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 20, rotation=0)
ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 15, rotation=90)
cbar = ax.collections[0].colorbar
cbar.set_label(label='log2 (n + 1)', size=20)
cbar.ax.tick_params(labelsize=20)
#ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('../figs/motif_analysis/heatmap_K562_CNN_distal_cisbp.png')

motifs_shared_GraphReg = np.intersect1d(motifs_GM_GraphReg, motifs_K562_GraphReg)
motifs_only_K562_GraphReg = np.setdiff1d(motifs_K562_GraphReg, motifs_shared_GraphReg)
motifs_only_GM_GraphReg = np.setdiff1d(motifs_GM_GraphReg, motifs_shared_GraphReg)

plt.figure(figsize = (10,10))
set1 = set(motifs_GM_GraphReg)
set2 = set(motifs_K562_GraphReg)
venn2([set1, set2], ('GM12878 motifs by GraphReg (distal)', 'K562 motifs by GraphReg (distal)'))
plt.savefig('../figs/motif_analysis/venn_GraphReg_distal_cisbp.png')

motifs_shared_CNN = np.intersect1d(motifs_GM_CNN, motifs_K562_CNN)
motifs_only_K562_CNN = np.setdiff1d(motifs_K562_CNN, motifs_shared_CNN)
motifs_only_GM_CNN = np.setdiff1d(motifs_GM_CNN, motifs_shared_CNN)

plt.figure(figsize = (10,10))
set1 = set(motifs_GM_CNN)
set2 = set(motifs_K562_CNN)
venn2([set1, set2], ('GM12878 motifs by CNN (distal)', 'K562 motifs by CNN (distal)'))
plt.savefig('../figs/motif_analysis/venn_CNN_distal_cisbp.png')

###########

data = np.zeros([len(genes_GM), len(motifs_only_GM_GraphReg)])
for i, g in enumerate(genes_GM):
    for j, m in enumerate(motifs_only_GM_GraphReg):
        data[i,j] = len(GM_GraphReg_df[((GM_GraphReg_df['gene_name']==g) & (GM_GraphReg_df['motif_alt_id']==m) & (GM_GraphReg_df['score']>=.001))])

df1 = pd.DataFrame(data=data, index=genes_GM, columns=motifs_only_GM_GraphReg)

data = np.zeros([len(genes_GM), len(motifs_only_GM_GraphReg)])
for i, g in enumerate(genes_GM):
    for j, m in enumerate(motifs_only_GM_GraphReg):
        data[i,j] = len(GM_GraphReg_df[((GM_GraphReg_df['gene_name']==g) & (GM_GraphReg_df['motif_alt_id']==m) & (GM_GraphReg_df['score']<.001))])

df2 = pd.DataFrame(data=data, index=genes_GM, columns=motifs_only_GM_GraphReg)

f,(ax1,ax2) = plt.subplots(1,2, figsize=(25,20))
g1 = sns.heatmap(df1,cmap="YlGnBu",cbar=False,ax=ax1, vmin=0, vmax=10, annot=True)
g1.set_aspect('equal')
g1.set_ylabel('GM12878 highly expressed genes', fontsize=30)
g1.set_xlabel('GM12878 motifs by GraphReg (distal)', fontsize=30)
g1.set_title('High saliency', fontsize=30)
g1.set_yticklabels(g1.get_ymajorticklabels(), fontsize = 20)
g1.set_xticklabels(g1.get_xmajorticklabels(), fontsize = 20)
g2 = sns.heatmap(df2,cmap="YlGnBu",ax=ax2, vmin=0, vmax=10, annot=True)
g2.set_aspect('equal')
#g2.set_ylabel('GM12878 highly expressed genes', fontsize=30)
g2.set_xlabel('GM12878 motifs by GraphReg (distal)', fontsize=30)
g2.set_title('Low saliency', fontsize=30)
g2.set_yticklabels(g2.get_ymajorticklabels(), fontsize = 20)
g2.set_xticklabels(g2.get_xmajorticklabels(), fontsize = 20)
cbar = g2.collections[0].colorbar
cbar.set_label(label='Number of motifs', size=30)
cbar.ax.tick_params(labelsize=20)
plt.tight_layout()
plt.savefig('../figs/motif_analysis/heatmap_GM_GraphReg_with_genes_distal_cisbp.png')

##########
data = np.zeros([len(genes_GM), len(motifs_only_GM_CNN)])
for i, g in enumerate(genes_GM):
    for j, m in enumerate(motifs_only_GM_CNN):
        data[i,j] = len(GM_CNN_df[((GM_CNN_df['gene_name']==g) & (GM_CNN_df['motif_alt_id']==m) & (GM_CNN_df['score']>=.001))])

df1 = pd.DataFrame(data=data, index=genes_GM, columns=motifs_only_GM_CNN)

data = np.zeros([len(genes_GM), len(motifs_only_GM_CNN)])
for i, g in enumerate(genes_GM):
    for j, m in enumerate(motifs_only_GM_CNN):
        data[i,j] = len(GM_CNN_df[((GM_CNN_df['gene_name']==g) & (GM_CNN_df['motif_alt_id']==m) & (GM_CNN_df['score']<.001))])

df2 = pd.DataFrame(data=data, index=genes_GM, columns=motifs_only_GM_CNN)

f,(ax1,ax2) = plt.subplots(1,2, figsize=(35,20))
g1 = sns.heatmap(df1,cmap="YlGnBu",cbar=False,ax=ax1, vmin=0, vmax=10, annot=True)
g1.set_aspect('equal')
g1.set_ylabel('GM12878 highly expressed genes', fontsize=30)
g1.set_xlabel('GM12878 motifs by CNN (distal)', fontsize=30)
g1.set_title('High saliency', fontsize=30)
g1.set_yticklabels(g1.get_ymajorticklabels(), fontsize = 20)
g1.set_xticklabels(g1.get_xmajorticklabels(), fontsize = 20)
g2 = sns.heatmap(df2,cmap="YlGnBu",ax=ax2, vmin=0, vmax=10, annot=True)
g2.set_aspect('equal')
#g2.set_ylabel('GM12878 highly expressed genes', fontsize=30)
g2.set_xlabel('GM12878 motifs by CNN (distal)', fontsize=30)
g2.set_title('Low saliency', fontsize=30)
g2.set_yticklabels(g2.get_ymajorticklabels(), fontsize = 20)
g2.set_xticklabels(g2.get_xmajorticklabels(), fontsize = 20)
cbar = g2.collections[0].colorbar
cbar.set_label(label='Number of motifs', size=30)
cbar.ax.tick_params(labelsize=20)
plt.tight_layout()
plt.savefig('../figs/motif_analysis/heatmap_GM_CNN_with_genes_distal_cisbp.png')

##########

data = np.zeros([len(genes_K562), len(motifs_only_K562_GraphReg)])
for i, g in enumerate(genes_K562):
    for j, m in enumerate(motifs_only_K562_GraphReg):
        data[i,j] = len(K562_GraphReg_df[((K562_GraphReg_df['gene_name']==g) & (K562_GraphReg_df['motif_alt_id']==m) & (K562_GraphReg_df['score']>=.001))])

df1 = pd.DataFrame(data=data, index=genes_K562, columns=motifs_only_K562_GraphReg)

data = np.zeros([len(genes_K562), len(motifs_only_K562_GraphReg)])
for i, g in enumerate(genes_K562):
    for j, m in enumerate(motifs_only_K562_GraphReg):
        data[i,j] = len(K562_GraphReg_df[((K562_GraphReg_df['gene_name']==g) & (K562_GraphReg_df['motif_alt_id']==m) & (K562_GraphReg_df['score']<.001))])

df2 = pd.DataFrame(data=data, index=genes_K562, columns=motifs_only_K562_GraphReg)

f,(ax1,ax2) = plt.subplots(1,2, figsize=(22,20))
g1 = sns.heatmap(df1,cmap="YlGnBu",cbar=False,ax=ax1, vmin=0, vmax=10, annot=True)
g1.set_aspect('equal')
g1.set_ylabel('K562 highly expressed genes', fontsize=30)
g1.set_xlabel('K562 motifs by GraphReg (distal)', fontsize=30)
g1.set_title('High saliency', fontsize=30)
g1.set_yticklabels(g1.get_ymajorticklabels(), fontsize = 20)
g1.set_xticklabels(g1.get_xmajorticklabels(), fontsize = 20)
g2 = sns.heatmap(df2,cmap="YlGnBu",ax=ax2, vmin=0, vmax=10, annot=True)
g2.set_aspect('equal')
#g2.set_ylabel('GM12878 highly expressed genes', fontsize=30)
g2.set_xlabel('K562 motifs by GraphReg (distal)', fontsize=30)
g2.set_title('Low saliency', fontsize=30)
g2.set_yticklabels(g2.get_ymajorticklabels(), fontsize = 20)
g2.set_xticklabels(g2.get_xmajorticklabels(), fontsize = 20)
cbar = g2.collections[0].colorbar
cbar.set_label(label='Number of motifs', size=30)
cbar.ax.tick_params(labelsize=20)
plt.tight_layout()
plt.savefig('../figs/motif_analysis/heatmap_K562_GraphReg_with_genes_distal_cisbp.png')

##########
data = np.zeros([len(genes_K562), len(motifs_only_K562_CNN)])
for i, g in enumerate(genes_K562):
    for j, m in enumerate(motifs_only_K562_CNN):
        data[i,j] = len(K562_CNN_df[((K562_CNN_df['gene_name']==g) & (K562_CNN_df['motif_alt_id']==m) & (K562_CNN_df['score']>=.001))])

df1 = pd.DataFrame(data=data, index=genes_K562, columns=motifs_only_K562_CNN)

data = np.zeros([len(genes_K562), len(motifs_only_K562_CNN)])
for i, g in enumerate(genes_K562):
    for j, m in enumerate(motifs_only_K562_CNN):
        data[i,j] = len(K562_CNN_df[((K562_CNN_df['gene_name']==g) & (K562_CNN_df['motif_alt_id']==m) & (K562_CNN_df['score']<.001))])

df2 = pd.DataFrame(data=data, index=genes_K562, columns=motifs_only_K562_CNN)


f,(ax1,ax2) = plt.subplots(1,2, figsize=(12,20))
g1 = sns.heatmap(df1,cmap="YlGnBu",cbar=False,ax=ax1, vmin=0, vmax=10, annot=True)
g1.set_aspect('equal')
g1.set_ylabel('K562 highly expressed genes', fontsize=30)
g1.set_xlabel('K562 motifs by CNN (distal)', fontsize=30)
g1.set_title('High saliency', fontsize=30)
g1.set_yticklabels(g1.get_ymajorticklabels(), fontsize = 20)
g1.set_xticklabels(g1.get_xmajorticklabels(), fontsize = 20)
g2 = sns.heatmap(df2,cmap="YlGnBu",ax=ax2, vmin=0, vmax=10, annot=True)
g2.set_aspect('equal')
#g2.set_ylabel('GM12878 highly expressed genes', fontsize=30)
g2.set_xlabel('K562 motifs by CNN (distal)', fontsize=30)
g2.set_title('Low saliency', fontsize=30)
g2.set_yticklabels(g2.get_ymajorticklabels(), fontsize = 20)
g2.set_xticklabels(g2.get_xmajorticklabels(), fontsize = 20)
cbar = g2.collections[0].colorbar
cbar.set_label(label='Number of motifs', size=30)
cbar.ax.tick_params(labelsize=20)
#plt.tight_layout()
plt.savefig('../figs/motif_analysis/heatmap_K562_CNN_with_genes_distal_cisbp.png')

##########

#%%
##### convert fimo.tsv to bed (for WashU browser)

##### GM12878
CNN_fimo_motifs_bed = open('/media/labuser/STORAGE/GraphReg/results/fimo/fimo_out_GM12878_CNN_distal/CNN_fimo_motifs_distal.bed', "w")
GraphReg_fimo_motifs_bed = open('/media/labuser/STORAGE/GraphReg/results/fimo/fimo_out_GM12878_GraphReg_distal/GraphReg_fimo_motifs_distal.bed', "w")

N = len(fimo_GM_GraphReg)
for i in range(N):
    motif = fimo_GM_GraphReg['motif_alt_id'].values[i]
    if motif in motifs_GM_GraphReg:
        strand = fimo_GM_GraphReg['strand'].values[i]
        seq_name = fimo_GM_GraphReg['sequence_name'].values[i]
        start = int(seq_name.split(':')[1].split('-')[0])
        end = int(seq_name.split(':')[1].split('-')[1])
        gene_name = seq_name.split(':')[0].split('_')[1]
        chr = seq_name.split(':')[0].split('_')[3]
        score = seq_name.split(':')[0].split('_')[2]

        if np.float64(score) >= .001:
            bed_name = 'motif:'+motif+'/gene:'+gene_name+'/score:'+str(score)
            line = chr+"\t"+str(start)+"\t"+str(end)+"\t"+bed_name+"\t"+str(score)+"\t"+strand
            GraphReg_fimo_motifs_bed.write(line)
            GraphReg_fimo_motifs_bed.write('\n')
GraphReg_fimo_motifs_bed.close()

N = len(fimo_GM_CNN)
for i in range(N):
    motif = fimo_GM_CNN['motif_alt_id'].values[i]
    if motif in motifs_GM_CNN:
        strand = fimo_GM_CNN['strand'].values[i]
        seq_name = fimo_GM_CNN['sequence_name'].values[i]
        start = int(seq_name.split(':')[1].split('-')[0])
        end = int(seq_name.split(':')[1].split('-')[1])
        gene_name = seq_name.split(':')[0].split('_')[1]
        chr = seq_name.split(':')[0].split('_')[3]
        score = seq_name.split(':')[0].split('_')[2]

        if np.float64(score) >= .001:
            bed_name = 'motif:'+motif+'/gene:'+gene_name+'/score:'+str(score)
            line = chr+"\t"+str(start)+"\t"+str(end)+"\t"+bed_name+"\t"+str(score)+"\t"+strand
            CNN_fimo_motifs_bed.write(line)
            CNN_fimo_motifs_bed.write('\n')
CNN_fimo_motifs_bed.close()

##### K562
CNN_fimo_motifs_bed = open('/media/labuser/STORAGE/GraphReg/results/fimo/fimo_out_K562_CNN_distal/CNN_fimo_motifs_distal.bed', "w")
GraphReg_fimo_motifs_bed = open('/media/labuser/STORAGE/GraphReg/results/fimo/fimo_out_K562_GraphReg_distal/GraphReg_fimo_motifs_distal.bed', "w")

N = len(fimo_K562_GraphReg)
for i in range(N):
    motif = fimo_K562_GraphReg['motif_alt_id'].values[i]
    if motif in motifs_K562_GraphReg:
        strand = fimo_K562_GraphReg['strand'].values[i]
        seq_name = fimo_K562_GraphReg['sequence_name'].values[i]
        start = int(seq_name.split(':')[1].split('-')[0])
        end = int(seq_name.split(':')[1].split('-')[1])
        gene_name = seq_name.split(':')[0].split('_')[1]
        chr = seq_name.split(':')[0].split('_')[3]
        score = seq_name.split(':')[0].split('_')[2]

        if np.float64(score) >= .001:
            bed_name = 'motif:'+motif+'/gene:'+gene_name+'/score:'+str(score)
            line = chr+"\t"+str(start)+"\t"+str(end)+"\t"+bed_name+"\t"+str(score)+"\t"+strand
            GraphReg_fimo_motifs_bed.write(line)
            GraphReg_fimo_motifs_bed.write('\n')
GraphReg_fimo_motifs_bed.close()

N = len(fimo_K562_CNN)
for i in range(N):
    motif = fimo_K562_CNN['motif_alt_id'].values[i]
    if motif in motifs_K562_CNN:
        strand = fimo_K562_CNN['strand'].values[i]
        seq_name = fimo_K562_CNN['sequence_name'].values[i]
        start = int(seq_name.split(':')[1].split('-')[0])
        end = int(seq_name.split(':')[1].split('-')[1])
        gene_name = seq_name.split(':')[0].split('_')[1]
        chr = seq_name.split(':')[0].split('_')[3]
        score = seq_name.split(':')[0].split('_')[2]

        if np.float64(score) >= .001:
            bed_name = 'motif:'+motif+'/gene:'+gene_name+'/score:'+str(score)
            line = chr+"\t"+str(start)+"\t"+str(end)+"\t"+bed_name+"\t"+str(score)+"\t"+strand
            CNN_fimo_motifs_bed.write(line)
            CNN_fimo_motifs_bed.write('\n')
CNN_fimo_motifs_bed.close()


