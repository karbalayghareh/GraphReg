import numpy as np
import pandas as pd
#pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import pickle
import os

#%%
##### extract features per gene

data_path = '/media/labuser/STORAGE/GraphReg'   # data path
organism = 'human'
genome='hg38'                                   # hg19/hg38
cell_line = 'GM12878'                              # K562/GM12878/mESC/hESC
fdr = '1'
L = 8
saliency_method = 'saliency'
model = 'combined'

epsilon = 0.01

##### TSS dataframe
#filename_tss = data_path+'/data/tss/'+organism+'/distal_reg_paper/'+genome+'/RefSeqCurated.170308.bed.CollapsedGeneBounds.hg38.TSS500bp.bed'
#tss_dataframe = pd.read_csv(filename_tss, header=None, delimiter='\t')
#tss_dataframe.columns = ["chr", "tss_1", "tss_2", "gene", "na", "strand"]
#tss_dataframe_sub = tss_dataframe[tss_dataframe['chr'].isin(['chr'+str(i) for i in range(1,23)]+['chrX'])].reset_index(drop=True)
#gene_names_list = tss_dataframe_sub['gene'].values
#chr_list = tss_dataframe_sub['chr'].values
#tss_list = tss_dataframe_sub['tss_1'].values.astype(np.int64)

##### candidate enhancers dataframe
#df_enhancers =  pd.read_csv(data_path+'/results/csv/distal_reg_paper/candidate_enhancers/{}/macs2_peaks.narrowPeak.sorted.candidateRegions_{}.bed'.format(cell_line, cell_line), header=None, delimiter='\t')
#df_enhancers.columns = ['chr', 'start', 'end']
#df_enhancers = df_enhancers[df_enhancers['chr'].isin(['chr'+str(i) for i in range(1,23)]+['chrX'])].reset_index(drop=True)
df_enhancers = pd.read_csv(data_path+'/results/csv/distal_reg_paper/candidate_enhancers/{}/EnhancerPredictionsAllPutative_{}.txt'.format(cell_line, cell_line), sep = '\t')
gene_names_list = np.unique(df_enhancers['TargetGene'].values)

##### load LR model
with open(data_path+'/results/csv/distal_reg_paper/{}/models/model_{}_FDR_{}_L_{}.pkl'.format(model, model, fdr, L),'rb') as f:
    clf = pickle.load(f)

for i, gene_name in enumerate(gene_names_list):
    print('gene {} = {}'.format(i, gene_name))
    if os.path.exists(data_path+'/results/csv/distal_reg_paper/gradients_per_gene/'+cell_line+'/grads_'+genome+'_FDR_'+fdr+'_'+gene_name+'.tsv'):
        df_grads_gene = pd.read_csv(data_path+'/results/csv/distal_reg_paper/gradients_per_gene/'+cell_line+'/grads_'+genome+'_FDR_'+fdr+'_'+gene_name+'.tsv', delimiter='\t')
        if len(df_grads_gene) > 0:
            enhancer_window_per_gene_start = df_grads_gene.loc[0,'end']
            enhancer_window_per_gene_end = df_grads_gene.loc[59999,'end']
            enhancer_window_per_gene_chr = df_grads_gene.loc[0,'chr']

            df_enhancers_per_gene = df_enhancers[(df_enhancers['TargetGene'] == gene_name) & (df_enhancers['chr'] == enhancer_window_per_gene_chr) & (df_enhancers['start'] >= enhancer_window_per_gene_start) & (df_enhancers['end'] <= enhancer_window_per_gene_end)].reset_index(drop=True)
            df_enhancers_per_gene.loc[:,'GraphReg.Score'] = 0.0

            X = np.zeros([1,19])
            for j in range(len(df_enhancers_per_gene)):
                enhancer_middle = (df_enhancers_per_gene.loc[j, 'start'] + df_enhancers_per_gene.loc[j, 'end']) // 2
                df_enhancer_middle = df_grads_gene[(df_grads_gene['start'] <= enhancer_middle) & (df_grads_gene['end'] > enhancer_middle)]
                df_tss_middle = df_grads_gene[(df_grads_gene['start'] <= df_grads_gene['TargetGeneTSS'].values[0]) & (df_grads_gene['end'] > df_grads_gene['TargetGeneTSS'].values[0])]
                df_enhancers_per_gene.loc[j, 'DistanceToTSS'] = X[0,18] = np.abs(enhancer_middle - df_grads_gene['TargetGeneTSS'].values[0])

                if (len(df_enhancer_middle) > 0 and len(df_tss_middle) > 0):
                    idx_mid = df_enhancer_middle.index[0]
                    idx_tss = df_tss_middle.index[0]

                    df_preds_around_enhancer = df_grads_gene.loc[idx_mid-L:idx_mid+L]
                    df_preds_around_tss = df_grads_gene.loc[idx_tss-L:idx_tss+L]

                    df_enhancers_per_gene.loc[j, 'H3K4me3_e_max_L_{}'.format(L)] = X[0,0] = np.max(2**(df_preds_around_enhancer['H3K4me3'].values) - 1)
                    df_enhancers_per_gene.loc[j, 'H3K27ac_e_max_L_{}'.format(L)] = X[0,1] =  np.max(2**(df_preds_around_enhancer['H3K27ac'].values) - 1)
                    df_enhancers_per_gene.loc[j, 'DNase_e_max_L_{}'.format(L)] = X[0,2] = np.max(2**(df_preds_around_enhancer['DNase'].values) - 1)
                    df_enhancers_per_gene.loc[j, 'H3K4me3_e_grad_max_L_{}'.format(L)] = X[0,3] = np.max(df_preds_around_enhancer['Grad_H3K4me3'].values)
                    df_enhancers_per_gene.loc[j, 'H3K27ac_e_grad_max_L_{}'.format(L)] = X[0,4] = np.max(df_preds_around_enhancer['Grad_H3K27ac'].values)
                    df_enhancers_per_gene.loc[j, 'DNase_e_grad_max_L_{}'.format(L)] = X[0,5] = np.max(df_preds_around_enhancer['Grad_DNase'].values)
                    df_enhancers_per_gene.loc[j, 'H3K4me3_e_grad_min_L_{}'.format(L)] = X[0,6] = np.min(df_preds_around_enhancer['Grad_H3K4me3'].values)
                    df_enhancers_per_gene.loc[j, 'H3K27ac_e_grad_min_L_{}'.format(L)] = X[0,7] = np.min(df_preds_around_enhancer['Grad_H3K27ac'].values)
                    df_enhancers_per_gene.loc[j, 'DNase_e_grad_min_L_{}'.format(L)] = X[0,8] = np.min(df_preds_around_enhancer['Grad_DNase'].values)

                    df_enhancers_per_gene.loc[j, 'H3K4me3_p_max_L_{}'.format(L)] = X[0,9] = np.max(2**(df_preds_around_tss['H3K4me3'].values) - 1)
                    df_enhancers_per_gene.loc[j, 'H3K27ac_p_max_L_{}'.format(L)] = X[0,10] = np.max(2**(df_preds_around_tss['H3K27ac'].values) - 1)
                    df_enhancers_per_gene.loc[j, 'DNase_p_max_L_{}'.format(L)] = X[0,11] = np.max(2**(df_preds_around_tss['DNase'].values) - 1)
                    df_enhancers_per_gene.loc[j, 'H3K4me3_p_grad_max_L_{}'.format(L)] = X[0,12] = np.max(df_preds_around_tss['Grad_H3K4me3'].values)
                    df_enhancers_per_gene.loc[j, 'H3K27ac_p_grad_max_L_{}'.format(L)] = X[0,13] = np.max(df_preds_around_tss['Grad_H3K27ac'].values)
                    df_enhancers_per_gene.loc[j, 'DNase_p_grad_max_L_{}'.format(L)] = X[0,14] = np.max(df_preds_around_tss['Grad_DNase'].values)
                    df_enhancers_per_gene.loc[j, 'H3K4me3_p_grad_min_L_{}'.format(L)] = X[0,15] = np.min(df_preds_around_tss['Grad_H3K4me3'].values)
                    df_enhancers_per_gene.loc[j, 'H3K27ac_p_grad_min_L_{}'.format(L)] = X[0,16] = np.min(df_preds_around_tss['Grad_H3K27ac'].values)
                    df_enhancers_per_gene.loc[j, 'DNase_p_grad_min_L_{}'.format(L)] = X[0,17] = np.min(df_preds_around_tss['Grad_DNase'].values)
                    
                    X = np.log(np.abs(X) + epsilon)
                    probs = clf.predict_proba(X)
                    df_enhancers_per_gene.loc[j, 'GraphReg.Score'] = probs[:,1]

            df_enhancers_per_gene = df_enhancers_per_gene.astype({'DistanceToTSS': 'int64'})
            df_enhancers_per_gene.to_csv(data_path+'/results/csv/distal_reg_paper/enhancer_preds_per_gene/'+cell_line+'/EG_preds_'+cell_line+'_'+genome+'_FDR_'+fdr+'_'+saliency_method+'_'+gene_name+'.tsv', sep = '\t', index=False)
            
#%%
##### extract features per gene
# data_path = '/media/labuser/STORAGE/GraphReg'   # data path
# organism = 'human'
# genome='hg38'                                   # hg19/hg38
# cell_line = 'K562'                              # K562/GM12878/hESC
# fdr = '1'
# L = 8
# saliency_method = 'saliency'

# df = pd.read_csv(data_path+'/results/csv/distal_reg_paper/enhancer_preds_per_gene/'+cell_line+'/GraphRegLR_EG_preds_'+cell_line+'_'+genome+'_FDR_'+fdr+'_L_'+str(L)+'_'+saliency_method+'.tsv', sep = '\t')
# df1 = df[(df['Score']>.1) & (df['DistanceToTSS']>20000)].reset_index(drop=True)

