import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import pickle

#%%
##### extract features

data_path = '/media/labuser/STORAGE/GraphReg'   # data path
organism = 'human'
genome='hg38'                                   # hg19/hg38
cell_line = 'K562'                              # K562/GM12878/mESC/hESC
dataset = 'combined'                    # 'fulco' or 'gasperini' or 'combined'

for fdr in ['001', '01', '1', '5', '9']:
    for saliency_method in ['saliency']:
        print('FDR: {} | {}'.format(fdr, saliency_method))

        #df_preds = pd.read_csv(data_path+'/results/csv/distal_reg_paper/'+dataset+'/EG_preds_'+cell_line+'_'+genome+'_FDR_'+fdr+'_'+saliency_method+'_'+dataset+'.tsv', delimiter='\t')

        ##### CRISPR dataframe
        if dataset == 'fulco':
            df_crispr = pd.read_csv(data_path+'/data/csv/CRISPR_benchmarking_data/EPCrisprBenchmark_Fulco2019_K562_GRCh38.tsv', delimiter='\t')
            df_crispr = df_crispr[df_crispr['ValidConnection']=="TRUE"].reset_index(drop=True)
        elif dataset == 'gasperini':
            df_crispr = pd.read_csv(data_path+'/data/csv/CRISPR_benchmarking_data/EPCrisprBenchmark_Gasperini2019_0.13gStd_0.8pwrAt15effect_GRCh38.tsv', delimiter='\t')
            df_crispr = df_crispr[df_crispr['ValidConnection']=="TRUE"].reset_index(drop=True)
        elif dataset == 'combined':
            df_crispr = pd.read_csv(data_path+'/data/csv/CRISPR_benchmarking_data/EPCrisprBenchmark_ensemble_data_GRCh38.tsv', delimiter='\t')

        for i in range(len(df_crispr)):
            print('fdr = {}, i = {}'.format(fdr, i))
            df_crispr_row = df_crispr.iloc[i]
            enhancer_middle = (df_crispr_row['chromStart'] + df_crispr_row['chromEnd']) // 2
            gene_name = df_crispr_row['measuredGeneSymbol']

            df_preds_gene = pd.read_csv(data_path+'/results/csv/distal_reg_paper/gradients_per_gene/'+cell_line+'/grads_'+genome+'_FDR_'+fdr+'_'+gene_name+'.tsv', delimiter='\t')

            #df_preds_gene = df_preds[df_preds['TargetGene'] == gene_name]
            if len(df_preds_gene) > 0:
                df_enhancer_middle = df_preds_gene[(df_preds_gene['start'] <= enhancer_middle) & (df_preds_gene['end'] > enhancer_middle)]
                df_tss_middle = df_preds_gene[(df_preds_gene['start'] <= df_preds_gene['TargetGeneTSS'].values[0]) & (df_preds_gene['end'] > df_preds_gene['TargetGeneTSS'].values[0])]
                df_crispr.loc[i, 'DistanceToTSS'] = np.abs(enhancer_middle - df_preds_gene['TargetGeneTSS'].values[0])
                
                if (len(df_enhancer_middle) > 0 and len(df_tss_middle) > 0):
                    idx_mid = df_enhancer_middle.index[0]
                    idx_tss = df_tss_middle.index[0]
                    for L in [2, 4, 8, 16, 32]:
                        df_preds_around_enhancer = df_preds_gene.loc[idx_mid-L:idx_mid+L]
                        df_preds_around_tss = df_preds_gene.loc[idx_tss-L:idx_tss+L]

                        df_crispr.loc[i, 'H3K4me3_e_max_L_{}'.format(L)] = np.max(2**(df_preds_around_enhancer['H3K4me3'].values) - 1)
                        df_crispr.loc[i, 'H3K27ac_e_max_L_{}'.format(L)] = np.max(2**(df_preds_around_enhancer['H3K27ac'].values) - 1)
                        df_crispr.loc[i, 'DNase_e_max_L_{}'.format(L)] = np.max(2**(df_preds_around_enhancer['DNase'].values) - 1)
                        df_crispr.loc[i, 'H3K4me3_e_grad_max_L_{}'.format(L)] = np.max(df_preds_around_enhancer['Grad_H3K4me3'].values)
                        df_crispr.loc[i, 'H3K27ac_e_grad_max_L_{}'.format(L)] = np.max(df_preds_around_enhancer['Grad_H3K27ac'].values)
                        df_crispr.loc[i, 'DNase_e_grad_max_L_{}'.format(L)] = np.max(df_preds_around_enhancer['Grad_DNase'].values)
                        df_crispr.loc[i, 'H3K4me3_e_grad_min_L_{}'.format(L)] = np.min(df_preds_around_enhancer['Grad_H3K4me3'].values)
                        df_crispr.loc[i, 'H3K27ac_e_grad_min_L_{}'.format(L)] = np.min(df_preds_around_enhancer['Grad_H3K27ac'].values)
                        df_crispr.loc[i, 'DNase_e_grad_min_L_{}'.format(L)] = np.min(df_preds_around_enhancer['Grad_DNase'].values)

                        df_crispr.loc[i, 'H3K4me3_p_max_L_{}'.format(L)] = np.max(2**(df_preds_around_tss['H3K4me3'].values) - 1)
                        df_crispr.loc[i, 'H3K27ac_p_max_L_{}'.format(L)] = np.max(2**(df_preds_around_tss['H3K27ac'].values) - 1)
                        df_crispr.loc[i, 'DNase_p_max_L_{}'.format(L)] = np.max(2**(df_preds_around_tss['DNase'].values) - 1)
                        df_crispr.loc[i, 'H3K4me3_p_grad_max_L_{}'.format(L)] = np.max(df_preds_around_tss['Grad_H3K4me3'].values)
                        df_crispr.loc[i, 'H3K27ac_p_grad_max_L_{}'.format(L)] = np.max(df_preds_around_tss['Grad_H3K27ac'].values)
                        df_crispr.loc[i, 'DNase_p_grad_max_L_{}'.format(L)] = np.max(df_preds_around_tss['Grad_DNase'].values)
                        df_crispr.loc[i, 'H3K4me3_p_grad_min_L_{}'.format(L)] = np.min(df_preds_around_tss['Grad_H3K4me3'].values)
                        df_crispr.loc[i, 'H3K27ac_p_grad_min_L_{}'.format(L)] = np.min(df_preds_around_tss['Grad_H3K27ac'].values)
                        df_crispr.loc[i, 'DNase_p_grad_min_L_{}'.format(L)] = np.min(df_preds_around_tss['Grad_DNase'].values)

        df_crispr.to_csv(data_path+'/results/csv/distal_reg_paper/'+dataset+'/features_'+cell_line+'_'+genome+'_'+dataset+'_FDR_'+fdr+'_'+saliency_method+'.tsv', sep = '\t', index=False)

#%%
##### classifier
from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.svm import SVC

data_path = '/media/labuser/STORAGE/GraphReg'   # data path
organism = 'human'
genome='hg38'                                   # hg19/hg38
cell_line = 'K562'                              # K562/GM12878/mESC/hESC
dataset = 'combined'                    # 'fulco' or 'gasperini' or 'combined'
save_model = True

epsilon = 0.01
scores = 0
plt.figure(figsize=(20,20))
for fdr in ['001', '01', '1', '5', '9']:
    for saliency_method in ['saliency']:
        if dataset in ['gasperini', 'combined']:
            df_crispr = pd.read_csv(data_path+'/results/csv/distal_reg_paper/'+dataset+'/features_'+cell_line+'_'+genome+'_'+dataset+'_FDR_'+fdr+'_'+saliency_method+'.tsv', delimiter = '\t')
            df_crispr = df_crispr[(~df_crispr['DNase_e_max_L_2'].isna()) & (~df_crispr['Regulated'].isna())].reset_index(drop=True)
        for L in [2, 4, 8, 16, 32]:
            print('FDR: {} | {} | L = {}'.format(fdr, saliency_method, L))
            if dataset == 'fulco':
                df_crispr = pd.read_csv(data_path+'/results/csv/distal_reg_paper/'+dataset+'/features_'+cell_line+'_'+genome+'_'+dataset+'_FDR_'+fdr+'_'+saliency_method+'_L_'+str(L)+'.tsv', delimiter = '\t')

            Y = df_crispr['Regulated'].values.astype(np.int64)
            X = np.zeros([Y.shape[0], 19])

            if dataset in ['gasperini', 'combined']:
                X[:,0] = df_crispr['H3K4me3_e_max_L_{}'.format(L)].values
                X[:,1] = df_crispr['H3K27ac_e_max_L_{}'.format(L)].values
                X[:,2] = df_crispr['DNase_e_max_L_{}'.format(L)].values
                X[:,3] = df_crispr['H3K4me3_e_grad_max_L_{}'.format(L)].values
                X[:,4] = df_crispr['H3K27ac_e_grad_max_L_{}'.format(L)].values
                X[:,5] = df_crispr['DNase_e_grad_max_L_{}'.format(L)].values
                X[:,6] = df_crispr['H3K4me3_e_grad_min_L_{}'.format(L)].values
                X[:,7] = df_crispr['H3K27ac_e_grad_min_L_{}'.format(L)].values
                X[:,8] = df_crispr['DNase_e_grad_min_L_{}'.format(L)].values
                X[:,9] = df_crispr['H3K4me3_p_max_L_{}'.format(L)].values
                X[:,10] = df_crispr['H3K27ac_p_max_L_{}'.format(L)].values
                X[:,11] = df_crispr['DNase_p_max_L_{}'.format(L)].values
                X[:,12] = df_crispr['H3K4me3_p_grad_max_L_{}'.format(L)].values
                X[:,13] = df_crispr['H3K27ac_p_grad_max_L_{}'.format(L)].values
                X[:,14] = df_crispr['DNase_p_grad_max_L_{}'.format(L)].values
                X[:,15] = df_crispr['H3K4me3_p_grad_min_L_{}'.format(L)].values
                X[:,16] = df_crispr['H3K27ac_p_grad_min_L_{}'.format(L)].values
                X[:,17] = df_crispr['DNase_p_grad_min_L_{}'.format(L)].values
                X[:,18] = df_crispr['DistanceToTSS'].values
            
            elif dataset == 'fulco':
                X[:,0] = df_crispr['H3K4me3_e_max'].values
                X[:,1] = df_crispr['H3K27ac_e_max'].values
                X[:,2] = df_crispr['DNase_e_max'].values
                X[:,3] = df_crispr['H3K4me3_e_grad_max'].values
                X[:,4] = df_crispr['H3K27ac_e_grad_max'].values
                X[:,5] = df_crispr['DNase_e_grad_max'].values
                X[:,6] = df_crispr['H3K4me3_e_grad_min'].values
                X[:,7] = df_crispr['H3K27ac_e_grad_min'].values
                X[:,8] = df_crispr['DNase_e_grad_min'].values
                X[:,9] = df_crispr['H3K4me3_p_max'].values
                X[:,10] = df_crispr['H3K27ac_p_max'].values
                X[:,11] = df_crispr['DNase_p_max'].values
                X[:,12] = df_crispr['H3K4me3_p_grad_max'].values
                X[:,13] = df_crispr['H3K27ac_p_grad_max'].values
                X[:,14] = df_crispr['DNase_p_grad_max'].values
                X[:,15] = df_crispr['H3K4me3_p_grad_min'].values
                X[:,16] = df_crispr['H3K27ac_p_grad_min'].values
                X[:,17] = df_crispr['DNase_p_grad_min'].values
                X[:,18] = df_crispr['DistanceToTSS'].values
            

            #X = np.sign(X) * (np.log(np.abs(X) + epsilon))
            X = np.log(np.abs(X) + epsilon)

            clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X, Y)
            # save model
            if save_model:
                with open(data_path+'/results/csv/distal_reg_paper/{}/models/model_{}_FDR_{}_L_{}.pkl'.format(dataset, dataset, fdr, L),'wb') as f:
                    pickle.dump(clf,f)

            probs = clf.predict_proba(X)
            scores = scores + probs[:,1]

            
            df_crispr['Score'] = probs[:,1]
            #df_crispr = df_crispr[df_crispr['DistanceToTSS'] >= 100000]
            #df_crispr = df_crispr[(df_crispr['DistanceToTSS'] >= 20000) & (df_crispr['DistanceToTSS'] < 100000)]

            Y_true = df_crispr['Regulated'].values.astype(np.int64)
            Y_pred = df_crispr['Score']

            precision, recall, thresholds = precision_recall_curve(Y_true, Y_pred)
            average_precision = average_precision_score(Y_true, Y_pred)
            aupr = auc(recall, precision)

            idx_recall_70_pct = np.argsort(np.abs(recall - 0.7))[0]
            recall_at_70_pct = recall[idx_recall_70_pct]
            precision_at_70_pct_recall = precision[idx_recall_70_pct]
            threshod_in_70_pct_recall = thresholds[idx_recall_70_pct]

            plt.plot(recall, precision, linewidth=3, label='FDR=0.{} | L={} | auPR={:6.4f} | Recall={:4.2f} | Precision={:4.2f} | Threshold={:4.2f}'.format(fdr, L, aupr, recall_at_70_pct, precision_at_70_pct_recall, threshod_in_70_pct_recall))
            
'''
df_crispr['Score'] = scores/5
#df_crispr = df_crispr[df_crispr['DistanceToTSS'] < 20000]
#df_crispr = df_crispr[(df_crispr['DistanceToTSS'] >= 20000) & (df_crispr['DistanceToTSS'] < 100000)]
#df_crispr = df_crispr[df_crispr['DistanceToTSS'] >= 100000]

Y_true = df_crispr['Regulated'].values.astype(np.int64)
Y_pred = df_crispr['Score']

precision, recall, thresholds = precision_recall_curve(Y_true, Y_pred)
average_precision = average_precision_score(Y_true, Y_pred)
aupr = auc(recall, precision)

plt.figure(figsize=(20,20))
plt.plot(recall, precision, linewidth=3, label='Ensemble | auPR={:6.4f}'.format(aupr))
'''
plt.title('{} | all = {} | positives = {}'.format(dataset, len(df_crispr), np.sum(df_crispr['Regulated']==True)), fontsize=40)
plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=40)
plt.xlabel("Recall", fontsize=40)
plt.ylabel("Precision", fontsize=40)
plt.tick_params(axis='x', labelsize=40)
plt.tick_params(axis='y', labelsize=40)
plt.grid()
plt.savefig(data_path+'/results/csv/distal_reg_paper/figs/PR_curve_LRScores_{}_all.pdf'.format(dataset), bbox_inches='tight')


#%%
##### cross-validation
from sklearn.linear_model import LogisticRegression

data_path = '/media/labuser/STORAGE/GraphReg'   # data path
organism = 'human'
genome='hg38'                                   # hg19/hg38
cell_line = 'K562'                              # K562/GM12878/mESC/hESC
dataset = 'gasperini'                    # 'fulco' or 'gasperini' or 'combined'

epsilon = 0.01
if dataset == 'fulco':
    n_folds = 101
elif  dataset == 'gasperini':
    n_folds = 376
elif dataset == 'combined':
    n_folds = 484

plt.figure(figsize=(20,20))
for fdr in ['001']: # ['001', '01', '1', '5', '9']:
    for saliency_method in ['saliency']:
        if dataset in ['gasperini', 'combined']:
            df_crispr = pd.read_csv(data_path+'/results/csv/distal_reg_paper/'+dataset+'/features_'+cell_line+'_'+genome+'_'+dataset+'_FDR_'+fdr+'_'+saliency_method+'.tsv', delimiter = '\t')
            df_crispr = df_crispr[(~df_crispr['DNase_e_max_L_2'].isna()) & (~df_crispr['Regulated'].isna())].reset_index(drop=True)
        for L in [8]: # [2, 4, 8, 16, 32]:
            print('FDR: {} | {} | L = {}'.format(fdr, saliency_method, L))
            if dataset == 'fulco':
                df_crispr = pd.read_csv(data_path+'/results/csv/distal_reg_paper/'+dataset+'/features_'+cell_line+'_'+genome+'_'+dataset+'_FDR_'+fdr+'_'+saliency_method+'_L_'+str(L)+'.tsv', delimiter = '\t')

            Y = df_crispr['Regulated'].values.astype(np.int64)
            X = np.zeros([Y.shape[0], 19])

            if dataset in ['gasperini', 'combined']:
                X[:,0] = df_crispr['H3K4me3_e_max_L_{}'.format(L)].values
                X[:,1] = df_crispr['H3K27ac_e_max_L_{}'.format(L)].values
                X[:,2] = df_crispr['DNase_e_max_L_{}'.format(L)].values
                X[:,3] = df_crispr['H3K4me3_e_grad_max_L_{}'.format(L)].values
                X[:,4] = df_crispr['H3K27ac_e_grad_max_L_{}'.format(L)].values
                X[:,5] = df_crispr['DNase_e_grad_max_L_{}'.format(L)].values
                X[:,6] = df_crispr['H3K4me3_e_grad_min_L_{}'.format(L)].values
                X[:,7] = df_crispr['H3K27ac_e_grad_min_L_{}'.format(L)].values
                X[:,8] = df_crispr['DNase_e_grad_min_L_{}'.format(L)].values
                X[:,9] = df_crispr['H3K4me3_p_max_L_{}'.format(L)].values
                X[:,10] = df_crispr['H3K27ac_p_max_L_{}'.format(L)].values
                X[:,11] = df_crispr['DNase_p_max_L_{}'.format(L)].values
                X[:,12] = df_crispr['H3K4me3_p_grad_max_L_{}'.format(L)].values
                X[:,13] = df_crispr['H3K27ac_p_grad_max_L_{}'.format(L)].values
                X[:,14] = df_crispr['DNase_p_grad_max_L_{}'.format(L)].values
                X[:,15] = df_crispr['H3K4me3_p_grad_min_L_{}'.format(L)].values
                X[:,16] = df_crispr['H3K27ac_p_grad_min_L_{}'.format(L)].values
                X[:,17] = df_crispr['DNase_p_grad_min_L_{}'.format(L)].values
                X[:,18] = df_crispr['DistanceToTSS'].values
            
            elif dataset == 'fulco':
                X[:,0] = df_crispr['H3K4me3_e_max'].values
                X[:,1] = df_crispr['H3K27ac_e_max'].values
                X[:,2] = df_crispr['DNase_e_max'].values
                X[:,3] = df_crispr['H3K4me3_e_grad_max'].values
                X[:,4] = df_crispr['H3K27ac_e_grad_max'].values
                X[:,5] = df_crispr['DNase_e_grad_max'].values
                X[:,6] = df_crispr['H3K4me3_e_grad_min'].values
                X[:,7] = df_crispr['H3K27ac_e_grad_min'].values
                X[:,8] = df_crispr['DNase_e_grad_min'].values
                X[:,9] = df_crispr['H3K4me3_p_max'].values
                X[:,10] = df_crispr['H3K27ac_p_max'].values
                X[:,11] = df_crispr['DNase_p_max'].values
                X[:,12] = df_crispr['H3K4me3_p_grad_max'].values
                X[:,13] = df_crispr['H3K27ac_p_grad_max'].values
                X[:,14] = df_crispr['DNase_p_grad_max'].values
                X[:,15] = df_crispr['H3K4me3_p_grad_min'].values
                X[:,16] = df_crispr['H3K27ac_p_grad_min'].values
                X[:,17] = df_crispr['DNase_p_grad_min'].values
                X[:,18] = df_crispr['DistanceToTSS'].values
            

            #X = np.sign(X) * (np.log(np.abs(X) + epsilon))
            X = np.log(np.abs(X) + epsilon)
            #X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

            n1 = np.sum(Y)
            n0 = len(Y) - n1

            b1 = n1//n_folds
            b0 = n0//n_folds

            idx = np.arange(len(Y))
            idx1 = np.where(Y==1)[0]
            idx0 = np.where(Y==0)[0]

            for c in range(n_folds):
                if c < n_folds - 1:
                    idx1_test = idx1[c*b1:(c+1)*b1]
                    idx0_test = idx0[c*b0:(c+1)*b0]
                    idx_test = np.hstack((idx1_test, idx0_test))
                    idx_train = np.delete(idx, idx_test)
                else:
                    idx1_test = idx1[c*b1:]
                    idx0_test = idx0[c*b0:]
                    idx_test = np.hstack((idx1_test, idx0_test))
                    idx_train = np.delete(idx, idx_test)

                X_test = X[idx_test, :]
                Y_test = Y[idx_test]
                X_train = X[idx_train, :]
                Y_train = Y[idx_train]

                clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
                probs = clf.predict_proba(X_test)
                df_crispr.loc[idx_test, 'Score'] = probs[:,1]

            if dataset == 'gasperini':
                df_crispr_sub = pd.concat([df_crispr.iloc[:,0:21], df_crispr.iloc[:,-1]], axis=1)
            elif dataset == 'fulco':
                df_crispr_sub = pd.concat([df_crispr.iloc[:,0:17], df_crispr.iloc[:,-1]], axis=1)
            elif dataset == 'combined':
                df_crispr_sub = pd.concat([df_crispr.iloc[:,0:26], df_crispr.iloc[:,-1]], axis=1)
            #df_crispr_sub = df_crispr[df_crispr['DistanceToTSS'] < 20000]
            #df_crispr_sub = df_crispr[(df_crispr['DistanceToTSS'] >= 20000) & (df_crispr['DistanceToTSS'] < 100000)]
            #df_crispr_sub = df_crispr[df_crispr['DistanceToTSS'] >= 100000]
            
            Y_true = df_crispr_sub['Regulated'].values.astype(np.int64)
            Y_pred = df_crispr_sub['Score']

            precision, recall, thresholds = precision_recall_curve(Y_true, Y_pred)
            average_precision = average_precision_score(Y_true, Y_pred)
            aupr = auc(recall, precision)

            idx_recall_70_pct = np.argsort(np.abs(recall - 0.7))[0]
            recall_at_70_pct = recall[idx_recall_70_pct]
            precision_at_70_pct_recall = precision[idx_recall_70_pct]
            threshod_in_70_pct_recall = thresholds[idx_recall_70_pct]

            plt.plot(recall, precision, linewidth=3, label='FDR=0.{} | L={} | auPR={:6.4f} | Recall={:4.2f} | Precision={:4.2f} | Threshold={:4.2f}'.format(fdr, L, aupr, recall_at_70_pct, precision_at_70_pct_recall, threshod_in_70_pct_recall))

plt.title('{} | all = {} | positives = {}'.format(dataset, len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=40)
plt.xlabel("Recall", fontsize=40)
plt.ylabel("Precision", fontsize=40)
plt.tick_params(axis='x', labelsize=40)
plt.tick_params(axis='y', labelsize=40)
plt.grid()
plt.savefig(data_path+'/results/csv/distal_reg_paper/figs/PR_curve_LRScores_{}FoldCV_{}_best_all.pdf'.format(n_folds, dataset), bbox_inches='tight')

# save to tsv
#df_crispr_sub.to_csv(data_path+'/results/csv/distal_reg_paper/'+dataset+'/EG_preds_'+cell_line+'_'+genome+'_'+dataset+'_FDR_'+fdr+'_'+saliency_method+'_L_'+str(L)+'.tsv', sep = '\t', index=False)


#%%
##### cross-validation by holding out chromosomes
from sklearn.linear_model import LogisticRegression

data_path = '/media/labuser/STORAGE/GraphReg'   # data path
organism = 'human'
genome='hg38'                                   # hg19/hg38
cell_line = 'K562'                              # K562/GM12878/mESC/hESC
dataset = 'combined'                    # 'fulco' or 'gasperini' or 'combined'
save_model = False
load_combined_model = True
epsilon = 0.01

chr_list = ['chr{}'.format(i) for i in range(1,23)] + ['chrX']

plt.figure(figsize=(20,20))
for fdr in ['1']: # ['001', '01', '1', '5', '9']:
    for saliency_method in ['saliency']:
        if dataset in ['gasperini', 'combined']:
            df_crispr = pd.read_csv(data_path+'/results/csv/distal_reg_paper/'+dataset+'/features_'+cell_line+'_'+genome+'_'+dataset+'_FDR_'+fdr+'_'+saliency_method+'.tsv', delimiter = '\t')
            df_crispr = df_crispr[(~df_crispr['DNase_e_max_L_2'].isna()) & (~df_crispr['Regulated'].isna())].reset_index(drop=True)
        for L in [8]: # [2, 4, 8, 16, 32]:
            print('FDR: {} | {} | L = {}'.format(fdr, saliency_method, L))
            if dataset == 'fulco':
                df_crispr = pd.read_csv(data_path+'/results/csv/distal_reg_paper/'+dataset+'/features_'+cell_line+'_'+genome+'_'+dataset+'_FDR_'+fdr+'_'+saliency_method+'_L_'+str(L)+'.tsv', delimiter = '\t')

            Y = df_crispr['Regulated'].values.astype(np.int64)
            X = np.zeros([Y.shape[0], 19])

            if dataset in ['gasperini', 'combined']:
                X[:,0] = df_crispr['H3K4me3_e_max_L_{}'.format(L)].values
                X[:,1] = df_crispr['H3K27ac_e_max_L_{}'.format(L)].values
                X[:,2] = df_crispr['DNase_e_max_L_{}'.format(L)].values
                X[:,3] = df_crispr['H3K4me3_e_grad_max_L_{}'.format(L)].values
                X[:,4] = df_crispr['H3K27ac_e_grad_max_L_{}'.format(L)].values
                X[:,5] = df_crispr['DNase_e_grad_max_L_{}'.format(L)].values
                X[:,6] = df_crispr['H3K4me3_e_grad_min_L_{}'.format(L)].values
                X[:,7] = df_crispr['H3K27ac_e_grad_min_L_{}'.format(L)].values
                X[:,8] = df_crispr['DNase_e_grad_min_L_{}'.format(L)].values
                X[:,9] = df_crispr['H3K4me3_p_max_L_{}'.format(L)].values
                X[:,10] = df_crispr['H3K27ac_p_max_L_{}'.format(L)].values
                X[:,11] = df_crispr['DNase_p_max_L_{}'.format(L)].values
                X[:,12] = df_crispr['H3K4me3_p_grad_max_L_{}'.format(L)].values
                X[:,13] = df_crispr['H3K27ac_p_grad_max_L_{}'.format(L)].values
                X[:,14] = df_crispr['DNase_p_grad_max_L_{}'.format(L)].values
                X[:,15] = df_crispr['H3K4me3_p_grad_min_L_{}'.format(L)].values
                X[:,16] = df_crispr['H3K27ac_p_grad_min_L_{}'.format(L)].values
                X[:,17] = df_crispr['DNase_p_grad_min_L_{}'.format(L)].values
                X[:,18] = df_crispr['DistanceToTSS'].values
            
            elif dataset == 'fulco':
                X[:,0] = df_crispr['H3K4me3_e_max'].values
                X[:,1] = df_crispr['H3K27ac_e_max'].values
                X[:,2] = df_crispr['DNase_e_max'].values
                X[:,3] = df_crispr['H3K4me3_e_grad_max'].values
                X[:,4] = df_crispr['H3K27ac_e_grad_max'].values
                X[:,5] = df_crispr['DNase_e_grad_max'].values
                X[:,6] = df_crispr['H3K4me3_e_grad_min'].values
                X[:,7] = df_crispr['H3K27ac_e_grad_min'].values
                X[:,8] = df_crispr['DNase_e_grad_min'].values
                X[:,9] = df_crispr['H3K4me3_p_max'].values
                X[:,10] = df_crispr['H3K27ac_p_max'].values
                X[:,11] = df_crispr['DNase_p_max'].values
                X[:,12] = df_crispr['H3K4me3_p_grad_max'].values
                X[:,13] = df_crispr['H3K27ac_p_grad_max'].values
                X[:,14] = df_crispr['DNase_p_grad_max'].values
                X[:,15] = df_crispr['H3K4me3_p_grad_min'].values
                X[:,16] = df_crispr['H3K27ac_p_grad_min'].values
                X[:,17] = df_crispr['DNase_p_grad_min'].values
                X[:,18] = df_crispr['DistanceToTSS'].values

            X = np.log(np.abs(X) + epsilon)
            idx = np.arange(len(Y))

            for chr in chr_list:
                idx_test = df_crispr[df_crispr['chrom']==chr].index.values
                if len(idx_test) > 0:
                    idx_train = np.delete(idx, idx_test)

                    X_test = X[idx_test, :]
                    Y_test = Y[idx_test]
                    X_train = X[idx_train, :]
                    Y_train = Y[idx_train]
                    
                    if dataset in ['gasperini', 'combined']:
                        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
                        if save_model:
                            with open(data_path+'/results/csv/distal_reg_paper/{}/models/model_{}_FDR_{}_L_{}_test_{}.pkl'.format(dataset, dataset, fdr, L, chr),'wb') as f:
                                pickle.dump(clf,f)
                    elif dataset == 'fulco':
                        if load_combined_model:
                            with open(data_path+'/results/csv/distal_reg_paper/combined/models/model_combined_FDR_{}_L_{}_test_{}.pkl'.format(fdr, L, chr),'rb') as f:
                                clf = pickle.load(f)
                        else:
                            clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)

                    probs = clf.predict_proba(X_test)
                    df_crispr.loc[idx_test, 'Score'] = probs[:,1]
            

            if dataset == 'gasperini':
                df_crispr_sub = pd.concat([df_crispr.iloc[:,0:21], df_crispr.iloc[:,-1]], axis=1)
            elif dataset == 'fulco':
                df_crispr_sub = pd.concat([df_crispr.iloc[:,0:17], df_crispr.iloc[:,-1]], axis=1)
            elif dataset == 'combined':
                df_crispr_sub = pd.concat([df_crispr.iloc[:,0:26], df_crispr.iloc[:,-1]], axis=1)
            #df_crispr_sub = df_crispr[df_crispr['DistanceToTSS'] < 20000]
            #df_crispr_sub = df_crispr[(df_crispr['DistanceToTSS'] >= 20000) & (df_crispr['DistanceToTSS'] < 100000)]
            #df_crispr_sub = df_crispr[df_crispr['DistanceToTSS'] >= 100000]
            
            Y_true = df_crispr_sub['Regulated'].values.astype(np.int64)
            Y_pred = df_crispr_sub['Score']

            precision, recall, thresholds = precision_recall_curve(Y_true, Y_pred)
            average_precision = average_precision_score(Y_true, Y_pred)
            aupr = auc(recall, precision)

            idx_recall_70_pct = np.argsort(np.abs(recall - 0.7))[0]
            recall_at_70_pct = recall[idx_recall_70_pct]
            precision_at_70_pct_recall = precision[idx_recall_70_pct]
            threshod_in_70_pct_recall = thresholds[idx_recall_70_pct]

            plt.plot(recall, precision, linewidth=3, label='FDR=0.{} | L={} | auPR={:6.4f} | Recall={:4.2f} | Precision={:4.2f} | Threshold={:4.2f}'.format(fdr, L, aupr, recall_at_70_pct, precision_at_70_pct_recall, threshod_in_70_pct_recall))

plt.title('{} | all = {} | positives = {}'.format(dataset, len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=40)
plt.xlabel("Recall", fontsize=40)
plt.ylabel("Precision", fontsize=40)
plt.tick_params(axis='x', labelsize=40)
plt.tick_params(axis='y', labelsize=40)
plt.grid()
#plt.savefig(data_path+'/results/csv/distal_reg_paper/figs/PR_curve_LRScores_CV_based_on_chrs_{}_best_all.pdf'.format(dataset), bbox_inches='tight')

# save to tsv
df_crispr_sub.to_csv(data_path+'/results/csv/distal_reg_paper/'+dataset+'/EG_preds_on_holdout_chrs_'+cell_line+'_'+genome+'_'+dataset+'_FDR_'+fdr+'_'+saliency_method+'_L_'+str(L)+'.tsv', sep = '\t', index=False)


# %%
##### Ensemble model - cross validation
from sklearn.linear_model import LogisticRegression

data_path = '/media/labuser/STORAGE/GraphReg'   # data path
organism = 'human'
genome='hg38'                                   # hg19/hg38
cell_line = 'K562'                              # K562/GM12878/mESC/hESC
dataset = 'fulco'                    # 'fulco' or 'gasperini' or 'combined'

epsilon = 0.01

if dataset == 'fulco':
    n_folds = 101
elif  dataset == 'gasperini':
    n_folds = 376
elif dataset == 'combined':
    n_folds = 484

cnt = 0
for c in range(n_folds):
    print('Cross fold = {}'.format(c+1))
    scores = 0
    for fdr in ['001', '01', '1', '5', '9']:
        for saliency_method in ['saliency']:
            if dataset in ['gasperini', 'combined']:
                df_crispr = pd.read_csv(data_path+'/results/csv/distal_reg_paper/'+dataset+'/features_'+cell_line+'_'+genome+'_'+dataset+'_FDR_'+fdr+'_'+saliency_method+'.tsv', delimiter = '\t')
                df_crispr = df_crispr[(~df_crispr['DNase_e_max_L_2'].isna()) & (~df_crispr['Regulated'].isna())].reset_index(drop=True)

            for L in [2, 4, 8, 16, 32]:
                cnt += 1

                if dataset == 'fulco':
                    df_crispr = pd.read_csv(data_path+'/results/csv/distal_reg_paper/'+dataset+'/features_'+cell_line+'_'+genome+'_'+dataset+'_FDR_'+fdr+'_'+saliency_method+'_L_'+str(L)+'.tsv', delimiter = '\t')

                if cnt == 1:
                    df_crispr_copy = df_crispr.copy()

                Y = df_crispr['Regulated'].values.astype(np.int64)
                X = np.zeros([Y.shape[0], 19])

                if dataset in ['gasperini', 'combined']:
                    X[:,0] = df_crispr['H3K4me3_e_max_L_{}'.format(L)].values
                    X[:,1] = df_crispr['H3K27ac_e_max_L_{}'.format(L)].values
                    X[:,2] = df_crispr['DNase_e_max_L_{}'.format(L)].values
                    X[:,3] = df_crispr['H3K4me3_e_grad_max_L_{}'.format(L)].values
                    X[:,4] = df_crispr['H3K27ac_e_grad_max_L_{}'.format(L)].values
                    X[:,5] = df_crispr['DNase_e_grad_max_L_{}'.format(L)].values
                    X[:,6] = df_crispr['H3K4me3_e_grad_min_L_{}'.format(L)].values
                    X[:,7] = df_crispr['H3K27ac_e_grad_min_L_{}'.format(L)].values
                    X[:,8] = df_crispr['DNase_e_grad_min_L_{}'.format(L)].values
                    X[:,9] = df_crispr['H3K4me3_p_max_L_{}'.format(L)].values
                    X[:,10] = df_crispr['H3K27ac_p_max_L_{}'.format(L)].values
                    X[:,11] = df_crispr['DNase_p_max_L_{}'.format(L)].values
                    X[:,12] = df_crispr['H3K4me3_p_grad_max_L_{}'.format(L)].values
                    X[:,13] = df_crispr['H3K27ac_p_grad_max_L_{}'.format(L)].values
                    X[:,14] = df_crispr['DNase_p_grad_max_L_{}'.format(L)].values
                    X[:,15] = df_crispr['H3K4me3_p_grad_min_L_{}'.format(L)].values
                    X[:,16] = df_crispr['H3K27ac_p_grad_min_L_{}'.format(L)].values
                    X[:,17] = df_crispr['DNase_p_grad_min_L_{}'.format(L)].values
                    X[:,18] = df_crispr['DistanceToTSS'].values
                
                elif dataset == 'fulco':
                    X[:,0] = df_crispr['H3K4me3_e_max'].values
                    X[:,1] = df_crispr['H3K27ac_e_max'].values
                    X[:,2] = df_crispr['DNase_e_max'].values
                    X[:,3] = df_crispr['H3K4me3_e_grad_max'].values
                    X[:,4] = df_crispr['H3K27ac_e_grad_max'].values
                    X[:,5] = df_crispr['DNase_e_grad_max'].values
                    X[:,6] = df_crispr['H3K4me3_e_grad_min'].values
                    X[:,7] = df_crispr['H3K27ac_e_grad_min'].values
                    X[:,8] = df_crispr['DNase_e_grad_min'].values
                    X[:,9] = df_crispr['H3K4me3_p_max'].values
                    X[:,10] = df_crispr['H3K27ac_p_max'].values
                    X[:,11] = df_crispr['DNase_p_max'].values
                    X[:,12] = df_crispr['H3K4me3_p_grad_max'].values
                    X[:,13] = df_crispr['H3K27ac_p_grad_max'].values
                    X[:,14] = df_crispr['DNase_p_grad_max'].values
                    X[:,15] = df_crispr['H3K4me3_p_grad_min'].values
                    X[:,16] = df_crispr['H3K27ac_p_grad_min'].values
                    X[:,17] = df_crispr['DNase_p_grad_min'].values
                    X[:,18] = df_crispr['DistanceToTSS'].values
                    
                X = np.log(np.abs(X) + epsilon)
                #X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

                n1 = np.sum(Y)
                n0 = len(Y) - n1

                b1 = n1//n_folds
                b0 = n0//n_folds

                idx = np.arange(len(Y))
                idx1 = np.where(Y==1)[0]
                idx0 = np.where(Y==0)[0]

                if c < n_folds - 1:
                    idx1_test = idx1[c*b1:(c+1)*b1]
                    idx0_test = idx0[c*b0:(c+1)*b0]
                    idx_test = np.hstack((idx1_test, idx0_test))
                    idx_train = np.delete(idx, idx_test)
                else:
                    idx1_test = idx1[c*b1:]
                    idx0_test = idx0[c*b0:]
                    idx_test = np.hstack((idx1_test, idx0_test))
                    idx_train = np.delete(idx, idx_test)

                X_test = X[idx_test, :]
                Y_test = Y[idx_test]
                X_train = X[idx_train, :]
                Y_train = Y[idx_train]

                clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
                probs = clf.predict_proba(X_test)
                scores = scores + probs[:,1]
    
    df_crispr_copy.loc[idx_test, 'Score'] = scores/25

df_crispr = df_crispr_copy
#df_crispr = df_crispr_copy[df_crispr_copy['DistanceToTSS'] < 20000]
#df_crispr = df_crispr_copy[(df_crispr_copy['DistanceToTSS'] >= 20000) & (df_crispr_copy['DistanceToTSS'] < 100000)]
#df_crispr = df_crispr_copy[df_crispr_copy['DistanceToTSS'] >= 100000]

Y_true = df_crispr['Regulated'].values.astype(np.int64)
Y_pred = df_crispr['Score']

precision, recall, thresholds = precision_recall_curve(Y_true, Y_pred)
average_precision = average_precision_score(Y_true, Y_pred)
aupr = auc(recall, precision)

idx_recall_70_pct = np.argsort(np.abs(recall - 0.7))[0]
recall_at_70_pct = recall[idx_recall_70_pct]
precision_at_70_pct_recall = precision[idx_recall_70_pct]
threshod_in_70_pct_recall = thresholds[idx_recall_70_pct]

plt.figure(figsize=(20,20))
plt.plot(recall, precision, linewidth=3, label='Ensemble | auPR={:6.4f} | Recall={:4.2f} | Precision={:4.2f} | Threshold={:4.2f}'.format(aupr, recall_at_70_pct, precision_at_70_pct_recall, threshod_in_70_pct_recall))
plt.title('{} | all = {} | positives = {}'.format(dataset, len(df_crispr), np.sum(df_crispr['Regulated']==True)), fontsize=40)
plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=40)
plt.xlabel("Recall", fontsize=40)
plt.ylabel("Precision", fontsize=40)
plt.tick_params(axis='x', labelsize=40)
plt.tick_params(axis='y', labelsize=40)
plt.grid()
plt.savefig(data_path+'/results/csv/distal_reg_paper/figs/PR_curve_LRScores_{}FoldCV_{}_ensemble_all.pdf'.format(n_folds, dataset), bbox_inches='tight')


# %%

##### generalization  (fulco --> gasperini / gasperini --> fulco)
from sklearn.linear_model import LogisticRegression

data_path = '/media/labuser/STORAGE/GraphReg'   # data path
organism = 'human'
genome='hg38'                                   # hg19/hg38
cell_line = 'K562'                              # K562/GM12878/mESC/hESC
dataset = 'fulco'                    # 'fulco' or 'gasperini'
model = 'combined'                          # 'fulco' or 'gasperini' or 'combined'

epsilon = 0.01
scores = 0
#plt.figure(figsize=(20,20))
for fdr in ['001', '01', '1', '5', '9']:
    for saliency_method in ['saliency']:
        if dataset == 'gasperini':
            df_crispr = pd.read_csv(data_path+'/results/csv/distal_reg_paper/'+dataset+'/features_'+cell_line+'_'+genome+'_'+dataset+'_FDR_'+fdr+'_'+saliency_method+'.tsv', delimiter = '\t')
            df_crispr = df_crispr.dropna().reset_index(drop=True)
        for L in [2, 4, 8, 16, 32]:
            print('FDR: {} | {} | L = {}'.format(fdr, saliency_method, L))
            if dataset == 'fulco':
                df_crispr = pd.read_csv(data_path+'/results/csv/distal_reg_paper/'+dataset+'/features_'+cell_line+'_'+genome+'_'+dataset+'_FDR_'+fdr+'_'+saliency_method+'_L_'+str(L)+'.tsv', delimiter = '\t')

            Y = df_crispr['Regulated'].values.astype(np.int64)
            X = np.zeros([Y.shape[0], 19])

            if dataset == 'gasperini':
                X[:,0] = df_crispr['H3K4me3_e_max_L_{}'.format(L)].values
                X[:,1] = df_crispr['H3K27ac_e_max_L_{}'.format(L)].values
                X[:,2] = df_crispr['DNase_e_max_L_{}'.format(L)].values
                X[:,3] = df_crispr['H3K4me3_e_grad_max_L_{}'.format(L)].values
                X[:,4] = df_crispr['H3K27ac_e_grad_max_L_{}'.format(L)].values
                X[:,5] = df_crispr['DNase_e_grad_max_L_{}'.format(L)].values
                X[:,6] = df_crispr['H3K4me3_e_grad_min_L_{}'.format(L)].values
                X[:,7] = df_crispr['H3K27ac_e_grad_min_L_{}'.format(L)].values
                X[:,8] = df_crispr['DNase_e_grad_min_L_{}'.format(L)].values
                X[:,9] = df_crispr['H3K4me3_p_max_L_{}'.format(L)].values
                X[:,10] = df_crispr['H3K27ac_p_max_L_{}'.format(L)].values
                X[:,11] = df_crispr['DNase_p_max_L_{}'.format(L)].values
                X[:,12] = df_crispr['H3K4me3_p_grad_max_L_{}'.format(L)].values
                X[:,13] = df_crispr['H3K27ac_p_grad_max_L_{}'.format(L)].values
                X[:,14] = df_crispr['DNase_p_grad_max_L_{}'.format(L)].values
                X[:,15] = df_crispr['H3K4me3_p_grad_min_L_{}'.format(L)].values
                X[:,16] = df_crispr['H3K27ac_p_grad_min_L_{}'.format(L)].values
                X[:,17] = df_crispr['DNase_p_grad_min_L_{}'.format(L)].values
                X[:,18] = df_crispr['DistanceToTSS'].values
            
            elif dataset == 'fulco':
                X[:,0] = df_crispr['H3K4me3_e_max'].values
                X[:,1] = df_crispr['H3K27ac_e_max'].values
                X[:,2] = df_crispr['DNase_e_max'].values
                X[:,3] = df_crispr['H3K4me3_e_grad_max'].values
                X[:,4] = df_crispr['H3K27ac_e_grad_max'].values
                X[:,5] = df_crispr['DNase_e_grad_max'].values
                X[:,6] = df_crispr['H3K4me3_e_grad_min'].values
                X[:,7] = df_crispr['H3K27ac_e_grad_min'].values
                X[:,8] = df_crispr['DNase_e_grad_min'].values
                X[:,9] = df_crispr['H3K4me3_p_max'].values
                X[:,10] = df_crispr['H3K27ac_p_max'].values
                X[:,11] = df_crispr['DNase_p_max'].values
                X[:,12] = df_crispr['H3K4me3_p_grad_max'].values
                X[:,13] = df_crispr['H3K27ac_p_grad_max'].values
                X[:,14] = df_crispr['DNase_p_grad_max'].values
                X[:,15] = df_crispr['H3K4me3_p_grad_min'].values
                X[:,16] = df_crispr['H3K27ac_p_grad_min'].values
                X[:,17] = df_crispr['DNase_p_grad_min'].values
                X[:,18] = df_crispr['DistanceToTSS'].values
            

            #X = np.sign(X) * (np.log(np.abs(X) + epsilon))
            X = np.log(np.abs(X) + epsilon)

            # load model
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/model_{}_FDR_{}_L_{}.pkl'.format(model, model, fdr, L),'rb') as f:
                clf = pickle.load(f)

            probs = clf.predict_proba(X)
            scores = scores + probs[:,1]

            '''
            df_crispr['Score'] = probs[:,1]
            #df_crispr = df_crispr[df_crispr['DistanceToTSS'] < 20000]
            #df_crispr = df_crispr[(df_crispr['DistanceToTSS'] >= 20000) & (df_crispr['DistanceToTSS'] < 100000)]
            #df_crispr = df_crispr[df_crispr['DistanceToTSS'] >= 100000]

            Y_true = df_crispr['Regulated'].values.astype(np.int64)
            Y_pred = df_crispr['Score']

            precision, recall, thresholds = precision_recall_curve(Y_true, Y_pred)
            average_precision = average_precision_score(Y_true, Y_pred)
            aupr = auc(recall, precision)

            plt.plot(recall, precision, linewidth=3, label='fdr 0.{} | {} | L={} | auPR={:6.4f}'.format(fdr, saliency_method, L, aupr))
            '''

df_crispr['Score'] = scores/25
#df_crispr = df_crispr[df_crispr['DistanceToTSS'] < 20000]
#df_crispr = df_crispr[(df_crispr['DistanceToTSS'] >= 20000) & (df_crispr['DistanceToTSS'] < 100000)]
#df_crispr = df_crispr[df_crispr['DistanceToTSS'] >= 100000]

Y_true = df_crispr['Regulated'].values.astype(np.int64)
Y_pred = df_crispr['Score']

precision, recall, thresholds = precision_recall_curve(Y_true, Y_pred)
average_precision = average_precision_score(Y_true, Y_pred)
aupr = auc(recall, precision)

plt.figure(figsize=(20,20))
plt.plot(recall, precision, linewidth=3, label='Ensemble | auPR={:6.4f}'.format(aupr))
plt.title('{} to {} | all = {} | positives = {}'.format(model, dataset, len(df_crispr), np.sum(df_crispr['Regulated']==True)), fontsize=40)
plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=40)
plt.xlabel("Recall", fontsize=40)
plt.ylabel("Precision", fontsize=40)
plt.tick_params(axis='x', labelsize=40)
plt.tick_params(axis='y', labelsize=40)
plt.grid()
#plt.savefig(data_path+'/results/csv/distal_reg_paper/figs/PR_curve_LRScores_{}_to_{}_ensemble_all.pdf'.format(model, dataset), bbox_inches='tight')

# %%
##### merge both datasets (best models from both datasets)

from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.svm import SVC

data_path = '/media/labuser/STORAGE/GraphReg'   # data path
organism = 'human'
genome='hg38'                                   # hg19/hg38
cell_line = 'K562'                              # K562/GM12878/mESC/hESC
dataset1 = 'gasperini'                    # 'fulco' or 'gasperini'
dataset2 = 'fulco'

df_crispr_1 = pd.read_csv(data_path+'/results/csv/distal_reg_paper/'+dataset1+'/EG_preds_K562_hg38_gasperini_FDR_001_saliency_L_32.tsv', delimiter = '\t')
df_crispr_2 = pd.read_csv(data_path+'/results/csv/distal_reg_paper/'+dataset2+'/EG_preds_K562_hg38_fulco_FDR_5_saliency_L_8.tsv', delimiter = '\t')

df_crispr = df_crispr_1[['Regulated', 'DistanceToTSS', 'Score']].append(df_crispr_2[['Regulated', 'DistanceToTSS', 'Score']]).reset_index(drop=True)

Y_true = df_crispr['Regulated'].values.astype(np.int64)
Y_pred = df_crispr['Score']

precision, recall, thresholds = precision_recall_curve(Y_true, Y_pred)
average_precision = average_precision_score(Y_true, Y_pred)
aupr = auc(recall, precision)

idx_recall_70_pct = np.argsort(np.abs(recall - 0.7))[0]
recall_at_70_pct = recall[idx_recall_70_pct]
precision_at_70_pct_recall = precision[idx_recall_70_pct]
threshod_in_70_pct_recall = thresholds[idx_recall_70_pct]

plt.figure(figsize=(20,20))
plt.plot(recall, precision, linewidth=3, label='auPR={:6.4f} | \n Recall={:4.2f} | Precision={:4.2f} | Threshold={:4.2f}'.format(aupr, recall_at_70_pct, precision_at_70_pct_recall, threshod_in_70_pct_recall))
plt.title('Merged datasets (best preds) | all = {} | positives = {}'.format(len(df_crispr), np.sum(df_crispr['Regulated']==True)), fontsize=40)
plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=40)
plt.xlabel("Recall", fontsize=40)
plt.ylabel("Precision", fontsize=40)
plt.tick_params(axis='x', labelsize=40)
plt.tick_params(axis='y', labelsize=40)
plt.grid()
plt.savefig(data_path+'/results/csv/distal_reg_paper/figs/PR_curve_LRScores_merged_datasets_best_CVpreds_all.pdf', bbox_inches='tight')

# %%
