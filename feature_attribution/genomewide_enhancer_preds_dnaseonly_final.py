import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LogisticRegression
import shap
import glob
import os

data_path = '/media/labuser/STORAGE/GraphReg'   # data path
organism = 'human'
genome='hg38'                                   # hg19/hg38
cell_line = 'GM12878'                              # K562/GM12878
save_to_csv = True
epsilon = 0.01
chr_list = ['chr{}'.format(i) for i in range(1,23)] + ['chrX']

#%%
########################################################################################## DNaseOnly ##########################################################################################

# df_enhancers = pd.read_csv(data_path+f'/results/csv/distal_reg_paper/EG_features/{cell_line}/DNaseOnly/{cell_line}_AllFeatures_NAfilled.DNaseOnly.tsv', delimiter = '\t')
# df_enhancers = df_enhancers.replace([np.inf, -np.inf], np.nan)
# df_enhancers = df_enhancers.fillna(0)

k = 0
for filepath in glob.glob(os.path.join('/media/labuser/STORAGE/GraphReg/results/csv/distal_reg_paper/EG_features/DNaseOnly_CellTypes/batch2', '*.gz')):
    k += 1
    print(k)
    print(filepath)

    df_enhancers = pd.read_csv(filepath, delimiter = '\t')
    df_enhancers = df_enhancers.replace([np.inf, -np.inf], np.nan)
    df_enhancers = df_enhancers.fillna(0)

    model = 'Full'
    features_list = ['numTSSEnhGene',
                    'distanceToTSS', 'normalizedDNase_enh', 'normalizedDNase_prom',
                    'numNearbyEnhancers', 'sumNearbyEnhancers', 'ubiquitousExpressedGene',
                    'numCandidateEnhGene', '3DContactAvgHicTrack2',
                    '3DContactAvgHicTrack2_squared',
                    'activityEnhDNaseOnlyAvgHicTrack2_squared',
                    'activityPromDNaseOnlyAvgHicTrack2', 'ABCScoreDNaseOnlyAvgHicTrack2']

    X = df_enhancers.loc[:,features_list]
    X = np.log(np.abs(X) + epsilon)

    for chr in chr_list:
        idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
        #print(f'Num test {chr} is {len(idx_test)}')
        if len(idx_test) > 0:
            X_test = X.loc[idx_test, :]

            with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/DNaseOnly/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
                clf = pickle.load(f)

            probs = clf.predict_proba(X_test)
            df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

    if save_to_csv:
        df_enhancers.to_csv(filepath, sep = '\t', index=False)


#%%
########################################################################################## DNaseH3K27acOnly ##########################################################################################

df_enhancers = pd.read_csv(data_path+f'/results/csv/distal_reg_paper/EG_features/{cell_line}/DNaseH3K27acOnly/{cell_line}_AllFeatures_NAfilled.DNaseH3K27acOnly.tsv', delimiter = '\t')
df_enhancers = df_enhancers.replace([np.inf, -np.inf], np.nan)
df_enhancers = df_enhancers.fillna(0)

model = 'Full'
features_list = ['pearsonCorrelation',
                'spearmanCorrelation', 'glsCoefficient', 'numTSSEnhGene',
                'distanceToTSS', 'normalizedDNase_enh', 'normalizedDNase_prom',
                'normalizedH3K27ac_enh', 'normalizedH3K27ac_prom', 'activity_enh',
                'activity_enh_squared', 'activity_prom', 'ABCNumerator', 'ABCScore',
                'ABCDenominator', 'numNearbyEnhancers', 'numNearbyEnhancers_10kb',
                'sumNearbyEnhancers', 'sumNearbyEnhancers_10kb', 'promCTCF', 'enhCTCF',
                'averageCorrWeighted', 'RamilWeighted', 'phastConMax', 'phyloPMax',
                'P2PromoterClass', 'ubiquitousExpressedGene']

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/DNaseH3K27acOnly/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
if save_to_csv:
    df_enhancers.to_csv(data_path+f'/results/csv/distal_reg_paper/EG_features/{cell_line}/DNaseH3K27acOnly/{cell_line}_AllFeatures_NAfilled.DNaseH3K27acOnly_withPreds.tsv', sep = '\t', index=False)


#%%
########################################################################################## EnhActivity ##########################################################################################

df_enhancers = pd.read_csv(data_path+f'/results/csv/distal_reg_paper/EG_features/{cell_line}/EnhActivity/{cell_line}_AllFeatures_NAfilled.tsv', delimiter = '\t')
df_enhancers = df_enhancers.replace([np.inf, -np.inf], np.nan)
df_enhancers = df_enhancers.fillna(0)

model = 'Full'
features_list = ['normalizedH3K27ac_enhActivity', 'normalizedH3K4me1_enhActivity',
                'normalizedH3K4me3_enhActivity', 'normalizedH3K27me3_enhActivity',
                'normalizedH3K9me3_enhActivity', 'normalizedH3K36me3_enhActivity',
                'normalizedCTCF_enhActivity', 'normalizedEP300_enhActivity', 'ABCScore',
                'distanceToTSS', '3DContact']

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/EnhActivity/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]
    
#%%
if save_to_csv:
    df_enhancers.to_csv(data_path+f'/results/csv/distal_reg_paper/EG_features/{cell_line}/EnhActivity/{cell_line}_AllFeatures_NAfilled_EnhActivity_withPreds.tsv', sep = '\t', index=False)

