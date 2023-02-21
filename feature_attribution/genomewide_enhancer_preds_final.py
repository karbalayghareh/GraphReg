import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LogisticRegression

data_path = '/media/labuser/STORAGE/GraphReg'   # data path
organism = 'human'
genome='hg38'                                   # hg19/hg38
cell_line = 'GM12878'                              # K562/GM12878
save_to_csv = False
save_fig = False
epsilon = 0.01

chr_list = ['chr{}'.format(i) for i in range(1,23)] + ['chrX']

############################################### Full model suggestions ###############################################

#%%
df_enhancers = pd.read_csv(data_path+f'/results/csv/distal_reg_paper/EG_features/{cell_line}/FullModel/{cell_line}_AllFeatures_NAfilled.FullModel.tsv', delimiter = '\t')
df_enhancers = df_enhancers.drop(columns=['GraphReg.Score'])
df_enhancers = df_enhancers.replace([np.inf, -np.inf], np.nan)
df_enhancers = df_enhancers.fillna(0)

model = 'FullModel'
features_list = ['EpiMapScore',
                'glsCoefficient', 'numTSSEnhGene', 'distanceToTSS',
                'normalizedDNase_enh', 'normalizedDNase_prom', 'normalizedH3K27ac_enh',
                'normalizedH3K27ac_prom', 'activity_enh', '3DContact',
                'activity_enh_squared', '3DContact_squared', 'activity_prom',
                'ABCNumerator', 'ABCScore', 'ABCDenominator', 'numNearbyEnhancers',
                'sumNearbyEnhancers', 'PEToutsideNormalized', 'PETcrossNormalized',
                'promCTCF', 'enhCTCF', 'H3K4me3_e_max_L_8', 'H3K4me3_e_grad_max_L_8',
                'H3K27ac_e_grad_max_L_8', 'DNase_e_grad_max_L_8',
                'H3K4me3_e_grad_min_L_8', 'H3K27ac_e_grad_min_L_8',
                'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8', 'averageCorrWeighted', 'phastConMax',
                'phyloPMax', 'P2PromoterClass', 'ubiquitousExpressedGene',
                'HiCLoopOutsideNormalized', 'HiCLoopCrossNormalized', 'inTAD', 'inCCD',
                'normalizedEP300_enhActivity']

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
if save_to_csv:
    df_enhancers.to_csv(data_path+f'/results/csv/distal_reg_paper/EG_features/{cell_line}/FullModel/{cell_line}_AllFeatures_NAfilled.FullModel_withPreds.tsv', sep = '\t', index=False)


#%%
model = 'Baseline'
features_list = ['distanceToTSS', '3DContact',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 'normalizedH3K27ac_prom', 'normalizedDNase_prom']

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'GraphReg'
features_list = ['distanceToTSS',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8']

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'GraphReg_NoGrads'
features_list = ['distanceToTSS',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_p_max_L_8', 'H3K27ac_p_max_L_8', 'DNase_p_max_L_8']

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'GraphReg+RamilWeighted'
features_list = ['distanceToTSS',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8', 'RamilWeighted']

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'GraphReg+ABCScore'
features_list = ['distanceToTSS',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8', 'ABCScore']

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'GraphReg+ABCScore+RamilWeighted'
features_list = ['distanceToTSS',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8', 'ABCScore', 'RamilWeighted']

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'GraphReg+ABCScore+PET'
features_list = ['distanceToTSS',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8', 'ABCScore', 'PEToutsideNormalized', 'PETcrossNormalized']

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]


#%%
model = 'GraphReg+ABCScore+CTCF'
features_list = ['distanceToTSS',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8', 'ABCScore', 'enhCTCF', 'promCTCF']

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]


#%%
model = 'GraphReg+ABCScore+PET+CTCF'
features_list = ['distanceToTSS',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8', 'ABCScore', 'PEToutsideNormalized', 'PETcrossNormalized', 'enhCTCF', 'promCTCF']

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'GraphReg+ABCScore+PET+CTCF+RamilWeighted'
features_list = ['distanceToTSS',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8', 'ABCScore', 'PEToutsideNormalized', 'PETcrossNormalized', 'enhCTCF', 'promCTCF', 'RamilWeighted']

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'GraphReg_ABCScore+PET+CTCF+RamilWeighted+sumNearbyEnhancers_10kb'
features_list = ['distanceToTSS',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8', 'ABCScore', 'PEToutsideNormalized', 'PETcrossNormalized', 
                'enhCTCF', 'promCTCF', 'RamilWeighted', 'sumNearbyEnhancers_10kb']

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]


#%%
##### Top 10 shap
model = 'Top10_shap_scores'
features_list = ['H3K27ac_p_grad_max_L_8', 'H3K4me3_p_grad_max_L_8', 'normalizedH3K27ac_prom',
                'normalizedDNase_prom', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8', 'H3K4me3_e_grad_max_L_8', 
                'sumNearbyEnhancers_10kb', 'distanceToTSS', 'DNase_p_grad_max_L_8']

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]


#%%
##### Top 15 shap
model = 'Top15_shap_scores'
features_list = ['H3K27ac_p_grad_max_L_8', 'H3K4me3_p_grad_max_L_8', 'normalizedH3K27ac_prom',
                'normalizedDNase_prom', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8', 'H3K4me3_e_grad_max_L_8', 
                'sumNearbyEnhancers_10kb', 'distanceToTSS', 'DNase_p_grad_max_L_8', 
                'numNearbyEnhancers', 'ABCScore', 'PETcrossNormalized', 'H3K27ac_p_max_L_8', 'DNase_p_grad_min_L_8']

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### Top 20 shap
model = 'Top20_shap_scores'
features_list = ['H3K27ac_p_grad_max_L_8', 'H3K4me3_p_grad_max_L_8', 'normalizedH3K27ac_prom',
                'normalizedDNase_prom', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8', 'H3K4me3_e_grad_max_L_8', 
                'sumNearbyEnhancers_10kb', 'distanceToTSS', 'DNase_p_grad_max_L_8', 
                'numNearbyEnhancers', 'ABCScore', 'PETcrossNormalized', 'H3K27ac_p_max_L_8', 'DNase_p_grad_min_L_8', 
                'H3K4me3_p_max_L_8', 'normalizedH3K27ac_enh', 'STARRseqABC', 'activityEnhDNaseOnlyAvgHicTrack1_squared',
                'ABCDenominator']

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### Top 25 shap
model = 'Top25_shap_scores'
features_list = ['H3K27ac_p_grad_max_L_8', 'H3K4me3_p_grad_max_L_8', 'normalizedH3K27ac_prom',
                'normalizedDNase_prom', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8', 'H3K4me3_e_grad_max_L_8', 
                'sumNearbyEnhancers_10kb', 'distanceToTSS', 'DNase_p_grad_max_L_8', 
                'numNearbyEnhancers', 'ABCScore', 'PETcrossNormalized', 'H3K27ac_p_max_L_8', 'DNase_p_grad_min_L_8', 
                'H3K4me3_p_max_L_8', 'normalizedH3K27ac_enh', 'STARRseqABC', 'activityEnhDNaseOnlyAvgHicTrack1_squared',
                'ABCDenominator', 'activity_enh_squared', '3DContactAvgHicTrack1', 'ubiquitousExpressedGene',
                'ABCScoreDNaseOnlyAvgHicTrack1', '3DContact']

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### Top 30 shap
model = 'Top30_shap_scores'
features_list = ['H3K27ac_p_grad_max_L_8', 'H3K4me3_p_grad_max_L_8', 'normalizedH3K27ac_prom',
                'normalizedDNase_prom', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8', 'H3K4me3_e_grad_max_L_8', 
                'sumNearbyEnhancers_10kb', 'distanceToTSS', 'DNase_p_grad_max_L_8', 
                'numNearbyEnhancers', 'ABCScore', 'PETcrossNormalized', 'H3K27ac_p_max_L_8', 'DNase_p_grad_min_L_8', 
                'H3K4me3_p_max_L_8', 'normalizedH3K27ac_enh', 'STARRseqABC', 'activityEnhDNaseOnlyAvgHicTrack1_squared',
                'ABCDenominator', 'activity_enh_squared', '3DContactAvgHicTrack1', 'ubiquitousExpressedGene',
                'ABCScoreDNaseOnlyAvgHicTrack1', '3DContact', 'activityPromDNaseOnlyAvgHicTrack1', 'H3K4me3_p_grad_min_L_8',
                'numTSSEnhGene', 'H3K27ac_e_grad_max_L_8', 'H3K27ac_e_grad_min_L_8']

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

############################################### Jesse suggestions ###############################################

#%%
model = 'ABCScore+numNearbyEnhancers_10kb+sumNearbyEnhancers_10kb'
features_list = ['ABCScore', 'numNearbyEnhancers_10kb', 'sumNearbyEnhancers_10kb']

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'Baseline+numNearbyEnhancers_10kb+sumNearbyEnhancers_10kb'
features_list = ['distanceToTSS', '3DContact',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 'normalizedH3K27ac_prom', 
                'normalizedDNase_prom', 'numNearbyEnhancers_10kb', 'sumNearbyEnhancers_10kb']

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'activity+3Dcontact'
features_list = ['activity_enh', 'activity_prom', '3DContact']

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'activity+3Dcontact+activity_squared'
features_list = ['activity_enh', 'activity_prom', '3DContact', 'activity_enh_squared']

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

############################################### Ramil suggestions ###############################################

#%%
model = 'Baseline+ABCScore'
features_list = ['distanceToTSS', '3DContact',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 'normalizedH3K27ac_prom', 
                'normalizedDNase_prom', 'ABCScore']

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'Baseline+ABCScore+RamilWeighted'
features_list = ['distanceToTSS', '3DContact',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 'normalizedH3K27ac_prom', 
                'normalizedDNase_prom', 'ABCScore', 'RamilWeighted']

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

############################################### Andreas suggestions ###############################################
#%%
##### 
features_list = ['normalizedDNase_enh', 'normalizedDNase_prom', '3DContact']
model = '+'.join(features_list)

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
features_list = ['normalizedH3K27ac_enh', 'normalizedH3K27ac_prom', '3DContact']
model = '+'.join(features_list)

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
features_list = ['normalizedDNase_enh', 'normalizedDNase_prom', 'normalizedH3K27ac_enh', 'normalizedH3K27ac_prom', '3DContact']
model = '+'.join(features_list)

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
features_list = ['activity_enh', 'activity_prom', '3DContact']
model = '+'.join(features_list)

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
features_list = ['normalizedDNase_enh', 'normalizedDNase_prom', 'distanceToTSS']
model = '+'.join(features_list)

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
features_list = ['normalizedH3K27ac_enh', 'normalizedH3K27ac_prom', 'distanceToTSS']
model = '+'.join(features_list)

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
features_list = ['normalizedDNase_enh', 'normalizedDNase_prom', 'normalizedH3K27ac_enh', 'normalizedH3K27ac_prom', 'distanceToTSS']
model = '+'.join(features_list)

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
features_list = ['activity_enh', 'activity_prom', 'distanceToTSS']
model = '+'.join(features_list)

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
features_list = ['activity_enh', 'activity_prom', 'distanceToTSS', '3DContact']
model = '+'.join(features_list)

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]


############################################### Wang suggestions ###############################################

#%%
##### 
features_list = ['distanceToTSS', '3DContact',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 
                'normalizedH3K27ac_prom', 'normalizedDNase_prom', 'PEToutsideNormalized']
model = 'Baseline+PEToutsideNormalized'

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
features_list = ['distanceToTSS', '3DContact',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 
                'normalizedH3K27ac_prom', 'normalizedDNase_prom', 'PETcrossNormalized']
model = 'Baseline+PETcrossNormalized'

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
features_list = ['distanceToTSS', '3DContact',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 
                'normalizedH3K27ac_prom', 'normalizedDNase_prom', 'PETcrossNormalized', 'PEToutsideNormalized']
model = 'Baseline+PETcrossNormalized+PEToutsideNormalized'

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
features_list = ['ABCScore', 'PEToutsideNormalized', 'PETcrossNormalized']
model = '+'.join(features_list)

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

############################################### Evelyn suggestions 1 ###############################################


#%%
##### 
features_list = ['ABCScore', 'averageCorrWeighted']
model = 'ABCScore+averageCorrWeighted'

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
features_list = ['ABCScore', 'distanceToTSS', '3DContact',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 
                'normalizedH3K27ac_prom', 'normalizedDNase_prom', 'averageCorrWeighted']

model = 'ABCScore+Baseline+averageCorrWeighted'

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
features_list = ['ABCScore', 'distanceToTSS',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8', 'averageCorrWeighted']

model = 'ABCScore+GraphReg+averageCorrWeighted'

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

############################################### Evelyn suggestions 2 ###############################################


#%%
##### 
features_list = ['ABCScore', 'P2PromoterClass']
model = 'ABCScore+P2PromoterClass'

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
features_list = ['ABCScore', 'distanceToTSS', '3DContact',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 
                'normalizedH3K27ac_prom', 'normalizedDNase_prom', 'P2PromoterClass']

model = 'ABCScore+Baseline+P2PromoterClass'

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
features_list = ['ABCScore', 'distanceToTSS',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8', 'P2PromoterClass']

model = 'ABCScore+GraphReg+P2PromoterClass'

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]


############################################### Evelyn suggestions 3 ###############################################

#%%
##### 
features_list = ['ABCScore', 'ubiquitousExpressedGene']
model = 'ABCScore+ubiquitousExpressedGene'

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
features_list = ['ABCScore', 'distanceToTSS', '3DContact',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 
                'normalizedH3K27ac_prom', 'normalizedDNase_prom', 'ubiquitousExpressedGene']

model = 'ABCScore+Baseline+ubiquitousExpressedGene'

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
features_list = ['ABCScore', 'distanceToTSS',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8', 'ubiquitousExpressedGene']

model = 'ABCScore+GraphReg+ubiquitousExpressedGene'

X = df_enhancers.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)

for chr in chr_list:
    idx_test = df_enhancers[df_enhancers['chr']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        X_test = X.loc[idx_test, :]

        with open(data_path+'/results/csv/distal_reg_paper/CRISPR_ensemble/models/final/model_{}_test_{}.pkl'.format(model, chr),'rb') as f:
            clf = pickle.load(f)

        probs = clf.predict_proba(X_test)
        df_enhancers.loc[idx_test, model+'.Score'] = probs[:,1]


#%%
if save_to_csv:
    df_enhancers.to_csv(data_path+f'/results/csv/distal_reg_paper/EG_features/{cell_line}/AllFeatures/{cell_line}_AllFeatures_NAfilled_withPreds.tsv', sep = '\t', index=False)

#%%
# prepare data for submission to ENCODE portal

cell_line = 'GM12878'
df_enhancers = pd.read_csv(data_path+f'/results/csv/distal_reg_paper/EG_features/{cell_line}/AllFeatures/{cell_line}_AllFeatures_NAfilled_withPreds.tsv', delimiter = '\t')
df_enhancers['GraphReg_LR_thresholded.Score'] = (df_enhancers['GraphReg.Score'] > 0.2247).astype(np.int32)   # Threshold corresponding to recall = 0.7
df_enhancers_final = df_enhancers[['chr', 'start', 'end', 'name', 'class', 'TargetGene', 
                                    'TargetGeneTSS', 'CellType', 'distanceToTSS', 'GraphReg.Score', 'GraphReg_LR_thresholded.Score',
                                    'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                                    'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                                    'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                                    'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                                    'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                                    'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                                    'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                                    'DNase_p_grad_min_L_8']]

df_enhancers_final = df_enhancers_final.rename(columns={'GraphReg.Score': 'GraphReg_LR.Score', 'distanceToTSS': 'DistanceToTSS'})
df_enhancers_final['CellType'] = cell_line
df_enhancers_final['DistanceToTSS'] = df_enhancers_final['DistanceToTSS'].astype(np.int64)

if save_to_csv:
    df_enhancers_final.to_csv(data_path+f'/results/csv/distal_reg_paper/EG_features/{cell_line}/AllFeatures/SubmittedToENCODE/GraphRegLR_Predictions_{cell_line}.tsv', sep = '\t', index=False)


cell_line = 'K562'
df_enhancers = pd.read_csv(data_path+f'/results/csv/distal_reg_paper/EG_features/{cell_line}/AllFeatures/{cell_line}_AllFeatures_NAfilled_withPreds.tsv', delimiter = '\t')
df_enhancers['GraphReg_LR_thresholded.Score'] = (df_enhancers['GraphReg.Score'] > 0.2247).astype(np.int32)   # Threshold corresponding to recall = 0.7
df_enhancers_final = df_enhancers[['chr', 'start', 'end', 'name', 'class', 'TargetGene', 
                                    'TargetGeneTSS', 'CellType', 'distanceToTSS', 'GraphReg.Score', 'GraphReg_LR_thresholded.Score',
                                    'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                                    'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                                    'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                                    'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                                    'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                                    'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                                    'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                                    'DNase_p_grad_min_L_8']]

df_enhancers_final = df_enhancers_final.rename(columns={'GraphReg.Score': 'GraphReg_LR.Score', 'distanceToTSS': 'DistanceToTSS'})
df_enhancers_final['CellType'] = cell_line
df_enhancers_final['DistanceToTSS'] = df_enhancers_final['DistanceToTSS'].astype(np.int64)

if save_to_csv:
    df_enhancers_final.to_csv(data_path+f'/results/csv/distal_reg_paper/EG_features/{cell_line}/AllFeatures/SubmittedToENCODE/GraphRegLR_Predictions_{cell_line}.tsv', sep = '\t', index=False)

