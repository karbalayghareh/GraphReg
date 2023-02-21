##### cross-validation by holding out chromosomes

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pickle
from sklearn.linear_model import LogisticRegression
#import shap
import scipy
#from statannotations.Annotator import Annotator

# Needed for Illustrator
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = ['Arial','Helvetica']


data_path = '/media/labuser/STORAGE/GraphReg'   # data path
organism = 'human'
genome='hg38'                                   # hg19/hg38
cell_line = 'K562'                              # K562/GM12878
dataset = 'CRISPR_ensemble'                    # 'fulco' or 'gasperini' or 'CRISPR_ensemble'
save_model = True
save_to_csv = False
compute_shap = False
epsilon = 0.01

chr_list = ['chr{}'.format(i) for i in range(1,23)] + ['chrX']

############################################### Full model suggestions ###############################################

#%%

RefSeqGenes = pd.read_csv(data_path+'/results/csv/distal_reg_paper/RefSeqGenes/RefSeqCurated.170308.bed.CollapsedGeneBounds.hg38.TSS500bp.bed', names = ['chr', 'start', 'end', 'gene', 'len', 'strand'], delimiter = '\t')

df_crispr_full = pd.read_csv(data_path+'/results/csv/distal_reg_paper/EG_features/K562/FullModel/EPCrisprBenchmark_ensemble_data_GRCh38.K562_AllFeatures_NAfilled.FullModel.tsv', delimiter = '\t')
df_crispr_full = df_crispr_full[df_crispr_full['measuredGeneSymbol'].isin(RefSeqGenes['gene'])].reset_index(drop=True)
df_crispr_full = df_crispr_full[~df_crispr_full['Regulated'].isna()].reset_index(drop=True)
df_crispr_full = df_crispr_full.drop(columns=['GraphReg.Score'])
df_crispr_full = df_crispr_full.replace([np.inf, -np.inf], np.nan)
df_crispr_full = df_crispr_full.fillna(0)

RefSeqGenes_sub = RefSeqGenes[RefSeqGenes['gene'].isin(df_crispr_full['measuredGeneSymbol'])].reset_index(drop=True)
df_crispr_full['TSS_from_universe'] = -1
for i, g in enumerate(RefSeqGenes_sub['gene'].values):
    idx = df_crispr_full[df_crispr_full['measuredGeneSymbol'] == g].index
    df_crispr_full.loc[idx, 'TSS_from_universe'] = (RefSeqGenes_sub.loc[i, 'start'] + RefSeqGenes_sub.loc[i, 'end'])//2

df_crispr_full['distance'] = np.abs((df_crispr_full['chromStart'] + df_crispr_full['chromEnd'])//2 - df_crispr_full['TSS_from_universe'])

model = 'ENCODE-E2G_Extended'
features_list = ['EpiMapScore',
                'glsCoefficient', 'numTSSEnhGene', 'distance',
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

X = df_crispr_full.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr_full['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
shap_values_all = np.empty([0,X.shape[1]])
X_test_all = pd.DataFrame(columns=features_list)
for chr in chr_list:
    idx_test = df_crispr_full[df_crispr_full['chrom']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr_full.loc[idx_test, model+'.Score'] = probs[:,1]

        if compute_shap:
            background = shap.kmeans(X_train, 10)
            explainer = shap.KernelExplainer(clf.predict_proba, background)
            shap_values = explainer.shap_values(X_test)
            shap_values_all = np.append(shap_values_all, shap_values[1], axis=0)
            X_test_all = X_test_all.append(X_test).reset_index(drop=True)

if compute_shap:
    fig=plt.gcf()
    shap.summary_plot(shap_values_all, X_test_all, plot_type='bar', max_display=50)
    fig.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/SHAP_scores_barplot_ENCODE_E2G_Extended_Ensemble.pdf', bbox_inches='tight')

    fig=plt.gcf()
    shap.summary_plot(shap_values_all, X_test_all, plot_type='dot', max_display=50)
    fig.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/SHAP_scores_dotplot_ENCODE_E2G_Extended_Ensemble.pdf', bbox_inches='tight')

#%%
model = 'ENCODE-E2G_Extended without gradients (12)'  # minus 12 features
features_list = ['EpiMapScore',
                'glsCoefficient', 'numTSSEnhGene', 'distance',
                'normalizedDNase_enh', 'normalizedDNase_prom', 'normalizedH3K27ac_enh',
                'normalizedH3K27ac_prom', 'activity_enh', '3DContact',
                'activity_enh_squared', '3DContact_squared', 'activity_prom',
                'ABCNumerator', 'ABCScore', 'ABCDenominator', 'numNearbyEnhancers',
                'sumNearbyEnhancers', 'PEToutsideNormalized', 'PETcrossNormalized',
                'promCTCF', 'enhCTCF', 'H3K4me3_e_max_L_8',
                'H3K4me3_p_max_L_8', 'averageCorrWeighted', 'phastConMax',
                'phyloPMax', 'P2PromoterClass', 'ubiquitousExpressedGene',
                'HiCLoopOutsideNormalized', 'HiCLoopCrossNormalized', 'inTAD', 'inCCD',
                'normalizedEP300_enhActivity']

X = df_crispr_full.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr_full['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
shap_values_all = np.empty([0,X.shape[1]])
X_test_all = pd.DataFrame(columns=features_list)
for chr in chr_list:
    idx_test = df_crispr_full[df_crispr_full['chrom']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr_full.loc[idx_test, model+'.Score'] = probs[:,1]

        if compute_shap:
            background = shap.kmeans(X_train, 10)
            explainer = shap.KernelExplainer(clf.predict_proba, background)
            shap_values = explainer.shap_values(X_test)
            shap_values_all = np.append(shap_values_all, shap_values[1], axis=0)
            X_test_all = X_test_all.append(X_test).reset_index(drop=True)

if compute_shap:
    shap.summary_plot(shap_values_all, X_test_all, plot_type='bar', max_display=50)
    shap.summary_plot(shap_values_all, X_test_all, plot_type='dot', max_display=50)

#%%
model = 'ENCODE-E2G_Extended without distance (3)'  # minus 3 feature

features_list = ['glsCoefficient',
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

X = df_crispr_full.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr_full['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
shap_values_all = np.empty([0,X.shape[1]])
X_test_all = pd.DataFrame(columns=features_list)
for chr in chr_list:
    idx_test = df_crispr_full[df_crispr_full['chrom']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr_full.loc[idx_test, model+'.Score'] = probs[:,1]

        if compute_shap:
            background = shap.kmeans(X_train, 10)
            explainer = shap.KernelExplainer(clf.predict_proba, background)
            shap_values = explainer.shap_values(X_test)
            shap_values_all = np.append(shap_values_all, shap_values[1], axis=0)
            X_test_all = X_test_all.append(X_test).reset_index(drop=True)

if compute_shap:
    shap.summary_plot(shap_values_all, X_test_all, plot_type='bar', max_display=50)
    shap.summary_plot(shap_values_all, X_test_all, plot_type='dot', max_display=50)

#%%
model = 'ENCODE-E2G_Extended without EP300 (1)' # minus 1 feature
features_list = ['EpiMapScore',
                'glsCoefficient', 'numTSSEnhGene', 'distance',
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
                'HiCLoopOutsideNormalized', 'HiCLoopCrossNormalized', 'inTAD', 'inCCD']

X = df_crispr_full.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr_full['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
shap_values_all = np.empty([0,X.shape[1]])
X_test_all = pd.DataFrame(columns=features_list)
for chr in chr_list:
    idx_test = df_crispr_full[df_crispr_full['chrom']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr_full.loc[idx_test, model+'.Score'] = probs[:,1]

        if compute_shap:
            background = shap.kmeans(X_train, 10)
            explainer = shap.KernelExplainer(clf.predict_proba, background)
            shap_values = explainer.shap_values(X_test)
            shap_values_all = np.append(shap_values_all, shap_values[1], axis=0)
            X_test_all = X_test_all.append(X_test).reset_index(drop=True)

if compute_shap:
    shap.summary_plot(shap_values_all, X_test_all, plot_type='bar', max_display=50)
    shap.summary_plot(shap_values_all, X_test_all, plot_type='dot', max_display=50)

#%%
model = 'ENCODE-E2G_Extended without correlations (3)' # minus 3 features
features_list = ['numTSSEnhGene', 'distance',
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
                'DNase_p_grad_min_L_8', 'phastConMax',
                'phyloPMax', 'P2PromoterClass', 'ubiquitousExpressedGene',
                'HiCLoopOutsideNormalized', 'HiCLoopCrossNormalized', 'inTAD', 'inCCD',
                'normalizedEP300_enhActivity']

X = df_crispr_full.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr_full['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
shap_values_all = np.empty([0,X.shape[1]])
X_test_all = pd.DataFrame(columns=features_list)
for chr in chr_list:
    idx_test = df_crispr_full[df_crispr_full['chrom']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr_full.loc[idx_test, model+'.Score'] = probs[:,1]

        if compute_shap:
            background = shap.kmeans(X_train, 10)
            explainer = shap.KernelExplainer(clf.predict_proba, background)
            shap_values = explainer.shap_values(X_test)
            shap_values_all = np.append(shap_values_all, shap_values[1], axis=0)
            X_test_all = X_test_all.append(X_test).reset_index(drop=True)

if compute_shap:
    shap.summary_plot(shap_values_all, X_test_all, plot_type='bar', max_display=50)
    shap.summary_plot(shap_values_all, X_test_all, plot_type='dot', max_display=50)

#%%
model = 'ENCODE-E2G_Extended without CTCF (5)'  # minus 5 features
features_list = ['EpiMapScore',
                'glsCoefficient', 'numTSSEnhGene', 'distance',
                'normalizedDNase_enh', 'normalizedDNase_prom', 'normalizedH3K27ac_enh',
                'normalizedH3K27ac_prom', 'activity_enh', '3DContact',
                'activity_enh_squared', '3DContact_squared', 'activity_prom',
                'ABCNumerator', 'ABCScore', 'ABCDenominator', 'numNearbyEnhancers',
                'sumNearbyEnhancers', 'H3K4me3_e_max_L_8', 'H3K4me3_e_grad_max_L_8',
                'H3K27ac_e_grad_max_L_8', 'DNase_e_grad_max_L_8',
                'H3K4me3_e_grad_min_L_8', 'H3K27ac_e_grad_min_L_8',
                'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8', 'averageCorrWeighted', 'phastConMax',
                'phyloPMax', 'P2PromoterClass', 'ubiquitousExpressedGene',
                'HiCLoopOutsideNormalized', 'HiCLoopCrossNormalized', 'inTAD',
                'normalizedEP300_enhActivity']

X = df_crispr_full.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr_full['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
shap_values_all = np.empty([0,X.shape[1]])
X_test_all = pd.DataFrame(columns=features_list)
for chr in chr_list:
    idx_test = df_crispr_full[df_crispr_full['chrom']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr_full.loc[idx_test, model+'.Score'] = probs[:,1]

        if compute_shap:
            background = shap.kmeans(X_train, 10)
            explainer = shap.KernelExplainer(clf.predict_proba, background)
            shap_values = explainer.shap_values(X_test)
            shap_values_all = np.append(shap_values_all, shap_values[1], axis=0)
            X_test_all = X_test_all.append(X_test).reset_index(drop=True)

if compute_shap:
    shap.summary_plot(shap_values_all, X_test_all, plot_type='bar', max_display=50)
    shap.summary_plot(shap_values_all, X_test_all, plot_type='dot', max_display=50)

#%%
model = 'ENCODE-E2G_Extended without Hi-C (20)'  # minus 20 features
features_list = ['EpiMapScore',
                'glsCoefficient', 'numTSSEnhGene', 'distance',
                'normalizedDNase_enh', 'normalizedDNase_prom', 'normalizedH3K27ac_enh',
                'normalizedH3K27ac_prom', 'activity_enh',
                'activity_enh_squared', 'activity_prom', 'numNearbyEnhancers',
                'sumNearbyEnhancers', 'PEToutsideNormalized', 'PETcrossNormalized',
                'promCTCF', 'enhCTCF', 'H3K4me3_e_max_L_8', 'H3K4me3_p_max_L_8', 'averageCorrWeighted', 'phastConMax',
                'phyloPMax', 'P2PromoterClass', 'ubiquitousExpressedGene', 'inCCD',
                'normalizedEP300_enhActivity']

X = df_crispr_full.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr_full['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
shap_values_all = np.empty([0,X.shape[1]])
X_test_all = pd.DataFrame(columns=features_list)
for chr in chr_list:
    idx_test = df_crispr_full[df_crispr_full['chrom']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr_full.loc[idx_test, model+'.Score'] = probs[:,1]

        if compute_shap:
            background = shap.kmeans(X_train, 10)
            explainer = shap.KernelExplainer(clf.predict_proba, background)
            shap_values = explainer.shap_values(X_test)
            shap_values_all = np.append(shap_values_all, shap_values[1], axis=0)
            X_test_all = X_test_all.append(X_test).reset_index(drop=True)

if compute_shap:
    shap.summary_plot(shap_values_all, X_test_all, plot_type='bar', max_display=50)
    shap.summary_plot(shap_values_all, X_test_all, plot_type='dot', max_display=50)

#%%
model = 'ENCODE-E2G_Extended without Hi-C and distance (23)'  # minus 23 features
features_list = ['glsCoefficient', 'normalizedDNase_enh', 'normalizedDNase_prom', 
                'normalizedH3K27ac_enh', 'normalizedH3K27ac_prom', 'activity_enh',
                'activity_enh_squared', 'activity_prom', 'numNearbyEnhancers',
                'sumNearbyEnhancers', 'PEToutsideNormalized', 'PETcrossNormalized',
                'promCTCF', 'enhCTCF', 'H3K4me3_e_max_L_8', 'H3K4me3_p_max_L_8', 'averageCorrWeighted', 'phastConMax',
                'phyloPMax', 'P2PromoterClass', 'ubiquitousExpressedGene', 'inCCD',
                'normalizedEP300_enhActivity']

X = df_crispr_full.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr_full['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
shap_values_all = np.empty([0,X.shape[1]])
X_test_all = pd.DataFrame(columns=features_list)
for chr in chr_list:
    idx_test = df_crispr_full[df_crispr_full['chrom']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr_full.loc[idx_test, model+'.Score'] = probs[:,1]

        if compute_shap:
            background = shap.kmeans(X_train, 10)
            explainer = shap.KernelExplainer(clf.predict_proba, background)
            shap_values = explainer.shap_values(X_test)
            shap_values_all = np.append(shap_values_all, shap_values[1], axis=0)
            X_test_all = X_test_all.append(X_test).reset_index(drop=True)

if compute_shap:
    shap.summary_plot(shap_values_all, X_test_all, plot_type='bar', max_display=50)
    shap.summary_plot(shap_values_all, X_test_all, plot_type='dot', max_display=50)

#%%
model = 'ENCODE-E2G_Extended without contact and distance (26)'  # minus 26 features
features_list = ['glsCoefficient', 
                'normalizedDNase_enh', 'normalizedDNase_prom', 'normalizedH3K27ac_enh',
                'normalizedH3K27ac_prom', 'activity_enh', 'activity_enh_squared', 'activity_prom', 'numNearbyEnhancers',
                'sumNearbyEnhancers', 'promCTCF', 'enhCTCF', 'H3K4me3_e_max_L_8', 
                'H3K4me3_p_max_L_8', 'averageCorrWeighted', 'phastConMax',
                'phyloPMax', 'P2PromoterClass', 'ubiquitousExpressedGene',
                'normalizedEP300_enhActivity']

X = df_crispr_full.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr_full['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
shap_values_all = np.empty([0,X.shape[1]])
X_test_all = pd.DataFrame(columns=features_list)
for chr in chr_list:
    idx_test = df_crispr_full[df_crispr_full['chrom']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr_full.loc[idx_test, model+'.Score'] = probs[:,1]

        if compute_shap:
            background = shap.kmeans(X_train, 10)
            explainer = shap.KernelExplainer(clf.predict_proba, background)
            shap_values = explainer.shap_values(X_test)
            shap_values_all = np.append(shap_values_all, shap_values[1], axis=0)
            X_test_all = X_test_all.append(X_test).reset_index(drop=True)

if compute_shap:
    shap.summary_plot(shap_values_all, X_test_all, plot_type='bar', max_display=50)
    shap.summary_plot(shap_values_all, X_test_all, plot_type='dot', max_display=50)

#%%
model = 'ENCODE-E2G_Extended without contact (23)'  # minus 23 features
features_list = ['EpiMapScore',
                'glsCoefficient', 'numTSSEnhGene', 'distance', 'normalizedDNase_enh', 'normalizedDNase_prom', 'normalizedH3K27ac_enh',
                'normalizedH3K27ac_prom', 'activity_enh', 'activity_enh_squared', 'activity_prom', 'numNearbyEnhancers',
                'sumNearbyEnhancers', 'promCTCF', 'enhCTCF', 'H3K4me3_e_max_L_8', 
                'H3K4me3_p_max_L_8', 'averageCorrWeighted', 'phastConMax',
                'phyloPMax', 'P2PromoterClass', 'ubiquitousExpressedGene',
                'normalizedEP300_enhActivity']

X = df_crispr_full.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr_full['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
shap_values_all = np.empty([0,X.shape[1]])
X_test_all = pd.DataFrame(columns=features_list)
for chr in chr_list:
    idx_test = df_crispr_full[df_crispr_full['chrom']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr_full.loc[idx_test, model+'.Score'] = probs[:,1]

        if compute_shap:
            background = shap.kmeans(X_train, 10)
            explainer = shap.KernelExplainer(clf.predict_proba, background)
            shap_values = explainer.shap_values(X_test)
            shap_values_all = np.append(shap_values_all, shap_values[1], axis=0)
            X_test_all = X_test_all.append(X_test).reset_index(drop=True)

if compute_shap:
    shap.summary_plot(shap_values_all, X_test_all, plot_type='bar', max_display=50)
    shap.summary_plot(shap_values_all, X_test_all, plot_type='dot', max_display=50)

#%%
model = 'ENCODE-E2G_Extended without ABC (3)'  # minus 3 features
features_list = ['EpiMapScore',
                'glsCoefficient', 'numTSSEnhGene', 'distance',
                'normalizedDNase_enh', 'normalizedDNase_prom', 'normalizedH3K27ac_enh',
                'normalizedH3K27ac_prom', 'activity_enh', '3DContact',
                'activity_enh_squared', '3DContact_squared', 'activity_prom', 'numNearbyEnhancers',
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

X = df_crispr_full.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr_full['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
shap_values_all = np.empty([0,X.shape[1]])
X_test_all = pd.DataFrame(columns=features_list)
for chr in chr_list:
    idx_test = df_crispr_full[df_crispr_full['chrom']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr_full.loc[idx_test, model+'.Score'] = probs[:,1]

        if compute_shap:
            background = shap.kmeans(X_train, 10)
            explainer = shap.KernelExplainer(clf.predict_proba, background)
            shap_values = explainer.shap_values(X_test)
            shap_values_all = np.append(shap_values_all, shap_values[1], axis=0)
            X_test_all = X_test_all.append(X_test).reset_index(drop=True)

if compute_shap:
    shap.summary_plot(shap_values_all, X_test_all, plot_type='bar', max_display=50)
    shap.summary_plot(shap_values_all, X_test_all, plot_type='dot', max_display=50)


#%%
model = 'ENCODE-E2G_Extended without number of nearby enhancers (2)'  # minus 2 features
features_list = ['EpiMapScore',
                'glsCoefficient', 'numTSSEnhGene', 'distance',
                'normalizedDNase_enh', 'normalizedDNase_prom', 'normalizedH3K27ac_enh',
                'normalizedH3K27ac_prom', 'activity_enh', '3DContact',
                'activity_enh_squared', '3DContact_squared', 'activity_prom',
                'ABCNumerator', 'ABCScore', 'ABCDenominator', 'PEToutsideNormalized', 'PETcrossNormalized',
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

X = df_crispr_full.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr_full['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
shap_values_all = np.empty([0,X.shape[1]])
X_test_all = pd.DataFrame(columns=features_list)
for chr in chr_list:
    idx_test = df_crispr_full[df_crispr_full['chrom']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr_full.loc[idx_test, model+'.Score'] = probs[:,1]

        if compute_shap:
            background = shap.kmeans(X_train, 10)
            explainer = shap.KernelExplainer(clf.predict_proba, background)
            shap_values = explainer.shap_values(X_test)
            shap_values_all = np.append(shap_values_all, shap_values[1], axis=0)
            X_test_all = X_test_all.append(X_test).reset_index(drop=True)

if compute_shap:
    shap.summary_plot(shap_values_all, X_test_all, plot_type='bar', max_display=50)
    shap.summary_plot(shap_values_all, X_test_all, plot_type='dot', max_display=50)

#%%
model = 'ENCODE-E2G_Extended without enhancer activity (24)'  # minus 24 features
features_list = ['numTSSEnhGene', 'distance',
                'normalizedDNase_prom',
                'normalizedH3K27ac_prom', '3DContact',
                '3DContact_squared', 'activity_prom', 'numNearbyEnhancers',
                'sumNearbyEnhancers', 'PEToutsideNormalized', 'PETcrossNormalized',
                'promCTCF', 'H3K4me3_p_max_L_8', 'averageCorrWeighted', 'phastConMax',
                'phyloPMax', 'P2PromoterClass', 'ubiquitousExpressedGene',
                'HiCLoopOutsideNormalized', 'HiCLoopCrossNormalized', 'inTAD', 'inCCD']

X = df_crispr_full.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr_full['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
shap_values_all = np.empty([0,X.shape[1]])
X_test_all = pd.DataFrame(columns=features_list)
for chr in chr_list:
    idx_test = df_crispr_full[df_crispr_full['chrom']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr_full.loc[idx_test, model+'.Score'] = probs[:,1]

        if compute_shap:
            background = shap.kmeans(X_train, 10)
            explainer = shap.KernelExplainer(clf.predict_proba, background)
            shap_values = explainer.shap_values(X_test)
            shap_values_all = np.append(shap_values_all, shap_values[1], axis=0)
            X_test_all = X_test_all.append(X_test).reset_index(drop=True)

if compute_shap:
    shap.summary_plot(shap_values_all, X_test_all, plot_type='bar', max_display=50)
    shap.summary_plot(shap_values_all, X_test_all, plot_type='dot', max_display=50)

#%%
model = 'ENCODE-E2G_Extended without promoter activity (23)'  # minus 23 features
features_list = ['numTSSEnhGene', 'distance',
                'normalizedDNase_enh', 'normalizedH3K27ac_enh',
                'activity_enh', '3DContact',
                'activity_enh_squared', '3DContact_squared',
                'ABCNumerator', 'ABCScore', 'numNearbyEnhancers',
                'sumNearbyEnhancers', 'PEToutsideNormalized', 'PETcrossNormalized',
                'enhCTCF', 'H3K4me3_e_max_L_8', 'phastConMax',
                'phyloPMax', 'HiCLoopOutsideNormalized', 'HiCLoopCrossNormalized', 'inTAD', 'inCCD',
                'normalizedEP300_enhActivity']

X = df_crispr_full.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr_full['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
shap_values_all = np.empty([0,X.shape[1]])
X_test_all = pd.DataFrame(columns=features_list)
for chr in chr_list:
    idx_test = df_crispr_full[df_crispr_full['chrom']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr_full.loc[idx_test, model+'.Score'] = probs[:,1]

        if compute_shap:
            background = shap.kmeans(X_train, 10)
            explainer = shap.KernelExplainer(clf.predict_proba, background)
            shap_values = explainer.shap_values(X_test)
            shap_values_all = np.append(shap_values_all, shap_values[1], axis=0)
            X_test_all = X_test_all.append(X_test).reset_index(drop=True)

if compute_shap:
    shap.summary_plot(shap_values_all, X_test_all, plot_type='bar', max_display=50)
    shap.summary_plot(shap_values_all, X_test_all, plot_type='dot', max_display=50)

#%%
model = 'ENCODE-E2G_Extended without promoter class (3)'  # minus 3 features
features_list = ['EpiMapScore',
                'glsCoefficient', 'numTSSEnhGene', 'distance',
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
                'DNase_p_grad_min_L_8', 'phastConMax',
                'phyloPMax', 'HiCLoopOutsideNormalized', 'HiCLoopCrossNormalized', 'inTAD', 'inCCD',
                'normalizedEP300_enhActivity']

X = df_crispr_full.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr_full['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
shap_values_all = np.empty([0,X.shape[1]])
X_test_all = pd.DataFrame(columns=features_list)
for chr in chr_list:
    idx_test = df_crispr_full[df_crispr_full['chrom']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr_full.loc[idx_test, model+'.Score'] = probs[:,1]

        if compute_shap:
            background = shap.kmeans(X_train, 10)
            explainer = shap.KernelExplainer(clf.predict_proba, background)
            shap_values = explainer.shap_values(X_test)
            shap_values_all = np.append(shap_values_all, shap_values[1], axis=0)
            X_test_all = X_test_all.append(X_test).reset_index(drop=True)

if compute_shap:
    shap.summary_plot(shap_values_all, X_test_all, plot_type='bar', max_display=50)
    shap.summary_plot(shap_values_all, X_test_all, plot_type='dot', max_display=50)

#%%
model = 'ENCODE-E2G_Extended without conservation (2)'  # minus 2 features
features_list = ['EpiMapScore',
                'glsCoefficient', 'numTSSEnhGene', 'distance',
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
                'DNase_p_grad_min_L_8', 'averageCorrWeighted', 'P2PromoterClass', 'ubiquitousExpressedGene',
                'HiCLoopOutsideNormalized', 'HiCLoopCrossNormalized', 'inTAD', 'inCCD',
                'normalizedEP300_enhActivity']

X = df_crispr_full.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr_full['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
shap_values_all = np.empty([0,X.shape[1]])
X_test_all = pd.DataFrame(columns=features_list)
for chr in chr_list:
    idx_test = df_crispr_full[df_crispr_full['chrom']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr_full.loc[idx_test, model+'.Score'] = probs[:,1]

        if compute_shap:
            background = shap.kmeans(X_train, 10)
            explainer = shap.KernelExplainer(clf.predict_proba, background)
            shap_values = explainer.shap_values(X_test)
            shap_values_all = np.append(shap_values_all, shap_values[1], axis=0)
            X_test_all = X_test_all.append(X_test).reset_index(drop=True)

if compute_shap:
    shap.summary_plot(shap_values_all, X_test_all, plot_type='bar', max_display=50)
    shap.summary_plot(shap_values_all, X_test_all, plot_type='dot', max_display=50)

#%%
if save_to_csv:
    df_crispr_full.to_csv(data_path+'/results/csv/distal_reg_paper/EG_features/K562/FullModel/EPCrisprBenchmark_ensemble_data_GRCh38.K562_AllFeatures_NAfilled.FullModel_withPreds.tsv', sep = '\t', index=False)


############################################### Alireza's suggestions (GraphReg models) ###############################################

#%%
RefSeqGenes = pd.read_csv(data_path+'/results/csv/distal_reg_paper/RefSeqGenes/RefSeqCurated.170308.bed.CollapsedGeneBounds.hg38.TSS500bp.bed', names = ['chr', 'start', 'end', 'gene', 'len', 'strand'], delimiter = '\t')

df_crispr = pd.read_csv(data_path+'/results/csv/distal_reg_paper/EG_features/K562/AllFeatures/EPCrisprBenchmark_ensemble_data_GRCh38.K562_AllFeatures_NAfilled.tsv', delimiter = '\t')
df_crispr = df_crispr[df_crispr['measuredGeneSymbol'].isin(RefSeqGenes['gene'])].reset_index(drop=True)
df_crispr = df_crispr[~df_crispr['Regulated'].isna()].reset_index(drop=True)
df_crispr = df_crispr.drop(columns=['GraphReg.Score'])
df_crispr = df_crispr.replace([np.inf, -np.inf], np.nan)
df_crispr = df_crispr.fillna(0)

RefSeqGenes_sub = RefSeqGenes[RefSeqGenes['gene'].isin(df_crispr['measuredGeneSymbol'])].reset_index(drop=True)
df_crispr['TSS_from_universe'] = -1
for i, g in enumerate(RefSeqGenes_sub['gene'].values):
    idx = df_crispr[df_crispr['measuredGeneSymbol'] == g].index
    df_crispr.loc[idx, 'TSS_from_universe'] = (RefSeqGenes_sub.loc[i, 'start'] + RefSeqGenes_sub.loc[i, 'end'])//2

df_crispr['distance'] = np.abs((df_crispr['chromStart'] + df_crispr['chromEnd'])//2 - df_crispr['TSS_from_universe'])

model = 'Baseline'
features_list = ['distance', '3DContact',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 'normalizedH3K27ac_prom', 'normalizedDNase_prom']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'GraphReg_LR'
features_list = ['distance',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'GraphReg_LR without H3K4me3'
features_list = ['distance',
                'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'GraphReg_LR without H3K27ac'
features_list = ['distance',
                'H3K4me3_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8',
                'DNase_p_grad_min_L_8']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'GraphReg_LR without DNase'
features_list = ['distance',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'GraphReg_LR without gradients'
features_list = ['distance',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_p_max_L_8', 'H3K27ac_p_max_L_8', 'DNase_p_max_L_8']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'GraphReg_LR without distance'
features_list = ['H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'GraphReg_LR without gradients and distance'
features_list = ['H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_p_max_L_8', 'H3K27ac_p_max_L_8', 'DNase_p_max_L_8']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'GraphReg+RamilWeighted'
model_list.append(model)
features_list = ['distance',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8', 'RamilWeighted']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'GraphReg+ABCScore'
model_list.append(model)
features_list = ['distance',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8', 'ABCScore']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'GraphReg+ABCScore+RamilWeighted'
model_list.append(model)
features_list = ['distance',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8', 'ABCScore', 'RamilWeighted']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
shap_values_all = np.empty([0,X.shape[1]])
X_test_all = pd.DataFrame(columns=features_list)
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'GraphReg+ABCScore+PET'
model_list.append(model)
features_list = ['distance',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8', 'ABCScore', 'PEToutsideNormalized', 'PETcrossNormalized']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]


#%%
model = 'GraphReg+ABCScore+CTCF'
model_list.append(model)
features_list = ['distance',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8', 'ABCScore', 'enhCTCF', 'promCTCF']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]


#%%
model = 'GraphReg+ABCScore+PET+CTCF'
model_list.append(model)
features_list = ['distance',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8', 'ABCScore', 'PEToutsideNormalized', 'PETcrossNormalized', 'enhCTCF', 'promCTCF']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'GraphReg+ABCScore+PET+CTCF+RamilWeighted'
model_list.append(model)
features_list = ['distance',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8', 'ABCScore', 'PEToutsideNormalized', 'PETcrossNormalized', 'enhCTCF', 'promCTCF', 'RamilWeighted']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'GraphReg+ABCScore+PET+CTCF+RamilWeighted+sumNearbyEnhancers_10kb'
model_list.append(model)
features_list = ['distance',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8', 'ABCScore', 'PEToutsideNormalized', 'PETcrossNormalized', 
                'enhCTCF', 'promCTCF', 'RamilWeighted', 'sumNearbyEnhancers_10kb']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'GraphReg+ABCScore+PET+CTCF+RamilWeighted+HiCLoop'
model_list.append(model)
features_list = ['distance',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8', 'ABCScore', 'PEToutsideNormalized', 'PETcrossNormalized', 
                'enhCTCF', 'promCTCF', 'RamilWeighted', 'HiCLoopCrossNormalized', 'HiCLoopOutsideNormalized']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'GraphReg+ABCScore+PET+CTCF+RamilWeighted+HiCLoop+inTAD+inCCD'
model_list.append(model)
features_list = ['distance',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8', 'ABCScore', 'PEToutsideNormalized', 'PETcrossNormalized', 
                'enhCTCF', 'promCTCF', 'RamilWeighted', 'HiCLoopCrossNormalized', 'HiCLoopOutsideNormalized',
                'inTAD', 'inCCD']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### find enhancers that some models fail to predict
df_crispr_sub = df_crispr[(df_crispr['distance'] >= 20000) & (df_crispr['distance'] < 200000)].reset_index(drop=True)
# classify based on threshold for 0.7 recall
Threshold_GraphReg = 0.22
Threshold_GraphReg_without_gradients = 0.20

df_crispr_sub['GraphReg_pred'] = (df_crispr_sub['GraphReg.Score'] > Threshold_GraphReg).values.astype(np.int64)
df_crispr_sub['GraphReg_without_Gradients_pred'] = (df_crispr_sub['GraphReg_without_Gradients.Score'] > Threshold_GraphReg_without_gradients).values.astype(np.int64)
df_crispr_sub['True_enhancer'] = df_crispr_sub['Regulated'].values.astype(np.int64)

df_crispr_sub_mismatch = df_crispr_sub[(df_crispr_sub['True_enhancer']==df_crispr_sub['GraphReg_pred']) & ((df_crispr_sub['True_enhancer']!=df_crispr_sub['GraphReg_without_Gradients_pred']))].reset_index(drop=True)
#df_crispr_sub_mismatch_v2 = df_crispr_sub[(df_crispr_sub['True_enhancer']==df_crispr_sub['GraphReg_without_Gradients_pred']) & ((df_crispr_sub['True_enhancer']!=df_crispr_sub['GraphReg_pred']))].reset_index(drop=True)

df_crispr_sub_mismatch_false_pos = df_crispr_sub_mismatch[df_crispr_sub_mismatch['ABCScore']>0.02].reset_index(drop=True)
df_crispr_sub_mismatch_false_neg = df_crispr_sub_mismatch[df_crispr_sub_mismatch['ABCScore']<0.01].reset_index(drop=True)



#%%
##### auPR curves

assert  all(df_crispr['name'] == df_crispr_full['name'])
if 'ENCODE-E2G_Extended.Score' not in df_crispr.columns:
    df_crispr = df_crispr.join(df_crispr_full[['ENCODE-E2G_Extended.Score',
                                            'ENCODE-E2G_Extended without gradients (12).Score',
                                            'ENCODE-E2G_Extended without distance (3).Score',
                                            'ENCODE-E2G_Extended without EP300 (1).Score',
                                            'ENCODE-E2G_Extended without correlations (3).Score',
                                            'ENCODE-E2G_Extended without CTCF (5).Score',
                                            'ENCODE-E2G_Extended without Hi-C (20).Score',
                                            'ENCODE-E2G_Extended without Hi-C and distance (23).Score',
                                            'ENCODE-E2G_Extended without contact and distance (26).Score',
                                            'ENCODE-E2G_Extended without contact (23).Score',
                                            'ENCODE-E2G_Extended without ABC (3).Score',
                                            'ENCODE-E2G_Extended without number of nearby enhancers (2).Score',
                                            'ENCODE-E2G_Extended without enhancer activity (24).Score',
                                            'ENCODE-E2G_Extended without promoter activity (20).Score',
                                            'ENCODE-E2G_Extended without promoter class (3).Score',
                                            'ENCODE-E2G_Extended without conservation (2).Score']])

sns.set_style("ticks")

name_analysis = 'GraphReg_Models_without_category_features'
model_list = ['ABC',
           'GraphReg_LR',
           'GraphReg_LR without gradients',
           'GraphReg_LR without distance',
           'GraphReg_LR without H3K4me3',
           'GraphReg_LR without H3K27ac',
           'GraphReg_LR without DNase',
           ]


'''
name_analysis = 'Models_without_gradients'
model_list = ['ABC',
            'ENCODE-E2G_Extended',
            'ENCODE-E2G_Extended without gradients (12)',
            'GraphReg_LR',
            'GraphReg_LR without gradients'
           ]
'''

save_fig = True
for i in range(6):
    df = pd.DataFrame()
    if i==0:
        df_crispr_sub = df_crispr
    elif i==1:
        df_crispr_sub = df_crispr[df_crispr['distance'] < 10000]
    elif i==2:
        df_crispr_sub = df_crispr[(df_crispr['distance'] >= 10000) & (df_crispr['distance'] < 100000)]
    elif i==3:
        df_crispr_sub = df_crispr[(df_crispr['distance'] >= 100000) & (df_crispr['distance'] <= 2500000)]
    elif i==4:
        df_crispr_sub = df_crispr[df_crispr['dataset'] == 'FlowFISH_K562']
    elif i==5:
        df_crispr_sub = df_crispr[df_crispr['dataset'] == 'Gasperini2019']

    Y_true = df_crispr_sub['Regulated'].values.astype(np.int64)
    plt.figure(figsize=(20,20))
    plt.grid()
    for model in model_list:
        if model == 'ABC':
            Y_pred = df_crispr_sub['ABCScore'].values
        else:
            Y_pred = df_crispr_sub[model+'.Score'].values

        if model=='ABC':
            color_code = '#4E79A7'
        elif model=='GraphReg_LR':
            color_code =  "#2E8B57"
        elif model=='GraphReg_LR without gradients':
            color_code = "lightgreen"
        elif model=='GraphReg_LR without distance':
            color_code = "#00EEEE"
        elif model=='GraphReg_LR without H3K4me3':
            color_code = "#EEAD0E"
        elif model=='GraphReg_LR without H3K27ac':
            color_code = "coral"
        elif model=='GraphReg_LR without DNase':
            color_code = "#838B83"
        elif model=='ENCODE-E2G_Extended':
            color_code = "#B07AA1"
        elif model=='ENCODE-E2G_Extended without gradients (12)':
            color_code = "pink"

        precision, recall, thresholds = precision_recall_curve(Y_true, Y_pred)
        average_precision = average_precision_score(Y_true, Y_pred)
        aupr = auc(recall, precision)

        idx_recall_70_pct = np.argsort(np.abs(recall - 0.7))[0]
        recall_at_70_pct = recall[idx_recall_70_pct]
        precision_at_70_pct_recall = precision[idx_recall_70_pct]
        threshod_in_70_pct_recall = thresholds[idx_recall_70_pct]

        #plt.plot(recall, precision, color=color_code, linewidth=5, label='{} || auPR={:6.4f} || Precision={:6.4f} || Threshold={:6.4f}'.format(model, aupr, precision_at_70_pct_recall, threshod_in_70_pct_recall))
        plt.plot(recall, precision, color=color_code, linewidth=5) 
        if i==0:
            plt.title('Ensemble | All | #EG = {} | #Positives = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==1:
            plt.title('Ensemble | [0,10kb) | #EG = {} | #Positives = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==2:
            plt.title('Ensemble | [10kb,100kb) | #EG = {} | #Positives = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==3:
            plt.title('Ensemble | [100kb,2.5Mb) | #EG = {} | #Positives = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==4:
            plt.title('Fulco | All | #EG = {} | #Positives = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==5:
            plt.title('Gasperini | All | #EG = {} | #Positives = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        
        #plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=40)
        plt.xlabel("Recall", fontsize=40)
        plt.ylabel("Precision", fontsize=40)
        plt.tick_params(axis='x', labelsize=40, length=10, width=5)
        plt.tick_params(axis='y', labelsize=40, length=10, width=5)
        plt.grid(False)
        if save_fig:
            if i==0:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_all.pdf', bbox_inches='tight')
            elif i==1:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_0-10k.pdf', bbox_inches='tight')
            elif i==2:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_10k-100k.pdf', bbox_inches='tight')
            elif i==3:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_100k-2500k.pdf', bbox_inches='tight')
            elif i==4:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_FlowFISH_all.pdf', bbox_inches='tight')
            elif i==5:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Gasperini_all.pdf', bbox_inches='tight')


#%%
##### bootstrapping (remove different categories of features in GraphReg_LR model)

sns.set_style("ticks")

# statistic functions for scipy.stats.bootstrap:

def my_statistic_aupr(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    aupr = auc(recall, precision)
    return aupr

def my_statistic_precision(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    idx_recall_70_pct = np.argsort(np.abs(recall - 0.7))[0]
    precision_at_70_pct_recall = precision[idx_recall_70_pct]
    return precision_at_70_pct_recall

def my_statistic_delta_aupr(y_true, y_pred_full, y_pred_ablated):
    precision_full, recall_full, thresholds_full = precision_recall_curve(y_true, y_pred_full)
    aupr_full = auc(recall_full, precision_full)

    precision_ablated, recall_ablated, thresholds_ablated = precision_recall_curve(y_true, y_pred_ablated)
    aupr_ablated = auc(recall_ablated, precision_ablated)

    delta_aupr = aupr_ablated - aupr_full
    return delta_aupr

def my_statistic_delta_precision(y_true, y_pred_full, y_pred_ablated):
    precision_full, recall_full, thresholds_full = precision_recall_curve(y_true, y_pred_full)
    idx_recall_full_70_pct = np.argsort(np.abs(recall_full - 0.7))[0]
    precision_full_at_70_pct_recall = precision_full[idx_recall_full_70_pct]

    precision_ablated, recall_ablated, thresholds_ablated = precision_recall_curve(y_true, y_pred_ablated)
    idx_recall_ablated_70_pct = np.argsort(np.abs(recall_ablated - 0.7))[0]
    precision_ablated_at_70_pct_recall = precision_ablated[idx_recall_ablated_70_pct]

    delta_precision = precision_ablated_at_70_pct_recall - precision_full_at_70_pct_recall
    return delta_precision

def bootstrap_pvalue(delta, res_delta):
    """ Bootstrap p values for delta (aupr/precision) """
    
    # Original delta
    orig_delta = delta
    
    # Generate boostrap distribution of delta under null hypothesis
    #delta_boot_distribution = res_delta.bootstrap_distribution - orig_delta  # important centering step to get sampling distribution under the null
    delta_boot_distribution = res_delta.bootstrap_distribution - res_delta.bootstrap_distribution.mean()

    # Calculate proportion of bootstrap samples with at least as strong evidence against null    
    pval = np.mean(np.abs(delta_boot_distribution) >= np.abs(orig_delta))
    
    return pval

name_analysis = 'GraphReg_Models_without_category_features'
model_list = ['GraphReg_LR',
            'GraphReg_LR without gradients',
            'GraphReg_LR without distance',
            'GraphReg_LR without H3K4me3',
            'GraphReg_LR without H3K27ac',
            'GraphReg_LR without DNase',
            ]

save_fig = False
df = pd.DataFrame(columns=['Distance Range', 'Model', 'ID', 'Delta auPR', 'Delta Precision'])
df_append = pd.DataFrame(columns=['Distance Range', 'Model', 'ID', 'Delta auPR', 'Delta Precision'])
for i in range(4):
    if i==0:
        df_crispr_sub = df_crispr
        Distance_Range = 'All'
    elif i==1:
        df_crispr_sub = df_crispr[df_crispr['distance'] < 10000]
        Distance_Range = '[0, 10kb)'
    elif i==2:
        df_crispr_sub = df_crispr[(df_crispr['distance'] >= 10000) & (df_crispr['distance'] < 100000)]
        Distance_Range = '[10kb, 100kb)'
    elif i==3:
        df_crispr_sub = df_crispr[(df_crispr['distance'] >= 100000) & (df_crispr['distance'] <= 2500000)]
        Distance_Range = '[100kb, 2.5Mb)'
    elif i==4:
        df_crispr_sub = df_crispr[df_crispr['dataset'] == 'FlowFISH_K562']
        Distance_Range = 'Fulco'
    elif i==5:
        df_crispr_sub = df_crispr[df_crispr['dataset'] == 'Gasperini2019']
        Distance_Range = 'Gasperini'

    ## scipy bootstrap
    Y_true = df_crispr_sub['Regulated'].values.astype(np.int64)
    Y_pred_full = df_crispr_sub[model_list[0]+'.Score'].values
    for model in model_list[1:]:
        print(f'i = {i} ,model = {model}')

        Y_pred_ablated = df_crispr_sub[model+'.Score'].values
        data = (Y_true, Y_pred_full, Y_pred_ablated)
        delta_aupr = my_statistic_delta_aupr(Y_true, Y_pred_full, Y_pred_ablated)
        delta_precision = my_statistic_delta_precision(Y_true, Y_pred_full, Y_pred_ablated)

        res_delta_aupr = scipy.stats.bootstrap(data, my_statistic_delta_aupr, n_resamples=1000, paired=True, confidence_level=0.95, method='percentile')
        res_delta_precision = scipy.stats.bootstrap(data, my_statistic_delta_precision, n_resamples=1000, paired=True, confidence_level=0.95, method='percentile')

        print(f'Delta auPR p-value = {bootstrap_pvalue(delta_aupr, res_delta_aupr)}')
        print(f'Delta precision p-value = {bootstrap_pvalue(delta_precision, res_delta_precision)}')
        print('######################################')

        df_append['Delta auPR'] = res_delta_aupr.bootstrap_distribution
        df_append['Delta Precision'] = res_delta_precision.bootstrap_distribution
        df_append['ID'] = np.arange(1000)
        df_append['Distance Range'] = Distance_Range
        df_append['Model'] = model
        df = pd.concat([df, df_append], ignore_index=True)

df_sub = df[df['Distance Range']=='All']
median_dict = {}
model_list = df['Model'].unique()
for model in model_list:
    if model != "ENCODE-E2G_Extended":
        median_dict[model] = np.median(df_sub[df_sub['Model']==model]['Delta auPR'].values)

sorted_list = sorted([(value,key) for (key,value) in median_dict.items()])
order = []
for _, model in enumerate(sorted_list):
    order.append(model[1])

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(22, 5))
ax1.grid('y')
g = sns.barplot(data=df, x='Distance Range', y='Delta auPR', hue='Model', ax=ax1, errorbar=("pi", 95), seed=None, hue_order=order, errwidth=2, capsize=0.03,
        palette={"ABC": "#4E79A7", "GraphReg_LR": "#2E8B57", 'GraphReg_LR without gradients': "lightgreen", 
        'GraphReg_LR without distance': "#00EEEE", 'GraphReg_LR without H3K4me3': "#EEAD0E", 
        'GraphReg_LR without H3K27ac': "coral", 'GraphReg_LR without DNase': "#838B83"})
sns.move_legend(ax1, "upper left", bbox_to_anchor=(1, 1))
# annotator = Annotator(ax1, data=df, x='Distance Range', y='auPR', hue='Model', 
#                                 pairs=[(("All", "GraphReg_LRScore"), ("All", "GraphReg_LRScore w/o gradients")),
#                                         (("[10kb, 100kb)", "GraphReg_LRScore"), ("[10kb, 100kb)", "GraphReg_LRScore w/o gradients")),
#                                         (("[100kb, 2.5Mb)", "GraphReg_LRScore"), ("[100kb, 2.5Mb)", "GraphReg_LRScore w/o gradients"))])
# annotator.configure(test='Wilcoxon', comparisons_correction='Benjamini-Hochberg', text_format='star', loc='inside', fontsize='x-large')
# annotator.apply_and_annotate()

ax1.yaxis.set_tick_params(labelsize=20)
ax1.xaxis.set_tick_params(labelsize=20)
#ax1.set_title('title', fontsize=20)
g.set_xlabel("",fontsize=20)
g.set_ylabel("Delta auPR",fontsize=20)
plt.setp(ax1.get_legend().get_texts(), fontsize='20')
plt.setp(ax1.get_legend().get_title(), fontsize='20')
plt.tight_layout()
if save_fig:
    plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/Barplot_Delta_auPR_{name_analysis}.pdf', bbox_inches='tight')


fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(22, 5))
ax1.grid('y')
g = sns.barplot(data=df, x='Distance Range', y='Delta Precision', hue='Model', ax=ax1, errorbar=("pi", 95), seed=None, hue_order=order, errwidth=2, capsize=0.03,
        palette={"ABC": "#4E79A7", "GraphReg_LR": "#2E8B57", 'GraphReg_LR without gradients': "lightgreen", 
        'GraphReg_LR without distance': "#00EEEE", 'GraphReg_LR without H3K4me3': "#EEAD0E", 
        'GraphReg_LR without H3K27ac': "coral", 'GraphReg_LR without DNase': "#838B83"})
sns.move_legend(ax1, "upper left", bbox_to_anchor=(1, 1))
# annotator = Annotator(ax1, data=df, x='Distance Range', y='Precision at Recall=0.7', hue='Model', 
#                                 pairs=[(("All", "GraphReg_LRScore"), ("All", "GraphReg_LRScore w/o gradients")),
#                                         (("[10kb, 100kb)", "GraphReg_LRScore"), ("[10kb, 100kb)", "GraphReg_LRScore w/o gradients"))])
# annotator.configure(test='Wilcoxon', comparisons_correction='Benjamini-Hochberg', text_format='star', loc='inside', fontsize='x-large')
# annotator.apply_and_annotate()

ax1.yaxis.set_tick_params(labelsize=20)
ax1.xaxis.set_tick_params(labelsize=20)
#ax1.set_title('title', fontsize=20)
g.set_xlabel("",fontsize=20)
g.set_ylabel("Delat Precision",fontsize=20)
plt.setp(ax1.get_legend().get_texts(), fontsize='20')
plt.setp(ax1.get_legend().get_title(), fontsize='20')
plt.tight_layout()
if save_fig:
    plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/Barplot_Delta_Precision_{name_analysis}.pdf', bbox_inches='tight')

#%%
##### bootstrapping (remove different categories of features in E2G_ext model)

sns.set_style("ticks")

# statistic functions for scipy.stats.bootstrap:

def my_statistic_aupr(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    aupr = auc(recall, precision)
    return aupr

def my_statistic_precision(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    idx_recall_70_pct = np.argsort(np.abs(recall - 0.7))[0]
    precision_at_70_pct_recall = precision[idx_recall_70_pct]
    return precision_at_70_pct_recall

def my_statistic_delta_aupr(y_true, y_pred_full, y_pred_ablated):
    precision_full, recall_full, thresholds_full = precision_recall_curve(y_true, y_pred_full)
    aupr_full = auc(recall_full, precision_full)

    precision_ablated, recall_ablated, thresholds_ablated = precision_recall_curve(y_true, y_pred_ablated)
    aupr_ablated = auc(recall_ablated, precision_ablated)

    delta_aupr = aupr_ablated - aupr_full
    return delta_aupr

def my_statistic_delta_precision(y_true, y_pred_full, y_pred_ablated):
    precision_full, recall_full, thresholds_full = precision_recall_curve(y_true, y_pred_full)
    idx_recall_full_70_pct = np.argsort(np.abs(recall_full - 0.7))[0]
    precision_full_at_70_pct_recall = precision_full[idx_recall_full_70_pct]

    precision_ablated, recall_ablated, thresholds_ablated = precision_recall_curve(y_true, y_pred_ablated)
    idx_recall_ablated_70_pct = np.argsort(np.abs(recall_ablated - 0.7))[0]
    precision_ablated_at_70_pct_recall = precision_ablated[idx_recall_ablated_70_pct]

    delta_precision = precision_ablated_at_70_pct_recall - precision_full_at_70_pct_recall
    return delta_precision

def bootstrap_pvalue(delta, res_delta):
    """ Bootstrap p values for delta (aupr/precision) """
    
    # Original delta
    orig_delta = delta
    
    # Generate boostrap distribution of delta under null hypothesis
    #delta_boot_distribution = res_delta.bootstrap_distribution - orig_delta  # important centering step to get sampling distribution under the null
    delta_boot_distribution = res_delta.bootstrap_distribution - res_delta.bootstrap_distribution.mean()

    # Calculate proportion of bootstrap samples with at least as strong evidence against null    
    pval = np.mean(np.abs(delta_boot_distribution) >= np.abs(orig_delta))
    
    return pval

'''
name_analysis = 'Models_without_category_features'
model_list = ['ENCODE-E2G_Extended',
            'ENCODE-E2G_Extended without gradients (12)',
            'ENCODE-E2G_Extended without distance (3)',
            'ENCODE-E2G_Extended without EP300 (1)',
            'ENCODE-E2G_Extended without correlations (3)',
            'ENCODE-E2G_Extended without CTCF (5)',
            'ENCODE-E2G_Extended without Hi-C (20)',
            'ENCODE-E2G_Extended without Hi-C and distance (23)',
            'ENCODE-E2G_Extended without contact and distance (26)',
            'ENCODE-E2G_Extended without contact (23)',
            'ENCODE-E2G_Extended without ABC (3)',
            'ENCODE-E2G_Extended without number of nearby enhancers (2)',
            'ENCODE-E2G_Extended without enhancer activity (24)',
            'ENCODE-E2G_Extended without promoter activity (20)',
            'ENCODE-E2G_Extended without promoter class (3)',
            'ENCODE-E2G_Extended without conservation (2)',
            ]
palette = 'tab20'
'''

name_analysis = 'Models_without_category_features_shortlist'
model_list = ['ENCODE-E2G_Extended',
            'ENCODE-E2G_Extended without gradients (12)',
            'ENCODE-E2G_Extended without correlations (3)',
            'ENCODE-E2G_Extended without CTCF (5)',
            'ENCODE-E2G_Extended without contact and distance (26)',
            'ENCODE-E2G_Extended without number of nearby enhancers (2)',
            'ENCODE-E2G_Extended without enhancer activity (24)',
            'ENCODE-E2G_Extended without promoter activity (20)',
            'ENCODE-E2G_Extended without promoter class (3)']
palette = 'tab10'


save_fig = False
df = pd.DataFrame(columns=['Distance Range', 'Model', 'ID', 'Delta auPR', 'Delta Precision'])
df_append = pd.DataFrame(columns=['Distance Range', 'Model', 'ID', 'Delta auPR', 'Delta Precision'])
for i in range(4):
    if i==0:
        df_crispr_sub = df_crispr
        Distance_Range = 'All'
    elif i==1:
        df_crispr_sub = df_crispr[df_crispr['distance'] < 10000]
        Distance_Range = '[0, 10kb)'
    elif i==2:
        df_crispr_sub = df_crispr[(df_crispr['distance'] >= 10000) & (df_crispr['distance'] < 100000)]
        Distance_Range = '[10kb, 100kb)'
    elif i==3:
        df_crispr_sub = df_crispr[(df_crispr['distance'] >= 100000) & (df_crispr['distance'] <= 2500000)]
        Distance_Range = '[100kb, 2.5Mb)'
    elif i==4:
        df_crispr_sub = df_crispr[df_crispr['dataset'] == 'FlowFISH_K562']
        Distance_Range = 'Fulco'
    elif i==5:
        df_crispr_sub = df_crispr[df_crispr['dataset'] == 'Gasperini2019']
        Distance_Range = 'Gasperini'

    ## scipy bootstrap
    Y_true = df_crispr_sub['Regulated'].values.astype(np.int64)
    Y_pred_full = df_crispr_sub[model_list[0]+'.Score'].values
    for model in model_list[1:]:
        print(f'i = {i} ,model = {model}')

        Y_pred_ablated = df_crispr_sub[model+'.Score'].values
        data = (Y_true, Y_pred_full, Y_pred_ablated)
        delta_aupr = my_statistic_delta_aupr(Y_true, Y_pred_full, Y_pred_ablated)
        delta_precision = my_statistic_delta_precision(Y_true, Y_pred_full, Y_pred_ablated)

        res_delta_aupr = scipy.stats.bootstrap(data, my_statistic_delta_aupr, n_resamples=1000, paired=True, confidence_level=0.95, method='percentile')
        res_delta_precision = scipy.stats.bootstrap(data, my_statistic_delta_precision, n_resamples=1000, paired=True, confidence_level=0.95, method='percentile')

        print(f'Delta auPR p-value = {bootstrap_pvalue(delta_aupr, res_delta_aupr)}')
        print(f'Delta precision p-value = {bootstrap_pvalue(delta_precision, res_delta_precision)}')
        print('######################################')

        df_append['Delta auPR'] = res_delta_aupr.bootstrap_distribution
        df_append['Delta Precision'] = res_delta_precision.bootstrap_distribution
        df_append['ID'] = np.arange(1000)
        df_append['Distance Range'] = Distance_Range
        df_append['Model'] = model
        df = pd.concat([df, df_append], ignore_index=True)

df_sub = df[df['Distance Range']=='All']
median_dict = {}
model_list = df['Model'].unique()
for model in model_list:
    if model != "ENCODE-E2G_Extended":
        median_dict[model] = np.median(df_sub[df_sub['Model']==model]['Delta auPR'].values)

sorted_list = sorted([(value,key) for (key,value) in median_dict.items()])
order = []
for _, model in enumerate(sorted_list):
    order.append(model[1])

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(25, 6))
ax1.grid('y')
g = sns.barplot(data=df, x='Distance Range', y='Delta auPR', hue='Model', ax=ax1, errorbar=("pi", 95), seed=None, hue_order=order, errwidth=1.5, capsize=0.03, palette=palette)
sns.move_legend(ax1, "upper left", bbox_to_anchor=(1, 1))

ax1.yaxis.set_tick_params(labelsize=20)
ax1.xaxis.set_tick_params(labelsize=20)
#ax1.set_ylim(bottom=None, top=None, emit=True, auto=False, ymin=-0.05, ymax=None)
#ax1.set_title('title', fontsize=20)
g.set_xlabel("",fontsize=20)
g.set_ylabel("Delta auPR",fontsize=20)
plt.setp(ax1.get_legend().get_texts(), fontsize='20')
plt.setp(ax1.get_legend().get_title(), fontsize='20')
plt.tight_layout()
if save_fig:
    plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/Barplot_Delta_auPR_{name_analysis}.pdf', bbox_inches='tight')


fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(25, 6))
ax1.grid('y')
g = sns.barplot(data=df, x='Distance Range', y='Delta Precision', hue='Model', ax=ax1, errorbar=("pi", 95), seed=None, hue_order=order, errwidth=1.5, capsize=0.03, palette=palette)
sns.move_legend(ax1, "upper left", bbox_to_anchor=(1, 1))


ax1.yaxis.set_tick_params(labelsize=20)
ax1.xaxis.set_tick_params(labelsize=20)
#ax1.set_title('title', fontsize=20)
g.set_xlabel("",fontsize=20)
g.set_ylabel("Delta Precision",fontsize=20)
plt.setp(ax1.get_legend().get_texts(), fontsize='20')
plt.setp(ax1.get_legend().get_title(), fontsize='20')
plt.tight_layout()
if save_fig:
    plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/Barplot_Delta_Precision_{name_analysis}.pdf', bbox_inches='tight')



############################################### Alireza's suggestions (ABC vs GraphReg vs Full models) ###############################################

#%%
name_analysis = 'Alireza_ABC_GraphReg_Full'
df_crispr['FullModel.Score'] = df_crispr_full['FullModel.Score']
df_crispr['FullModel_minus_EP300.Score'] = df_crispr_full['FullModel_minus_EP300.Score']
model_list = ['ABC', 'GraphReg', 'GraphReg+ABCScore', 'FullModel_minus_EP300', 'FullModel']
save_fig = True
for i in range(6):
    if i==0:
        df_crispr_sub = df_crispr
    elif i==1:
        df_crispr_sub = df_crispr[df_crispr['distance'] < 20000]
    elif i==2:
        df_crispr_sub = df_crispr[(df_crispr['distance'] >= 20000) & (df_crispr['distance'] < 200000)]
    elif i==3:
        df_crispr_sub = df_crispr[df_crispr['distance'] >= 200000]
    elif i==4:
        df_crispr_sub = df_crispr[df_crispr['dataset'] == 'FlowFISH_K562']
    elif i==5:
        df_crispr_sub = df_crispr[df_crispr['dataset'] == 'Gasperini2019']

    Y_true = df_crispr_sub['Regulated'].values.astype(np.int64)
    plt.figure(figsize=(20,20))
    for model in model_list:
        if model == 'ABC':
            Y_pred = df_crispr_sub['ABCScore'].values
        else:
            Y_pred = df_crispr_sub[model+'.Score'].values

        precision, recall, thresholds = precision_recall_curve(Y_true, Y_pred)
        average_precision = average_precision_score(Y_true, Y_pred)
        aupr = auc(recall, precision)

        idx_recall_70_pct = np.argsort(np.abs(recall - 0.7))[0]
        recall_at_70_pct = recall[idx_recall_70_pct]
        precision_at_70_pct_recall = precision[idx_recall_70_pct]
        threshod_in_70_pct_recall = thresholds[idx_recall_70_pct]

        plt.plot(recall, precision, linewidth=3, label='model={} || auPR={:6.4f} || Precision={:4.2f} || Threshold={:4.2f}'.format(model, aupr, precision_at_70_pct_recall, threshod_in_70_pct_recall))
        if i==0:
            plt.title('Ensemble | [0,inf) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==1:
            plt.title('Ensemble | [0,20k) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==2:
            plt.title('Ensemble | [20k,200k) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==3:
            plt.title('Ensemble | [200k,inf) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==4:
            plt.title('FlowFISH | [0,inf) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==5:
            plt.title('Gasperini | [0,inf) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        
        plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=40)
        plt.xlabel("Recall", fontsize=40)
        plt.ylabel("Precision", fontsize=40)
        plt.tick_params(axis='x', labelsize=40)
        plt.tick_params(axis='y', labelsize=40)
        plt.grid()
        if save_fig:
            if i==0:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_all.pdf', bbox_inches='tight')
            elif i==1:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_0-20k.pdf', bbox_inches='tight')
            elif i==2:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_20k-200k.pdf', bbox_inches='tight')
            elif i==3:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_200k-end.pdf', bbox_inches='tight')
            elif i==4:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_FlowFISH_all.pdf', bbox_inches='tight')
            elif i==5:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Gasperini_all.pdf', bbox_inches='tight')
        

############################################### Maya's suggestions ###############################################

#%%
df_crispr = pd.read_csv(data_path+'/results/csv/distal_reg_paper/EG_features/K562/AllFeatures/EPCrisprBenchmark_ensemble_data_GRCh38.K562_AllFeatures_NAfilled.tsv', delimiter = '\t')
df_crispr = df_crispr[~df_crispr['Regulated'].isna()].reset_index(drop=True)
df_crispr = df_crispr.drop(columns=['GraphReg.Score'])
df_crispr = df_crispr.replace([np.inf, -np.inf], np.nan)
df_crispr = df_crispr.fillna(0)

model = 'Baseline'
features_list = ['distance', '3DContact',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 'normalizedH3K27ac_prom', 'normalizedDNase_prom']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%

model = 'Baseline+ABCDenominator'
features_list = ['distance', '3DContact',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 'normalizedH3K27ac_prom', 'normalizedDNase_prom',
                'ABCDenominator']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%

model = 'Baseline+numNearbyEnhancers'
features_list = ['distance', '3DContact',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 'normalizedH3K27ac_prom', 'normalizedDNase_prom',
                'numNearbyEnhancers']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%

model = 'Baseline+sumNearbyEnhancers'
features_list = ['distance', '3DContact',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 'normalizedH3K27ac_prom', 'normalizedDNase_prom',
                'sumNearbyEnhancers']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%

model = 'Baseline+numNearbyEnhancers+sumNearbyEnhancers'
features_list = ['distance', '3DContact',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 'normalizedH3K27ac_prom', 'normalizedDNase_prom',
                'numNearbyEnhancers', 'sumNearbyEnhancers']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%

model = 'Baseline+sumNearbyEnhancers_10kb'
features_list = ['distance', '3DContact',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 'normalizedH3K27ac_prom', 'normalizedDNase_prom',
                'sumNearbyEnhancers_10kb']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%

model = 'Baseline+H3K27ac_e_grad_max_L_8+sumNearbyEnhancers'
features_list = ['distance', '3DContact',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 'normalizedH3K27ac_prom', 'normalizedDNase_prom',
                'H3K27ac_e_grad_max_L_8', 'sumNearbyEnhancers']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%

model = 'Baseline+H3K27ac_e_grad_max_L_8+H3K27ac_p_grad_max_L_8+sumNearbyEnhancers'
features_list = ['distance', '3DContact',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 'normalizedH3K27ac_prom', 'normalizedDNase_prom',
                'H3K27ac_e_grad_max_L_8', 'H3K27ac_p_grad_max_L_8', 'sumNearbyEnhancers']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%

model = 'ABCScore+ABCDenominator'
features_list = ['ABCScore', 'ABCDenominator']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%

model = 'ABCScore+numNearbyEnhancers'
features_list = ['ABCScore', 'numNearbyEnhancers']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%

model = 'ABCScore+sumNearbyEnhancers'
features_list = ['ABCScore', 'sumNearbyEnhancers']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%

model = 'ABCScore+numNearbyEnhancers+sumNearbyEnhancers'
features_list = ['ABCScore', 'numNearbyEnhancers', 'sumNearbyEnhancers']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%

model = 'ABCScore+sumNearbyEnhancers_10kb'
features_list = ['ABCScore', 'sumNearbyEnhancers_10kb']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%

model = 'ABCScore+H3K27ac_e_grad_max_L_8+sumNearbyEnhancers'
features_list = ['ABCScore', 'H3K27ac_e_grad_max_L_8', 'sumNearbyEnhancers']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
name_analysis = 'Maya'
model_list = ['ABC', 'ABCScore+ABCDenominator', 'ABCScore+numNearbyEnhancers', 'ABCScore+sumNearbyEnhancers', 
                'ABCScore+numNearbyEnhancers+sumNearbyEnhancers', 'ABCScore+sumNearbyEnhancers_10kb', 'ABCScore+H3K27ac_e_grad_max_L_8+sumNearbyEnhancers',
                'Baseline', 'Baseline+ABCDenominator', 'Baseline+numNearbyEnhancers', 'Baseline+sumNearbyEnhancers', 
                'Baseline+numNearbyEnhancers+sumNearbyEnhancers', 'Baseline+sumNearbyEnhancers_10kb', 'Baseline+H3K27ac_e_grad_max_L_8+sumNearbyEnhancers']

save_fig = True
for i in range(6):
    if i==0:
        df_crispr_sub = df_crispr
    elif i==1:
        df_crispr_sub = df_crispr[df_crispr['distance'] < 20000]
    elif i==2:
        df_crispr_sub = df_crispr[(df_crispr['distance'] >= 20000) & (df_crispr['distance'] < 200000)]
    elif i==3:
        df_crispr_sub = df_crispr[df_crispr['distance'] >= 200000]
    elif i==4:
        df_crispr_sub = df_crispr[df_crispr['dataset'] == 'FlowFISH_K562']
    elif i==5:
        df_crispr_sub = df_crispr[df_crispr['dataset'] == 'Gasperini2019']

    Y_true = df_crispr_sub['Regulated'].values.astype(np.int64)
    plt.figure(figsize=(20,20))
    for model in model_list:
        if model == 'ABC':
            Y_pred = df_crispr_sub['ABCScore'].values
        else:
            Y_pred = df_crispr_sub[model+'.Score'].values

        precision, recall, thresholds = precision_recall_curve(Y_true, Y_pred)
        average_precision = average_precision_score(Y_true, Y_pred)
        aupr = auc(recall, precision)

        idx_recall_70_pct = np.argsort(np.abs(recall - 0.7))[0]
        recall_at_70_pct = recall[idx_recall_70_pct]
        precision_at_70_pct_recall = precision[idx_recall_70_pct]
        threshod_in_70_pct_recall = thresholds[idx_recall_70_pct]

        plt.plot(recall, precision, linewidth=3, label='model={} || auPR={:6.4f} || Precision={:4.2f} || Threshold={:4.2f}'.format(model, aupr, precision_at_70_pct_recall, threshod_in_70_pct_recall))
        if i==0:
            plt.title('Ensemble | [0,inf) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==1:
            plt.title('Ensemble | [0,20k) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==2:
            plt.title('Ensemble | [20k,200k) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==3:
            plt.title('Ensemble | [200k,inf) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==4:
            plt.title('FlowFISH | [0,inf) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==5:
            plt.title('Gasperini | [0,inf) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        
        plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=40)
        plt.xlabel("Recall", fontsize=40)
        plt.ylabel("Precision", fontsize=40)
        plt.tick_params(axis='x', labelsize=40)
        plt.tick_params(axis='y', labelsize=40)
        plt.grid()
        if save_fig:
            if i==0:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_all.pdf', bbox_inches='tight')
            elif i==1:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_0-20k.pdf', bbox_inches='tight')
            elif i==2:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_20k-200k.pdf', bbox_inches='tight')
            elif i==3:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_200k-end.pdf', bbox_inches='tight')
            elif i==4:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_FlowFISH_all.pdf', bbox_inches='tight')
            elif i==5:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Gasperini_all.pdf', bbox_inches='tight')
        


############################################### Jesse suggestions ###############################################

#%%
model_list = ['ABC']
model = 'ABCScore+numNearbyEnhancers_10kb+sumNearbyEnhancers_10kb'
model_list.append(model)
features_list = ['ABCScore', 'numNearbyEnhancers_10kb', 'sumNearbyEnhancers_10kb']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'Baseline+numNearbyEnhancers_10kb+sumNearbyEnhancers_10kb'
model_list.append(model)
features_list = ['distance', '3DContact',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 'normalizedH3K27ac_prom', 
                'normalizedDNase_prom', 'numNearbyEnhancers_10kb', 'sumNearbyEnhancers_10kb']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'activity+3Dcontact'
model_list.append(model)
features_list = ['activity_enh', 'activity_prom', '3DContact']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'activity+3Dcontact+activity_squared'
model_list.append(model)
features_list = ['activity_enh', 'activity_prom', '3DContact', 'activity_enh_squared']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
name_analysis = 'Jesse'
save_fig = True
for i in range(6):
    if i==0:
        df_crispr_sub = df_crispr
    elif i==1:
        df_crispr_sub = df_crispr[df_crispr['distance'] < 20000]
    elif i==2:
        df_crispr_sub = df_crispr[(df_crispr['distance'] >= 20000) & (df_crispr['distance'] < 200000)]
    elif i==3:
        df_crispr_sub = df_crispr[df_crispr['distance'] >= 200000]
    elif i==4:
        df_crispr_sub = df_crispr[df_crispr['dataset'] == 'FlowFISH_K562']
    elif i==5:
        df_crispr_sub = df_crispr[df_crispr['dataset'] == 'Gasperini2019']

    Y_true = df_crispr_sub['Regulated'].values.astype(np.int64)
    plt.figure(figsize=(20,20))
    for model in model_list:
        if model == 'ABC':
            Y_pred = df_crispr_sub['ABCScore'].values
        else:
            Y_pred = df_crispr_sub[model+'.Score'].values

        precision, recall, thresholds = precision_recall_curve(Y_true, Y_pred)
        average_precision = average_precision_score(Y_true, Y_pred)
        aupr = auc(recall, precision)

        idx_recall_70_pct = np.argsort(np.abs(recall - 0.7))[0]
        recall_at_70_pct = recall[idx_recall_70_pct]
        precision_at_70_pct_recall = precision[idx_recall_70_pct]
        threshod_in_70_pct_recall = thresholds[idx_recall_70_pct]

        plt.plot(recall, precision, linewidth=3, label='model={} || auPR={:6.4f} || Precision={:4.2f} || Threshold={:4.2f}'.format(model, aupr, precision_at_70_pct_recall, threshod_in_70_pct_recall))
        if i==0:
            plt.title('Ensemble | [0,inf) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==1:
            plt.title('Ensemble | [0,20k) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==2:
            plt.title('Ensemble | [20k,200k) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==3:
            plt.title('Ensemble | [200k,inf) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==4:
            plt.title('FlowFISH | [0,inf) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==5:
            plt.title('Gasperini | [0,inf) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        
        plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=40)
        plt.xlabel("Recall", fontsize=40)
        plt.ylabel("Precision", fontsize=40)
        plt.tick_params(axis='x', labelsize=40)
        plt.tick_params(axis='y', labelsize=40)
        plt.grid()
        if save_fig:
            if i==0:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_all.pdf', bbox_inches='tight')
            elif i==1:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_0-20k.pdf', bbox_inches='tight')
            elif i==2:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_20k-200k.pdf', bbox_inches='tight')
            elif i==3:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_200k-end.pdf', bbox_inches='tight')
            elif i==4:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_FlowFISH_all.pdf', bbox_inches='tight')
            elif i==5:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Gasperini_all.pdf', bbox_inches='tight')
 

############################################### Ramil suggestions ###############################################

#%%

df_crispr = pd.read_csv(data_path+'/results/csv/distal_reg_paper/EG_features/K562/AllFeatures/EPCrisprBenchmark_ensemble_data_GRCh38.K562_AllFeatures_NAfilled.tsv', delimiter = '\t')
df_crispr = df_crispr[~df_crispr['Regulated'].isna()].reset_index(drop=True)
df_crispr = df_crispr.drop(columns=['GraphReg.Score'])
df_crispr = df_crispr.replace([np.inf, -np.inf], np.nan)
df_crispr = df_crispr.fillna(0)

model_list = ['ABC']
model = 'Baseline+ABCScore'
model_list.append(model)
features_list = ['distance', '3DContact',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 'normalizedH3K27ac_prom', 
                'normalizedDNase_prom', 'ABCScore']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'Baseline+ABCScore+RamilWeighted'
model_list.append(model)
features_list = ['distance', '3DContact',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 'normalizedH3K27ac_prom', 
                'normalizedDNase_prom', 'ABCScore', 'RamilWeighted']

X = df_crispr.loc[:,features_list]
#X[features_list[:-1]] = np.log(np.abs(X[features_list[:-1]]) + epsilon)
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]


#%%
model = 'GraphReg'
features_list = ['distance',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'GraphReg+pearsonCorrelation'
model_list.append(model)
features_list = ['distance',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8', 'pearsonCorrelation']

X = df_crispr.loc[:,features_list]
X.iloc[:,:-1] = np.log(np.abs(X.iloc[:,:-1]) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'GraphReg+spearmanCorrelation'
model_list.append(model)
features_list = ['distance',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8', 'spearmanCorrelation']

X = df_crispr.loc[:,features_list]
X.iloc[:,:-1] = np.log(np.abs(X.iloc[:,:-1]) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'ABCScoreAvgHicTrack1'
model_list.append(model)
features_list = ['ABCScoreAvgHicTrack1']

X = df_crispr.loc[:,features_list]
#X.iloc[:,:-1] = np.log(np.abs(X.iloc[:,:-1]) + epsilon)
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'ABCScoreAvgHicTrack1+pearsonCorrelation'
model_list.append(model)
features_list = ['ABCScoreAvgHicTrack1', 'pearsonCorrelation']

X = df_crispr.loc[:,features_list]
X.iloc[:,:-1] = np.log(np.abs(X.iloc[:,:-1]) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'ABCScoreAvgHicTrack1+spearmanCorrelation'
model_list.append(model)
features_list = ['ABCScoreAvgHicTrack1', 'spearmanCorrelation']

X = df_crispr.loc[:,features_list]
X.iloc[:,:-1] = np.log(np.abs(X.iloc[:,:-1]) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]


#%%
model = 'GraphReg+ABCScore'
model_list.append(model)
features_list = ['distance',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8', 'ABCScore']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'GraphReg+RamilWeighted'
model_list.append(model)
features_list = ['distance',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8', 'RamilWeighted']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'GraphReg+ABCScore+RamilWeighted'
model_list.append(model)
features_list = ['distance',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8', 'ABCScore', 'RamilWeighted']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
shap_values_all = np.empty([0,X.shape[1]])
X_test_all = pd.DataFrame(columns=features_list)
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]
#%%
model = 'Baseline'
features_list = ['distance', '3DContact',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 'normalizedH3K27ac_prom', 
                'normalizedDNase_prom']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'Baseline+glsCoefficient'
features_list = ['distance', '3DContact',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 'normalizedH3K27ac_prom', 
                'normalizedDNase_prom', 'glsCoefficient']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'ABCScore+glsCoefficient'
features_list = ['ABCScore', 'glsCoefficient']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'Baseline+pearsonCorrelation'
model_list.append(model)
features_list = ['distance', '3DContact',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 'normalizedH3K27ac_prom', 
                'normalizedDNase_prom', 'pearsonCorrelation']

X = df_crispr.loc[:,features_list]
X.iloc[:,:-1] = np.log(np.abs(X.iloc[:,:-1]) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]  

#%%
model = 'Baseline+spearmanCorrelation'
model_list.append(model)
features_list = ['distance', '3DContact',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 'normalizedH3K27ac_prom', 
                'normalizedDNase_prom', 'spearmanCorrelation']

X = df_crispr.loc[:,features_list]
X.iloc[:,:-1] = np.log(np.abs(X.iloc[:,:-1]) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]  

#%%
model = 'Baseline+glsCoefficient'
features_list = ['distance', '3DContact',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 'normalizedH3K27ac_prom', 
                'normalizedDNase_prom', 'glsCoefficient']

X = df_crispr.loc[:,features_list]
X.iloc[:,:-1] = np.log(np.abs(X.iloc[:,:-1]) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'GraphReg+glsCoefficient'
features_list = ['distance',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8', 'glsCoefficient']

X = df_crispr.loc[:,features_list]
X.iloc[:,:-1] = np.log(np.abs(X.iloc[:,:-1]) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'ABCScoreAvgHicTrack2+glsCoefficient'
features_list = ['ABCScoreAvgHicTrack2', 'glsCoefficient']

X = df_crispr.loc[:,features_list]
X.iloc[:,:-1] = np.log(np.abs(X.iloc[:,:-1]) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
#name_analysis = 'Ramil_1'
#model_list = ['Baseline', 'Baseline+pearsonCorrelation', 'Baseline+spearmanCorrelation']
#name_analysis = 'Ramil_2'
#model_list = ['GraphReg', 'GraphReg+pearsonCorrelation', 'GraphReg+spearmanCorrelation']
#name_analysis = 'Ramil_3'
#model_list = ['ABCScoreAvgHicTrack1', 'ABCScoreAvgHicTrack1+pearsonCorrelation', 'ABCScoreAvgHicTrack1+spearmanCorrelation']
#name_analysis = 'Ramil_4'
#model_list = ['Baseline', 'Baseline+glsCoefficient', 'GraphReg', 'GraphReg+glsCoefficient']
#name_analysis = 'Ramil_5'
#model_list = ['ABCScoreAvgHicTrack2', 'ABCScoreAvgHicTrack2+glsCoefficient']
name_analysis = 'Ramil_6'
model_list = ['Baseline', 'Baseline+glsCoefficient', 'ABC', 'ABCScore+glsCoefficient', 'FullModel', 'FullModel_minus_glsCoefficient']
save_fig = True
for i in range(6):
    if i==0:
        df_crispr_sub = df_crispr
    elif i==1:
        df_crispr_sub = df_crispr[df_crispr['distance'] < 20000]
    elif i==2:
        df_crispr_sub = df_crispr[(df_crispr['distance'] >= 20000) & (df_crispr['distance'] < 200000)]
    elif i==3:
        df_crispr_sub = df_crispr[df_crispr['distance'] >= 200000]
    elif i==4:
        df_crispr_sub = df_crispr[df_crispr['dataset'] == 'FlowFISH_K562']
    elif i==5:
        df_crispr_sub = df_crispr[df_crispr['dataset'] == 'Gasperini2019']

    Y_true = df_crispr_sub['Regulated'].values.astype(np.int64)
    plt.figure(figsize=(20,20))
    for model in model_list:
        if model == 'ABC':
            Y_pred = df_crispr_sub['ABCScore'].values
        elif model == 'ABCScoreAvgHicTrack2':
            Y_pred = df_crispr_sub['ABCScoreAvgHicTrack2'].values
        else:
            Y_pred = df_crispr_sub[model+'.Score'].values

        precision, recall, thresholds = precision_recall_curve(Y_true, Y_pred)
        average_precision = average_precision_score(Y_true, Y_pred)
        aupr = auc(recall, precision)

        idx_recall_70_pct = np.argsort(np.abs(recall - 0.7))[0]
        recall_at_70_pct = recall[idx_recall_70_pct]
        precision_at_70_pct_recall = precision[idx_recall_70_pct]
        threshod_in_70_pct_recall = thresholds[idx_recall_70_pct]

        plt.plot(recall, precision, linewidth=3, label='model={} || auPR={:6.4f} || Precision={:4.2f} || Threshold={:4.2f}'.format(model, aupr, precision_at_70_pct_recall, threshod_in_70_pct_recall))
        if i==0:
            plt.title('Ensemble | [0,inf) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==1:
            plt.title('Ensemble | [0,20k) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==2:
            plt.title('Ensemble | [20k,200k) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==3:
            plt.title('Ensemble | [200k,inf) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==4:
            plt.title('FlowFISH | [0,inf) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==5:
            plt.title('Gasperini | [0,inf) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        
        plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=40)
        plt.xlabel("Recall", fontsize=40)
        plt.ylabel("Precision", fontsize=40)
        plt.tick_params(axis='x', labelsize=40)
        plt.tick_params(axis='y', labelsize=40)
        plt.grid()3
        if save_fig:
            if i==0:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_all.pdf', bbox_inches='tight')
            elif i==1:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_0-20k.pdf', bbox_inches='tight')
            elif i==2:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_20k-200k.pdf', bbox_inches='tight')
            elif i==3:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_200k-end.pdf', bbox_inches='tight')
            elif i==4:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_FlowFISH_all.pdf', bbox_inches='tight')
            elif i==5:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Gasperini_all.pdf', bbox_inches='tight')
        

#%%
if save_to_csv:
    df_crispr.to_csv(data_path+'/results/csv/distal_reg_paper/EG_features/K562/AllFeatures/EPCrisprBenchmark_ensemble_data_GRCh38.K562_AllFeatures_NAfilled_withPreds_Ramil_v2.tsv', sep = '\t', index=False)

############################################### Andreas suggestions ###############################################
#%%
##### 
model_list = ['ABC']
features_list = ['normalizedDNase_enh', 'normalizedDNase_prom', '3DContact']
model = '+'.join(features_list)
model_list.append(model)

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
features_list = ['normalizedH3K27ac_enh', 'normalizedH3K27ac_prom', '3DContact']
model = '+'.join(features_list)
model_list.append(model)

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
features_list = ['normalizedDNase_enh', 'normalizedDNase_prom', 'normalizedH3K27ac_enh', 'normalizedH3K27ac_prom', '3DContact']
model = '+'.join(features_list)
model_list.append(model)

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
features_list = ['activity_enh', 'activity_prom', '3DContact']
model = '+'.join(features_list)
model_list.append(model)

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
features_list = ['normalizedDNase_enh', 'normalizedDNase_prom', 'distance']
model = '+'.join(features_list)
model_list.append(model)

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
features_list = ['normalizedH3K27ac_enh', 'normalizedH3K27ac_prom', 'distance']
model = '+'.join(features_list)
model_list.append(model)

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
features_list = ['normalizedDNase_enh', 'normalizedDNase_prom', 'normalizedH3K27ac_enh', 'normalizedH3K27ac_prom', 'distance']
model = '+'.join(features_list)
model_list.append(model)

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
features_list = ['activity_enh', 'activity_prom', 'distance']
model = '+'.join(features_list)
model_list.append(model)

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
features_list = ['activity_enh', 'activity_prom', 'distance', '3DContact']
model = '+'.join(features_list)
model_list.append(model)

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]


#%%
name_analysis = 'Andreas'
save_fig = True
for i in range(6):
    if i==0:
        df_crispr_sub = df_crispr
    elif i==1:
        df_crispr_sub = df_crispr[df_crispr['distance'] < 20000]
    elif i==2:
        df_crispr_sub = df_crispr[(df_crispr['distance'] >= 20000) & (df_crispr['distance'] < 200000)]
    elif i==3:
        df_crispr_sub = df_crispr[df_crispr['distance'] >= 200000]
    elif i==4:
        df_crispr_sub = df_crispr[df_crispr['dataset'] == 'FlowFISH_K562']
    elif i==5:
        df_crispr_sub = df_crispr[df_crispr['dataset'] == 'Gasperini2019']

    Y_true = df_crispr_sub['Regulated'].values.astype(np.int64)
    plt.figure(figsize=(20,20))
    for model in model_list:
        if model == 'ABC':
            Y_pred = df_crispr_sub['ABCScore'].values
        else:
            Y_pred = df_crispr_sub[model+'.Score'].values

        precision, recall, thresholds = precision_recall_curve(Y_true, Y_pred)
        average_precision = average_precision_score(Y_true, Y_pred)
        aupr = auc(recall, precision)

        idx_recall_70_pct = np.argsort(np.abs(recall - 0.7))[0]
        recall_at_70_pct = recall[idx_recall_70_pct]
        precision_at_70_pct_recall = precision[idx_recall_70_pct]
        threshod_in_70_pct_recall = thresholds[idx_recall_70_pct]

        plt.plot(recall, precision, linewidth=3, label='model={} || auPR={:6.4f} || Precision={:4.2f} || Threshold={:4.2f}'.format(model, aupr, precision_at_70_pct_recall, threshod_in_70_pct_recall))
        if i==0:
            plt.title('Ensemble | [0,inf) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==1:
            plt.title('Ensemble | [0,20k) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==2:
            plt.title('Ensemble | [20k,200k) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==3:
            plt.title('Ensemble | [200k,inf) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==4:
            plt.title('FlowFISH | [0,inf) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==5:
            plt.title('Gasperini | [0,inf) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        
        plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=40)
        plt.xlabel("Recall", fontsize=40)
        plt.ylabel("Precision", fontsize=40)
        plt.tick_params(axis='x', labelsize=40)
        plt.tick_params(axis='y', labelsize=40)
        plt.grid()
        if save_fig:
            if i==0:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_all.pdf', bbox_inches='tight')
            elif i==1:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_0-20k.pdf', bbox_inches='tight')
            elif i==2:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_20k-200k.pdf', bbox_inches='tight')
            elif i==3:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_200k-end.pdf', bbox_inches='tight')
            elif i==4:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_FlowFISH_all.pdf', bbox_inches='tight')
            elif i==5:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Gasperini_all.pdf', bbox_inches='tight')
        




############################################### Wang suggestions ###############################################

#%%
##### Baseline model
model = 'Baseline'
features_list = ['distance', '3DContact',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 'normalizedH3K27ac_prom', 'normalizedDNase_prom']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### Baseline2 model
model = 'BaselineWithoutHiC'
features_list = ['distance',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 'normalizedH3K27ac_prom', 'normalizedDNase_prom']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
model = 'Baseline+PEToutsideNormalized'
features_list = ['distance', '3DContact',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 
                'normalizedH3K27ac_prom', 'normalizedDNase_prom', 'PEToutsideNormalized']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
model = 'BaselineWithoutHiC+PEToutsideNormalized'
features_list = ['distance',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 
                'normalizedH3K27ac_prom', 'normalizedDNase_prom', 'PEToutsideNormalized']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
#####
model = 'Baseline+PETcrossNormalized'
features_list = ['distance', '3DContact',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 
                'normalizedH3K27ac_prom', 'normalizedDNase_prom', 'PETcrossNormalized']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
#####
model = 'BaselineWithoutHiC+PETcrossNormalized'
features_list = ['distance',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 
                'normalizedH3K27ac_prom', 'normalizedDNase_prom', 'PETcrossNormalized']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
model = 'Baseline+PETcrossNormalized+PEToutsideNormalized'
features_list = ['distance', '3DContact',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 
                'normalizedH3K27ac_prom', 'normalizedDNase_prom', 'PETcrossNormalized', 'PEToutsideNormalized']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
model = 'BaselineWithoutHiC+PETcrossNormalized+PEToutsideNormalized'
features_list = ['distance',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 
                'normalizedH3K27ac_prom', 'normalizedDNase_prom', 'PETcrossNormalized', 'PEToutsideNormalized']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
features_list = ['ABCScore', 'PEToutsideNormalized', 'PETcrossNormalized']
model = '+'.join(features_list)
model_list.append(model)

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
features_list = ['distance',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8', 'PEToutsideNormalized', 'PETcrossNormalized']

model = 'GraphReg+PEToutsideNormalized+PETcrossNormalized'
model_list.append(model)

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
features_list = ['distance',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8', 'ABCScore', 'PEToutsideNormalized', 'PETcrossNormalized']

model = 'GraphReg+ABCScore+PEToutsideNormalized+PETcrossNormalized'
model_list.append(model)

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]


#%%
# name_analysis = 'Wang'
# model_list = ['Baseline',
#               'Baseline+PEToutsideNormalized',
#               'Baseline+PETcrossNormalized',
#               'Baseline+PETcrossNormalized+PEToutsideNormalized']

name_analysis = 'Wang_v2'
model_list = ['Baseline',
              'BaselineWithoutHiC',
              'Baseline+PEToutsideNormalized',
              'BaselineWithoutHiC+PEToutsideNormalized',
              'Baseline+PETcrossNormalized',
              'BaselineWithoutHiC+PETcrossNormalized',
              'Baseline+PETcrossNormalized+PEToutsideNormalized',
              'BaselineWithoutHiC+PETcrossNormalized+PEToutsideNormalized']

save_fig = True
for i in range(6):
    df = pd.DataFrame()
    if i==0:
        df_crispr_sub = df_crispr
    elif i==1:
        df_crispr_sub = df_crispr[df_crispr['distance'] < 10000]
    elif i==2:
        df_crispr_sub = df_crispr[(df_crispr['distance'] >= 10000) & (df_crispr['distance'] < 100000)]
    elif i==3:
        df_crispr_sub = df_crispr[df_crispr['distance'] >= 100000]
    elif i==4:
        df_crispr_sub = df_crispr[df_crispr['dataset'] == 'FlowFISH_K562']
    elif i==5:
        df_crispr_sub = df_crispr[df_crispr['dataset'] == 'Gasperini2019']

    Y_true = df_crispr_sub['Regulated'].values.astype(np.int64)
    plt.figure(figsize=(20,20))
    plt.grid()
    for model in model_list:
        if model == 'ABC':
            Y_pred = df_crispr_sub['ABCScore'].values
        else:
            Y_pred = df_crispr_sub[model+'.Score'].values

        precision, recall, thresholds = precision_recall_curve(Y_true, Y_pred)
        average_precision = average_precision_score(Y_true, Y_pred)
        aupr = auc(recall, precision)

        idx_recall_70_pct = np.argsort(np.abs(recall - 0.7))[0]
        recall_at_70_pct = recall[idx_recall_70_pct]
        precision_at_70_pct_recall = precision[idx_recall_70_pct]
        threshod_in_70_pct_recall = thresholds[idx_recall_70_pct]

        plt.plot(recall, precision, linewidth=5, label='{} || auPR={:6.4f} || Precision={:4.2f} || Threshold={:4.2f}'.format(model, aupr, precision_at_70_pct_recall, threshod_in_70_pct_recall))
        if i==0:
            plt.title('Ensemble | All | #EG = {} | #Positives = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==1:
            plt.title('Ensemble | [0,10kb) | #EG = {} | #Positives = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==2:
            plt.title('Ensemble | [10kb,100kb) | #EG = {} | #Positives = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==3:
            plt.title('Ensemble | [100kb,2.5Mb) | #EG = {} | #Positives = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==4:
            plt.title('Fulco | All | #EG = {} | #Positives = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==5:
            plt.title('Gasperini | All | #EG = {} | #Positives = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        
        plt.legend(bbox_to_anchor=(0, -.1), loc='upper left', borderaxespad=0, fontsize=40)
        plt.xlabel("Recall", fontsize=40)
        plt.ylabel("Precision", fontsize=40)
        plt.tick_params(axis='x', labelsize=40)
        plt.tick_params(axis='y', labelsize=40)
        plt.grid(False)
        if save_fig:
            if i==0:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_all.pdf', bbox_inches='tight')
            elif i==1:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_0-10k.pdf', bbox_inches='tight')
            elif i==2:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_10k-100k.pdf', bbox_inches='tight')
            elif i==3:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_100k-2500k.pdf', bbox_inches='tight')
            elif i==4:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_FlowFISH_all.pdf', bbox_inches='tight')
            elif i==5:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Gasperini_all.pdf', bbox_inches='tight')


############################################### Evelyn suggestions 1 ###############################################

#%%
##### Baseline model
model_list = ['ABC']
model = 'Baseline'
model_list.append(model)
features_list = ['distance', '3DContact',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 'normalizedH3K27ac_prom', 'normalizedDNase_prom']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### GraphReg/LR model
model = 'GraphReg'
model_list.append(model)
features_list = ['distance',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
features_list = ['ABCScore', 'averageCorrWeighted']
model = 'ABCScore+averageCorrWeighted'
model_list.append(model)

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'GraphReg+averageCorrWeighted'
model_list.append(model)
features_list = ['distance',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8', 'averageCorrWeighted']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
features_list = ['ABCScore', 'distance', '3DContact',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 
                'normalizedH3K27ac_prom', 'normalizedDNase_prom']

model = 'ABCScore+Baseline'
model_list.append(model)

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]
        
#%%
##### 
features_list = ['ABCScore', 'distance', '3DContact',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 
                'normalizedH3K27ac_prom', 'normalizedDNase_prom', 'averageCorrWeighted']

model = 'ABCScore+Baseline+averageCorrWeighted'
model_list.append(model)

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
features_list = ['ABCScore', 'distance',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8', 'averageCorrWeighted']

model = 'ABCScore+GraphReg+averageCorrWeighted'
model_list.append(model)

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
name_analysis = 'Evelyn1'
save_fig = True
for i in range(6):
    if i==0:
        df_crispr_sub = df_crispr
    elif i==1:
        df_crispr_sub = df_crispr[df_crispr['distance'] < 20000]
    elif i==2:
        df_crispr_sub = df_crispr[(df_crispr['distance'] >= 20000) & (df_crispr['distance'] < 200000)]
    elif i==3:
        df_crispr_sub = df_crispr[df_crispr['distance'] >= 200000]
    elif i==4:
        df_crispr_sub = df_crispr[df_crispr['dataset'] == 'FlowFISH_K562']
    elif i==5:
        df_crispr_sub = df_crispr[df_crispr['dataset'] == 'Gasperini2019']

    Y_true = df_crispr_sub['Regulated'].values.astype(np.int64)
    plt.figure(figsize=(20,20))
    for model in model_list:
        if model == 'ABC':
            Y_pred = df_crispr_sub['ABCScore'].values
        else:
            Y_pred = df_crispr_sub[model+'.Score'].values

        precision, recall, thresholds = precision_recall_curve(Y_true, Y_pred)
        average_precision = average_precision_score(Y_true, Y_pred)
        aupr = auc(recall, precision)

        idx_recall_70_pct = np.argsort(np.abs(recall - 0.7))[0]
        recall_at_70_pct = recall[idx_recall_70_pct]
        precision_at_70_pct_recall = precision[idx_recall_70_pct]
        threshod_in_70_pct_recall = thresholds[idx_recall_70_pct]

        plt.plot(recall, precision, linewidth=3, label='model={} || auPR={:6.4f} || Precision={:4.2f} || Threshold={:4.2f}'.format(model, aupr, precision_at_70_pct_recall, threshod_in_70_pct_recall))
        if i==0:
            plt.title('Ensemble | [0,inf) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==1:
            plt.title('Ensemble | [0,20k) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==2:
            plt.title('Ensemble | [20k,200k) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==3:
            plt.title('Ensemble | [200k,inf) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==4:
            plt.title('FlowFISH | [0,inf) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==5:
            plt.title('Gasperini | [0,inf) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        
        plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=40)
        plt.xlabel("Recall", fontsize=40)
        plt.ylabel("Precision", fontsize=40)
        plt.tick_params(axis='x', labelsize=40)
        plt.tick_params(axis='y', labelsize=40)
        plt.grid()
        if save_fig:
            if i==0:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_all.pdf', bbox_inches='tight')
            elif i==1:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_0-20k.pdf', bbox_inches='tight')
            elif i==2:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_20k-200k.pdf', bbox_inches='tight')
            elif i==3:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_200k-end.pdf', bbox_inches='tight')
            elif i==4:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_FlowFISH_all.pdf', bbox_inches='tight')
            elif i==5:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Gasperini_all.pdf', bbox_inches='tight')
            

############################################### Evelyn suggestions 2 ###############################################

#%%
##### Baseline model
model_list = ['ABC']
model = 'Baseline'
model_list.append(model)
features_list = ['distance', '3DContact',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 'normalizedH3K27ac_prom', 'normalizedDNase_prom']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### GraphReg/LR model
model = 'GraphReg'
model_list.append(model)
features_list = ['distance',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
features_list = ['ABCScore', 'P2PromoterClass']
model = 'ABCScore+P2PromoterClass'
model_list.append(model)

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
features_list = ['ABCScore', 'distance', '3DContact',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 
                'normalizedH3K27ac_prom', 'normalizedDNase_prom', 'P2PromoterClass']

model = 'ABCScore+Baseline+P2PromoterClass'
model_list.append(model)

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
features_list = ['ABCScore', 'distance',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8', 'P2PromoterClass']

model = 'ABCScore+GraphReg+P2PromoterClass'
model_list.append(model)

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
name_analysis = 'Evelyn2'
save_fig = True
for i in range(6):
    if i==0:
        df_crispr_sub = df_crispr
    elif i==1:
        df_crispr_sub = df_crispr[df_crispr['distance'] < 20000]
    elif i==2:
        df_crispr_sub = df_crispr[(df_crispr['distance'] >= 20000) & (df_crispr['distance'] < 200000)]
    elif i==3:
        df_crispr_sub = df_crispr[df_crispr['distance'] >= 200000]
    elif i==4:
        df_crispr_sub = df_crispr[df_crispr['dataset'] == 'FlowFISH_K562']
    elif i==5:
        df_crispr_sub = df_crispr[df_crispr['dataset'] == 'Gasperini2019']

    Y_true = df_crispr_sub['Regulated'].values.astype(np.int64)
    plt.figure(figsize=(20,20))
    for model in model_list:
        if model == 'ABC':
            Y_pred = df_crispr_sub['ABCScore'].values
        else:
            Y_pred = df_crispr_sub[model+'.Score'].values

        precision, recall, thresholds = precision_recall_curve(Y_true, Y_pred)
        average_precision = average_precision_score(Y_true, Y_pred)
        aupr = auc(recall, precision)

        idx_recall_70_pct = np.argsort(np.abs(recall - 0.7))[0]
        recall_at_70_pct = recall[idx_recall_70_pct]
        precision_at_70_pct_recall = precision[idx_recall_70_pct]
        threshod_in_70_pct_recall = thresholds[idx_recall_70_pct]

        plt.plot(recall, precision, linewidth=3, label='model={} || auPR={:6.4f} || Precision={:4.2f} || Threshold={:4.2f}'.format(model, aupr, precision_at_70_pct_recall, threshod_in_70_pct_recall))
        if i==0:
            plt.title('Ensemble | [0,inf) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==1:
            plt.title('Ensemble | [0,20k) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==2:
            plt.title('Ensemble | [20k,200k) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==3:
            plt.title('Ensemble | [200k,inf) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==4:
            plt.title('FlowFISH | [0,inf) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==5:
            plt.title('Gasperini | [0,inf) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        
        plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=40)
        plt.xlabel("Recall", fontsize=40)
        plt.ylabel("Precision", fontsize=40)
        plt.tick_params(axis='x', labelsize=40)
        plt.tick_params(axis='y', labelsize=40)
        plt.grid()
        if save_fig:
            if i==0:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_all.pdf', bbox_inches='tight')
            elif i==1:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_0-20k.pdf', bbox_inches='tight')
            elif i==2:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_20k-200k.pdf', bbox_inches='tight')
            elif i==3:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_200k-end.pdf', bbox_inches='tight')
            elif i==4:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_FlowFISH_all.pdf', bbox_inches='tight')
            elif i==5:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Gasperini_all.pdf', bbox_inches='tight')
            

############################################### Evelyn suggestions 3 ###############################################

#%%
##### Baseline model
model_list = ['ABC']
model = 'Baseline'
model_list.append(model)
features_list = ['distance', '3DContact',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 'normalizedH3K27ac_prom', 'normalizedDNase_prom']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### GraphReg/LR model
model = 'GraphReg'
model_list.append(model)
features_list = ['distance',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    #print(f'Num test {chr} is {len(idx_test)}')
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
features_list = ['ABCScore', 'ubiquitousExpressedGene']
model = 'ABCScore+ubiquitousExpressedGene'
model_list.append(model)

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
features_list = ['ABCScore', 'distance', '3DContact',
                'normalizedH3K27ac_enh', 'normalizedDNase_enh', 
                'normalizedH3K27ac_prom', 'normalizedDNase_prom', 'ubiquitousExpressedGene']

model = 'ABCScore+Baseline+ubiquitousExpressedGene'
model_list.append(model)

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
##### 
features_list = ['ABCScore', 'distance',
                'H3K4me3_e_max_L_8', 'H3K27ac_e_max_L_8', 'DNase_e_max_L_8',
                'H3K4me3_e_grad_max_L_8', 'H3K27ac_e_grad_max_L_8',
                'DNase_e_grad_max_L_8', 'H3K4me3_e_grad_min_L_8',
                'H3K27ac_e_grad_min_L_8', 'DNase_e_grad_min_L_8', 'H3K4me3_p_max_L_8',
                'H3K27ac_p_max_L_8', 'DNase_p_max_L_8', 'H3K4me3_p_grad_max_L_8',
                'H3K27ac_p_grad_max_L_8', 'DNase_p_grad_max_L_8',
                'H3K4me3_p_grad_min_L_8', 'H3K27ac_p_grad_min_L_8',
                'DNase_p_grad_min_L_8', 'ubiquitousExpressedGene']

model = 'ABCScore+GraphReg+ubiquitousExpressedGene'
model_list.append(model)

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
for chr in chr_list:
    idx_test = df_crispr[df_crispr['chrom']==chr].index.values
    if len(idx_test) > 0:
        idx_train = np.delete(idx, idx_test)

        X_test = X.loc[idx_test, :]
        Y_test = Y[idx_test]
        X_train = X.loc[idx_train, :]
        Y_train = Y[idx_train]
        
        clf = LogisticRegression(random_state=0, class_weight=None, solver='lbfgs', max_iter=100000).fit(X_train, Y_train)
        if save_model:
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
name_analysis = 'Evelyn3'
save_fig = True
for i in range(6):
    if i==0:
        df_crispr_sub = df_crispr
    elif i==1:
        df_crispr_sub = df_crispr[df_crispr['distance'] < 20000]
    elif i==2:
        df_crispr_sub = df_crispr[(df_crispr['distance'] >= 20000) & (df_crispr['distance'] < 200000)]
    elif i==3:
        df_crispr_sub = df_crispr[df_crispr['distance'] >= 200000]
    elif i==4:
        df_crispr_sub = df_crispr[df_crispr['dataset'] == 'FlowFISH_K562']
    elif i==5:
        df_crispr_sub = df_crispr[df_crispr['dataset'] == 'Gasperini2019']

    Y_true = df_crispr_sub['Regulated'].values.astype(np.int64)
    plt.figure(figsize=(20,20))
    for model in model_list:
        if model == 'ABC':
            Y_pred = df_crispr_sub['ABCScore'].values
        else:
            Y_pred = df_crispr_sub[model+'.Score'].values

        precision, recall, thresholds = precision_recall_curve(Y_true, Y_pred)
        average_precision = average_precision_score(Y_true, Y_pred)
        aupr = auc(recall, precision)

        idx_recall_70_pct = np.argsort(np.abs(recall - 0.7))[0]
        recall_at_70_pct = recall[idx_recall_70_pct]
        precision_at_70_pct_recall = precision[idx_recall_70_pct]
        threshod_in_70_pct_recall = thresholds[idx_recall_70_pct]

        plt.plot(recall, precision, linewidth=3, label='model={} || auPR={:6.4f} || Precision={:4.2f} || Threshold={:4.2f}'.format(model, aupr, precision_at_70_pct_recall, threshod_in_70_pct_recall))
        if i==0:
            plt.title('Ensemble | [0,inf) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==1:
            plt.title('Ensemble | [0,20k) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==2:
            plt.title('Ensemble | [20k,200k) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==3:
            plt.title('Ensemble | [200k,inf) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==4:
            plt.title('FlowFISH | [0,inf) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        elif i==5:
            plt.title('Gasperini | [0,inf) | #EG = {} | #Regulated = {}'.format(len(df_crispr_sub), np.sum(df_crispr_sub['Regulated']==True)), fontsize=40)
        
        plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=40)
        plt.xlabel("Recall", fontsize=40)
        plt.ylabel("Precision", fontsize=40)
        plt.tick_params(axis='x', labelsize=40)
        plt.tick_params(axis='y', labelsize=40)
        plt.grid()
        if save_fig:
            if i==0:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_all.pdf', bbox_inches='tight')
            elif i==1:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_0-20k.pdf', bbox_inches='tight')
            elif i==2:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_20k-200k.pdf', bbox_inches='tight')
            elif i==3:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Ensemble_200k-end.pdf', bbox_inches='tight')
            elif i==4:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_FlowFISH_all.pdf', bbox_inches='tight')
            elif i==5:
                plt.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/PR_curve_{name_analysis}_Gasperini_all.pdf', bbox_inches='tight')
            


#%%
if save_to_csv:
    df_crispr.to_csv(data_path+'/results/csv/distal_reg_paper/EG_features/K562/AllFeatures/EPCrisprBenchmark_ensemble_data_GRCh38.K562_AllFeatures_NAfilled_withPreds.tsv', sep = '\t', index=False)

