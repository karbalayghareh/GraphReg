##### cross-validation by holding out chromosomes

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
import matplotlib
import pickle
from sklearn.linear_model import LogisticRegression
#import shap

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

########################################################################################## DNaseOnly ##########################################################################################

#%%
RefSeqGenes = pd.read_csv(data_path+'/results/csv/distal_reg_paper/RefSeqGenes/RefSeqCurated.170308.bed.CollapsedGeneBounds.hg38.TSS500bp.bed', names = ['chr', 'start', 'end', 'gene', 'len', 'strand'], delimiter = '\t')

#df_crispr = pd.read_csv(data_path+'/results/csv/distal_reg_paper/EG_features/K562/DNaseOnly/EPCrisprBenchmark_ensemble_data_GRCh38.K562_AllFeatures_NAfilled.DNaseOnly.tsv', delimiter = '\t')
df_crispr = pd.read_csv(data_path+'/results/csv/distal_reg_paper/EG_features/K562/DNaseOnly/EPCrisprBenchmark_ensemble_data_GRCh38.K562_ENCDO000AAD_ENCFF325RTP_DNaseOnly_features_NAfilled.tsv', delimiter = '\t')
df_crispr = df_crispr[df_crispr['measuredGeneSymbol'].isin(RefSeqGenes['gene'])].reset_index(drop=True)
df_crispr = df_crispr[~df_crispr['Regulated'].isna()].reset_index(drop=True)
df_crispr = df_crispr.replace([np.inf, -np.inf], np.nan)
df_crispr = df_crispr.fillna(0)

RefSeqGenes_sub = RefSeqGenes[RefSeqGenes['gene'].isin(df_crispr['measuredGeneSymbol'])].reset_index(drop=True)
df_crispr['TSS_from_universe'] = -1
for i, g in enumerate(RefSeqGenes_sub['gene'].values):
    idx = df_crispr[df_crispr['measuredGeneSymbol'] == g].index
    df_crispr.loc[idx, 'TSS_from_universe'] = (RefSeqGenes_sub.loc[i, 'start'] + RefSeqGenes_sub.loc[i, 'end'])//2

df_crispr['distance'] = np.abs((df_crispr['chromStart'] + df_crispr['chromEnd'])//2 - df_crispr['TSS_from_universe'])

model = 'ENCODE-E2G'
features_list = ['numTSSEnhGene',
                'distance', 'normalizedDNase_enh', 'normalizedDNase_prom',
                'numNearbyEnhancers', 'sumNearbyEnhancers', 'ubiquitousExpressedGene',
                'numCandidateEnhGene', '3DContactAvgHicTrack2',
                '3DContactAvgHicTrack2_squared',
                'activityEnhDNaseOnlyAvgHicTrack2_squared',
                'activityPromDNaseOnlyAvgHicTrack2', 'ABCScoreDNaseOnlyAvgHicTrack2']
print(len(features_list))
X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
shap_values_all = np.empty([0,X.shape[1]])
X_test_all = pd.DataFrame(columns=features_list)
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
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/DNaseOnly/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

        if compute_shap:
            background = shap.kmeans(X_train, 10)
            explainer = shap.KernelExplainer(clf.predict_proba, background)
            shap_values = explainer.shap_values(X_test)
            shap_values_all = np.append(shap_values_all, shap_values[1], axis=0)
            X_test_all = X_test_all.append(X_test).reset_index(drop=True)

if compute_shap:
    fig=plt.gcf()
    shap.summary_plot(shap_values_all, X_test_all, plot_type='bar', max_display=50)
    fig.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/SHAP_scores_barplot_ENCODE_E2G_Ensemble.pdf', bbox_inches='tight')

    fig=plt.gcf()
    shap.summary_plot(shap_values_all, X_test_all, plot_type='dot', max_display=50)
    fig.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/SHAP_scores_dotplot_ENCODE_E2G_Ensemble.pdf', bbox_inches='tight')

#%%
model = 'ENCODE-E2G without distance (2)'
features_list = ['normalizedDNase_enh', 'normalizedDNase_prom',
                'numNearbyEnhancers', 'sumNearbyEnhancers', 'ubiquitousExpressedGene',
                'numCandidateEnhGene', '3DContactAvgHicTrack2',
                '3DContactAvgHicTrack2_squared',
                'activityEnhDNaseOnlyAvgHicTrack2_squared',
                'activityPromDNaseOnlyAvgHicTrack2', 'ABCScoreDNaseOnlyAvgHicTrack2']
print(len(features_list))
X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
shap_values_all = np.empty([0,X.shape[1]])
X_test_all = pd.DataFrame(columns=features_list)
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
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/DNaseOnly/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

        if compute_shap:
            background = shap.kmeans(X_train, 10)
            explainer = shap.KernelExplainer(clf.predict_proba, background)
            shap_values = explainer.shap_values(X_test)
            shap_values_all = np.append(shap_values_all, shap_values[1], axis=0)
            X_test_all = X_test_all.append(X_test).reset_index(drop=True)

if compute_shap:
    fig=plt.gcf()
    shap.summary_plot(shap_values_all, X_test_all, plot_type='bar', max_display=50)
    fig.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/SHAP_scores_barplot_ENCODE_E2G_Ensemble.pdf', bbox_inches='tight')

    fig=plt.gcf()
    shap.summary_plot(shap_values_all, X_test_all, plot_type='dot', max_display=50)
    fig.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/SHAP_scores_dotplot_ENCODE_E2G_Ensemble.pdf', bbox_inches='tight')

#%%
model = 'ENCODE-E2G without contact (3)'
features_list = ['numTSSEnhGene',
                'distance', 'normalizedDNase_enh', 'normalizedDNase_prom',
                'numNearbyEnhancers', 'sumNearbyEnhancers', 'ubiquitousExpressedGene',
                'numCandidateEnhGene', 'activityEnhDNaseOnlyAvgHicTrack2_squared',
                'activityPromDNaseOnlyAvgHicTrack2']
print(len(features_list))
X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
shap_values_all = np.empty([0,X.shape[1]])
X_test_all = pd.DataFrame(columns=features_list)
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
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/DNaseOnly/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

        if compute_shap:
            background = shap.kmeans(X_train, 10)
            explainer = shap.KernelExplainer(clf.predict_proba, background)
            shap_values = explainer.shap_values(X_test)
            shap_values_all = np.append(shap_values_all, shap_values[1], axis=0)
            X_test_all = X_test_all.append(X_test).reset_index(drop=True)

if compute_shap:
    fig=plt.gcf()
    shap.summary_plot(shap_values_all, X_test_all, plot_type='bar', max_display=50)
    fig.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/SHAP_scores_barplot_ENCODE_E2G_Ensemble.pdf', bbox_inches='tight')

    fig=plt.gcf()
    shap.summary_plot(shap_values_all, X_test_all, plot_type='dot', max_display=50)
    fig.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/SHAP_scores_dotplot_ENCODE_E2G_Ensemble.pdf', bbox_inches='tight')

#%%
model = 'ENCODE-E2G without contact and distance (5)'
features_list = ['normalizedDNase_enh', 'normalizedDNase_prom',
                'numNearbyEnhancers', 'sumNearbyEnhancers', 'ubiquitousExpressedGene',
                'numCandidateEnhGene', 'activityEnhDNaseOnlyAvgHicTrack2_squared',
                'activityPromDNaseOnlyAvgHicTrack2']
print(len(features_list))
X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
shap_values_all = np.empty([0,X.shape[1]])
X_test_all = pd.DataFrame(columns=features_list)
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
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/DNaseOnly/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

        if compute_shap:
            background = shap.kmeans(X_train, 10)
            explainer = shap.KernelExplainer(clf.predict_proba, background)
            shap_values = explainer.shap_values(X_test)
            shap_values_all = np.append(shap_values_all, shap_values[1], axis=0)
            X_test_all = X_test_all.append(X_test).reset_index(drop=True)

if compute_shap:
    fig=plt.gcf()
    shap.summary_plot(shap_values_all, X_test_all, plot_type='bar', max_display=50)
    fig.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/SHAP_scores_barplot_ENCODE_E2G_Ensemble.pdf', bbox_inches='tight')

    fig=plt.gcf()
    shap.summary_plot(shap_values_all, X_test_all, plot_type='dot', max_display=50)
    fig.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/SHAP_scores_dotplot_ENCODE_E2G_Ensemble.pdf', bbox_inches='tight')

#%%
model = 'ENCODE-E2G without ABC (1)'
features_list = ['numTSSEnhGene',
                'distance', 'normalizedDNase_enh', 'normalizedDNase_prom',
                'numNearbyEnhancers', 'sumNearbyEnhancers', 'ubiquitousExpressedGene',
                'numCandidateEnhGene', '3DContactAvgHicTrack2',
                '3DContactAvgHicTrack2_squared',
                'activityEnhDNaseOnlyAvgHicTrack2_squared',
                'activityPromDNaseOnlyAvgHicTrack2']
print(len(features_list))
X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
shap_values_all = np.empty([0,X.shape[1]])
X_test_all = pd.DataFrame(columns=features_list)
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
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/DNaseOnly/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

        if compute_shap:
            background = shap.kmeans(X_train, 10)
            explainer = shap.KernelExplainer(clf.predict_proba, background)
            shap_values = explainer.shap_values(X_test)
            shap_values_all = np.append(shap_values_all, shap_values[1], axis=0)
            X_test_all = X_test_all.append(X_test).reset_index(drop=True)

if compute_shap:
    fig=plt.gcf()
    shap.summary_plot(shap_values_all, X_test_all, plot_type='bar', max_display=50)
    fig.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/SHAP_scores_barplot_ENCODE_E2G_Ensemble.pdf', bbox_inches='tight')

    fig=plt.gcf()
    shap.summary_plot(shap_values_all, X_test_all, plot_type='dot', max_display=50)
    fig.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/SHAP_scores_dotplot_ENCODE_E2G_Ensemble.pdf', bbox_inches='tight')

#%%
model = 'ENCODE-E2G without number of nearby enhancers (3)'
features_list = ['numTSSEnhGene',
                'distance', 'normalizedDNase_enh', 'normalizedDNase_prom',
                'ubiquitousExpressedGene', '3DContactAvgHicTrack2',
                '3DContactAvgHicTrack2_squared',
                'activityEnhDNaseOnlyAvgHicTrack2_squared',
                'activityPromDNaseOnlyAvgHicTrack2', 'ABCScoreDNaseOnlyAvgHicTrack2']
print(len(features_list))
X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
shap_values_all = np.empty([0,X.shape[1]])
X_test_all = pd.DataFrame(columns=features_list)
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
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/DNaseOnly/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

        if compute_shap:
            background = shap.kmeans(X_train, 10)
            explainer = shap.KernelExplainer(clf.predict_proba, background)
            shap_values = explainer.shap_values(X_test)
            shap_values_all = np.append(shap_values_all, shap_values[1], axis=0)
            X_test_all = X_test_all.append(X_test).reset_index(drop=True)

if compute_shap:
    fig=plt.gcf()
    shap.summary_plot(shap_values_all, X_test_all, plot_type='bar', max_display=50)
    fig.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/SHAP_scores_barplot_ENCODE_E2G_Ensemble.pdf', bbox_inches='tight')

    fig=plt.gcf()
    shap.summary_plot(shap_values_all, X_test_all, plot_type='dot', max_display=50)
    fig.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/SHAP_scores_dotplot_ENCODE_E2G_Ensemble.pdf', bbox_inches='tight')

#%%
model = 'ENCODE-E2G without enhancer activity (3)'
features_list = ['numTSSEnhGene',
                'distance', 'normalizedDNase_prom',
                'numNearbyEnhancers', 'sumNearbyEnhancers', 'ubiquitousExpressedGene',
                'numCandidateEnhGene', '3DContactAvgHicTrack2',
                '3DContactAvgHicTrack2_squared', 'activityPromDNaseOnlyAvgHicTrack2']
print(len(features_list))
X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
shap_values_all = np.empty([0,X.shape[1]])
X_test_all = pd.DataFrame(columns=features_list)
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
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/DNaseOnly/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

        if compute_shap:
            background = shap.kmeans(X_train, 10)
            explainer = shap.KernelExplainer(clf.predict_proba, background)
            shap_values = explainer.shap_values(X_test)
            shap_values_all = np.append(shap_values_all, shap_values[1], axis=0)
            X_test_all = X_test_all.append(X_test).reset_index(drop=True)

if compute_shap:
    fig=plt.gcf()
    shap.summary_plot(shap_values_all, X_test_all, plot_type='bar', max_display=50)
    fig.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/SHAP_scores_barplot_ENCODE_E2G_Ensemble.pdf', bbox_inches='tight')

    fig=plt.gcf()
    shap.summary_plot(shap_values_all, X_test_all, plot_type='dot', max_display=50)
    fig.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/SHAP_scores_dotplot_ENCODE_E2G_Ensemble.pdf', bbox_inches='tight')


#%%
model = 'ENCODE-E2G without promoter activity (2)'
features_list = ['numTSSEnhGene',
                'distance', 'normalizedDNase_enh',
                'numNearbyEnhancers', 'sumNearbyEnhancers', 'ubiquitousExpressedGene',
                'numCandidateEnhGene', '3DContactAvgHicTrack2', '3DContactAvgHicTrack2_squared',
                'activityEnhDNaseOnlyAvgHicTrack2_squared', 'ABCScoreDNaseOnlyAvgHicTrack2']
print(len(features_list))
X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
shap_values_all = np.empty([0,X.shape[1]])
X_test_all = pd.DataFrame(columns=features_list)
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
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/DNaseOnly/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

        if compute_shap:
            background = shap.kmeans(X_train, 10)
            explainer = shap.KernelExplainer(clf.predict_proba, background)
            shap_values = explainer.shap_values(X_test)
            shap_values_all = np.append(shap_values_all, shap_values[1], axis=0)
            X_test_all = X_test_all.append(X_test).reset_index(drop=True)

if compute_shap:
    fig=plt.gcf()
    shap.summary_plot(shap_values_all, X_test_all, plot_type='bar', max_display=50)
    fig.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/SHAP_scores_barplot_ENCODE_E2G_Ensemble.pdf', bbox_inches='tight')

    fig=plt.gcf()
    shap.summary_plot(shap_values_all, X_test_all, plot_type='dot', max_display=50)
    fig.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/SHAP_scores_dotplot_ENCODE_E2G_Ensemble.pdf', bbox_inches='tight')

#%%
model = 'ENCODE-E2G without promoter class (1)'
features_list = ['numTSSEnhGene',
                'distance', 'normalizedDNase_enh', 'normalizedDNase_prom',
                'numNearbyEnhancers', 'sumNearbyEnhancers',
                'numCandidateEnhGene', '3DContactAvgHicTrack2',
                '3DContactAvgHicTrack2_squared',
                'activityEnhDNaseOnlyAvgHicTrack2_squared',
                'activityPromDNaseOnlyAvgHicTrack2', 'ABCScoreDNaseOnlyAvgHicTrack2']
print(len(features_list))
X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
shap_values_all = np.empty([0,X.shape[1]])
X_test_all = pd.DataFrame(columns=features_list)
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
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/DNaseOnly/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

        if compute_shap:
            background = shap.kmeans(X_train, 10)
            explainer = shap.KernelExplainer(clf.predict_proba, background)
            shap_values = explainer.shap_values(X_test)
            shap_values_all = np.append(shap_values_all, shap_values[1], axis=0)
            X_test_all = X_test_all.append(X_test).reset_index(drop=True)

if compute_shap:
    fig=plt.gcf()
    shap.summary_plot(shap_values_all, X_test_all, plot_type='bar', max_display=50)
    fig.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/SHAP_scores_barplot_ENCODE_E2G_Ensemble.pdf', bbox_inches='tight')

    fig=plt.gcf()
    shap.summary_plot(shap_values_all, X_test_all, plot_type='dot', max_display=50)
    fig.savefig(data_path+f'/results/csv/distal_reg_paper/figs/final/SHAP_scores_dotplot_ENCODE_E2G_Ensemble.pdf', bbox_inches='tight')

#%%
name_analysis = 'DNaseOnly'
model_list = ['ABCScoreDNaseOnlyAvgHicTrack2', 'Full']
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
        if model == 'ABCScoreDNaseOnlyAvgHicTrack1':
            Y_pred = df_crispr_sub['ABCScoreDNaseOnlyAvgHicTrack1'].values
        elif model == 'ABCScoreDNaseOnlyAvgHicTrack2':
            Y_pred = df_crispr_sub['ABCScoreDNaseOnlyAvgHicTrack2'].values
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
        
if save_to_csv:
    #df_crispr.to_csv(data_path+'/results/csv/distal_reg_paper/EG_features/K562/DNaseOnly/EPCrisprBenchmark_ensemble_data_GRCh38.K562_AllFeatures_NAfilled.DNaseOnly_withPreds.tsv', sep = '\t', index=False)
    df_crispr.to_csv(data_path+'/results/csv/distal_reg_paper/EG_features/K562/DNaseOnly/EPCrisprBenchmark_ensemble_data_GRCh38.K562_ENCDO000AAD_ENCFF325RTP_DNaseOnly_features_NAfilled_withPreds.tsv', sep = '\t', index=False)

#%%
##### auPR curves

sns.set_style("ticks")

name_analysis = 'ENCODE_E2G_models_without_category_features'
model_list = ['ENCODE-E2G',
            'ENCODE-E2G without distance (2)',
            'ENCODE-E2G without contact (3)',
            'ENCODE-E2G without contact and distance (5)',
            'ENCODE-E2G without ABC (1)',
            'ENCODE-E2G without number of nearby enhancers (3)',
            'ENCODE-E2G without enhancer activity (3)',
            'ENCODE-E2G without promoter activity (2)',
            'ENCODE-E2G without promoter class (1)'
            ]

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

        if model=='ENCODE-E2G':
            color_code = 'black'
        elif model=='ENCODE-E2G without distance (2)':
            color_code =  "#9467bd"
        elif model=='ENCODE-E2G without contact (3)':
            color_code = "#2ca02c"
        elif model=='ENCODE-E2G without contact and distance (5)':
            color_code = "#1f77b4"
        elif model=='ENCODE-E2G without ABC (1)':
            color_code = "#d62728"
        elif model=='ENCODE-E2G without number of nearby enhancers (3)':
            color_code = "#8c564b"
        elif model=='ENCODE-E2G without enhancer activity (3)':
            color_code = "#ff7f0e"
        elif model=='ENCODE-E2G without promoter activity (2)':
            color_code = "#7f7f7f"
        elif model=='ENCODE-E2G without promoter class (1)':
            color_code = "#e377c2"

        precision, recall, thresholds = precision_recall_curve(Y_true, Y_pred)
        average_precision = average_precision_score(Y_true, Y_pred)
        aupr = auc(recall, precision)

        idx_recall_70_pct = np.argsort(np.abs(recall - 0.7))[0]
        recall_at_70_pct = recall[idx_recall_70_pct]
        precision_at_70_pct_recall = precision[idx_recall_70_pct]
        threshod_in_70_pct_recall = thresholds[idx_recall_70_pct]

        #plt.plot(recall, precision, linewidth=5, color=color_code, label='{} || auPR={:6.4f} || Precision={:4.2f} || Threshold={:4.2f}'.format(model, aupr, precision_at_70_pct_recall, threshod_in_70_pct_recall))
        #plt.plot(recall, precision, linewidth=5, color=color_code, label='{}'.format(model))
        plt.plot(recall, precision, linewidth=5, color=color_code)
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
##### bootstrapping (remove different categories of features in E2G model)

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

name_analysis = 'ENCODE-E2G_Models_without_category_features'
model_list = ['ENCODE-E2G',
            'ENCODE-E2G without distance (2)',
            'ENCODE-E2G without contact (3)',
            'ENCODE-E2G without contact and distance (5)',
            'ENCODE-E2G without ABC (1)',
            'ENCODE-E2G without number of nearby enhancers (3)',
            'ENCODE-E2G without enhancer activity (3)',
            'ENCODE-E2G without promoter activity (2)',
            'ENCODE-E2G without promoter class (1)'
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

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(25, 6))
ax1.grid('y')
g = sns.barplot(data=df, x='Distance Range', y='Delta auPR', hue='Model', ax=ax1, errorbar=("pi", 95), seed=None, hue_order=order, errwidth=1.5, capsize=0.03, palette='tab10')
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
g = sns.barplot(data=df, x='Distance Range', y='Delta Precision', hue='Model', ax=ax1, errorbar=("pi", 95), seed=None, hue_order=order, errwidth=1.5, capsize=0.03, palette='tab10')
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


########################################################################################## DNaseH3K27acOnly ##########################################################################################


# %%
df_crispr = pd.read_csv(data_path+'/results/csv/distal_reg_paper/EG_features/K562/DNaseH3K27acOnly/EPCrisprBenchmark_ensemble_data_GRCh38.K562_AllFeatures_NAfilled.DNaseH3K27acOnly.tsv', delimiter = '\t')
df_crispr = df_crispr[~df_crispr['Regulated'].isna()].reset_index(drop=True)
df_crispr = df_crispr.replace([np.inf, -np.inf], np.nan)
df_crispr = df_crispr.fillna(0)


model = 'Full'
features_list = ['pearsonCorrelation',
                'spearmanCorrelation', 'glsCoefficient', 'numTSSEnhGene',
                'distance', 'normalizedDNase_enh', 'normalizedDNase_prom',
                'normalizedH3K27ac_enh', 'normalizedH3K27ac_prom', 'activity_enh',
                'activity_enh_squared', 'activity_prom', 'ABCNumerator', 'ABCScore',
                'ABCDenominator', 'numNearbyEnhancers', 'numNearbyEnhancers_10kb',
                'sumNearbyEnhancers', 'sumNearbyEnhancers_10kb', 'promCTCF', 'enhCTCF',
                'averageCorrWeighted', 'RamilWeighted', 'phastConMax', 'phyloPMax',
                'P2PromoterClass', 'ubiquitousExpressedGene']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
shap_values_all = np.empty([0,X.shape[1]])
X_test_all = pd.DataFrame(columns=features_list)
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
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/DNaseH3K27acOnly/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

        if compute_shap:
            background = shap.kmeans(X_train, 10)
            explainer = shap.KernelExplainer(clf.predict_proba, background)
            shap_values = explainer.shap_values(X_test)
            shap_values_all = np.append(shap_values_all, shap_values[1], axis=0)
            X_test_all = X_test_all.append(X_test).reset_index(drop=True)

if compute_shap:
    shap.summary_plot(shap_values_all, X_test_all, plot_type='bar', max_display=70)
    shap.summary_plot(shap_values_all, X_test_all, plot_type='dot', max_display=70)

# %%
name_analysis = 'DNaseH3K27acOnly'
model_list = ['ABC', 'Full']
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


if save_to_csv:
    df_crispr.to_csv(data_path+'/results/csv/distal_reg_paper/EG_features/K562/DNaseH3K27acOnly/EPCrisprBenchmark_ensemble_data_GRCh38.K562_AllFeatures_NAfilled.DNaseH3K27acOnly_withPreds.tsv', sep = '\t', index=False)

########################################################################################## EnhActivity ##########################################################################################


# %%
df_crispr = pd.read_csv(data_path+'/results/csv/distal_reg_paper/EG_features/K562/EnhActivity/EPCrisprBenchmark_ensemble_data_GRCh38.K562_AllFeatures_NAfilled.tsv', delimiter = '\t')
df_crispr = df_crispr[~df_crispr['Regulated'].isna()].reset_index(drop=True)
df_crispr = df_crispr.replace([np.inf, -np.inf], np.nan)
df_crispr = df_crispr.fillna(0)

#%%
model = 'Full'
features_list = ['normalizedH3K27ac_enhActivity', 'normalizedH3K4me1_enhActivity',
                'normalizedH3K4me3_enhActivity', 'normalizedH3K27me3_enhActivity',
                'normalizedH3K9me3_enhActivity', 'normalizedH3K36me3_enhActivity',
                'normalizedCTCF_enhActivity', 'normalizedEP300_enhActivity', 'ABCScore',
                'distance', '3DContact']

X = df_crispr.loc[:,features_list]
X = np.log(np.abs(X) + epsilon)
Y = df_crispr['Regulated'].values.astype(np.int64)

idx = np.arange(len(Y))
shap_values_all = np.empty([0,X.shape[1]])
X_test_all = pd.DataFrame(columns=features_list)
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
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/EnhActivity/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

        if compute_shap:
            background = shap.kmeans(X_train, 10)
            explainer = shap.KernelExplainer(clf.predict_proba, background)
            shap_values = explainer.shap_values(X_test)
            shap_values_all = np.append(shap_values_all, shap_values[1], axis=0)
            X_test_all = X_test_all.append(X_test).reset_index(drop=True)

if compute_shap:
    shap.summary_plot(shap_values_all, X_test_all, plot_type='bar', max_display=70)
    shap.summary_plot(shap_values_all, X_test_all, plot_type='dot', max_display=70)

#%%
model = 'normalizedH3K27ac_enhActivity+distance+3DContact'
features_list = ['normalizedH3K27ac_enhActivity', 'distance', '3DContact']

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
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/EnhActivity/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'normalizedH3K4me1_enhActivity+distance+3DContact'
features_list = ['normalizedH3K4me1_enhActivity', 'distance', '3DContact']

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
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/EnhActivity/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'normalizedH3K4me3_enhActivity+distance+3DContact'
features_list = ['normalizedH3K4me3_enhActivity', 'distance', '3DContact']

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
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/EnhActivity/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'normalizedH3K27me3_enhActivity+distance+3DContact'
features_list = ['normalizedH3K27me3_enhActivity', 'distance', '3DContact']

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
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/EnhActivity/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'normalizedH3K9me3_enhActivity+distance+3DContact'
features_list = ['normalizedH3K9me3_enhActivity', 'distance', '3DContact']

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
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/EnhActivity/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'normalizedH3K36me3_enhActivity+distance+3DContact'
features_list = ['normalizedH3K36me3_enhActivity', 'distance', '3DContact']

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
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/EnhActivity/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'normalizedCTCF_enhActivity+distance+3DContact'
features_list = ['normalizedCTCF_enhActivity', 'distance', '3DContact']

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
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/EnhActivity/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

#%%
model = 'normalizedEP300_enhActivity+distance+3DContact'
features_list = ['normalizedEP300_enhActivity', 'distance', '3DContact']

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
            with open(data_path+'/results/csv/distal_reg_paper/{}/models/final/EnhActivity/model_{}_test_{}.pkl'.format(dataset, model, chr),'wb') as f:
                pickle.dump(clf,f)

        probs = clf.predict_proba(X_test)
        df_crispr.loc[idx_test, model+'.Score'] = probs[:,1]

# %%
name_analysis = 'EnhActivity'
model_list = ['ABC',
                    'normalizedH3K27ac_enhActivity+distance+3DContact',
                    'normalizedH3K4me1_enhActivity+distance+3DContact',
                    'normalizedH3K4me3_enhActivity+distance+3DContact',
                    'normalizedH3K27me3_enhActivity+distance+3DContact',
                    'normalizedH3K9me3_enhActivity+distance+3DContact',
                    'normalizedH3K36me3_enhActivity+distance+3DContact',
                    'normalizedCTCF_enhActivity+distance+3DContact',
                    'normalizedEP300_enhActivity+distance+3DContact',
                    'Full']
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


if save_to_csv:
    df_crispr.to_csv(data_path+'/results/csv/distal_reg_paper/EG_features/K562/EnhActivity/EPCrisprBenchmark_ensemble_data_GRCh38.K562_AllFeatures_NAfilled_EnhActivity_withPreds.tsv', sep = '\t', index=False)


# %%
