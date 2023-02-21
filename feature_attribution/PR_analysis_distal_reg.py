import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt

data_path = '/media/labuser/STORAGE/GraphReg'   # data path
organism = 'human'
genome='hg38'                                   # hg19/hg38
cell_line = 'K562'                              # K562/GM12878/mESC/hESC
dataset = 'fulco'                    # 'fulco' or 'gasperini'

##### TSS dataframe
df_tss = pd.read_csv(data_path+'/data/tss/'+organism+'/distal_reg_paper/'+genome+'/RefSeqCurated.170308.bed.CollapsedGeneBounds.hg38.TSS500bp.bed', header=None, delimiter='\t')
df_tss.columns = ["chr", "tss_1", "tss_2", "gene", "na", "strand"]

##### CRISPR dataframe
if dataset == 'fulco':
    df_crispr = pd.read_csv(data_path+'/data/csv/CRISPR_benchmarking_data/EPCrisprBenchmark_Fulco2019_K562_GRCh38.tsv', delimiter='\t')
    df_crispr = df_crispr[df_crispr['ValidConnection']=="TRUE"].reset_index(drop=True)
    gene_list = np.unique(df_crispr['measuredGeneSymbol'].values)
    print('Number of E-G pais in Fulco dataset: {}'.format(len(df_crispr)))
    print('Number of genes in Fulco dataset: {}'.format(len(gene_list)))
elif dataset == 'gasperini':
    df_crispr = pd.read_csv(data_path+'/data/csv/CRISPR_benchmarking_data/EPCrisprBenchmark_Gasperini2019_0.13gStd_0.8pwrAt15effect_GRCh38.tsv', delimiter='\t')
    df_crispr = df_crispr[df_crispr['ValidConnection']=="TRUE"].reset_index(drop=True)
    gene_list = np.unique(df_crispr['measuredGeneSymbol'].values)
    print('Number of E-G pais in Gasperini dataset: {}'.format(len(df_crispr)))
    print('Number of genes in Gasperini dataset: {}'.format(len(gene_list)))

chr_list = []
tss_list = []
for g in gene_list:
    chr_list.append(df_crispr[df_crispr['measuredGeneSymbol']==g]['chrom'].values[0])
    tss_list.append(df_crispr[df_crispr['measuredGeneSymbol']==g]['startTSS'].values[0])

df = pd.DataFrame({'chr': chr_list, 'tss': tss_list, 'gene': gene_list})
if dataset == 'fulco':
    df.loc[3,'tss'] = df_tss[df_tss['gene']=='C19orf43']['tss_1'].values[0]   # Replace TSS of C19orf43
df = df.sort_values(by=['chr', 'tss']).reset_index(drop=True)

gene_list = df['gene'].values
chr_list = df['chr'].values
tss_list = df['tss'].values.astype(np.int64)

##### Enhancer predictions

df_crispr['DistanceToTSS'] = 0
cnt = 0
for fdr in ['1']:   # ['001', '01', '1', '5', '9']:
    for saliency_method in ['saliency']:
        for L in [2, 5, 10, 20, 40, 50]:
            cnt += 1
            print('FDR: {} | {} | L = {}'.format(fdr, saliency_method, L))

            df_preds = pd.read_csv(data_path+'/results/csv/distal_reg_paper/'+dataset+'/EG_preds_'+cell_line+'_'+genome+'_FDR_'+fdr+'_'+saliency_method+'_'+dataset+'.tsv', delimiter='\t')

            df_crispr['Score_fdr_{}_{}_L_{}'.format(fdr, saliency_method, L)] = 0.0
            for i in range(len(df_crispr)):
                df_crispr_row = df_crispr.iloc[i]
                enhancer_middle = (df_crispr_row['chromStart'] + df_crispr_row['chromEnd']) // 2
                gene_name = df_crispr_row['measuredGeneSymbol']

                df_preds_gene = df_preds[df_preds['TargetGene'] == gene_name]
                scores_gene = np.sqrt((np.exp(df_preds_gene['DNase']) - 1) * np.abs(df_preds_gene['Grad_DNase']) * (np.exp(df_preds_gene['H3K27ac']) - 1) * np.abs(df_preds_gene['Grad_H3K27ac']))
                df_enhancer_middle = df_preds_gene[(df_preds_gene['start'] <= enhancer_middle) & (df_preds_gene['end'] > enhancer_middle)]
                
                if cnt == 1:
                    df_crispr.loc[i, 'DistanceToTSS'] = np.abs(enhancer_middle - df_preds_gene['TargetGeneTSS'].values[0])
                    
                if (len(df_enhancer_middle) > 0 and np.max(scores_gene) > 0):
                    # normalization of scores for each gene
                    #scores_gene = (scores_gene - np.mean(scores_gene)) / np.std(scores_gene)
                    idx_mid = df_enhancer_middle.index[0]
                    scores_gene_around_enhancer = scores_gene.loc[idx_mid-L:idx_mid+L]
                    final_score = np.max(scores_gene_around_enhancer)/np.max(scores_gene)
                    #final_score = np.max(scores_gene_around_enhancer)
                else:
                    final_score = 0.0

                df_crispr.loc[i, 'Score_fdr_{}_{}_L_{}'.format(fdr, saliency_method, L)] = final_score

df_crispr.to_csv(data_path+'/results/csv/distal_reg_paper/'+dataset+'/EG_preds_only_enhancers_'+cell_line+'_'+genome+'_'+dataset+'.tsv', sep = '\t', index=False)



##### load
df_crispr = pd.read_csv(data_path+'/results/csv/distal_reg_paper/'+dataset+'/EG_preds_only_enhancers_'+cell_line+'_'+genome+'_'+dataset+'.tsv', delimiter = '\t')
df_crispr = df_crispr[df_crispr['DistanceToTSS'] < 100000]
#df_crispr = df_crispr[(df_crispr['DistanceToTSS'] >= 20000) & (df_crispr['DistanceToTSS'] < 100000)]
#df_crispr = df_crispr[df_crispr['DistanceToTSS'] >= 500000]


plt.figure(figsize=(25,25))
for fdr in ['1']:
    for saliency_method in ['saliency']:
        for L in [2, 5, 10, 20, 40, 50]:
            print('FDR: {} | {} | L = {}'.format(fdr, saliency_method, L))

            Y_true = df_crispr['Regulated'].values.astype(np.int32)
            Y_pred = df_crispr['Score_fdr_{}_{}_L_{}'.format(fdr, saliency_method, L)].values

            precision, recall, thresholds = precision_recall_curve(Y_true, Y_pred)
            average_precision = average_precision_score(Y_true, Y_pred)
            aupr = auc(recall, precision)

            plt.plot(recall, precision, linewidth=3, label='fdr 0.{} | {} | L={} | auPR={:6.4f}'.format(fdr, saliency_method, L, aupr))

plt.title('Fulco | all = {} | positives = {}'.format(len(df_crispr), np.sum(df_crispr['Regulated']==True)), fontsize=40)
plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=40)
plt.xlabel("Recall", fontsize=40)
plt.ylabel("Precision", fontsize=40)
plt.tick_params(axis='x', labelsize=40)
plt.tick_params(axis='y', labelsize=40)


plt.savefig(data_path+'/results/csv/distal_reg_paper/PR_curve_{}_500K-end.pdf'.format(dataset), bbox_inches='tight')
