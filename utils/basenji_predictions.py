import numpy as np
import pandas as pd
import time
import os
from scipy.stats import spearmanr
import tensorflow as tf


resolution = '5kb'     # 5kb/10kb
organism = 'human'     # human/mouse
genome = 'hg19'        # hg38/hg19/mm10
cell_line = 'GM12878'  # GM12878/K562
data_path = '/media/labuser/STORAGE/GraphReg'


def poisson_loss(y_true, mu_pred):
    nll = tf.reduce_mean(tf.math.lgamma(y_true + 1) + mu_pred - y_true * tf.math.log(mu_pred))
    return nll


df_basenji_preds_5k = pd.DataFrame(columns=['chr', 'genes', 'n_tss', 'tss', 'true_cage', 'pred_cage_basenji'])

for i in range(2,22):
    print('chr: ', i)
    tss_pos = np.load(data_path+'/data/tss/'+organism+'/'+genome+'/tss_pos_chr'+str(i)+'.npy', allow_pickle=True)
    gene_names_all = np.load(data_path+'/data/tss/'+organism+'/'+genome+'/tss_gene_chr'+str(i)+'.npy', allow_pickle=True)
    n_tss = np.load(data_path+'/data/tss/'+organism+'/'+genome+'/tss_bins_chr'+str(i)+'.npy', allow_pickle=True)

    tss_pos = tss_pos[tss_pos>0]
    print('tss_pos: ', len(tss_pos), tss_pos[0:10])
    gene_names_all = gene_names_all[gene_names_all != ""]
    print('gene_names_all: ', len(gene_names_all), gene_names_all[0:10])
    n_tss = n_tss[n_tss>=1]
    print('n_tss: ', len(n_tss), n_tss[0:10])

    if i in [2,12]:
        basenji_preds = pd.read_csv('/home/labuser/Codes/basenji/output/'+cell_line+'/predictions_bedgraph_1/bedgraph/preds_t0.bedgraph', names=['chr', 'start', 'end', 'value'], sep='\t')
        basenji_preds = basenji_preds.sort_values(by=['chr', 'start']).reset_index(drop=True)
        basenji_true = pd.read_csv('/home/labuser/Codes/basenji/output/'+cell_line+'/predictions_bedgraph_1/bedgraph/targets_t0.bedgraph', names=['chr', 'start', 'end', 'value'], sep='\t')
        basenji_true = basenji_true.sort_values(by=['chr', 'start']).reset_index(drop=True)
    elif i in [3,13]:
        basenji_preds = pd.read_csv('/home/labuser/Codes/basenji/output/'+cell_line+'/predictions_bedgraph_2/bedgraph/preds_t0.bedgraph', names=['chr', 'start', 'end', 'value'], sep='\t')
        basenji_preds = basenji_preds.sort_values(by=['chr', 'start']).reset_index(drop=True)
        basenji_true = pd.read_csv('/home/labuser/Codes/basenji/output/'+cell_line+'/predictions_bedgraph_2/bedgraph/targets_t0.bedgraph', names=['chr', 'start', 'end', 'value'], sep='\t')
        basenji_true = basenji_true.sort_values(by=['chr', 'start']).reset_index(drop=True)
    elif i in [4,14]:
        basenji_preds = pd.read_csv('/home/labuser/Codes/basenji/output/'+cell_line+'/predictions_bedgraph_3/bedgraph/preds_t0.bedgraph', names=['chr', 'start', 'end', 'value'], sep='\t')
        basenji_preds = basenji_preds.sort_values(by=['chr', 'start']).reset_index(drop=True)
        basenji_true = pd.read_csv('/home/labuser/Codes/basenji/output/'+cell_line+'/predictions_bedgraph_3/bedgraph/targets_t0.bedgraph', names=['chr', 'start', 'end', 'value'], sep='\t')
        basenji_true = basenji_true.sort_values(by=['chr', 'start']).reset_index(drop=True)
    elif i in [5,15]:
        basenji_preds = pd.read_csv('/home/labuser/Codes/basenji/output/'+cell_line+'/predictions_bedgraph_4/bedgraph/preds_t0.bedgraph', names=['chr', 'start', 'end', 'value'], sep='\t')
        basenji_preds = basenji_preds.sort_values(by=['chr', 'start']).reset_index(drop=True)
        basenji_true = pd.read_csv('/home/labuser/Codes/basenji/output/'+cell_line+'/predictions_bedgraph_4/bedgraph/targets_t0.bedgraph', names=['chr', 'start', 'end', 'value'], sep='\t')
        basenji_true = basenji_true.sort_values(by=['chr', 'start']).reset_index(drop=True)
    elif i in [6,16]:
        basenji_preds = pd.read_csv('/home/labuser/Codes/basenji/output/'+cell_line+'/predictions_bedgraph_5/bedgraph/preds_t0.bedgraph', names=['chr', 'start', 'end', 'value'], sep='\t')
        basenji_preds = basenji_preds.sort_values(by=['chr', 'start']).reset_index(drop=True)
        basenji_true = pd.read_csv('/home/labuser/Codes/basenji/output/'+cell_line+'/predictions_bedgraph_5/bedgraph/targets_t0.bedgraph', names=['chr', 'start', 'end', 'value'], sep='\t')
        basenji_true = basenji_true.sort_values(by=['chr', 'start']).reset_index(drop=True)
    elif i in [7,17]:
        basenji_preds = pd.read_csv('/home/labuser/Codes/basenji/output/'+cell_line+'/predictions_bedgraph_6/bedgraph/preds_t0.bedgraph', names=['chr', 'start', 'end', 'value'], sep='\t')
        basenji_preds = basenji_preds.sort_values(by=['chr', 'start']).reset_index(drop=True)
        basenji_true = pd.read_csv('/home/labuser/Codes/basenji/output/'+cell_line+'/predictions_bedgraph_6/bedgraph/targets_t0.bedgraph', names=['chr', 'start', 'end', 'value'], sep='\t')
        basenji_true = basenji_true.sort_values(by=['chr', 'start']).reset_index(drop=True)
    elif i in [8,18]:
        basenji_preds = pd.read_csv('/home/labuser/Codes/basenji/output/'+cell_line+'/predictions_bedgraph_7/bedgraph/preds_t0.bedgraph', names=['chr', 'start', 'end', 'value'], sep='\t')
        basenji_preds = basenji_preds.sort_values(by=['chr', 'start']).reset_index(drop=True)
        basenji_true = pd.read_csv('/home/labuser/Codes/basenji/output/'+cell_line+'/predictions_bedgraph_7/bedgraph/targets_t0.bedgraph', names=['chr', 'start', 'end', 'value'], sep='\t')
        basenji_true = basenji_true.sort_values(by=['chr', 'start']).reset_index(drop=True)
    elif i in [9,19]:
        basenji_preds = pd.read_csv('/home/labuser/Codes/basenji/output/'+cell_line+'/predictions_bedgraph_8/bedgraph/preds_t0.bedgraph', names=['chr', 'start', 'end', 'value'], sep='\t')
        basenji_preds = basenji_preds.sort_values(by=['chr', 'start']).reset_index(drop=True)
        basenji_true = pd.read_csv('/home/labuser/Codes/basenji/output/'+cell_line+'/predictions_bedgraph_8/bedgraph/targets_t0.bedgraph', names=['chr', 'start', 'end', 'value'], sep='\t')
        basenji_true = basenji_true.sort_values(by=['chr', 'start']).reset_index(drop=True)
    elif i in [10,20]:
        basenji_preds = pd.read_csv('/home/labuser/Codes/basenji/output/'+cell_line+'/predictions_bedgraph_9/bedgraph/preds_t0.bedgraph', names=['chr', 'start', 'end', 'value'], sep='\t')
        basenji_preds = basenji_preds.sort_values(by=['chr', 'start']).reset_index(drop=True)
        basenji_true = pd.read_csv('/home/labuser/Codes/basenji/output/'+cell_line+'/predictions_bedgraph_9/bedgraph/targets_t0.bedgraph', names=['chr', 'start', 'end', 'value'], sep='\t')
        basenji_true = basenji_true.sort_values(by=['chr', 'start']).reset_index(drop=True)
    elif i in [11,21]:
        basenji_preds = pd.read_csv('/home/labuser/Codes/basenji/output/'+cell_line+'/predictions_bedgraph_10/bedgraph/preds_t0.bedgraph', names=['chr', 'start', 'end', 'value'], sep='\t')
        basenji_preds = basenji_preds.sort_values(by=['chr', 'start']).reset_index(drop=True)
        basenji_true = pd.read_csv('/home/labuser/Codes/basenji/output/'+cell_line+'/predictions_bedgraph_10/bedgraph/targets_t0.bedgraph', names=['chr', 'start', 'end', 'value'], sep='\t')
        basenji_true = basenji_true.sort_values(by=['chr', 'start']).reset_index(drop=True)


    for pos, g, n in zip(tss_pos, gene_names_all, n_tss):
        bin_start_5k = pos - np.mod(pos, 5000)
        bin_end_5k = bin_start_5k + 5000

        df_start = basenji_preds[(basenji_preds['chr']=='chr'+str(i)) & (np.abs(basenji_preds['start'] - bin_start_5k) <= 64)]
        df_end = basenji_preds[(basenji_preds['chr']=='chr'+str(i)) & (np.abs(basenji_preds['end'] - bin_end_5k) <= 64)]

        if len(df_start) > 0 and len(df_end) > 0:

            basenji_start_idx = df_start.index[0]
            basenji_end_idx = df_end.index[0]
            pred_cage_basenji = np.sum(basenji_preds.iloc[basenji_start_idx:basenji_end_idx+1].value)
            true_cage = np.sum(basenji_true.iloc[basenji_start_idx:basenji_end_idx+1].value)
            dict_row = {'chr': 'chr'+str(i), 'genes': g, 'n_tss': n, 'tss': pos, 'true_cage': true_cage, 'pred_cage_basenji': pred_cage_basenji}
            df_basenji_preds_5k = df_basenji_preds_5k.append(dict_row, ignore_index=True)

df_basenji_preds_5k.to_csv(data_path+'/results/csv/cage_prediction/seq_models/cage_predictions_basenji_'+cell_line+'.csv', sep="\t", index=False)


##### Add n_contact and tss_distance_from_center to the dataframe and save #####
df_basenji_preds_5k = pd.read_csv(data_path+'/results/csv/cage_prediction/seq_models/cage_predictions_basenji_'+cell_line+'.csv', sep="\t")
df_graphreg = pd.read_csv(data_path+'/results/csv/cage_prediction/seq_models/cage_predictions_seq_e2e_models_'+cell_line+'_HiChIP_FDR_1.csv', sep="\t")

df_basenji_preds_5k['n_contact'] = 0
df_basenji_preds_5k['tss_distance_from_center'] = 0

for i in range(len(df_basenji_preds_5k)):

    df_tmp = df_graphreg[(df_graphreg['genes'] == df_basenji_preds_5k.loc[i,'genes']) & (df_graphreg['tss'] == df_basenji_preds_5k.loc[i,'tss'])]
    df_basenji_preds_5k.loc[i,'n_contact'] = df_tmp['n_contact'].values
    df_basenji_preds_5k.loc[i,'tss_distance_from_center'] = df_tmp['tss_distance_from_center'].values
    df_basenji_preds_5k.loc[i,'true_cage'] = df_tmp['true_cage'].values


df_basenji_preds_5k = df_basenji_preds_5k.reindex(columns=['chr', 'genes', 'n_tss', 'tss', 'tss_distance_from_center', 'n_contact', 'true_cage', 'pred_cage_basenji'])
df_basenji_preds_5k.to_csv(data_path+'/results/csv/cage_prediction/seq_models/cage_predictions_basenji_'+cell_line+'.csv', sep="\t", index=False)


###### Compute R for cage predictions of basenji #####

NLL = np.zeros([10,4])
R = np.zeros([10,4])
SP = np.zeros([10,4])
n_gene = np.zeros([10,4])

cell_line = 'GM12878'
train_mode = 'e2e'
assay_type = 'none'
qval = 1

df_basenji_preds_5k = pd.read_csv(data_path+'/results/csv/cage_prediction/seq_models/cage_predictions_basenji_'+cell_line+'.csv', sep="\t")

df = pd.DataFrame(columns=['cell', 'Method', 'Train_mode', 'Set', 'valid_chr', 'test_chr', 'n_gene_test', '3D_data', 'FDR', 'R','NLL'])

for i in range(1,1+10):
    print('i: ', i)
    if organism == 'mouse' and i==9:
        iv2 = i+10
        it2 = 1
    elif organism == 'mouse' and i==10:
        iv2 = 1
        it2 = 2
    else:
        iv2 = i+10
        it2 = i+11
    valid_chr_list = [i, iv2]
    test_chr_list = [i+1, it2]
    test_chr = ['chr'+str(c) for c in test_chr_list]

    test_chr_str = [str(i) for i in test_chr_list]
    test_chr_str = ','.join(test_chr_str)
    valid_chr_str = [str(i) for i in valid_chr_list]
    valid_chr_str = ','.join(valid_chr_str)

    y_gene = df_basenji_preds_5k[df_basenji_preds_5k['chr'].isin(test_chr)]['true_cage'].values
    y_hat_gene_basenji = df_basenji_preds_5k[df_basenji_preds_5k['chr'].isin(test_chr)]['pred_cage_basenji'].values
    n_contacts = df_basenji_preds_5k[df_basenji_preds_5k['chr'].isin(test_chr)]['n_contact'].values

    for j in range(4):
        if j==0:
            min_expression = 0 
            min_contact = 0
        elif j==1:
            min_expression = 5 
            min_contact = 0
        elif j==2:
            min_expression = 5 
            min_contact = 1
        else:
            min_expression = 5 
            min_contact = 5

        idx = np.where(np.logical_and(n_contacts >= min_contact, y_gene >= min_expression))[0]
        y_gene_idx = y_gene[idx]
        y_hat_gene_basenji_idx = y_hat_gene_basenji[idx]

        NLL[i-1,j] = poisson_loss(y_gene_idx, y_hat_gene_basenji_idx).numpy()
        R[i-1,j] = np.corrcoef(np.log2(y_gene_idx+1),np.log2(y_hat_gene_basenji_idx+1))[0,1]
        SP[i-1,j] = spearmanr(np.log2(y_gene_idx+1),np.log2(y_hat_gene_basenji_idx+1))[0]
        n_gene[i-1,j] = len(y_gene_idx)

    if  R[i-1,0] is not np.nan and R[i-1,0] < np.inf:
        df = df.append({'cell': cell_line, 'Method': 'Basenji', 'Train_mode': train_mode, 'Set': 'All', 'valid_chr': valid_chr_str, 'test_chr': test_chr_str, 
                        'n_gene_test': n_gene[i-1,0], '3D_data': assay_type, 'FDR': qval, 'R': R[i-1,0], 'NLL': NLL[i-1,0]}, ignore_index=True)
        df = df.append({'cell': cell_line, 'Method': 'Basenji', 'Train_mode': train_mode, 'Set': 'Expressed', 'valid_chr': valid_chr_str, 'test_chr': test_chr_str, 
                        'n_gene_test': n_gene[i-1,1], '3D_data': assay_type, 'FDR': qval, 'R': R[i-1,1], 'NLL': NLL[i-1,1]}, ignore_index=True)
        df = df.append({'cell': cell_line, 'Method': 'Basenji', 'Train_mode': train_mode, 'Set': 'Interacted', 'valid_chr': valid_chr_str, 'test_chr': test_chr_str, 
                        'n_gene_test': n_gene[i-1,2], '3D_data': assay_type, 'FDR': qval, 'R': R[i-1,2], 'NLL': NLL[i-1,2]}, ignore_index=True)

df.to_csv(data_path+'/results/csv/cage_prediction/seq_models/R_NLL_basenji_'+cell_line+'.csv', sep="\t", index=False)