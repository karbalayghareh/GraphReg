from __future__ import division
import sys
sys.path.insert(0,'../train')
  
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from gat_layer import GraphAttention
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow import keras
import matplotlib.pyplot as plt
import time
from scipy.stats import spearmanr
from scipy.stats import wilcoxon
from scipy.stats import ranksums
from tensorflow.keras import regularizers
import pyBigWig
from tensorflow.keras import backend as K
from adjustText import adjust_text
import matplotlib.patches as mpatches
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap
from statannot import add_stat_annotation
import seaborn as sns
import pandas as pd
import mygene
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
import statsmodels.stats.multitest as fdr
# Needed for Illustrator
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = ['Arial','Helvetica']

data_path = '/media/labuser/STORAGE/GraphReg'   # data path
qval = .1                                       # 0.1, 0.01, 0.001
assay_type = 'HiChIP'                           # HiChIP, HiC, MicroC, HiCAR

if qval == 0.1:
    FDR = '1'
elif qval == 0.01:
    FDR = '01'
elif qval == 0.001:
    FDR = '001'

def poisson_loss(y_true, mu_pred):
    nll = tf.reduce_mean(tf.math.lgamma(y_true + 1) + mu_pred - y_true * tf.math.log(mu_pred))
    return nll

def parse_proto(example_protos):
      features = {
        'last_batch': tf.io.FixedLenFeature([1], tf.int64),
        'adj': tf.io.FixedLenFeature([], tf.string),
        #'adj_real': tf.io.FixedLenFeature([], tf.string),
        'tss_idx': tf.io.FixedLenFeature([], tf.string),
        'X_1d': tf.io.FixedLenFeature([], tf.string),
        'Y': tf.io.FixedLenFeature([], tf.string),
        'bin_idx': tf.io.FixedLenFeature([], tf.string),
        'sequence': tf.io.FixedLenFeature([], tf.string)
        }

      parsed_features = tf.io.parse_example(example_protos, features=features)

      last_batch = parsed_features['last_batch']

      adj = tf.io.decode_raw(parsed_features['adj'], tf.float16)
      adj = tf.cast(adj, tf.float32)

      #adj_real = tf.io.decode_raw(parsed_features['adj_real'], tf.float16)
      #adj_real = tf.cast(adj_real, tf.float32)

      tss_idx = tf.io.decode_raw(parsed_features['tss_idx'], tf.float16)
      tss_idx = tf.cast(tss_idx, tf.float32)

      X_epi = tf.io.decode_raw(parsed_features['X_1d'], tf.float16)
      X_epi = tf.cast(X_epi, tf.float32)

      Y = tf.io.decode_raw(parsed_features['Y'], tf.float16)
      Y = tf.cast(Y, tf.float32)

      bin_idx = tf.io.decode_raw(parsed_features['bin_idx'], tf.int64)
      bin_idx = tf.cast(bin_idx, tf.int64)

      seq = tf.io.decode_raw(parsed_features['sequence'], tf.float64)
      seq = tf.cast(seq, tf.float32)

      return {'seq': seq, 'last_batch': last_batch, 'X_epi': X_epi, 'Y': Y, 'adj': adj, 'tss_idx': tss_idx, 'bin_idx': bin_idx}

def file_to_records(filename):
        return tf.data.TFRecordDataset(filename, compression_type='ZLIB')

def dataset_iterator(file_name, batch_size):
    dataset = tf.data.Dataset.list_files(file_name)
    dataset = dataset.flat_map(file_to_records)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(parse_proto)
    iterator = dataset.make_one_shot_iterator()
    return iterator

def read_tf_record_1shot(iterator):
    try:
        next_datum = iterator.get_next()
        data_exist = True
    except tf.errors.OutOfRangeError:
        data_exist = False
    if data_exist:
        T = 400
        b = 5000
        F = 4
        seq = next_datum['seq']
        batch_size = tf.shape(seq)[0]
        seq = tf.reshape(seq, [60, 100000, F])
        adj = next_datum['adj']
        adj = tf.reshape(adj, [batch_size, 3*T, 3*T])

        last_batch = next_datum['last_batch']
        tss_idx = next_datum['tss_idx']
        tss_idx = tf.reshape(tss_idx, [3*T])
        bin_idx = next_datum['bin_idx']
        bin_idx = tf.reshape(bin_idx, [3*T])

        if last_batch==0:
            idx = tf.range(T, 2*T)
        else:
            idx = tf.range(T, 3*T)

        #bin_idx = tf.gather(bin_idx, idx)
        num_zero = np.sum(bin_idx.numpy()==0)
        if (num_zero == T+T//2+1 and bin_idx[0] == 0):
            start = bin_idx[T+T//2].numpy()
            end = bin_idx[-1].numpy()+5000
            pos1 = np.arange(start, end, 100).astype(int)
            pad = -np.flip(np.arange(100, 3000000+100, 100).astype(int))   # -3000000, -2999900, ... , -200, -100
            pos = np.append(pad, pos1).astype(int)

        elif (num_zero == T//2+1 and bin_idx[0] == 0):
            start = bin_idx[T//2].numpy()
            end = bin_idx[-1].numpy()+5000
            pos1 = np.arange(start, end, 100).astype(int)
            pad = -np.flip(np.arange(100, 1000000+100, 100).astype(int))   # -1000000, -999900, ... , -200, -100
            pos = np.append(pad, pos1).astype(int)

        elif bin_idx[-1] == 0:
            start = bin_idx[0].numpy()
            i0 = np.where(bin_idx.numpy()==0)[0][0]
            end = bin_idx[i0-1].numpy()+5000
            #print('end: ', end)
            pos1 = np.arange(start, end, 100).astype(int)
            l = 60000 - len(pos1)
            pad = 10**15 * np.ones(l)
            pos = np.append(pos1, pad).astype(int)

        else:
            start = bin_idx[0].numpy()
            end = bin_idx[-1].numpy()+5000
            pos = np.arange(start, end, 100).astype(int)

        Y = next_datum['Y']
        Y = tf.reshape(Y, [batch_size, 3*T, 50])
        Y = tf.reduce_sum(Y, axis=2)
        Y = tf.reshape(Y, [batch_size, 3*T])
        #Y = tf.gather(Y, idx, axis=1)

        X_epi = next_datum['X_epi']
        X_epi = tf.reshape(X_epi, [1, 60000, 3])
        X_epi = tf.math.log(X_epi + 1)

    else:
        X_epi = 0
        seq = 0
        Y = 0
        adj = 0
        tss_idx = 0
        idx = 0
        pos = 0
        last_batch = 10
    return data_exist, seq, X_epi, Y, adj, idx, tss_idx, pos, last_batch

def calculate_loss(model_gat, model_cnn, chr_list, cell_line, organism, genome, batch_size, TF_positions_df, TFs):
    loss_gat_all = np.array([])
    loss_cnn_all = np.array([])
    Y_hat_gat_all = np.array([])
    Y_hat_cnn_all = np.array([])
    loss_gat_ko_all = np.array([])
    loss_cnn_ko_all = np.array([])
    Y_hat_gat_ko_all = np.array([])
    Y_hat_cnn_ko_all = np.array([])
    Y_all = np.array([])
    y_gene = np.array([])
    y_hat_gene_gat = np.array([])
    y_hat_gene_cnn = np.array([])
    y_hat_gene_gat_ko_list = [np.array([])] * len(TFs)
    y_hat_gene_cnn_ko_list = [np.array([])] * len(TFs)
    chr_pos = []
    gene_pos = np.array([])
    gene_names = np.array([])
    n_contacts = np.array([])

    y_bw = np.array([])
    y_pred_gat_bw = np.array([])
    y_pred_cnn_bw = np.array([])
    chroms = np.array([])
    starts = np.array([])
    ends = np.array([])
    T = 400

    for i in chr_list:
        start_time = time.time()
        print('chr :', i)
        chrm = 'chr'+str(i)
        file_name = data_path+'/data/tfrecords/tfr_seq_'+cell_line+'_'+assay_type+'_FDR_'+FDR+'_chr'+str(i)+'.tfr'
        iterator = dataset_iterator(file_name, batch_size)
        tss_pos = np.load(data_path+'/data/tss/'+organism+'/'+genome+'/tss_pos_chr'+str(i)+'.npy', allow_pickle=True)
        gene_names_all = np.load(data_path+'/data/tss/'+organism+'/'+genome+'/tss_gene_chr'+str(i)+'.npy', allow_pickle=True)
        tss_pos = tss_pos[tss_pos>0]
        print('tss_pos: ', len(tss_pos))
        gene_names_all = gene_names_all[gene_names_all != ""]
        print('gene_names_all: ', len(gene_names_all), gene_names_all[0:10])
        pos_bw = np.array([])
        y_bw_ = np.array([])
        y_pred_gat_bw_ = np.array([])
        y_pred_cnn_bw_ = np.array([])
        print(tss_pos.shape[0], tss_pos[0:100])
        while True:
            data_exist, seq, X_epi, Y, adj, idx, tss_idx, pos, last_batch = read_tf_record_1shot(iterator)

            if data_exist:
                if tf.reduce_sum(tf.gather(tss_idx, idx)) > 0:
                    Y_hat_cnn, _, _, _, _ = model_cnn(seq)
                    Y_hat_gat, _, _, _, _, _ = model_gat([seq, adj])

                    chr_pos.append('chr'+str(i)+'_'+str(pos[20000]))

                    ## extract gene tss's ##
                    row_sum = tf.squeeze(tf.reduce_sum(adj, axis=-1))
                    num_contacts = np.repeat(row_sum.numpy().ravel(), 50)

                    if last_batch == 0:
                        tss_pos_1 = tss_pos[np.logical_and(tss_pos >= pos[20000], tss_pos < pos[40000])]
                    else:
                        tss_pos_1 = tss_pos[np.logical_and(tss_pos >= pos[20000], tss_pos < pos[-1])]

                    for j in range(len(tss_pos_1)):
                        idx_tss = np.where(pos == int(np.floor(tss_pos_1[j]/100)*100))[0][0]
                        #print(idx_tss)
                        idx_gene = np.where(tss_pos == tss_pos_1[j])[0]
                        
                        y_true_ = np.repeat(Y.numpy().ravel(), 50)
                        y_hat_gat_ = np.repeat(Y_hat_gat.numpy().ravel(), 50)
                        y_hat_cnn_ = np.repeat(Y_hat_cnn.numpy().ravel(), 50)

                        y_gene = np.append(y_gene, y_true_[idx_tss]) 
                        y_hat_gene_gat = np.append(y_hat_gene_gat, y_hat_gat_[idx_tss]) 
                        y_hat_gene_cnn = np.append(y_hat_gene_cnn, y_hat_cnn_[idx_tss]) 
                        gene_pos = np.append(gene_pos, 'chr'+str(i)+'_tss_'+str(tss_pos_1[j]))
                        gene_names = np.append(gene_names, gene_names_all[idx_gene]) 
                        n_contacts = np.append(n_contacts, num_contacts[idx_tss])

                    ############# Knock out the motifs of the TF ###################
                    for i_tf in range(len(TFs)):
                        TF_positions_df_sub = TF_positions_df[((TF_positions_df['TF']==TFs[i_tf]) & (TF_positions_df['chr']==chrm) & (TF_positions_df['start']>pos[0]) & (TF_positions_df['end']<pos[0]+6000000))].reset_index(drop=True)
                        #print('pos[0]: ', pos[0])
                        #print('TF_positions_df_sub: ', TF_positions_df_sub)
                        if len(TF_positions_df_sub)>0:
                            seq_np = np.copy(seq.numpy())
                            for ii in range(len(TF_positions_df_sub)):
                                j = (TF_positions_df_sub['start'].values[ii] - pos[0])
                                k = j // 100000
                                L = TF_positions_df_sub['end'].values[ii] - TF_positions_df_sub['start'].values[ii]
                                if j%100000 < 100000-20:
                                    start_mut_seq = j%100000
                                    end_mut_seq = start_mut_seq + L
                                    seq_np[k,start_mut_seq:end_mut_seq,:] = 0.
                            
                            seq_mut = tf.convert_to_tensor(seq_np)
                            Y_hat_cnn_ko, _, _, _, _ = model_cnn(seq_mut)
                            Y_hat_gat_ko, _, _, _, _, _ = model_gat([seq_mut, adj])
                        else: 
                            Y_hat_cnn_ko = Y_hat_cnn
                            Y_hat_gat_ko = Y_hat_gat

                        for j in range(len(tss_pos_1)):
                            idx_tss = np.where(pos == int(np.floor(tss_pos_1[j]/100)*100))[0][0]
                            
                            y_hat_gat_ko_ = np.repeat(Y_hat_gat_ko.numpy().ravel(), 50)
                            y_hat_cnn_ko_ = np.repeat(Y_hat_cnn_ko.numpy().ravel(), 50)

                            y_hat_gene_gat_ko_list[i_tf] = np.append(y_hat_gene_gat_ko_list[i_tf], y_hat_gat_ko_[idx_tss]) 
                            y_hat_gene_cnn_ko_list[i_tf] = np.append(y_hat_gene_cnn_ko_list[i_tf], y_hat_cnn_ko_[idx_tss]) 
                    
            else:
                break

    print('len of test/valid Y: ', len(y_gene))
    return y_gene, y_hat_gene_gat, y_hat_gene_gat_ko_list, y_hat_gene_cnn, y_hat_gene_cnn_ko_list, chr_pos, gene_pos, gene_names, n_contacts


#################### load model ####################
batch_size = 1
organism = 'human'            # human/mouse
cell_line = 'K562'            # K562/GM12878/mESC
genome = 'hg19'               # hg19


TF_positions_df = pd.read_csv(data_path+'/results/fimo/peaks_H3K27ac_K562/TF_positions_unique.bed', sep="\t")
TF_positions_df.columns = ['chr', 'start', 'end', 'TF', '-log10(pval)', 'strand']
TFs = np.unique(TF_positions_df['TF'].values)
print('TFs: ', TFs)

N = 2                         # Number of replicates (1,...,10)
col_name = ['True_CAGE', 'n_contact']
for i in range(1,1+N):
    col_name.append('WT_'+str(i))
for i in range(1, 1+N):
    col_name.append('KO_'+str(i))

df_GraphReg_dict = {}
df_CNN_dict = {}
chr_list = list(np.arange(1,1+22))
#chr_list = [22]

for i in range(1,1+N):
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

    test_chr_str = [str(i) for i in test_chr_list]
    test_chr_str = ','.join(test_chr_str)
    valid_chr_str = [str(i) for i in valid_chr_list]
    valid_chr_str = ','.join(valid_chr_str)

    model_name_gat = data_path+'/models/'+cell_line+'/Seq-GraphReg_e2e_'+cell_line+'_'+assay_type+'_FDR_'+FDR+'_valid_chr_'+valid_chr_str+'_test_chr_'+test_chr_str+'.h5'
    model_gat = tf.keras.models.load_model(model_name_gat, custom_objects={'GraphAttention': GraphAttention})
    model_gat.trainable = False
    model_gat._name = 'Seq-GraphReg_e2e'
    #model_gat.summary()

    model_name = data_path+'/models/'+cell_line+'/Seq-CNN_e2e_'+cell_line+'_'+assay_type+'_FDR_'+FDR+'_valid_chr_'+valid_chr_str+'_test_chr_'+test_chr_str+'.h5'
    model_cnn = tf.keras.models.load_model(model_name)
    model_cnn.trainable = False
    model_cnn._name = 'Seq-CNN_e2e'
    #model_cnn.summary()

    y_gene, y_hat_gene_gat, y_hat_gene_gat_ko_list, y_hat_gene_cnn, y_hat_gene_cnn_ko_list, _, _, gene_names, n_contacts = calculate_loss(model_gat, model_cnn, 
                        chr_list, cell_line, organism, genome, batch_size, TF_positions_df, TFs)

    for k in range(len(TFs)):
        if i == 1:
            df_GraphReg_dict[k] = pd.DataFrame(columns=col_name)
            df_GraphReg_dict[k]['True_CAGE'] = y_gene
            df_GraphReg_dict[k]['n_contact'] = n_contacts
            df_GraphReg_dict[k].index = gene_names
            
            df_CNN_dict[k] = pd.DataFrame(columns=col_name)
            df_CNN_dict[k]['True_CAGE'] = y_gene
            df_CNN_dict[k]['n_contact'] = n_contacts
            df_CNN_dict[k].index = gene_names

        df_GraphReg_dict[k]['WT_'+str(i)] = y_hat_gene_gat
        df_GraphReg_dict[k]['KO_'+str(i)] = y_hat_gene_gat_ko_list[k]

        df_CNN_dict[k]['WT_'+str(i)] = y_hat_gene_cnn
        df_CNN_dict[k]['KO_'+str(i)] = y_hat_gene_cnn_ko_list[k]


for k in range(len(TFs)):
    df_GraphReg_dict[k].to_csv(data_path+'/results/csv/insilico_TF_KO/Seq-GraphReg_TF_KO_'+TFs[k]+'.tsv', sep='\t')
    df_CNN_dict[k].to_csv(data_path+'/results/csv/insilico_TF_KO/Seq-CNN_TF_KO_'+TFs[k]+'.tsv', sep='\t')


######### load and do DESeq2 analysis #########
import numpy as np
import pandas as pd
# import mygene
# import matplotlib.pyplot as plt
import rpy2
from diffexpr.py_deseq import py_DESeq2

TF_positions_df = pd.read_csv(data_path+'/results/fimo/peaks_H3K27ac_K562/TF_positions_unique.bed', sep="\t")
TF_positions_df.columns = ['chr', 'start', 'end', 'TF', '-log10(pval)', 'strand']
TFs = np.unique(TF_positions_df['TF'].values)
print('TFs: ', TFs)
TFs = np.delete(TFs,9)

for k in range(len(TFs)):
    df_GraphReg = pd.read_csv(data_path+'/results/csv/insilico_TF_KO/Seq-GraphReg_TF_KO_'+TFs[k]+'.tsv', sep='\t')
    df_GraphReg.columns = ['gene_name', 'True_CAGE', 'n_contact', 'WT_1', 'WT_2', 'KO_1', 'KO_2']

    df_GraphReg_deseq = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
    df_GraphReg_deseq['id'] = df_GraphReg['gene_name']
    df_GraphReg_deseq['A_1'] = df_GraphReg['WT_1'].astype(np.int64)
    df_GraphReg_deseq['A_2'] = df_GraphReg['WT_2'].astype(np.int64)
    df_GraphReg_deseq['B_1'] = df_GraphReg['KO_1'].astype(np.int64)
    df_GraphReg_deseq['B_2'] = df_GraphReg['KO_2'].astype(np.int64)

    sample_df_GraphReg_deseq = pd.DataFrame({'samplename': df_GraphReg_deseq.columns}) \
            .query('samplename != "id"')\
            .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
            .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
    sample_df_GraphReg_deseq.index = sample_df_GraphReg_deseq.samplename

    dds = py_DESeq2(count_matrix = df_GraphReg_deseq,
               design_matrix = sample_df_GraphReg_deseq,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
    dds.run_deseq() 
    dds.get_deseq_result(contrast = ['sample','B','A'])
    res = dds.deseq_result 
    res.head()
    res = res[res['pvalue']<=1]
    res = res.sort_values(by=['pvalue']).reset_index(drop=True)
    res.to_csv(data_path+'/results/csv/CRISPRi_K562_DESeq_results/'+TFs[k]+'_KO_GraphReg.tsv', sep='\t')


    df_CNN = pd.read_csv(data_path+'/results/csv/insilico_TF_KO/Seq-CNN_TF_KO_'+TFs[k]+'.tsv', sep='\t')
    df_CNN.columns = ['gene_name', 'True_CAGE', 'n_contact', 'WT_1', 'WT_2', 'KO_1', 'KO_2']

    df_CNN_deseq = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
    df_CNN_deseq['id'] = df_CNN['gene_name']
    df_CNN_deseq['A_1'] = df_CNN['WT_1'].astype(np.int64)
    df_CNN_deseq['A_2'] = df_CNN['WT_2'].astype(np.int64)
    df_CNN_deseq['B_1'] = df_CNN['KO_1'].astype(np.int64)
    df_CNN_deseq['B_2'] = df_CNN['KO_2'].astype(np.int64)

    sample_df_CNN_deseq = pd.DataFrame({'samplename': df_CNN_deseq.columns}) \
            .query('samplename != "id"')\
            .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
            .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
    sample_df_CNN_deseq.index = sample_df_CNN_deseq.samplename

    dds = py_DESeq2(count_matrix = df_CNN_deseq,
               design_matrix = sample_df_CNN_deseq,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
    dds.run_deseq() 
    dds.get_deseq_result(contrast = ['sample','B','A'])
    res = dds.deseq_result 
    res.head()
    res = res[res['pvalue']<=1]
    res = res.sort_values(by=['pvalue']).reset_index(drop=True)
    res.to_csv(data_path+'/results/csv/CRISPRi_K562_DESeq_results/'+TFs[k]+'_KO_CNN.tsv', sep='\t')


######### Compare experimental TF knock out vs in-silico TF knock out of Seq-GraphReg and Seq-CNN #########
TF_positions_df = pd.read_csv(data_path+'/results/fimo/peaks_H3K27ac_K562/TF_positions_unique.bed', sep="\t")
TF_positions_df.columns = ['chr', 'start', 'end', 'TF', '-log10(pval)', 'strand']
TFs = np.unique(TF_positions_df['TF'].values)
TFs = np.delete(TFs,9)
print('TFs: ', TFs)
mg = mygene.MyGeneInfo()

for k in range(len(TFs)):
    df_true = pd.read_csv(data_path+'/results/csv/CRISPRi_K562_DESeq_results/'+TFs[k]+'_KO.tsv', sep='\t')
    df_GraphReg_deseq = pd.read_csv(data_path+'/results/csv/CRISPRi_K562_DESeq_results/'+TFs[k]+'_KO_GraphReg.tsv', sep='\t')
    df_CNN_deseq = pd.read_csv(data_path+'/results/csv/CRISPRi_K562_DESeq_results/'+TFs[k]+'_KO_CNN.tsv', sep='\t')

    df_GraphReg = pd.read_csv(data_path+'/results/csv/insilico_TF_KO/Seq-GraphReg_TF_KO_'+TFs[k]+'.tsv', sep='\t')
    df_GraphReg.columns = ['gene_name', 'True_CAGE', 'n_contact', 'WT_1', 'WT_2', 'KO_1', 'KO_2']


    df_true.insert(0, 'gene_name', np.repeat('NA', len(df_true)))
    df_true['id'] = df_true['id'].str.split('.', expand=True)[0]
    ens_genes = np.unique(df_true['id'].values)

    gene_symbols = mg.querymany(ens_genes, scope='ensembl.gene', fields='symbol', species='human')
    j = 0
    for dict in gene_symbols:
        j = j + 1
        print(j)
        ens = dict['query']
        if 'symbol' in dict:
            df_true.loc[df_true['id']==ens, 'gene_name'] = dict['symbol']

    genes_shared = np.intersect1d(df_GraphReg_deseq['id'].values, df_CNN_deseq['id'].values)
    genes_shared = np.intersect1d(genes_shared, df_true['gene_name'].values)

    df = pd.DataFrame(index=genes_shared, columns=['gene_name', 'true_CAGE', 'n_contacts', 'baseMean_true', 'log2FoldChange_true', 'pvalue_true', 
                                                   'baseMean_GraphReg', 'log2FoldChange_GraphReg', 'pvalue_GraphReg',
                                                   'baseMean_CNN', 'log2FoldChange_CNN', 'pvalue_CNN'])

    N = len(genes_shared)
    p=0
    for j in range(N):
        print(j)
        df.loc[genes_shared[j], 'gene_name'] = genes_shared[j]
        df.loc[genes_shared[j], 'true_CAGE'] = df_GraphReg[df_GraphReg['gene_name']==genes_shared[j]]['True_CAGE'].values[0]
        df.loc[genes_shared[j], 'n_contacts'] = df_GraphReg[df_GraphReg['gene_name']==genes_shared[j]]['n_contact'].values[0]

        df.loc[genes_shared[j], 'baseMean_true'] = df_true[df_true['gene_name']==genes_shared[j]]['baseMean'].values[0]
        df.loc[genes_shared[j], 'log2FoldChange_true'] = df_true[df_true['gene_name']==genes_shared[j]]['log2FoldChange'].values[0]
        df.loc[genes_shared[j], 'pvalue_true'] = df_true[df_true['gene_name']==genes_shared[j]]['pvalue'].values[0]

        df.loc[genes_shared[j], 'baseMean_GraphReg'] = df_GraphReg_deseq[df_GraphReg_deseq['id']==genes_shared[j]]['baseMean'].values[0]
        df.loc[genes_shared[j], 'log2FoldChange_GraphReg'] = df_GraphReg_deseq[df_GraphReg_deseq['id']==genes_shared[j]]['log2FoldChange'].values[0]
        df.loc[genes_shared[j], 'pvalue_GraphReg'] = df_GraphReg_deseq[df_GraphReg_deseq['id']==genes_shared[j]]['pvalue'].values[0]

        df.loc[genes_shared[j], 'baseMean_CNN'] = df_CNN_deseq[df_CNN_deseq['id']==genes_shared[j]]['baseMean'].values[0]
        df.loc[genes_shared[j], 'log2FoldChange_CNN'] = df_CNN_deseq[df_CNN_deseq['id']==genes_shared[j]]['log2FoldChange'].values[0]
        df.loc[genes_shared[j], 'pvalue_CNN'] = df_CNN_deseq[df_CNN_deseq['id']==genes_shared[j]]['pvalue'].values[0]

    df.to_csv(data_path+'/results/csv/insilico_TF_KO/TF_KO_'+TFs[k]+'_all.tsv', sep='\t')


######################## plot figures ########################
TF_positions_df = pd.read_csv(data_path+'/results/fimo/peaks_H3K27ac_K562/TF_positions_unique.bed', sep="\t")
TF_positions_df.columns = ['chr', 'start', 'end', 'TF', '-log10(pval)', 'strand']
TFs = np.unique(TF_positions_df['TF'].values)
TFs = np.delete(TFs,9)
print('TFs: ', TFs)

### log2FC distribution ###
n_min = 0
CAGE_min = 20

mean_gr = np.array([])
mean_cnn = np.array([])
mean_bg = np.array([])
frac_gr = np.array([])
frac_cnn = np.array([])

N = 100
TF_hm = []
pred_genes_gr = []
pred_genes_cnn = []
mat_gr = []
mat_cnn = []
p_adj_thr = .05
df_sns = pd.DataFrame(columns=['Method', 'TF_KO'])
for k in range(len(TFs)):
    df = pd.read_csv(data_path+'/results/csv/insilico_TF_KO/TF_KO_'+TFs[k]+'_all.tsv', sep='\t')
    _, pvals_adjusted, _, _ = fdr.multipletests(df['pvalue_true'].values, alpha=p_adj_thr, method='fdr_bh', is_sorted=False)
    df['pvalue_true'] = pvals_adjusted

    df_pos = df[((df['true_CAGE']>=CAGE_min) & (df['n_contacts']>=n_min) & (df['pvalue_true'] <= p_adj_thr) & (df['log2FoldChange_true'] < 0))]
    df_pos['class'] = 1

    if len(df_pos) >= 200:
        TF_hm.append(TFs[k])
        print('TF: ', TFs[k])
        print('len positives: ', len(df_pos))

        df_sub = df[((df['true_CAGE']>=CAGE_min) & (df['n_contacts']>=n_min) & (df['pvalue_true'] <= p_adj_thr))]
        print('len df_sub: ', len(df_sub))

        df_pos_gr = df_sub.sort_values(by=['log2FoldChange_GraphReg'],ascending=True).reset_index(drop=True)[0:N]
        pred_genes_gr.append(df_pos_gr['gene_name'].values[0:20])
        mat_gr.append(df_pos_gr['log2FoldChange_true'].values[0:20])

        df_pos_cnn = df_sub.sort_values(by=['log2FoldChange_CNN'],ascending=True).reset_index(drop=True)[0:N]
        pred_genes_cnn.append(df_pos_cnn['gene_name'].values[0:20])
        mat_cnn.append(df_pos_cnn['log2FoldChange_true'].values[0:20])

        mean_gr = np.append(mean_gr, np.mean(df_pos_gr['log2FoldChange_true'].values))
        mean_cnn = np.append(mean_cnn, np.mean(df_pos_cnn['log2FoldChange_true'].values))
        mean_bg = np.append(mean_bg, np.mean(df_sub['log2FoldChange_true'].values))

        frac_gr = np.append(frac_gr, len(df_pos_gr[df_pos_gr['log2FoldChange_true']<0])/N)
        print('frac_gr: ', len(df_pos_gr[df_pos_gr['log2FoldChange_true']<0])/N)

        frac_cnn = np.append(frac_cnn, len(df_pos_cnn[df_pos_cnn['log2FoldChange_true']<0])/N)
        print('frac_cnn: ', len(df_pos_cnn[df_pos_cnn['log2FoldChange_true']<0])/N)

        df_sns = df_sns.append({'Mean': np.mean(df_pos_gr['log2FoldChange_true'].values),
            'Fraction': len(df_pos_gr[df_pos_gr['log2FoldChange_true']<0])/N, 'Method': 'Seq-GraphReg', 'TF_KO': TFs[k]}, ignore_index=True)
        df_sns = df_sns.append({'Mean': np.mean(df_pos_cnn['log2FoldChange_true'].values), 
            'Fraction': len(df_pos_cnn[df_pos_cnn['log2FoldChange_true']<0])/N, 'Method': 'Seq-CNN', 'TF_KO': TFs[k]}, ignore_index=True)
        df_sns = df_sns.append({'Mean': np.mean(df_sub['log2FoldChange_true'].values), 
            'Fraction': len(df_sub[df_sub['log2FoldChange_true']<0])/len(df_sub), 'Method': 'Baseline', 'TF_KO': TFs[k]}, ignore_index=True)

print('len TF_hm: ', len(TF_hm))

### plot boxplot ###
fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
b=sns.boxplot(x='Method', y='Mean', data=df_sns, palette={"Seq-GraphReg": "orange", "Seq-CNN": "deepskyblue", "Baseline": "green"}, ax=ax1)
add_stat_annotation(ax1, data=df_sns, x='Method', y='Mean',
                box_pairs=[(("Seq-GraphReg"), ("Seq-CNN")), (("Seq-GraphReg"), ("Baseline")), (("Seq-CNN"), ("Baseline"))],
                test='Wilcoxon', text_format='star', loc='inside', verbose=0, fontsize='x-large')
b=sns.swarmplot(x='Method', y='Mean', data=df_sns, color = 'black', alpha=.3, size=8, ax=ax1)
ax1.yaxis.set_tick_params(labelsize=20)
ax1.xaxis.set_tick_params(labelsize=20)
ax1.set_title(r'$n \geq$'+str(n_min), fontsize=20)
b.set_xlabel("",fontsize=20)
b.set_ylabel("Mean Log2FC",fontsize=20)
#plt.setp(ax1.get_legend().get_texts(), fontsize='15')
#plt.setp(ax1.get_legend().get_title(), fontsize='15')
#ax1.set_ylim((.4,.75))
plt.tight_layout()
plt.savefig('../figs/TF_KO/boxplot_min_CAGE_'+str(CAGE_min)+'_n_min_'+str(n_min)+'_mean.pdf')

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
b=sns.boxplot(x='Method', y='Fraction', data=df_sns, palette={"Seq-GraphReg": "orange", "Seq-CNN": "deepskyblue", "Baseline": "green"}, ax=ax1)
add_stat_annotation(ax1, data=df_sns, x='Method', y='Fraction',
                box_pairs=[(("Seq-GraphReg"), ("Seq-CNN")), (("Seq-GraphReg"), ("Baseline")), (("Seq-CNN"), ("Baseline"))],
                test='Wilcoxon', text_format='star', loc='inside', verbose=0, fontsize='x-large')
b=sns.swarmplot(x='Method', y='Fraction', data=df_sns, color = 'black', alpha=.3, size=8, ax=ax1)
ax1.yaxis.set_tick_params(labelsize=20)
ax1.xaxis.set_tick_params(labelsize=20)
ax1.set_title(r'$n \geq$'+str(n_min), fontsize=20)
b.set_xlabel("",fontsize=20)
b.set_ylabel("Precision",fontsize=20)
#plt.setp(ax1.get_legend().get_texts(), fontsize='15')
#plt.setp(ax1.get_legend().get_title(), fontsize='15')
#ax1.set_ylim((.4,.75))
plt.tight_layout()
plt.savefig('../figs/TF_KO/boxplot_min_CAGE_'+str(CAGE_min)+'_n_min_'+str(n_min)+'_Fraction.pdf')

### plot heatmap ###
mean = np.vstack((mean_gr, mean_cnn))
df_hm = pd.DataFrame(data=mean, index=['Seq-GraphReg', 'Seq-CNN'], columns=TF_hm)
df_hm = df_hm.sort_values(by=['Seq-GraphReg'], axis = 1)
plt.figure(figsize = (20,3.5))
ax = sns.heatmap(df_hm, xticklabels=1, yticklabels=1, cmap="coolwarm", annot=df_hm, annot_kws={'rotation': 90})
ax.set_title(r'$n \geq$'+str(n_min), fontsize=20)
ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 20, rotation=0)
ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 20, rotation=90)
cbar = ax.collections[0].colorbar
cbar.set_label(label='Mean Log2FC', size=20)
cbar.ax.tick_params(labelsize=20)
#ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('../figs/TF_KO/heatmap_min_CAGE_'+str(CAGE_min)+'_n_min_'+str(n_min)+'_mean.pdf')

Fraction = np.vstack((frac_gr, frac_cnn))
df_hm = pd.DataFrame(data=Fraction, index=['Seq-GraphReg', 'Seq-CNN'], columns=TF_hm)
df_hm = df_hm.sort_values(by=['Seq-GraphReg'], ascending=False, axis = 1)
plt.figure(figsize = (20,3.5))
ax = sns.heatmap(df_hm, xticklabels=1, yticklabels=1, cmap="YlGnBu", annot=df_hm, annot_kws={'rotation': 90})
ax.set_title(r'$n \geq$'+str(n_min), fontsize=20)
ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 20, rotation=0)
ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 20, rotation=90)
cbar = ax.collections[0].colorbar
cbar.set_label(label='Precision', size=20)
cbar.ax.tick_params(labelsize=20)
#ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('../figs/TF_KO/heatmap_min_CAGE_'+str(CAGE_min)+'_n_min_'+str(n_min)+'_Fraction.pdf')


### Gene by TF heatmaps ###
mat_gr_all = np.zeros([20, len(TF_hm)])
mat_cnn_all = np.zeros([20, len(TF_hm)])
pred_genes_gr_mat = np.repeat("NANANANANA", 20*len(TF_hm)).reshape([20, len(TF_hm)])
pred_genes_cnn_mat = np.repeat("NANANANANA", 20*len(TF_hm)).reshape([20, len(TF_hm)])

for k in range(len(TF_hm)):
    mat_gr_all[:,k] = mat_gr[k]
    pred_genes_gr_mat[:,k] = pred_genes_gr[k]

    mat_cnn_all[:,k] = mat_cnn[k]
    pred_genes_cnn_mat[:,k] = pred_genes_cnn[k]

df_topgenes_gr = pd.DataFrame(mat_gr_all, columns=TF_hm)
plt.figure(figsize = (33,20))
ax = sns.heatmap(df_topgenes_gr, xticklabels=1, yticklabels=1, cmap="coolwarm", annot=pred_genes_gr_mat, fmt='', vmin=-1.5, vmax=1.5, linewidths=1)
ax.set_title('Top 20 predicted downregulated genes by Seq-GraphReg when n >= '+str(n_min)+', precision = '+'%.2f'%(np.sum(mat_gr_all<0)/len(mat_gr_all.ravel())), fontsize=30)
ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 20, rotation=0)
ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 20, rotation=90)
cbar = ax.collections[0].colorbar
cbar.set_label(label='Log2FC', size=30)
cbar.ax.tick_params(labelsize=30)
plt.tight_layout()
plt.savefig('../figs/TF_KO/topgenes_GraphReg_min_CAGE_'+str(CAGE_min)+'_n_min_'+str(n_min)+'.pdf')

df_topgenes_cnn = pd.DataFrame(mat_cnn_all, columns=TF_hm)
plt.figure(figsize = (33,20))
ax = sns.heatmap(df_topgenes_cnn, xticklabels=1, yticklabels=1, cmap="coolwarm", annot=pred_genes_cnn_mat, fmt='', vmin=-1.5, vmax=1.5, linewidths=1)
ax.set_title('Top 20 predicted downregulated genes by Seq-CNN when n >= '+str(n_min)+', precision = '+'%.2f'%(np.sum(mat_cnn_all<0)/len(mat_cnn_all.ravel())), fontsize=30)
ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 20, rotation=0)
ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 20, rotation=90)
cbar = ax.collections[0].colorbar
cbar.set_label(label='Log2FC', size=30)
cbar.ax.tick_params(labelsize=30)
plt.tight_layout()
plt.savefig('../figs/TF_KO/topgenes_CNN_min_CAGE_'+str(CAGE_min)+'_n_min_'+str(n_min)+'.pdf')

### plot clustermaps ###
'''
pred_genes_gr = np.unique(pred_genes_gr)
pred_genes_cnn = np.unique(pred_genes_cnn)
gene_by_tf_matrix_gr = np.zeros([len(pred_genes_gr), len(TF_hm)])
gene_by_tf_matrix_cnn = np.zeros([len(pred_genes_cnn), len(TF_hm)])

for k in range(len(TF_hm)):
    df = pd.read_csv(data_path+'/results/csv/insilico_TF_KO/TF_KO_'+TF_hm[k]+'_all.tsv', sep='\t')
    df.index = df['gene_name'].values
    gene_by_tf_matrix_gr[:,k] = df.loc[pred_genes_gr, ['log2FoldChange_true']].values.ravel()

    df = pd.read_csv(data_path+'/results/csv/insilico_TF_KO/TF_KO_'+TF_hm[k]+'_all.tsv', sep='\t')
    df.index = df['gene_name'].values
    gene_by_tf_matrix_cnn[:,k] = df.loc[pred_genes_cnn, ['log2FoldChange_true']].values.ravel()


df_gene_by_tf_gr = pd.DataFrame(data=gene_by_tf_matrix_gr, index=pred_genes_gr, columns=TF_hm)
df_gene_by_tf_gr = df_gene_by_tf_gr.dropna()
df_gene_by_tf_cnn = pd.DataFrame(data=gene_by_tf_matrix_cnn, index=pred_genes_cnn, columns=TF_hm)
df_gene_by_tf_cnn = df_gene_by_tf_cnn.dropna()

g = sns.clustermap(df_gene_by_tf_gr, figsize=(10, 20), cmap="coolwarm", vmin=-1, vmax=1)
plt.savefig('../figs/TF_KO/clustermap_GraphReg_min_CAGE_'+str(CAGE_min)+'_n_min_'+str(n_min)+'.png')


g = sns.clustermap(df_gene_by_tf_cnn, figsize=(10, 20), cmap="coolwarm", vmin=-1, vmax=1)
plt.savefig('../figs/TF_KO/clustermap_CNN_min_CAGE_'+str(CAGE_min)+'_n_min_'+str(n_min)+'.png')

pred_genes_shared = np.intersect1d(pred_genes_gr, pred_genes_cnn)
pred_genes_gr_only = np.setdiff1d(pred_genes_gr, pred_genes_shared)
pred_genes_cnn_only = np.setdiff1d(pred_genes_cnn, pred_genes_shared)

g = sns.clustermap(df_gene_by_tf_gr.loc[pred_genes_gr_only].dropna(), figsize=(10, 20), cmap="coolwarm", vmin=-1, vmax=1)
plt.savefig('../figs/TF_KO/clustermap_GraphReg_only_min_CAGE_'+str(CAGE_min)+'_n_min_'+str(n_min)+'.png')

g = sns.clustermap(df_gene_by_tf_gr.loc[pred_genes_shared].dropna(), figsize=(10, 20), cmap="coolwarm", vmin=-1, vmax=1)
plt.savefig('../figs/TF_KO/clustermap_shared_min_CAGE_'+str(CAGE_min)+'_n_min_'+str(n_min)+'.png')

g = sns.clustermap(df_gene_by_tf_cnn.loc[pred_genes_cnn_only].dropna(), figsize=(10, 20), cmap="coolwarm", vmin=-1, vmax=1)
plt.savefig('../figs/TF_KO/clustermap_CNN_only_min_CAGE_'+str(CAGE_min)+'_n_min_'+str(n_min)+'.png')
'''



### Look at TFs for MYC ###
p_adj_thr = 0.05
df_MYC = pd.DataFrame(columns=df.columns)
df_MYC['TF'] = TFs

for k in range(len(TFs)):
    df = pd.read_csv(data_path+'/results/csv/insilico_TF_KO/TF_KO_'+TFs[k]+'_all.tsv', sep='\t')
    _, pvals_adjusted, _, _ = fdr.multipletests(df['pvalue_true'].values, alpha=p_adj_thr, method='fdr_bh', is_sorted=False)
    df['pvalue_true'] = pvals_adjusted

    df_MYC.loc[k,df.columns] = df[df['gene_name']=='MYC'].values

df_MYC_decrease = df_MYC[((df_MYC['pvalue_true'] <= p_adj_thr) & (df_MYC['log2FoldChange_true'] < 0))]
df_MYC_increase = df_MYC[((df_MYC['pvalue_true'] <= p_adj_thr) & (df_MYC['log2FoldChange_true'] > 0))]


### Look at some TF-gene pairs ###
'''
# SP1-RAC2
p_adj_thr = 0.05
df = pd.read_csv(data_path+'/results/csv/insilico_TF_KO/TF_KO_SP1_all.tsv', sep='\t')
_, pvals_adjusted, _, _ = fdr.multipletests(df['pvalue_true'].values, alpha=p_adj_thr, method='fdr_bh', is_sorted=False)
df['pvalue_true'] = pvals_adjusted

# NRF1-ANKRD9
p_adj_thr = 0.05
df = pd.read_csv(data_path+'/results/csv/insilico_TF_KO/TF_KO_NRF1_all.tsv', sep='\t')
_, pvals_adjusted, _, _ = fdr.multipletests(df['pvalue_true'].values, alpha=p_adj_thr, method='fdr_bh', is_sorted=False)
df['pvalue_true'] = pvals_adjusted
'''

# JUND-TCF3
p_adj_thr = 0.05
df = pd.read_csv(data_path+'/results/csv/insilico_TF_KO/TF_KO_JUND_all.tsv', sep='\t')
_, pvals_adjusted, _, _ = fdr.multipletests(df['pvalue_true'].values, alpha=p_adj_thr, method='fdr_bh', is_sorted=False)
df['pvalue_true'] = pvals_adjusted


##### scatter plots #####
TF = 'USF2'
n_min = 5
CAGE_min = 20
p_adj_thr = 0.05
df = pd.read_csv(data_path+'/results/csv/insilico_TF_KO/TF_KO_'+TF+'_all.tsv', sep='\t')
_, pvals_adjusted, _, _ = fdr.multipletests(df['pvalue_true'].values, alpha=p_adj_thr, method='fdr_bh', is_sorted=False)
df['pvalue_true'] = pvals_adjusted

df_sub = df[((df['true_CAGE']>=CAGE_min) & (df['n_contacts']>=n_min) & (df['pvalue_true'] <= p_adj_thr))]
print('len df_sub: ', len(df_sub))

sns.set_style("whitegrid")
fig, ax = plt.subplots()
g = sns.scatterplot(data=df_sub, x="log2FoldChange_true", y="log2FoldChange_GraphReg", alpha=.5, palette=plt.cm.get_cmap('viridis_r'), ax=ax)
ax.set_title('{}'.format(TF))
ax.set_xlabel('log2FC True')
ax.set_ylabel('log2FC GrapgReg')
plt.tight_layout()
plt.savefig('../figs/TF_KO/scatterplot_logfc_true_vs_graphreg_'+TF+'.pdf')
plt.close()

sns.set_style("whitegrid")
fig, ax = plt.subplots()
g = sns.scatterplot(data=df_sub, x="log2FoldChange_true", y="log2FoldChange_CNN", alpha=.5, palette=plt.cm.get_cmap('viridis_r'), ax=ax)
ax.set_title('{}'.format(TF))
ax.set_xlabel('log2FC True')
ax.set_ylabel('log2FC CNN')
plt.tight_layout()
plt.savefig('../figs/TF_KO/scatterplot_logfc_true_vs_cnn_'+TF+'.pdf')
plt.close()