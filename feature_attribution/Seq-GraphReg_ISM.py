from __future__ import division
from os import write
import sys
sys.path.insert(0,'../train')
  
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from gat_layer import GraphAttention
import numpy as np
import scipy as sp
import pandas as pd
import tensorflow as tf
import seaborn as sns
from tensorflow.keras import datasets, layers, models
from tensorflow import keras
import matplotlib.pyplot as plt
import time
import pyBigWig
from tensorflow.keras import backend as K

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
            pos1 = np.arange(start, end, 1).astype(int)
            pad = -np.flip(np.arange(1, 3000000+1, 1).astype(int))   # -3000000, ... , -2, -1
            pos = np.append(pad, pos1).astype(int)

        elif (num_zero == T//2+1 and bin_idx[0] == 0):
            start = bin_idx[T//2].numpy()
            end = bin_idx[-1].numpy()+5000
            pos1 = np.arange(start, end, 1).astype(int)
            pad = -np.flip(np.arange(1, 1000000+1, 1).astype(int))   # -1000000, -999999, ... , -2, -1
            pos = np.append(pad, pos1).astype(int)

        elif bin_idx[-1] == 0:
            start = bin_idx[0].numpy()
            i0 = np.where(bin_idx.numpy()==0)[0][0]
            end = bin_idx[i0-1].numpy()+5000
            pos1 = np.arange(start, end, 1).astype(int)
            l = 6000000 - len(pos1)
            pad = 10**15 * np.ones(l)
            pos = np.append(pos1, pad).astype(int)

        else:
            start = bin_idx[0].numpy()
            end = bin_idx[-1].numpy()+5000
            pos = np.arange(start, end, 1).astype(int)

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

def calculate_loss(model_cnn_base, model_gat, model_cnn, gene_names_list, gene_tss_list, chr_list, cell_line, organism, genome, batch_size, write_bw, condition, JUND_positions):

    y_gene = np.array([])
    y_hat_gene_gat = np.array([])
    y_hat_gene_cnn = np.array([])
    n_contacts = np.array([])

    chroms = np.array([])
    starts = np.array([])
    ends = np.array([])
    T = 400

    for i, chr in enumerate(chr_list):
        print('chr: ', chr)
        chrm = 'chr'+str(chr)
        file_name = data_path+'/data/tfrecords/tfr_seq_'+cell_line+'_'+assay_type+'_FDR_'+FDR+'_chr'+str(chr)+'.tfr'
        iterator = dataset_iterator(file_name, batch_size)
        tss_pos = np.load(data_path+'/data/tss/'+organism+'/'+genome+'/tss_pos_chr'+str(chr)+'.npy', allow_pickle=True)
        gene_names_all = np.load(data_path+'/data/tss/'+organism+'/'+genome+'/tss_gene_chr'+str(chr)+'.npy', allow_pickle=True)
        tss_pos = tss_pos[tss_pos>0]
        gene_names_all = gene_names_all[gene_names_all != ""]

        print('tss_pos: ', len(tss_pos), tss_pos[0:10])
        print('gene_names_all: ', len(gene_names_all), gene_names_all[0:10])

        while True:
            data_exist, seq, X_epi, Y, adj, idx, tss_idx, pos, last_batch = read_tf_record_1shot(iterator)

            if data_exist:
                if tf.reduce_sum(tf.gather(tss_idx, idx)) > 0:
                    if (pos[-1] < 1e15  and gene_tss_list[i] >= pos[0]+2000000 and gene_tss_list[i] < pos[0]+4000000):
                        if type(model_cnn_base) is list:
                            Y_hat_cnn, _, _, _, _ = model_cnn(seq)
                            Y_hat_gat, _, _, _, _, _ = model_gat([seq, adj])
                        else:
                            H = []
                            for jj in range(0,60,10):
                                seq_batch = seq[jj:jj+10]
                                _,_,_, h = model_cnn_base(seq_batch)
                                H.append(h)
                            x_in = K.concatenate(H, axis = 0)
                            x_in = K.reshape(x_in, [1, 60000, 64])

                            Y_hat_cnn = model_cnn(x_in)
                            Y_hat_gat, _ = model_gat([x_in, adj])

                        explain_output_idx = np.floor((gene_tss_list[i]-pos[0])/5000).astype('int64')

                        y_gene = Y.numpy().ravel()[explain_output_idx]
                        y_hat_gene_gat = Y_hat_gat.numpy().ravel()[explain_output_idx]
                        y_hat_gene_cnn = Y_hat_cnn.numpy().ravel()[explain_output_idx]

                        row_sum = tf.squeeze(tf.reduce_sum(adj, axis=-1))
                        n_contacts = row_sum[explain_output_idx]

                        ################ ISM ################

                        ISM_scores_gat = np.zeros([60,100000,4])
                        ISM_scores_cnn = np.zeros([60,100000,4])
                        I = np.eye(4)
                        for k in range(60):
                            for j in range(100000):
                                bp_position = k*100000 + j + pos[0]
                                dist_from_motif = np.min(np.abs(bp_position - JUND_positions))
                                if dist_from_motif <=50:
                                    print('bp_position: ', bp_position)
                                    seq_np = np.copy(seq.numpy())
                                    for r in range(4):
                                        seq_np[k,j,:] = I[r]
                                        seq_mut = tf.convert_to_tensor(seq_np)

                                        if type(model_cnn_base) is list:
                                            Y_hat_cnn_ism, _, _, _, _ = model_cnn(seq_mut)
                                            Y_hat_gat_ism, _, _, _, _, _ = model_gat([seq_mut, adj])
                                        else:
                                            H = []
                                            for jj in range(0,60,10):
                                                seq_batch = seq_mut[jj:jj+10]
                                                _,_,_, h = model_cnn_base(seq_batch)
                                                H.append(h)
                                            x_in = K.concatenate(H, axis = 0)
                                            x_in = K.reshape(x_in, [1, 60000, 64])

                                            Y_hat_cnn_ism = model_cnn(x_in)
                                            Y_hat_gat_ism, _ = model_gat([x_in, adj])

                                        y_hat_gene_gat_ism = Y_hat_gat_ism.numpy().ravel()[explain_output_idx]
                                        y_hat_gene_cnn_ism = Y_hat_cnn_ism.numpy().ravel()[explain_output_idx]

                                        ISM_scores_gat[k,j,r] = y_hat_gene_gat_ism - y_hat_gene_gat
                                        ISM_scores_cnn[k,j,r] = y_hat_gene_cnn_ism - y_hat_gene_cnn
                                else:
                                    ISM_scores_gat[k,j,:] = 0
                                    ISM_scores_cnn[k,j,:] = 0
                            

                        sparse_matrix_gat_ism = sp.sparse.csr_matrix(ISM_scores_gat.flatten())
                        sp.sparse.save_npz(data_path+'/results/numpy/feature_attribution/Seq-GraphReg_BP_ISM_'+condition+'_'+cell_line+'_'+gene_names_list[i]+'.npz', sparse_matrix_gat_ism)

                        sparse_matrix_cnn_ism = sp.sparse.csr_matrix(ISM_scores_cnn.flatten())
                        sp.sparse.save_npz(data_path+'/results/numpy/feature_attribution/Seq-CNN_BP_ISM_'+condition+'_'+cell_line+'_'+gene_names_list[i]+'.npz', sparse_matrix_cnn_ism)
                        
                        ISM_scores_gat_sum = -np.sum(ISM_scores_gat, axis=2)
                        ISM_scores_gat_sum = np.reshape(ISM_scores_gat_sum, [6000000])

                        ISM_scores_cnn_sum = -np.sum(ISM_scores_cnn, axis=2)
                        ISM_scores_cnn_sum = np.reshape(ISM_scores_cnn_sum, [6000000])

                        ########## Write bigwig saliency files #########
                        if write_bw == True and organism == 'human' and genome == 'hg19':
                                header = [("chr1", 249250621), ("chr2", 243199373), ("chr3", 198022430), ("chr4", 191154276), ("chr5", 180915260), ("chr6", 171115067),
                                            ("chr7", 159138663), ("chr8", 146364022), ("chr9", 141213431), ("chr10", 135534747), ("chr11", 135006516), ("chr12", 133851895),
                                            ("chr13", 115169878), ("chr14", 107349540), ("chr15", 102531392), ("chr16", 90354753), ("chr17", 81195210), ("chr18", 78077248),
                                            ("chr19", 59128983), ("chr20", 63025520), ("chr21", 48129895), ("chr22", 51304566)]

                        if write_bw == True and organism == 'human' and genome == 'hg38':
                                header = [("chr1", 248956422), ("chr2", 242193529), ("chr3", 198295559), ("chr4", 190214555), ("chr5", 181538259), ("chr6", 170805979),
                                            ("chr7", 159345973), ("chr8", 145138636), ("chr9", 138394717), ("chr10", 133797422), ("chr11", 135086622), ("chr12", 133275309),
                                            ("chr13", 114364328), ("chr14", 107043718), ("chr15", 101991189), ("chr16", 90338345), ("chr17", 83257441), ("chr18", 80373285),
                                            ("chr19", 58617616), ("chr20", 64444167), ("chr21", 46709983), ("chr22", 50818468)]

                        if write_bw == True and organism == 'mouse':
                                header = [("chr1", 195465000), ("chr2", 182105000), ("chr3", 160030000), ("chr4", 156500000), ("chr5", 151825000), ("chr6", 149730000),
                                        ("chr7", 145435000), ("chr8", 129395000), ("chr9", 124590000), ("chr10", 130685000), ("chr11", 122075000), ("chr12", 120120000),
                                        ("chr13", 120415000), ("chr14", 124895000), ("chr15", 104035000), ("chr16", 98200000), ("chr17", 94980000), ("chr18", 90695000),
                                        ("chr19", 61425000)]
   

                        if write_bw == True:
                            bw_GraphReg = pyBigWig.open(data_path+'/results/bigwig/feature_attribution/Seq-models_basepair/Seq-GraphReg_BP_ISM_'+condition+'_'+cell_line+'_'+gene_names_list[i]+'.bw', "w")
                            bw_GraphReg.addHeader(header)
                            bw_CNN = pyBigWig.open(data_path+'/results/bigwig/feature_attribution/Seq-models_basepair/Seq-CNN_BP_ISM_'+condition+'_'+cell_line+'_'+gene_names_list[i]+'.bw', "w")
                            bw_CNN.addHeader(header)

                            starts = pos.astype(np.int64)
                            idx_pos = np.where(starts>0)[0]
                            starts = starts[idx_pos]
                            ends = starts + 1
                            ends = ends.astype(np.int64)
                            chroms = np.array([chrm] * len(idx_pos))

                            bw_GraphReg.addEntries(chroms, starts, ends=ends, values=ISM_scores_gat_sum.ravel()[idx_pos])
                            bw_CNN.addEntries(chroms, starts, ends=ends, values=ISM_scores_cnn_sum.ravel()[idx_pos])

                            bw_GraphReg.close()
                            bw_CNN.close()
                        #break
            else:
                break

    return y_gene, y_hat_gene_gat, y_hat_gene_cnn, n_contacts, ISM_scores_gat, ISM_scores_cnn


##### load models #####

data_path = '/media/labuser/STORAGE/GraphReg'
cell_line = 'K562'
organism = 'human'
genome = 'hg19'
assay_type = 'HiChIP'
fdr = '1'
valid_chr_str = '9,19'
test_chr_str = '10,20'
batch_size = 1
write_bw = True
e2e = False
gene_names_list = ['TCF3']

if e2e:
    condition = 'e2e'
else:
    condition = 'fft'

if organism == 'human' and genome == 'hg19':
    filename_tss = data_path+'/data/tss/'+organism+'/'+genome+'/hg19_gencodev19_tss.bed'
elif organism == 'human' and genome == 'hg38':
    filename_tss = data_path+'/data/tss/'+organism+'/'+genome+'/gencode.v38.annotation.gtf.tss.bed'
elif organism == 'mouse':
    filename_tss = data_path+'/data/tss/'+organism+'/'+genome+'/mm10_gencode_vM9_tss.bed'

tss_dataframe = pd.read_csv(filename_tss, header=None, delimiter='\t')
tss_dataframe.columns = ["chr", "tss_1", "tss_2", "ens", "gene", "strand", "type"]

gene_tss_list = []
chr_list = []

for gene in gene_names_list:
    df = tss_dataframe[tss_dataframe["gene"] == gene]
    gene_tss_list.append(df['tss_1'].values[0])
    chr_list.append(int(df['chr'].values[0][3:]))


if not e2e:
    model_name_cnn_base = data_path+'/models/'+cell_line+'/Seq-CNN_base_nodilation_fft_'+cell_line+'_valid_chr_'+valid_chr_str+'_test_chr_'+test_chr_str+'.h5'
    model_name_gat = data_path+'/models/'+cell_line+'/Seq-GraphReg_nodilation_fft_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'_valid_chr_'+valid_chr_str+'_test_chr_'+test_chr_str+'.h5'
    model_name_cnn = data_path+'/models/'+cell_line+'/Seq-CNN_nodilation_fft_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'_valid_chr_'+valid_chr_str+'_test_chr_'+test_chr_str+'.h5'
    
    model_cnn_base = tf.keras.models.load_model(model_name_cnn_base)
    model_cnn_base.trainable = False
    model_cnn_base._name = 'Seq-CNN_base'
    
    model_gat = tf.keras.models.load_model(model_name_gat, custom_objects={'GraphAttention': GraphAttention})
    model_gat.trainable = False
    model_gat._name = 'Seq-GraphReg'

    model_cnn = tf.keras.models.load_model(model_name_cnn)
    model_cnn.trainable = False
    model_cnn._name = 'Seq-CNN'

else:
    model_name_gat = data_path+'/models/'+cell_line+'/Seq-GraphReg_e2e_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'_valid_chr_'+valid_chr_str+'_test_chr_'+test_chr_str+'.h5'
    model_gat = tf.keras.models.load_model(model_name_gat, custom_objects={'GraphAttention': GraphAttention})
    model_gat.trainable = False
    model_gat._name = 'Seq-GraphReg_e2e'

    model_name = data_path+'/models/'+cell_line+'/Seq-CNN_e2e_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'_valid_chr_'+valid_chr_str+'_test_chr_'+test_chr_str+'.h5'
    model_cnn = tf.keras.models.load_model(model_name)
    model_cnn.trainable = False
    model_cnn._name = 'Seq-CNN_e2e'

    model_cnn_base = []

TF_positions_df = pd.read_csv(data_path+'/results/fimo/peaks_H3K27ac_K562/TF_positions_unique.bed', sep="\t")
TF_positions_df.columns = ['chr', 'start', 'end', 'TF', '-log10(pval)', 'strand']
df_JUND = TF_positions_df[(TF_positions_df['TF']=='JUND') & (TF_positions_df['chr']=='chr19') & (TF_positions_df['start']<4000000)]
JUND_positions = df_JUND['start'].values
print('len JUND TF motifs: ', len(JUND_positions))

y_gene, y_hat_gene_gat, y_hat_gene_cnn, n_contacts, ISM_scores_gat, ISM_scores_cnn = calculate_loss(model_cnn_base, model_gat, model_cnn, gene_names_list, gene_tss_list, chr_list, cell_line, organism, genome, batch_size, write_bw, condition, JUND_positions)


##### plot ISM heatmaps #####
sparse_matrix_gat_ism = sp.sparse.load_npz(data_path+'/results/numpy/feature_attribution/Seq-GraphReg_BP_ISM_'+condition+'_'+cell_line+'_'+gene_names_list[0]+'.npz')
dense_matrix_gat_ism = np.array(sparse_matrix_gat_ism.todense())
dense_matrix_gat_ism = np.reshape(dense_matrix_gat_ism, [60, 100000, 4])

pos0 = -1000000
candidate_region_start = 2945347
candidate_region_end = 2945447

K = (candidate_region_start - pos0) // 100000
candidate_region_start_j = candidate_region_start%100000
candidate_region_end_j = candidate_region_end%100000

ISM_candidate_region = dense_matrix_gat_ism[K, candidate_region_start_j:candidate_region_end_j-1, :]
ISM_candidate_region = ISM_candidate_region.transpose()
ISM_candidate_region = np.roll(ISM_candidate_region, 1, axis=0)
df_ISM = pd.DataFrame(data=ISM_candidate_region, index=['A', 'C', 'G', 'T'])

fig, ax1 = plt.subplots(figsize=(20, 2))
min_ISM = np.min(ISM_candidate_region)
sns.heatmap(df_ISM, cmap="coolwarm", xticklabels=False, ax=ax1, vmax=-min_ISM, vmin=min_ISM,
            cbar_kws={"orientation": "vertical"})
plt.tight_layout()
fig.savefig('../figs/Seq-models/final/scatterplot/heatmap_ISM_TCF3_enhancer_chr19_2945347-2945447.pdf', bbox_inches='tight')
