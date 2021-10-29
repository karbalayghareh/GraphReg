from __future__ import division
import sys
import os
sys.path.insert(0,'../train')

from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from gat_layer import GraphAttention
import numpy as np
import pandas as pd
import seaborn as sns
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
from adjustText import adjust_text
import matplotlib.patches as mpatches
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap
from statannot import add_stat_annotation

##### Input 
batch_size = 1
organism = 'human'            # human/mouse
genome='hg19'                 # hg19/hg38/mm10
cell_line = 'K562'            # K562/GM12878/mESC/hESC
write_bw = False              # write the predicted CAGE to bigwig files
prediction = True
logfold = False
load_np = False
plot_violin = False
plot_box = False
plot_scatter = False
save_R_NLL_to_csv = True
data_path = '/media/labuser/STORAGE/GraphReg'   # data path
qval = .1                                       # 0.1, 0.01, 0.001
assay_type = 'HiChIP'                           # HiChIP, HiC, MicroC, HiCAR

if qval == 0.1:
    fdr = '1'
elif qval == 0.01:
    fdr = '01'
elif qval == 0.001:
    fdr = '001'


def log2(x):
  numerator = tf.math.log(x)
  denominator = tf.math.log(tf.constant(2.))
  return numerator / denominator

def poisson_loss(y_true, mu_pred):
    nll = tf.reduce_mean(tf.math.lgamma(y_true + 1) + mu_pred - y_true * tf.math.log(mu_pred))
    return nll

def poisson_loss_individual(y_true, mu_pred):
    nll = tf.math.lgamma(y_true + 1) + mu_pred - y_true * tf.math.log(mu_pred)
    return nll

def parse_proto(example_protos):
      features = {
        'last_batch': tf.io.FixedLenFeature([1], tf.int64),
        'adj': tf.io.FixedLenFeature([], tf.string),
        #'adj_real': tf.io.FixedLenFeature([], tf.string),
        'tss_idx': tf.io.FixedLenFeature([], tf.string),
        'X_1d': tf.io.FixedLenFeature([], tf.string),
        'Y': tf.io.FixedLenFeature([], tf.string),
        'bin_idx': tf.io.FixedLenFeature([], tf.string)
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

      return {'last_batch': last_batch, 'X_epi': X_epi, 'Y': Y, 'adj': adj, 'tss_idx': tss_idx, 'bin_idx': bin_idx}

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
        b = 50
        F = 3 
        X_epi = next_datum['X_epi']
        batch_size = tf.shape(X_epi)[0]
        X_epi = tf.reshape(X_epi, [batch_size, 3*T*b, F])
        X_avg = tf.reshape(X_epi, [3*T, b, F])
        X_avg = tf.reduce_mean(X_avg, axis=1)
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
            pos1 = np.arange(start, end, 100).astype(int)
            l = 60000 - len(pos1)
            pad = 10**15 * np.ones(l)
            pos = np.append(pos1, pad).astype(int)

        else:
            start = bin_idx[0].numpy()
            end = bin_idx[-1].numpy()+5000
            pos = np.arange(start, end, 100).astype(int)

        Y = next_datum['Y']
        Y = tf.reshape(Y, [batch_size, 3*T, b])
        Y = tf.reduce_sum(Y, axis=2)
        Y = tf.reshape(Y, [batch_size, 3*T])

    else:
        X_epi = 0
        X_avg = 0
        Y = 0
        adj = 0
        tss_idx = 0
        idx = 0
        pos = 0
        last_batch = 10
    return data_exist, X_epi, X_avg, Y, adj, idx, tss_idx, pos, last_batch

def calculate_loss(model_gat, model_cnn, chr_list, valid_chr, test_chr, cell_line, organism, genome, batch_size, write_bw):
    loss_gat_all = np.array([])
    loss_cnn_all = np.array([])
    Y_hat_gat_all = np.array([])
    Y_hat_cnn_all = np.array([])
    Y_all = np.array([])
    y_gene = np.array([])
    y_hat_gene_gat = np.array([])
    y_hat_gene_cnn = np.array([])
    chr_pos = []
    gene_pos = np.array([])
    gene_names = np.array([])
    gene_tss = np.array([])
    gene_chr = np.array([])
    n_contacts = np.array([])
    n_tss_in_bin = np.array([])

    x_h3k4me3 = np.array([])
    x_h3k27ac = np.array([])
    x_dnase = np.array([])

    y_bw = np.array([])
    y_pred_gat_bw = np.array([])
    y_pred_cnn_bw = np.array([])
    chroms = np.array([])
    starts = np.array([])
    ends = np.array([])
    T = 400
    test_chr_str = [str(i) for i in test_chr]
    test_chr_str = ','.join(test_chr_str)
    valid_chr_str = [str(i) for i in valid_chr]
    valid_chr_str = ','.join(valid_chr_str)

    
    if write_bw == True and organism == 'human' and genome == 'hg19':
        # human
        chr_length = [("chr1", 249250621), ("chr2", 243199373), ("chr3", 198022430), ("chr4", 191154276), ("chr5", 180915260), ("chr6", 171115067),
                        ("chr7", 159138663), ("chr8", 146364022), ("chr9", 141213431), ("chr10", 135534747), ("chr11", 135006516), ("chr12", 133851895),
                        ("chr13", 115169878), ("chr14", 107349540), ("chr15", 102531392), ("chr16", 90354753), ("chr17", 81195210), ("chr18", 78077248),
                        ("chr19", 59128983), ("chr20", 63025520), ("chr21", 48129895), ("chr22", 51304566)]
        bw_y_true = pyBigWig.open(data_path+'/results/bigwig/cage_prediction/Epi-models/CAGE_'+cell_line+'_valid_chr_'+valid_chr_str+'_test_chr_'+test_chr_str+'.bw', "w")
        bw_y_true.addHeader(chr_length)
        bw_y_pred_gat = pyBigWig.open(data_path+'/results/bigwig/cage_prediction/Epi-models/Epi-GraphReg_CAGE_pred_'+cell_line+'_valid_chr_'+valid_chr_str+'_test_chr_'+test_chr_str+'.bw', "w")
        bw_y_pred_gat.addHeader(chr_length)
        bw_y_pred_cnn = pyBigWig.open(data_path+'/results/bigwig/cage_prediction/Epi-models/Epi-CNN_CAGE_pred_'+cell_line+'_valid_chr_'+valid_chr_str+'_test_chr_'+test_chr_str+'.bw', "w")
        bw_y_pred_cnn.addHeader(chr_length)

    if write_bw == True and organism == 'human' and genome == 'hg38':
        # human
        chr_length = [("chr1", 248956422), ("chr2", 242193529), ("chr3", 198295559), ("chr4", 190214555), ("chr5", 181538259), ("chr6", 170805979),
                        ("chr7", 159345973), ("chr8", 145138636), ("chr9", 138394717), ("chr10", 133797422), ("chr11", 135086622), ("chr12", 133275309),
                        ("chr13", 114364328), ("chr14", 107043718), ("chr15", 101991189), ("chr16", 90338345), ("chr17", 83257441), ("chr18", 80373285),
                        ("chr19", 58617616), ("chr20", 64444167), ("chr21", 46709983), ("chr22", 50818468)]
        bw_y_true = pyBigWig.open(data_path+'/results/bigwig/cage_prediction/Epi-models/CAGE_'+cell_line+'_valid_chr_'+valid_chr_str+'_test_chr_'+test_chr_str+'.bw', "w")
        bw_y_true.addHeader(chr_length)
        bw_y_pred_gat = pyBigWig.open(data_path+'/results/bigwig/cage_prediction/Epi-models/Epi-GraphReg_CAGE_pred_'+cell_line+'_valid_chr_'+valid_chr_str+'_test_chr_'+test_chr_str+'.bw', "w")
        bw_y_pred_gat.addHeader(chr_length)
        bw_y_pred_cnn = pyBigWig.open(data_path+'/results/bigwig/cage_prediction/Epi-models/Epi-CNN_CAGE_pred_'+cell_line+'_valid_chr_'+valid_chr_str+'_test_chr_'+test_chr_str+'.bw', "w")
        bw_y_pred_cnn.addHeader(chr_length)

    if write_bw == True and organism == 'mouse':
        # mouse:
        chr_length = [("chr1", 195465000), ("chr2", 182105000), ("chr3", 160030000), ("chr4", 156500000), ("chr5", 151825000), ("chr6", 149730000),
                                        ("chr7", 145435000), ("chr8", 129395000), ("chr9", 124590000), ("chr10", 130685000), ("chr11", 122075000), ("chr12", 120120000),
                                        ("chr13", 120415000), ("chr14", 124895000), ("chr15", 104035000), ("chr16", 98200000), ("chr17", 94980000), ("chr18", 90695000),
                                        ("chr19", 61425000)]
        bw_y_true = pyBigWig.open(data_path+'/results/bigwig/cage_prediction/Epi-models/CAGE_'+cell_line+'_valid_chr_'+valid_chr_str+'_test_chr_'+test_chr_str+'.bw', "w")
        bw_y_true.addHeader(chr_length)
        bw_y_pred_gat = pyBigWig.open(data_path+'/results/bigwig/cage_prediction/Epi-models/Epi-GraphReg_CAGE_pred_'+cell_line+'_valid_chr_'+valid_chr_str+'_test_chr_'+test_chr_str+'.bw', "w")
        bw_y_pred_gat.addHeader(chr_length)
        bw_y_pred_cnn = pyBigWig.open(data_path+'/results/bigwig/cage_prediction/Epi-models/Epi-CNN_CAGE_pred_'+cell_line+'_valid_chr_'+valid_chr_str+'_test_chr_'+test_chr_str+'.bw', "w")
        bw_y_pred_cnn.addHeader(chr_length)

    for i in chr_list:
        print('chr :', i)
        file_name = data_path+'/data/tfrecords/tfr_epi_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'_chr'+str(i)+'.tfr'
        iterator = dataset_iterator(file_name, batch_size)
        tss_pos = np.load(data_path+'/data/tss/'+organism+'/'+genome+'/tss_pos_chr'+str(i)+'.npy', allow_pickle=True)
        gene_names_all = np.load(data_path+'/data/tss/'+organism+'/'+genome+'/tss_gene_chr'+str(i)+'.npy', allow_pickle=True)
        n_tss = np.load(data_path+'/data/tss/'+organism+'/'+genome+'/tss_bins_chr'+str(i)+'.npy', allow_pickle=True)

        tss_pos = tss_pos[tss_pos>0]
        print('tss_pos: ', len(tss_pos), tss_pos[0:10])
        gene_names_all = gene_names_all[gene_names_all != ""]
        print('gene_names_all: ', len(gene_names_all), gene_names_all[0:10])
        n_tss = n_tss[n_tss>=1]
        print('n_tss: ', len(n_tss), n_tss[0:10])

        pos_bw = np.array([])
        y_bw_ = np.array([])
        y_pred_gat_bw_ = np.array([])
        y_pred_cnn_bw_ = np.array([])
        print(tss_pos.shape[0], tss_pos[0:100])
        while True:
            data_exist, X_epi, X_avg, Y, adj, idx, tss_idx, pos, last_batch = read_tf_record_1shot(iterator)

            ### Creating BigWig files for true and predicted CAGE tracks ###
            if write_bw == True:
                if data_exist:
                    pos_mid = pos[20000:40000]
                    if (pos_mid[-1] < 10**15):
                        pos_bw = np.append(pos_bw, pos_mid)
                        Y_hat_cnn = model_cnn(X_epi)
                        Y_hat_gat, att = model_gat([X_epi, adj])
                        #att_1 = att[0]
                        #att_2 = att[1]
                        #att_3 = att[2]

                        Y_idx = tf.gather(Y, tf.range(T, 2*T), axis=1)
                        Y_hat_cnn_idx = tf.gather(Y_hat_cnn, tf.range(T, 2*T), axis=1)
                        Y_hat_gat_idx = tf.gather(Y_hat_gat, tf.range(T, 2*T), axis=1)

                        y1 = np.repeat(Y_idx.numpy().ravel(), 50)
                        y2 = np.repeat(Y_hat_gat_idx.numpy().ravel(), 50)
                        y3 = np.repeat(Y_hat_cnn_idx.numpy().ravel(), 50)
                        y_bw_ = np.append(y_bw_, y1)
                        y_pred_gat_bw_= np.append(y_pred_gat_bw_, y2)
                        y_pred_cnn_bw_ = np.append(y_pred_cnn_bw_, y3)

            if data_exist:
                if tf.reduce_sum(tf.gather(tss_idx, idx)) > 0:
                    Y_hat_cnn = model_cnn(X_epi)
                    Y_hat_gat, att = model_gat([X_epi, adj])
                    #att_1 = att[0]
                    #att_2 = att[1]
                    #att_3 = att[2]

                    Y_idx = tf.gather(Y, idx, axis=1)
                    Y_hat_cnn_idx = tf.gather(Y_hat_cnn, idx, axis=1)
                    Y_hat_gat_idx = tf.gather(Y_hat_gat, idx, axis=1)

                    loss_gat = poisson_loss(Y_idx,Y_hat_gat_idx)
                    loss_cnn = poisson_loss(Y_idx,Y_hat_cnn_idx)
                    loss_gat_all = np.append(loss_gat_all, loss_gat.numpy())
                    loss_cnn_all = np.append(loss_cnn_all, loss_cnn.numpy())

                    Y_hat_gat_all = np.append(Y_hat_gat_all, Y_hat_gat_idx.numpy().ravel())
                    Y_hat_cnn_all = np.append(Y_hat_cnn_all, Y_hat_cnn_idx.numpy().ravel())
                    Y_all = np.append(Y_all, Y_idx.numpy().ravel())

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
                        idx_gene = np.where(tss_pos == tss_pos_1[j])[0]
                        
                        y_true_ = np.repeat(Y.numpy().ravel(), 50)
                        y_hat_gat_ = np.repeat(Y_hat_gat.numpy().ravel(), 50)
                        y_hat_cnn_ = np.repeat(Y_hat_cnn.numpy().ravel(), 50)

                        x_h3k4me3_ = np.repeat(X_avg.numpy()[:,0], 50)
                        x_h3k27ac_ = np.repeat(X_avg.numpy()[:,1], 50)
                        x_dnase_ = np.repeat(X_avg.numpy()[:,2], 50)

                        y_gene = np.append(y_gene, y_true_[idx_tss])                    # + y_true_[idx_tss-50] + y_true_[idx_tss+50])
                        y_hat_gene_gat = np.append(y_hat_gene_gat, y_hat_gat_[idx_tss]) # + y_hat_gat_[idx_tss-50] + y_hat_gat_[idx_tss+50])
                        y_hat_gene_cnn = np.append(y_hat_gene_cnn, y_hat_cnn_[idx_tss]) # + y_hat_cnn_[idx_tss-50] + y_hat_cnn_[idx_tss+50])
                        gene_pos = np.append(gene_pos, 'chr'+str(i)+'_tss_'+str(tss_pos_1[j]))
                        gene_tss = np.append(gene_tss, tss_pos_1[j])
                        gene_chr = np.append(gene_chr, 'chr'+str(i))
                        gene_names = np.append(gene_names, gene_names_all[idx_gene]) 
                        n_tss_in_bin = np.append(n_tss_in_bin, n_tss[idx_gene])
                        n_contacts = np.append(n_contacts, num_contacts[idx_tss])

                        x_h3k4me3 = np.append(x_h3k4me3, x_h3k4me3_[idx_tss])
                        x_h3k27ac = np.append(x_h3k27ac, x_h3k27ac_[idx_tss])
                        x_dnase = np.append(x_dnase, x_dnase_[idx_tss])

            else:
                if write_bw == True:
                    assert len(pos_bw) == len(y_bw_) == len(y_pred_gat_bw_)
                    chroms_ = np.array(["chr"+str(i)] * len(pos_bw))
                    print('chr'+str(i), len(pos_bw))
                    starts_ = pos_bw.astype(np.int64)
                    ends_ = starts_ + 100
                    ends_ = ends_.astype(np.int64)

                    chroms = np.append(chroms, chroms_)
                    starts = np.append(starts, starts_)
                    ends = np.append(ends, ends_)
                    y_bw = np.append(y_bw, y_bw_)
                    y_pred_gat_bw = np.append(y_pred_gat_bw, y_pred_gat_bw_)
                    y_pred_cnn_bw = np.append(y_pred_cnn_bw, y_pred_cnn_bw_)
                break

    if write_bw == True:
        starts = starts.astype(np.int64)
        idx_pos = np.where(starts>0)[0]
        starts = starts[idx_pos]
        ends = ends.astype(np.int64)[idx_pos]
        y_bw = y_bw.astype(np.float64)[idx_pos]
        y_pred_gat_bw = y_pred_gat_bw.astype(np.float64)[idx_pos]
        y_pred_cnn_bw = y_pred_cnn_bw.astype(np.float64)[idx_pos]
        chroms = chroms[idx_pos]
        print(len(chroms), len(starts), len(ends), len(y_bw))
        print(chroms)
        print(starts)
        print(ends)
        print(y_bw)
        bw_y_true.addEntries(chroms, starts, ends=ends, values=y_bw)
        bw_y_pred_gat.addEntries(chroms, starts, ends=ends, values=y_pred_gat_bw)
        bw_y_pred_cnn.addEntries(chroms, starts, ends=ends, values=y_pred_cnn_bw)
        bw_y_true.close()
        bw_y_pred_gat.close()
        bw_y_pred_cnn.close()

    print('len of test/valid Y: ', len(y_gene))
    return y_gene, y_hat_gene_gat, y_hat_gene_cnn, chr_pos, gene_pos, gene_names, gene_tss, gene_chr, n_contacts, n_tss_in_bin, x_h3k4me3, x_h3k27ac, x_dnase 


############################################################# load model #############################################################


def set_axis_style(ax, labels, positions_tick):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_tick_params(labelsize=15)
    ax.set_xticks(positions_tick)
    ax.set_xticklabels(labels, fontsize=20)

def add_label(violin, labels, label):
    color = violin["bodies"][0].get_facecolor().flatten()
    labels.append((mpatches.Patch(color=color), label))

if prediction == True:
    valid_loss_gat = np.zeros([10,4])
    valid_rho_gat = np.zeros([10,4])
    valid_sp_gat = np.zeros([10,4])
    valid_loss_cnn = np.zeros([10,4])
    valid_rho_cnn = np.zeros([10,4])
    valid_sp_cnn = np.zeros([10,4])
    n_gene = np.zeros([10,4])
    df_all_predictions = pd.DataFrame(columns=['chr', 'genes', 'n_tss', 'tss', 'tss_distance_from_center', 'n_contact', 'average_dnase', 'average_h3k27ac', 'average_h3k4me3', 'true_cage', 'pred_cage_epi_graphreg', 'pred_cage_epi_cnn', 'nll_epi_graphreg', 'nll_epi_cnn', 'delta_nll'])

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
        chr_list = test_chr_list.copy()
        chr_list.sort()

        test_chr_str = [str(i) for i in test_chr_list]
        test_chr_str = ','.join(test_chr_str)
        valid_chr_str = [str(i) for i in valid_chr_list]
        valid_chr_str = ','.join(valid_chr_str)

        if load_np == True:
            y_gene = np.load(data_path+'/results/numpy/cage_prediction/true_cage_'+cell_line+'_'+str(i)+'.npy')
            y_hat_gene_gat = np.load(data_path+'/results/numpy/cage_prediction/Epi-GraphReg_predicted_cage_'+cell_line+'_'+str(i)+'.npy')
            y_hat_gene_cnn = np.load(data_path+'/results/numpy/cage_prediction/Epi-CNN_predicted_cage_'+cell_line+'_'+str(i)+'.npy')
            n_contacts = np.load(data_path+'/results/numpy/cage_prediction/n_contacts_'+cell_line+'_'+str(i)+'.npy')
            gene_names = np.load(data_path+'/results/numpy/cage_prediction/gene_names_'+cell_line+'_'+str(i)+'.npy')
            gene_tss = np.load(data_path+'/results/numpy/cage_prediction/gene_tss_'+cell_line+'_'+str(i)+'.npy')
            gene_chr = np.load(data_path+'/results/numpy/cage_prediction/gene_chr_'+cell_line+'_'+str(i)+'.npy')
        else:
            model_name_gat = data_path+'/models/'+cell_line+'/Epi-GraphReg_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'_valid_chr_'+valid_chr_str+'_test_chr_'+test_chr_str+'.h5'
            model_gat = tf.keras.models.load_model(model_name_gat, custom_objects={'GraphAttention': GraphAttention})
            model_gat.trainable = False
            model_gat._name = 'Epi-GraphReg'
            #model_gat.summary()

            model_name = data_path+'/models/'+cell_line+'/Epi-CNN_'+cell_line+'_valid_chr_'+valid_chr_str+'_test_chr_'+test_chr_str+'.h5'
            model_cnn = tf.keras.models.load_model(model_name)
            model_cnn.trainable = False
            model_cnn._name = 'Epi-CNN'
            #model_cnn.summary()

            y_gene, y_hat_gene_gat, y_hat_gene_cnn, _, _, gene_names, gene_tss, gene_chr, n_contacts, n_tss_in_bin, x_h3k4me3, x_h3k27ac, x_dnase = calculate_loss(model_gat, model_cnn, 
                    chr_list, valid_chr_list, test_chr_list, cell_line, organism, genome, batch_size, write_bw)

            np.save(data_path+'/results/numpy/cage_prediction/true_cage_'+cell_line+'_'+str(i)+'.npy', y_gene)
            np.save(data_path+'/results/numpy/cage_prediction/Epi-GraphReg_predicted_cage_'+cell_line+'_'+str(i)+'.npy', y_hat_gene_gat)
            np.save(data_path+'/results/numpy/cage_prediction/Epi-CNN_predicted_cage_'+cell_line+'_'+str(i)+'.npy', y_hat_gene_cnn)
            np.save(data_path+'/results/numpy/cage_prediction/n_contacts_'+cell_line+'_'+str(i)+'.npy', n_contacts)
            np.save(data_path+'/results/numpy/cage_prediction/gene_names_'+cell_line+'_'+str(i)+'.npy', gene_names)
            np.save(data_path+'/results/numpy/cage_prediction/gene_tss_'+cell_line+'_'+str(i)+'.npy', gene_tss)
            np.save(data_path+'/results/numpy/cage_prediction/gene_chr_'+cell_line+'_'+str(i)+'.npy', gene_chr)

        df_tmp = pd.DataFrame(columns=['chr', 'genes', 'n_tss', 'tss', 'tss_distance_from_center', 'n_contact', 'average_dnase', 'average_h3k27ac', 'average_h3k4me3', 'true_cage', 'pred_cage_epi_graphreg', 'pred_cage_epi_cnn', 'nll_epi_graphreg', 'nll_epi_cnn', 'delta_nll'])
        df_tmp['chr'] = gene_chr
        df_tmp['genes'] = gene_names
        df_tmp['n_tss'] = n_tss_in_bin.astype(np.int64)
        df_tmp['tss'] = gene_tss.astype(np.int64)
        df_tmp['tss_distance_from_center'] = np.abs(np.mod(gene_tss, 5000) - 2500).astype(np.int64)
        df_tmp['n_contact'] = n_contacts.astype(np.int64)
        df_tmp['average_dnase'] = x_dnase
        df_tmp['average_h3k27ac'] = x_h3k27ac
        df_tmp['average_h3k4me3'] = x_h3k4me3
        df_tmp['true_cage'] = y_gene
        df_tmp['pred_cage_epi_graphreg'] = y_hat_gene_gat
        df_tmp['pred_cage_epi_cnn'] = y_hat_gene_cnn
        df_tmp['nll_epi_graphreg'] = poisson_loss_individual(y_gene, y_hat_gene_gat).numpy()
        df_tmp['nll_epi_cnn'] = poisson_loss_individual(y_gene, y_hat_gene_cnn).numpy()
        df_tmp['delta_nll'] = poisson_loss_individual(y_gene, y_hat_gene_cnn).numpy() - poisson_loss_individual(y_gene, y_hat_gene_gat).numpy()    # if delta_nll > 0 then GraphReg prediction is better than CNN

        df_all_predictions = df_all_predictions.append(df_tmp).reset_index(drop=True)

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
            y_hat_gene_gat_idx = y_hat_gene_gat[idx]
            y_hat_gene_cnn_idx = y_hat_gene_cnn[idx]

            valid_loss_gat[i-1,j] = poisson_loss(y_gene_idx, y_hat_gene_gat_idx).numpy()
            valid_rho_gat[i-1,j] = np.corrcoef(np.log2(y_gene_idx+1),np.log2(y_hat_gene_gat_idx+1))[0,1]
            valid_sp_gat[i-1,j] = spearmanr(np.log2(y_gene_idx+1),np.log2(y_hat_gene_gat_idx+1))[0]
            valid_loss_cnn[i-1,j] = poisson_loss(y_gene_idx, y_hat_gene_cnn_idx).numpy()
            valid_rho_cnn[i-1,j] = np.corrcoef(np.log2(y_gene_idx+1),np.log2(y_hat_gene_cnn_idx+1))[0,1]
            valid_sp_cnn[i-1,j] = spearmanr(np.log2(y_gene_idx+1), np.log2(y_hat_gene_cnn_idx+1))[0]

            n_gene[i-1,j] = len(y_gene_idx)

            print('NLL GAT: ', valid_loss_gat, ' rho: ', valid_rho_gat, ' sp: ', valid_sp_gat)
            print('NLL CNN: ', valid_loss_cnn, ' rho: ', valid_rho_cnn, ' sp: ', valid_sp_cnn)

    print('Mean Loss GAT: ', np.mean(valid_loss_gat, axis=0), ' +/- ', np.std(valid_loss_gat, axis=0), ' std')
    print('Mean Loss CNN: ', np.mean(valid_loss_cnn, axis=0), ' +/- ', np.std(valid_loss_cnn, axis=0), ' std \n')

    print('Mean R GAT: ', np.mean(valid_rho_gat, axis=0), ' +/- ', np.std(valid_rho_gat, axis=0), ' std')
    print('Mean R CNN: ', np.mean(valid_rho_cnn, axis=0), ' +/- ', np.std(valid_rho_cnn, axis=0), ' std \n')

    print('Mean SP GAT: ', np.mean(valid_sp_gat, axis=0), ' +/- ', np.std(valid_sp_gat, axis=0), ' std')
    print('Mean SP CNN: ', np.mean(valid_sp_cnn, axis=0), ' +/- ', np.std(valid_sp_cnn, axis=0), ' std')

    w_loss = np.zeros(4)
    w_rho = np.zeros(4)
    w_sp = np.zeros(4)
    p_loss = np.zeros(4)
    p_rho = np.zeros(4)
    p_sp = np.zeros(4)
    for j in range(4):
        w_loss[j], p_loss[j] = wilcoxon(valid_loss_gat[:,j], valid_loss_cnn[:,j], alternative='less')
        w_rho[j], p_rho[j] = wilcoxon(valid_rho_gat[:,j], valid_rho_cnn[:,j], alternative='greater')
        w_sp[j], p_sp[j] = wilcoxon(valid_sp_gat[:,j], valid_sp_cnn[:,j], alternative='greater')

    print('Wilcoxon Loss: ', w_loss, ' , p_values: ', p_loss)
    print('Wilcoxon R: ', w_rho, ' , p_values: ', p_rho)
    print('Wilcoxon SP: ', w_sp, ' , p_values: ', p_sp)

    # write the prediction to csv file
    df_all_predictions.to_csv(data_path+'/results/csv/cage_prediction/'+cell_line+'_cage_predictions_epi_models_'+assay_type+'_FDR_'+fdr+'.csv', sep="\t", index=False)


    ##### write R and NLL for different 3D graphs and FDRs #####
    if save_R_NLL_to_csv:
        df = pd.DataFrame(columns=['cell', 'Method', 'Set', 'valid_chr', 'test_chr', 'n_gene_test', '3D_data', 'FDR', 'R','NLL'])
        
        for i in range(1,1+10):
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
            chr_list = test_chr_list.copy()
            chr_list.sort()

            test_chr_str = [str(i) for i in test_chr_list]
            test_chr_str = ','.join(test_chr_str)
            valid_chr_str = [str(i) for i in valid_chr_list]
            valid_chr_str = ','.join(valid_chr_str)

            df = df.append({'cell': cell_line, 'Method': 'Epi-GraphReg', 'Set': 'All', 'valid_chr': valid_chr_str, 'test_chr': test_chr_str, 
                            'n_gene_test': n_gene[i-1,0], '3D_data': assay_type, 'FDR': qval, 'R': valid_rho_gat[i-1,0], 'NLL': valid_loss_gat[i-1,0]}, ignore_index=True)
            df = df.append({'cell': cell_line, 'Method': 'Epi-GraphReg', 'Set': 'Expressed', 'valid_chr': valid_chr_str, 'test_chr': test_chr_str, 
                            'n_gene_test': n_gene[i-1,1], '3D_data': assay_type, 'FDR': qval, 'R': valid_rho_gat[i-1,1], 'NLL': valid_loss_gat[i-1,1]}, ignore_index=True)
            df = df.append({'cell': cell_line, 'Method': 'Epi-GraphReg', 'Set': 'Interacted', 'valid_chr': valid_chr_str, 'test_chr': test_chr_str, 
                            'n_gene_test': n_gene[i-1,2], '3D_data': assay_type, 'FDR': qval, 'R': valid_rho_gat[i-1,2], 'NLL': valid_loss_gat[i-1,2]}, ignore_index=True)

            df = df.append({'cell': cell_line, 'Method': 'Epi-CNN', 'Set': 'All', 'valid_chr': valid_chr_str, 'test_chr': test_chr_str, 
                        'n_gene_test': n_gene[i-1,0], '3D_data': assay_type, 'FDR': qval, 'R': valid_rho_cnn[i-1,0], 'NLL': valid_loss_cnn[i-1,0]}, ignore_index=True)
            df = df.append({'cell': cell_line, 'Method': 'Epi-CNN', 'Set': 'Expressed', 'valid_chr': valid_chr_str, 'test_chr': test_chr_str, 
                        'n_gene_test': n_gene[i-1,1], '3D_data': assay_type, 'FDR': qval, 'R': valid_rho_cnn[i-1,1], 'NLL': valid_loss_cnn[i-1,1]}, ignore_index=True)
            df = df.append({'cell': cell_line, 'Method': 'Epi-CNN', 'Set': 'Interacted', 'valid_chr': valid_chr_str, 'test_chr': test_chr_str, 
                        'n_gene_test': n_gene[i-1,2], '3D_data': assay_type, 'FDR': qval, 'R': valid_rho_cnn[i-1,2], 'NLL': valid_loss_cnn[i-1,2]}, ignore_index=True)

        df.to_csv(data_path+'/results/csv/cage_prediction/'+cell_line+'_R_NLL_epi_models_'+assay_type+'_FDR_'+fdr+'.csv', sep="\t", index=False)
        

    ##### plot violin plots #####
    if plot_violin == True:
        labels = []
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

        ax1.set_title(cell_line, fontsize=20)
        ax1.set_ylabel('R', fontsize=20)
        positions1 = np.array([1,3,5,7])
        parts11 = ax1.violinplot(valid_rho_gat, positions=positions1, showmeans = False, showextrema = True, showmedians = True)
        for pc in parts11['bodies']:
            pc.set_facecolor('orange')
            pc.set_edgecolor('black')
            pc.set_alpha(1)
        for partname in ('cbars','cmins','cmaxes','cmedians'):
            vp = parts11[partname]
            vp.set_edgecolor('black')
            vp.set_linewidth(1)
        add_label(parts11, labels, "Epi-GraphReg") 

        positions2 = positions1 + .75
        parts12 = ax1.violinplot(valid_rho_cnn, positions=positions2, showmeans = False, showextrema = True, showmedians = True)
        for pc in parts12['bodies']:
            pc.set_facecolor('deepskyblue')
            pc.set_edgecolor('black')
            pc.set_alpha(1)
        for partname in ('cbars','cmins','cmaxes','cmedians'):
            vp = parts12[partname]
            vp.set_edgecolor('black')
            vp.set_linewidth(1)
        add_label(parts12, labels, "Epi-CNN")    

        tick_labels = ['Set A', 'Set B', 'Set C', 'Set D']
        positions_tick = (positions1 + positions2)/2
        set_axis_style(ax1, tick_labels, positions_tick)

        ax1.grid(axis='y')
        ax1.legend(*zip(*labels), loc=3, fontsize=15)
        ax1.set_ylim((0.4,1))

        for i in range(4):
            if p_rho[i] <= 0.05:
                x1, x2 = positions1[i], positions2[i]
                y, h, col = np.max(np.append(valid_rho_gat[:,i],valid_rho_cnn[:,i])) + .02, .02, 'k'
                ax1.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
                ax1.text((x1+x2)*.5, y+h, "p="+"{:4.2e}".format(p_rho[i]), ha='center', va='bottom', color=col, fontsize=15)

        ax2.set_ylabel('NLL', fontsize=20)
        positions1 = np.array([1,3,5,7])
        parts21 = ax2.violinplot(valid_loss_gat, positions=positions1, showmeans = False, showextrema = True, showmedians = True)
        for pc in parts21['bodies']:
            pc.set_facecolor('orange')
            pc.set_edgecolor('black')
            pc.set_alpha(1)
        for partname in ('cbars','cmins','cmaxes','cmedians'):
            vp = parts21[partname]
            vp.set_edgecolor('black')
            vp.set_linewidth(1)

        positions2 = positions1 + .75
        parts22 = ax2.violinplot(valid_loss_cnn, positions=positions2, showmeans = False, showextrema = True, showmedians = True)
        for pc in parts22['bodies']:
            pc.set_facecolor('deepskyblue')
            pc.set_edgecolor('black')
            pc.set_alpha(1)
        for partname in ('cbars','cmins','cmaxes','cmedians'):
            vp = parts22[partname]
            vp.set_edgecolor('black')
            vp.set_linewidth(1)

        tick_labels = ['Set A', 'Set B', 'Set C', 'Set D']
        positions_tick = (positions1 + positions2)/2
        set_axis_style(ax2, tick_labels, positions_tick)

        ax2.grid(axis='y')
        k = 2000
        ax2.set_ylim((0,k))
        for i in range(4):
            if p_loss[i] <= 0.05:
                x1, x2 = positions1[i], positions2[i]
                y, h, col = np.max(np.append(valid_loss_gat[:,i],valid_loss_cnn[:,i])) + k/40, k/40, 'k'
                ax2.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
                ax2.text((x1+x2)*.5, y+h, "p="+"{:4.2e}".format(p_loss[i]), ha='center', va='bottom', color=col, fontsize=15)
        plt.savefig('../figs/Epi-models/violinplot_'+cell_line+'.png')


    ##### plot boxplots (All gene sets) #####
    if plot_box == True:
        df = pd.DataFrame(columns=['R','NLL','Method','Set'])
        for i in range(10):
            df = df.append({'R': valid_rho_gat[i,0], 'NLL': valid_loss_gat[i,0], 'Method': 'Epi-GraphReg', 'Set': 'A'}, ignore_index=True)
            df = df.append({'R': valid_rho_gat[i,1], 'NLL': valid_loss_gat[i,1], 'Method': 'Epi-GraphReg', 'Set': 'B'}, ignore_index=True)
            df = df.append({'R': valid_rho_gat[i,2], 'NLL': valid_loss_gat[i,2], 'Method': 'Epi-GraphReg', 'Set': 'C'}, ignore_index=True)
            df = df.append({'R': valid_rho_gat[i,3], 'NLL': valid_loss_gat[i,3], 'Method': 'Epi-GraphReg', 'Set': 'D'}, ignore_index=True)

            df = df.append({'R': valid_rho_cnn[i,0], 'NLL': valid_loss_cnn[i,0], 'Method': 'Epi-CNN', 'Set': 'A'}, ignore_index=True)
            df = df.append({'R': valid_rho_cnn[i,1], 'NLL': valid_loss_cnn[i,1], 'Method': 'Epi-CNN', 'Set': 'B'}, ignore_index=True)
            df = df.append({'R': valid_rho_cnn[i,2], 'NLL': valid_loss_cnn[i,2], 'Method': 'Epi-CNN', 'Set': 'C'}, ignore_index=True)
            df = df.append({'R': valid_rho_cnn[i,3], 'NLL': valid_loss_cnn[i,3], 'Method': 'Epi-CNN', 'Set': 'D'}, ignore_index=True)

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
        #ax1.set_title(cell_line_train+' to '+cell_line_test, fontsize=20)
        #ax1.set_title(cell_line_train, fontsize=20)
        b=sns.boxplot(x='Set', y='R', hue='Method', data=df, palette={"Epi-GraphReg": "orange", "Epi-CNN": "deepskyblue"}, order=['A', 'B', 'C', 'D'], ax=ax1)
        
        add_stat_annotation(ax1, data=df, x='Set', y='R', hue='Method',
                        box_pairs=[(("A", "Epi-GraphReg"), ("A", "Epi-CNN")),
                                    (("B", "Epi-GraphReg"), ("B", "Epi-CNN")),
                                    (("C", "Epi-GraphReg"), ("C", "Epi-CNN")),
                                    (("D", "Epi-GraphReg"), ("D", "Epi-CNN"))],
                        test='Wilcoxon', text_format='star', loc='inside', verbose=0, order=['A', 'B', 'C', 'D'], fontsize='x-large', comparisons_correction=None)
        

        ax1.yaxis.set_tick_params(labelsize=20)
        ax1.xaxis.set_tick_params(labelsize=20)
        b.set_xlabel("Set",fontsize=20)
        b.set_ylabel("R",fontsize=20)
        plt.setp(ax1.get_legend().get_texts(), fontsize='15')
        plt.setp(ax1.get_legend().get_title(), fontsize='15')
        #ax1.set_ylim((.4,.75))

        b = sns.boxplot(x='Set', y='NLL', hue='Method', data=df, palette={"Epi-GraphReg": "orange", "Epi-CNN": "deepskyblue"}, order=['A', 'B', 'C', 'D'], ax=ax2)
        
        add_stat_annotation(ax2, data=df, x='Set', y='NLL', hue='Method',
                        box_pairs=[(("A", "Epi-GraphReg"), ("A", "Epi-CNN")),
                                    (("B", "Epi-GraphReg"), ("B", "Epi-CNN")),
                                    (("C", "Epi-GraphReg"), ("C", "Epi-CNN")),
                                    (("D", "Epi-GraphReg"), ("D", "Epi-CNN"))],
                        test='Wilcoxon', text_format='star', loc='inside', verbose=2, order=['A', 'B', 'C', 'D'], fontsize='x-large', comparisons_correction=None)
        

        ax2.yaxis.set_tick_params(labelsize=20)
        ax2.xaxis.set_tick_params(labelsize=20)
        b.set_xlabel("Set",fontsize=20)
        b.set_ylabel("NLL",fontsize=20)
        plt.setp(ax2.get_legend().get_texts(), fontsize='15')
        plt.setp(ax2.get_legend().get_title(), fontsize='15')
        #ax2.set_ylim((250,900))

        #fig.tight_layout()
        fig.suptitle(cell_line, fontsize=25)
        #fig.suptitle(cell_line_train+' to '+cell_line_test, fontsize=25)
        fig.tight_layout(rect=[0, 0, 1, .93])
        plt.savefig('../figs/Epi-models/boxplot_all_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.pdf')


        ##### plot boxplots (for gene sets C and D) #####
        df = pd.DataFrame(columns=['R','NLL','Method','Set'])
        for i in range(10):
            df = df.append({'R': valid_rho_gat[i,0], 'NLL': valid_loss_gat[i,0], 'Method': 'Epi-GraphReg', 'Set': 'A'}, ignore_index=True)
            df = df.append({'R': valid_rho_gat[i,1], 'NLL': valid_loss_gat[i,1], 'Method': 'Epi-GraphReg', 'Set': 'B'}, ignore_index=True)
            df = df.append({'R': valid_rho_gat[i,2], 'NLL': valid_loss_gat[i,2], 'Method': 'Epi-GraphReg', 'Set': 'C'}, ignore_index=True)
            df = df.append({'R': valid_rho_gat[i,3], 'NLL': valid_loss_gat[i,3], 'Method': 'Epi-GraphReg', 'Set': 'D'}, ignore_index=True)

            df = df.append({'R': valid_rho_cnn[i,0], 'NLL': valid_loss_cnn[i,0], 'Method': 'Epi-CNN', 'Set': 'A'}, ignore_index=True)
            df = df.append({'R': valid_rho_cnn[i,1], 'NLL': valid_loss_cnn[i,1], 'Method': 'Epi-CNN', 'Set': 'B'}, ignore_index=True)
            df = df.append({'R': valid_rho_cnn[i,2], 'NLL': valid_loss_cnn[i,2], 'Method': 'Epi-CNN', 'Set': 'C'}, ignore_index=True)
            df = df.append({'R': valid_rho_cnn[i,3], 'NLL': valid_loss_cnn[i,3], 'Method': 'Epi-CNN', 'Set': 'D'}, ignore_index=True)

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
        #ax1.set_title(cell_line_train+' to '+cell_line_test, fontsize=20)
        #ax1.set_title(cell_line_train, fontsize=20)
        b=sns.boxplot(x='Set', y='R', hue='Method', data=df, palette={"Epi-GraphReg": "orange", "Epi-CNN": "deepskyblue"}, order=['C', 'D'], ax=ax1)
        add_stat_annotation(ax1, data=df, x='Set', y='R', hue='Method',
                        box_pairs=[(("C", "Epi-GraphReg"), ("C", "Epi-CNN")),
                                    (("D", "Epi-GraphReg"), ("D", "Epi-CNN"))],
                        test='Wilcoxon', text_format='star', loc='inside', verbose=0, order=['C', 'D'], fontsize='x-large', comparisons_correction=None)
        ax1.yaxis.set_tick_params(labelsize=20)
        ax1.xaxis.set_tick_params(labelsize=20)
        b.set_xlabel("Set",fontsize=20)
        b.set_ylabel("R",fontsize=20)
        plt.setp(ax1.get_legend().get_texts(), fontsize='15')
        plt.setp(ax1.get_legend().get_title(), fontsize='15')
        #ax1.set_ylim((.4,.75))

        b = sns.boxplot(x='Set', y='NLL', hue='Method', data=df, palette={"Epi-GraphReg": "orange", "Epi-CNN": "deepskyblue"}, order=['C', 'D'], ax=ax2)
        add_stat_annotation(ax2, data=df, x='Set', y='NLL', hue='Method',
                        box_pairs=[(("C", "Epi-GraphReg"), ("C", "Epi-CNN")),
                                    (("D", "Epi-GraphReg"), ("D", "Epi-CNN"))],
                        test='Wilcoxon', text_format='star', loc='inside', verbose=2, order=['C', 'D'], fontsize='x-large', comparisons_correction=None)

        ax2.yaxis.set_tick_params(labelsize=20)
        ax2.xaxis.set_tick_params(labelsize=20)
        b.set_xlabel("Set",fontsize=20)
        b.set_ylabel("NLL",fontsize=20)
        plt.setp(ax2.get_legend().get_texts(), fontsize='15')
        plt.setp(ax2.get_legend().get_title(), fontsize='15')
        #ax2.set_ylim((250,900))

        #fig.tight_layout()
        fig.suptitle(cell_line, fontsize=25)
        #fig.suptitle(cell_line_train+' to '+cell_line_test, fontsize=25)
        fig.tight_layout(rect=[0, 0, 1, .93])
        plt.savefig('../figs/Epi-models/boxplot_CD_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.pdf')


    ##### scatter plots #####
    if plot_scatter == True:
        for i in range(1,1+10):
            y_gene = np.load(data_path+'/results/numpy/cage_prediction/true_cage_'+cell_line+'_'+str(i)+'.npy')
            y_hat_gene_gat = np.load(data_path+'/results/numpy/cage_prediction/Epi-GraphReg_predicted_cage_'+cell_line+'_'+str(i)+'.npy')
            y_hat_gene_cnn = np.load(data_path+'/results/numpy/cage_prediction/Epi-CNN_predicted_cage_'+cell_line+'_'+str(i)+'.npy')
            n_contacts = np.load(data_path+'/results/numpy/cage_prediction/n_contacts_'+cell_line+'_'+str(i)+'.npy')
            idx_0 = np.where(n_contacts==0)[0]
            idx_1 = np.where(n_contacts>=1)[0]
            n_contacts_idx_1 = n_contacts[idx_1]
            valid_loss_gat_n0 = poisson_loss(y_gene[idx_0], y_hat_gene_gat[idx_0]).numpy()
            valid_rho_gat_n0 = np.corrcoef(np.log2(y_gene[idx_0]+1),np.log2(y_hat_gene_gat[idx_0]+1))[0,1]
            valid_loss_gat_n1 = poisson_loss(y_gene[idx_1], y_hat_gene_gat[idx_1]).numpy()
            valid_rho_gat_n1 = np.corrcoef(np.log2(y_gene[idx_1]+1),np.log2(y_hat_gene_gat[idx_1]+1))[0,1]

            valid_loss_cnn_n0 = poisson_loss(y_gene[idx_0], y_hat_gene_cnn[idx_0]).numpy()
            valid_rho_cnn_n0 = np.corrcoef(np.log2(y_gene[idx_0]+1),np.log2(y_hat_gene_cnn[idx_0]+1))[0,1]
            valid_loss_cnn_n1 = poisson_loss(y_gene[idx_1], y_hat_gene_cnn[idx_1]).numpy()
            valid_rho_cnn_n1 = np.corrcoef(np.log2(y_gene[idx_1]+1),np.log2(y_hat_gene_cnn[idx_1]+1))[0,1]

            plt.figure(figsize=(9,8))
            cm = plt.cm.get_cmap('viridis_r')
            idx=np.argsort(n_contacts_idx_1)
            sc = plt.scatter(np.log2(y_gene[idx]+1),np.log2(y_hat_gene_gat[idx]+1), c=np.log2(n_contacts_idx_1[idx]+1), s=100, cmap=cm, alpha=.7, edgecolors='')
            plt.xlim((-.5,15))
            plt.ylim((-.5,15))
            plt.title('Epi-GraphReg, '+cell_line, fontsize=20)
            plt.xlabel("log2 (true + 1)", fontsize=20)
            plt.ylabel("log2 (pred + 1)", fontsize=20)
            plt.tick_params(axis='x', labelsize=15)
            plt.tick_params(axis='y', labelsize=15)
            plt.grid(alpha=.5)
            props = dict(boxstyle='round', facecolor='white', alpha=1)
            #plt.text(0,15, 'n=0: R= '+"{:5.3f}".format(valid_rho_gat_n0) + ', NLL= '+str(np.float16(valid_loss_gat_n0))+'\n'+
            #                 'n>0: R= '+"{:5.3f}".format(valid_rho_gat_n1) + ', NLL= '+str(np.float16(valid_loss_gat_n1)), 
            # horizontalalignment='left', verticalalignment='top', bbox=props, fontsize=20)
            plt.text(0,14.5, 'n>0: R= '+"{:5.3f}".format(valid_rho_gat_n1) + ', NLL= '+str(np.float16(valid_loss_gat_n1)), 
                horizontalalignment='left', verticalalignment='top', bbox=props, fontsize=20)
            cbar = plt.colorbar(sc)
            cbar.set_label(label='log2 (n + 1)', size=20)
            cbar.ax.tick_params(labelsize=15)
            #plt.show()
            plt.tight_layout()
            plt.savefig('../figs/Epi-models/scatter_plots/Epi-GraphReg_scatterplot_'+cell_line+'_'+str(i)+'.png')

            plt.figure(figsize=(9,8))
            cm = plt.cm.get_cmap('viridis_r')
            idx=np.argsort(n_contacts_idx_1)
            sc = plt.scatter(np.log2(y_gene[idx]+1),np.log2(y_hat_gene_cnn[idx]+1), c=np.log2(n_contacts_idx_1[idx]+1), s=100, cmap=cm, alpha=.7, edgecolors='')
            plt.xlim((-.5,15))
            plt.ylim((-.5,15))
            plt.title('Epi-CNN, '+cell_line, fontsize=20)
            plt.xlabel("log2 (true + 1)", fontsize=20)
            plt.ylabel("log2 (pred + 1)", fontsize=20)
            plt.tick_params(axis='x', labelsize=15)
            plt.tick_params(axis='y', labelsize=15)
            plt.grid(alpha=.5)
            props = dict(boxstyle='round', facecolor='white', alpha=1)
            #plt.text(0,15, 'n=0: R= '+"{:5.3f}".format(valid_rho_cnn_n0) + ', NLL= '+str(np.float16(valid_loss_cnn_n0))+'\n'+
            #                 'n>0: R= '+"{:5.3f}".format(valid_rho_cnn_n1) + ', NLL= '+str(np.float16(valid_loss_cnn_n1)),
            # horizontalalignment='left', verticalalignment='top', bbox=props, fontsize=20)
            plt.text(0,14.5, 'n>0: R= '+"{:5.3f}".format(valid_rho_cnn_n1) + ', NLL= '+str(np.float16(valid_loss_cnn_n1)),
                horizontalalignment='left', verticalalignment='top', bbox=props, fontsize=20)
            cbar = plt.colorbar(sc)
            cbar.set_label(label='log2 (n + 1)', size=20)
            cbar.ax.tick_params(labelsize=15)
            #plt.show()
            plt.tight_layout()
            plt.savefig('../figs/Epi-models/scatter_plots/Epi-CNN_scatterplot_'+cell_line+'_'+str(i)+'.png')


#################### log fold change ####################

if logfold == True:
    cell_line_1 = 'GM12878'
    cell_line_2 = 'K562'
    organism = 'human'
    R_gat = np.zeros([10,4])
    R_cnn = np.zeros([10,4])
    SP_gat = np.zeros([10,4])
    SP_cnn = np.zeros([10,4])
    MSE_gat = np.zeros([10,4])
    MSE_cnn = np.zeros([10,4])
    for i in range(1,1+10):
        print('i: ', i)
        iv2 = i+10
        it2 = i+11
        valid_chr_list = [i, iv2]
        test_chr_list = [i+1, it2]
        chr_list = test_chr_list.copy()
        chr_list.sort()

        test_chr_str = [str(i) for i in test_chr_list]
        test_chr_str = ','.join(test_chr_str)
        valid_chr_str = [str(i) for i in valid_chr_list]
        valid_chr_str = ','.join(valid_chr_str)

        if load_np == True:
            y_gene_1 = np.load(data_path+'/results/numpy/cage_prediction/true_cage_'+cell_line_1+'_'+str(i)+'.npy')
            y_hat_gene_gat_1 = np.load(data_path+'/results/numpy/cage_prediction/Epi-GraphReg_predicted_cage_'+cell_line_1+'_'+str(i)+'.npy')
            y_hat_gene_cnn_1 = np.load(data_path+'/results/numpy/cage_prediction/Epi-CNN_predicted_cage_'+cell_line_1+'_'+str(i)+'.npy')
            n_contacts_1 = np.load(data_path+'/results/numpy/cage_prediction/n_contacts_'+cell_line_1+'_'+str(i)+'.npy')
            gene_names = np.load(data_path+'/results/numpy/cage_prediction/gene_names_'+cell_line_1+'_'+str(i)+'.npy')
            gene_tss = np.load(data_path+'/results/numpy/cage_prediction/gene_tss_'+cell_line_1+'_'+str(i)+'.npy')
            gene_chr = np.load(data_path+'/results/numpy/cage_prediction/gene_chr_'+cell_line_1+'_'+str(i)+'.npy')
        else:
            model_name_gat = data_path+'/models/'+cell_line_1+'/Epi-GraphReg_'+cell_line_1+'_'+assay_type+'_FDR_'+fdr+'_valid_chr_'+valid_chr_str+'_test_chr_'+test_chr_str+'.h5'
            model_gat_1 = tf.keras.models.load_model(model_name_gat, custom_objects={'GraphAttention': GraphAttention})
            model_gat_1.trainable = False
            model_gat_1._name = 'Epi-GraphReg'
            #model_gat_1.summary()

            model_name = data_path+'/models/'+cell_line_1+'/Epi-CNN_'+cell_line_1+'_valid_chr_'+valid_chr_str+'_test_chr_'+test_chr_str+'.h5'
            model_cnn_1 = tf.keras.models.load_model(model_name)
            model_cnn_1.trainable = False
            model_cnn_1._name = 'Epi-CNN'
            #model_cnn_1.summary()

            y_gene_1, y_hat_gene_gat_1, y_hat_gene_cnn_1, _, _, gene_names, gene_tss, gene_chr, n_contacts_1, _, _, _, _ = calculate_loss(model_gat_1, model_cnn_1, 
                    chr_list, valid_chr_list, test_chr_list, cell_line_1, organism, batch_size, write_bw)
        
        if load_np == True:
            y_gene_2 = np.load(data_path+'/results/numpy/cage_prediction/true_cage_'+cell_line_2+'_'+str(i)+'.npy')
            y_hat_gene_gat_2 = np.load(data_path+'/results/numpy/cage_prediction/Epi-GraphReg_predicted_cage_'+cell_line_2+'_'+str(i)+'.npy')
            y_hat_gene_cnn_2 = np.load(data_path+'/results/numpy/cage_prediction/Epi-CNN_predicted_cage_'+cell_line_2+'_'+str(i)+'.npy')
            n_contacts_2 = np.load(data_path+'/results/numpy/cage_prediction/n_contacts_'+cell_line_2+'_'+str(i)+'.npy')
            gene_names = np.load(data_path+'/results/numpy/cage_prediction/gene_names_'+cell_line_2+'_'+str(i)+'.npy')
            gene_tss = np.load(data_path+'/results/numpy/cage_prediction/gene_tss_'+cell_line_2+'_'+str(i)+'.npy')
            gene_chr = np.load(data_path+'/results/numpy/cage_prediction/gene_chr_'+cell_line_2+'_'+str(i)+'.npy')
        else:
            model_name_gat = data_path+'/models/'+cell_line_2+'/Epi-GraphReg_'+cell_line_2+'_'+assay_type+'_FDR_'+fdr+'_valid_chr_'+valid_chr_str+'_test_chr_'+test_chr_str+'.h5'
            model_gat_2 = tf.keras.models.load_model(model_name_gat, custom_objects={'GraphAttention': GraphAttention})
            model_gat_2.trainable = False
            model_gat_2._name = 'Epi-GraphReg'
            #model_gat_2.summary()

            model_name = data_path+'/models/'+cell_line_2+'/Epi-CNN_'+cell_line_2+'_valid_chr_'+valid_chr_str+'_test_chr_'+test_chr_str+'.h5'
            model_cnn_2 = tf.keras.models.load_model(model_name)
            model_cnn_2.trainable = False
            model_cnn_2._name = 'Epi-CNN'
            #model_cnn_2.summary()
            
            y_gene_2, y_hat_gene_gat_2, y_hat_gene_cnn_2, _, _, gene_names, gene_tss, gene_chr, n_contacts_2, _, _, _, _ = calculate_loss(model_gat_2, model_cnn_2, 
                    chr_list, valid_chr_list, test_chr_list,  cell_line_2, organism, batch_size, write_bw)

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

            idx_1 = np.where(np.logical_and(n_contacts_1 >= min_contact, y_gene_1 >= min_expression))[0]
            idx_2 = np.where(np.logical_and(n_contacts_2 >= min_contact, y_gene_2 >= min_expression))[0]
            #idx = np.union1d(idx_1, idx_2)
            idx = np.intersect1d(idx_1, idx_2)

            y_gene_1_idx = y_gene_1[idx]
            y_gene_2_idx = y_gene_2[idx]
            y_hat_gene_gat_1_idx = y_hat_gene_gat_1[idx]
            y_hat_gene_gat_2_idx = y_hat_gene_gat_2[idx]
            y_hat_gene_cnn_1_idx = y_hat_gene_cnn_1[idx]
            y_hat_gene_cnn_2_idx = y_hat_gene_cnn_2[idx]

            log_fc_true = np.log2((y_gene_1_idx+1)/(y_gene_2_idx+1))
            log_fc_gat = np.log2((y_hat_gene_gat_1_idx+1)/(y_hat_gene_gat_2_idx+1))
            log_fc_cnn = np.log2((y_hat_gene_cnn_1_idx+1)/(y_hat_gene_cnn_2_idx+1))

            R_gat[i-1,j] = np.corrcoef(log_fc_true, log_fc_gat)[0,1]
            SP_gat[i-1,j] = spearmanr(log_fc_true, log_fc_gat)[0]
            MSE_gat[i-1,j] = np.mean((log_fc_true-log_fc_gat)**2)
            R_cnn[i-1,j] = np.corrcoef(log_fc_true, log_fc_cnn)[0,1]
            SP_cnn[i-1,j] = spearmanr(log_fc_true, log_fc_cnn)[0]
            MSE_cnn[i-1,j] = np.mean((log_fc_true-log_fc_cnn)**2)

    print('Mean LogFC R GAT: ', np.mean(R_gat, axis=0), ' +/- ', np.std(R_gat, axis=0), ' std')
    print('Mean LogFC R CNN: ', np.mean(R_cnn, axis=0), ' +/- ', np.std(R_cnn, axis=0), ' std \n')

    print('Mean LogFC MSE GAT: ', np.mean(MSE_gat, axis=0), ' +/- ', np.std(MSE_gat, axis=0), ' std')
    print('Mean LogFC MSE CNN: ', np.mean(MSE_cnn, axis=0), ' +/- ', np.std(MSE_cnn, axis=0), ' std')

    w_loss = np.zeros(4)
    w_rho = np.zeros(4)
    w_sp = np.zeros(4)
    p_loss = np.zeros(4)
    p_rho = np.zeros(4)
    p_sp = np.zeros(4)
    for j in range(4):
        w_loss[j], p_loss[j] = wilcoxon(MSE_gat[:,j], MSE_cnn[:,j], alternative='less')
        w_rho[j], p_rho[j] = wilcoxon(R_gat[:,j], R_cnn[:,j], alternative='greater')
        w_sp[j], p_sp[j] = wilcoxon(SP_gat[:,j], SP_cnn[:,j], alternative='greater')

    print('LogFC Wilcoxon Loss: ', w_loss, ' , p_values: ', p_loss)
    print('LogFC Wilcoxon R: ', w_rho, ' , p_values: ', p_rho)
    print('LogFC Wilcoxon SP: ', w_sp, ' , p_values: ', p_sp)

    ##### plot violin plots #####
    if plot_violin == True:
        labels = []
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

        ax1.set_title('LogFC (GM12878/K562)', fontsize=20)
        ax1.set_ylabel('R', fontsize=20)
        positions1 = np.array([1,3,5,7])
        parts11 = ax1.violinplot(R_gat, positions=positions1, showmeans = False, showextrema = True, showmedians = True)
        for pc in parts11['bodies']:
            pc.set_facecolor('orange')
            pc.set_edgecolor('black')
            pc.set_alpha(1)
        for partname in ('cbars','cmins','cmaxes','cmedians'):
            vp = parts11[partname]
            vp.set_edgecolor('black')
            vp.set_linewidth(1)
        add_label(parts11, labels, "Epi-GraphReg") 

        positions2 = positions1 + .75
        parts12 = ax1.violinplot(R_cnn, positions=positions2, showmeans = False, showextrema = True, showmedians = True)
        for pc in parts12['bodies']:
            pc.set_facecolor('deepskyblue')
            pc.set_edgecolor('black')
            pc.set_alpha(1)
        for partname in ('cbars','cmins','cmaxes','cmedians'):
            vp = parts12[partname]
            vp.set_edgecolor('black')
            vp.set_linewidth(1)
        add_label(parts12, labels, "Epi-CNN")    

        tick_labels = ['Set A', 'Set BB', 'Set CC', 'Set DD']
        positions_tick = (positions1 + positions2)/2
        set_axis_style(ax1, tick_labels, positions_tick)

        ax1.grid(axis='y')
        ax1.legend(*zip(*labels), loc=2, fontsize=15)
        ax1.set_ylim((0.2,1))

        for i in range(4):
            if p_rho[i] <= 0.05:
                x1, x2 = positions1[i], positions2[i]
                y, h, col = np.max(np.append(R_gat[:,i],R_cnn[:,i])) + .02, .02, 'k'
                ax1.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
                ax1.text((x1+x2)*.5, y+h, "p="+"{:4.2e}".format(p_rho[i]), ha='center', va='bottom', color=col, fontsize=15)

        ax2.set_ylabel('MSE', fontsize=20)
        positions1 = np.array([1,3,5,7])
        parts21 = ax2.violinplot(MSE_gat, positions=positions1, showmeans = False, showextrema = True, showmedians = True)
        for pc in parts21['bodies']:
            pc.set_facecolor('orange')
            pc.set_edgecolor('black')
            pc.set_alpha(1)
        for partname in ('cbars','cmins','cmaxes','cmedians'):
            vp = parts21[partname]
            vp.set_edgecolor('black')
            vp.set_linewidth(1)

        positions2 = positions1 + .75
        parts22 = ax2.violinplot(MSE_cnn, positions=positions2, showmeans = False, showextrema = True, showmedians = True)
        for pc in parts22['bodies']:
            pc.set_facecolor('deepskyblue')
            pc.set_edgecolor('black')
            pc.set_alpha(1)
        for partname in ('cbars','cmins','cmaxes','cmedians'):
            vp = parts22[partname]
            vp.set_edgecolor('black')
            vp.set_linewidth(1)

        tick_labels = ['Set A', 'Set BB', 'Set CC', 'Set DD']
        positions_tick = (positions1 + positions2)/2
        set_axis_style(ax2, tick_labels, positions_tick)

        ax2.grid(axis='y')
        k = 6
        ax2.set_ylim((0,k))
        for i in range(4):
            if p_loss[i] <= 0.05:
                x1, x2 = positions1[i], positions2[i]
                y, h, col = np.max(np.append(MSE_gat[:,i],MSE_cnn[:,i])) + k/40, k/40, 'k'
                ax2.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
                ax2.text((x1+x2)*.5, y+h, "p="+"{:4.2e}".format(p_loss[i]), ha='center', va='bottom', color=col, fontsize=15)

        #plt.show()
        plt.savefig('../figs/Epi-models/violinplot_LogFC_'+cell_line_1+'_'+cell_line_2+'.png')

    ##### plot box plots (all gene sets) #####
    if plot_box == True:
        df = pd.DataFrame(columns=['R','MSE','Method','Set'])
        for i in range(10):
            df = df.append({'R': R_gat[i,0], 'MSE': MSE_gat[i,0], 'Method': 'Epi-GraphReg', 'Set': 'A'}, ignore_index=True)
            df = df.append({'R': R_gat[i,1], 'MSE': MSE_gat[i,1], 'Method': 'Epi-GraphReg', 'Set': 'BB'}, ignore_index=True)
            df = df.append({'R': R_gat[i,2], 'MSE': MSE_gat[i,2], 'Method': 'Epi-GraphReg', 'Set': 'CC'}, ignore_index=True)
            df = df.append({'R': R_gat[i,3], 'MSE': MSE_gat[i,3], 'Method': 'Epi-GraphReg', 'Set': 'DD'}, ignore_index=True)

            df = df.append({'R': R_cnn[i,0], 'MSE': MSE_cnn[i,0], 'Method': 'Epi-CNN', 'Set': 'A'}, ignore_index=True)
            df = df.append({'R': R_cnn[i,1], 'MSE': MSE_cnn[i,1], 'Method': 'Epi-CNN', 'Set': 'BB'}, ignore_index=True)
            df = df.append({'R': R_cnn[i,2], 'MSE': MSE_cnn[i,2], 'Method': 'Epi-CNN', 'Set': 'CC'}, ignore_index=True)
            df = df.append({'R': R_cnn[i,3], 'MSE': MSE_cnn[i,3], 'Method': 'Epi-CNN', 'Set': 'DD'}, ignore_index=True)

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
        #ax1.set_title(cell_line_train+' to '+cell_line_test, fontsize=20)
        #ax1.set_title(cell_line_train, fontsize=20)
        b=sns.boxplot(x='Set', y='R', hue='Method', data=df, palette={"Epi-GraphReg": "orange", "Epi-CNN": "deepskyblue"}, order=['A', 'BB', 'CC', 'DD'], ax=ax1)
        add_stat_annotation(ax1, data=df, x='Set', y='R', hue='Method',
                        box_pairs=[(("A", "Epi-GraphReg"), ("A", "Epi-CNN")),
                                    (("BB", "Epi-GraphReg"), ("BB", "Epi-CNN")),
                                    (("CC", "Epi-GraphReg"), ("CC", "Epi-CNN")),
                                    (("DD", "Epi-GraphReg"), ("DD", "Epi-CNN"))],
                        test='Wilcoxon', text_format='star', loc='inside', verbose=0, order=['A', 'BB', 'CC', 'DD'], fontsize='x-large', comparisons_correction=None)
        ax1.yaxis.set_tick_params(labelsize=20)
        ax1.xaxis.set_tick_params(labelsize=20)
        b.set_xlabel("Set",fontsize=20)
        b.set_ylabel("R",fontsize=20)
        plt.setp(ax1.get_legend().get_texts(), fontsize='15')
        plt.setp(ax1.get_legend().get_title(), fontsize='15')
        #ax1.set_ylim((.4,.75))

        b = sns.boxplot(x='Set', y='MSE', hue='Method', data=df, palette={"Epi-GraphReg": "orange", "Epi-CNN": "deepskyblue"}, order=['A', 'BB', 'CC', 'DD'], ax=ax2)
        add_stat_annotation(ax2, data=df, x='Set', y='MSE', hue='Method',
                        box_pairs=[(("A", "Epi-GraphReg"), ("A", "Epi-CNN")),
                                    (("BB", "Epi-GraphReg"), ("BB", "Epi-CNN")),
                                    (("CC", "Epi-GraphReg"), ("CC", "Epi-CNN")),
                                    (("DD", "Epi-GraphReg"), ("DD", "Epi-CNN"))],
                        test='Wilcoxon', text_format='star', loc='inside', verbose=2, order=['A', 'BB', 'CC', 'DD'], fontsize='x-large', comparisons_correction=None)

        ax2.yaxis.set_tick_params(labelsize=20)
        ax2.xaxis.set_tick_params(labelsize=20)
        b.set_xlabel("Set",fontsize=20)
        b.set_ylabel("MSE",fontsize=20)
        plt.setp(ax2.get_legend().get_texts(), fontsize='15')
        plt.setp(ax2.get_legend().get_title(), fontsize='15')
        #ax2.set_ylim((1.15,3.4))

        #fig.tight_layout()
        fig.suptitle('LogFC (GM12878/K562)', fontsize=25)
        fig.tight_layout(rect=[0, 0, 1, .93])
        plt.savefig('../figs/Epi-models/boxplot_LogFC_'+cell_line_1+'_'+cell_line_2+'_all.png')

    ##### scatter plots #####
    if plot_scatter == True:
        for i in range(1,11):
            y_gene_1 = np.load(data_path+'/results/numpy/cage_prediction/true_cage_'+cell_line_1+'_'+str(i)+'.npy')
            y_hat_gene_gat_1 = np.load(data_path+'/results/numpy/cage_prediction/Epi-GraphReg_predicted_cage_'+cell_line_1+'_'+str(i)+'.npy')
            y_hat_gene_cnn_1 = np.load(data_path+'/results/numpy/cage_prediction/Epi-CNN_predicted_cage_'+cell_line_1+'_'+str(i)+'.npy')
            n_contacts_1 = np.load(data_path+'/results/numpy/cage_prediction/n_contacts_'+cell_line_1+'_'+str(i)+'.npy')

            y_gene_2 = np.load(data_path+'/results/numpy/cage_prediction/true_cage_'+cell_line_2+'_'+str(i)+'.npy')
            y_hat_gene_gat_2 = np.load(data_path+'/results/numpy/cage_prediction/Epi-GraphReg_predicted_cage_'+cell_line_2+'_'+str(i)+'.npy')
            y_hat_gene_cnn_2 = np.load(data_path+'/results/numpy/cage_prediction/Epi-CNN_predicted_cage_'+cell_line_2+'_'+str(i)+'.npy')
            n_contacts_2 = np.load(data_path+'/results/numpy/cage_prediction/n_contacts_'+cell_line_2+'_'+str(i)+'.npy')

            n_contacts = np.minimum(n_contacts_1, n_contacts_2)
            idx_0 = np.where(n_contacts==0)[0]
            idx_1 = np.where(n_contacts>0)[0]
            n_contacts_idx_1 = n_contacts[idx_1]

            #n_contacts_ratio = np.log2((n_contacts_1+1)/(n_contacts_2+1))

            log_fc_true = np.log2((y_gene_1+1)/(y_gene_2+1))
            log_fc_gat = np.log2((y_hat_gene_gat_1+1)/(y_hat_gene_gat_2+1))
            log_fc_cnn = np.log2((y_hat_gene_cnn_1+1)/(y_hat_gene_cnn_2+1))

            R_gat_m0 = np.corrcoef(log_fc_true[idx_0], log_fc_gat[idx_0])[0,1]
            R_gat_m1 = np.corrcoef(log_fc_true[idx_1], log_fc_gat[idx_1])[0,1]
            MSE_gat_m0 = np.mean((log_fc_true[idx_0]-log_fc_gat[idx_0])**2)
            MSE_gat_m1 = np.mean((log_fc_true[idx_1]-log_fc_gat[idx_1])**2)
            R_cnn_m0 = np.corrcoef(log_fc_true[idx_0], log_fc_cnn[idx_0])[0,1]
            R_cnn_m1 = np.corrcoef(log_fc_true[idx_1], log_fc_cnn[idx_1])[0,1]
            MSE_cnn_m0 = np.mean((log_fc_true[idx_0]-log_fc_cnn[idx_0])**2)
            MSE_cnn_m1 = np.mean((log_fc_true[idx_1]-log_fc_cnn[idx_1])**2)

            plt.figure(figsize=(8,7))
            cm = plt.cm.get_cmap('viridis_r')
            idx=np.argsort(n_contacts_idx_1)
            sc = plt.scatter(log_fc_true[idx], log_fc_gat[idx], c=np.log2(n_contacts_idx_1[idx]+1), s=100, cmap=cm, alpha=.7, edgecolors='')
            plt.xlim((-11,11))
            plt.ylim((-11,11))
            plt.title('Epi-GraphReg, '+cell_line_1+'/'+cell_line_2, fontsize=20)
            plt.xlabel("log2 (FC true)", fontsize=20)
            plt.ylabel("log2 (FC pred)", fontsize=20)
            plt.tick_params(axis='x', labelsize=15)
            plt.tick_params(axis='y', labelsize=15)
            plt.grid(alpha=.5)
            props = dict(boxstyle='round', facecolor='white', alpha=1)
            #plt.text(-14,14, 'm=0: R= '+"{:5.3f}".format(R_gat_m0) + ', MSE= '+str(np.float16(MSE_gat_m0))+'\n'+
            #                 'm>0: R= '+"{:5.3f}".format(R_gat_m1) + ', MSE= '+str(np.float16(MSE_gat_m1)), 
            # horizontalalignment='left', verticalalignment='top', bbox=props, fontsize=20)
            plt.text(-10,10, 'm>0: R= '+"{:5.3f}".format(R_gat_m1) + ', MSE= '+str(np.float16(MSE_gat_m1)), 
            horizontalalignment='left', verticalalignment='top', bbox=props, fontsize=20)
            cbar = plt.colorbar(sc)
            cbar.set_label(label='log2 (m + 1)', size=20)
            cbar.ax.tick_params(labelsize=15)
            plt.clim(1,7)
            plt.tight_layout()
            #plt.show()
            plt.savefig('../figs/Epi-models/scatter_plots/Epi-GraphReg_scatterplot_LogFC_'+cell_line_1+'_'+cell_line_2+'_model_'+str(i)+'.png')

            plt.figure(figsize=(8,7))
            cm = plt.cm.get_cmap('viridis_r')
            idx=np.argsort(n_contacts_idx_1)
            sc = plt.scatter(log_fc_true[idx], log_fc_cnn[idx], c=np.log2(n_contacts_idx_1[idx]+1), s=100, cmap=cm, alpha=.7, edgecolors='')
            plt.xlim((-11,11))
            plt.ylim((-11,11))
            plt.title('Epi-CNN, '+cell_line_1+'/'+cell_line_2, fontsize=20)
            plt.xlabel("log2 (FC true)", fontsize=20)
            plt.ylabel("log2 (FC pred)", fontsize=20)
            plt.tick_params(axis='x', labelsize=15)
            plt.tick_params(axis='y', labelsize=15)
            plt.grid(alpha=.5)
            props = dict(boxstyle='round', facecolor='white', alpha=1)
            #plt.text(-14,14, 'm=0: R= '+"{:5.3f}".format(R_cnn_m0) + ', MSE= '+str(np.float16(MSE_cnn_m0))+'\n'+
            #                 'm>0: R= '+"{:5.3f}".format(R_cnn_m1) + ', MSE= '+str(np.float16(MSE_cnn_m1)),
            # horizontalalignment='left', verticalalignment='top', bbox=props, fontsize=20)
            plt.text(-10,10, 'm>0: R= '+"{:5.3f}".format(R_cnn_m1) + ', MSE= '+str(np.float16(MSE_cnn_m1)),
            horizontalalignment='left', verticalalignment='top', bbox=props, fontsize=20)
            cbar = plt.colorbar(sc)
            cbar.set_label(label='log2 (m + 1)', size=20)
            cbar.ax.tick_params(labelsize=15)
            plt.clim(1,7)
            #plt.show()
            plt.tight_layout()
            plt.savefig('../figs/Epi-models/scatter_plots/Epi-CNN_scatterplot_LogFC_'+cell_line_1+'_'+cell_line_2+'_model_'+str(i)+'.png')
