
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
assay_type = 'HiC'                           # HiChIP, HiC, MicroC, HiCAR

if qval == 0.1:
    fdr = '1'
elif qval == 0.01:
    fdr = '01'
elif qval == 0.001:
    fdr = '001'

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
        #Y = tf.reshape(Y, [batch_size, 3*T, b])
        #Y = tf.reduce_sum(Y, axis=2)
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
        file_name = data_path+'/data/tfrecords/tfr_epi_RPGC_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'_chr'+str(i)+'.tfr'
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
    valid_loss_gat = np.zeros([64,4])
    valid_rho_gat = np.zeros([64,4])
    valid_sp_gat = np.zeros([64,4])
    valid_loss_cnn = np.zeros([1,4])
    valid_rho_cnn = np.zeros([1,4])
    valid_sp_cnn = np.zeros([1,4])
    n_gene = np.zeros([1,4])
    df_all_predictions = pd.DataFrame(columns=['epoch', 'chr', 'genes', 'n_tss', 'tss', 'tss_distance_from_center', 'n_contact', 'average_dnase', 'average_h3k27ac', 'average_h3k4me3', 'true_cage', 'pred_cage_epi_graphreg', 'pred_cage_epi_cnn', 'nll_epi_graphreg', 'nll_epi_cnn', 'delta_nll'])

    for i in range(1,1+64):
        print('i: ', i)

        valid_chr_list = [1, 11]
        test_chr_list = [2, 12]
        chr_list = test_chr_list.copy()
        chr_list.sort()

        test_chr_str = [str(i) for i in test_chr_list]
        test_chr_str = ','.join(test_chr_str)
        valid_chr_str = [str(i) for i in valid_chr_list]
        valid_chr_str = ','.join(valid_chr_str)

        if i == 1:
            epoch = 0
            model_name_gat = data_path+'/models/'+cell_line+'/distal_reg/Epi-GraphReg_RPGC_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'_valid_chr_'+valid_chr_str+'_test_chr_'+test_chr_str+'.h5'
        else:
            epoch = (i-1) * 10
            model_name_gat = data_path+'/models/'+cell_line+'/distal_reg/Epi-GraphReg_RPGC_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'_valid_chr_'+valid_chr_str+'_test_chr_'+test_chr_str+'_epoch_'+str(epoch)+'.h5'
        model_gat = tf.keras.models.load_model(model_name_gat, custom_objects={'GraphAttention': GraphAttention})
        model_gat.trainable = False
        model_gat._name = 'Epi-GraphReg'
        #model_gat.summary()

        model_name = data_path+'/models/'+cell_line+'/Epi-CNN_RPGC_'+cell_line+'_valid_chr_'+valid_chr_str+'_test_chr_'+test_chr_str+'.h5'
        model_cnn = tf.keras.models.load_model(model_name)
        model_cnn.trainable = False
        model_cnn._name = 'Epi-CNN'
        #model_cnn.summary()

        y_gene, y_hat_gene_gat, y_hat_gene_cnn, _, _, gene_names, gene_tss, gene_chr, n_contacts, n_tss_in_bin, x_h3k4me3, x_h3k27ac, x_dnase = calculate_loss(model_gat, model_cnn, 
                chr_list, valid_chr_list, test_chr_list, cell_line, organism, genome, batch_size, write_bw)


        df_tmp = pd.DataFrame(columns=['epoch', 'chr', 'genes', 'n_tss', 'tss', 'tss_distance_from_center', 'n_contact', 'average_dnase', 'average_h3k27ac', 'average_h3k4me3', 'true_cage', 'pred_cage_epi_graphreg', 'pred_cage_epi_cnn', 'nll_epi_graphreg', 'nll_epi_cnn', 'delta_nll'])
        df_tmp['epoch'] = epoch * np.ones([len(gene_names)])
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
            valid_loss_cnn[0,j] = poisson_loss(y_gene_idx, y_hat_gene_cnn_idx).numpy()
            valid_rho_cnn[0,j] = np.corrcoef(np.log2(y_gene_idx+1),np.log2(y_hat_gene_cnn_idx+1))[0,1]
            valid_sp_cnn[0,j] = spearmanr(np.log2(y_gene_idx+1), np.log2(y_hat_gene_cnn_idx+1))[0]

            n_gene[0,j] = len(y_gene_idx)

            print('NLL GAT: ', valid_loss_gat, ' rho: ', valid_rho_gat, ' sp: ', valid_sp_gat)
            print('NLL CNN: ', valid_loss_cnn, ' rho: ', valid_rho_cnn, ' sp: ', valid_sp_cnn)

    # write the prediction to csv file
    df_all_predictions.to_csv(data_path+'/results/csv/cage_prediction/epi_models/cage_predictions_epi_models_distal_reg_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.csv', sep="\t", index=False)


    ##### write R and NLL for different 3D graphs and FDRs #####
    if save_R_NLL_to_csv:
        df = pd.DataFrame(columns=['epoch', 'cell', 'Method', 'Set', 'valid_chr', 'test_chr', 'n_gene_test', '3D_data', 'FDR', 'R','NLL'])
        
        for i in range(1,1+64):

            valid_chr_list = [1, 11]
            test_chr_list = [2, 12]
            chr_list = test_chr_list.copy()
            chr_list.sort()

            test_chr_str = [str(i) for i in test_chr_list]
            test_chr_str = ','.join(test_chr_str)
            valid_chr_str = [str(i) for i in valid_chr_list]
            valid_chr_str = ','.join(valid_chr_str)

            if i == 1:
                epoch = 0
            else:
                epoch = (i-1) * 10

            df = df.append({'epoch': epoch, 'cell': cell_line, 'Method': 'Epi-GraphReg', 'Set': 'All', 'valid_chr': valid_chr_str, 'test_chr': test_chr_str, 
                            'n_gene_test': n_gene[0,0], '3D_data': assay_type, 'FDR': qval, 'R': valid_rho_gat[i-1,0], 'NLL': valid_loss_gat[i-1,0]}, ignore_index=True)
            df = df.append({'epoch': epoch, 'cell': cell_line, 'Method': 'Epi-GraphReg', 'Set': 'Expressed', 'valid_chr': valid_chr_str, 'test_chr': test_chr_str, 
                            'n_gene_test': n_gene[0,1], '3D_data': assay_type, 'FDR': qval, 'R': valid_rho_gat[i-1,1], 'NLL': valid_loss_gat[i-1,1]}, ignore_index=True)
            df = df.append({'epoch': epoch, 'cell': cell_line, 'Method': 'Epi-GraphReg', 'Set': 'Interacted', 'valid_chr': valid_chr_str, 'test_chr': test_chr_str, 
                            'n_gene_test': n_gene[0,2], '3D_data': assay_type, 'FDR': qval, 'R': valid_rho_gat[i-1,2], 'NLL': valid_loss_gat[i-1,2]}, ignore_index=True)

            df = df.append({'epoch': epoch, 'cell': cell_line, 'Method': 'Epi-CNN', 'Set': 'All', 'valid_chr': valid_chr_str, 'test_chr': test_chr_str, 
                        'n_gene_test': n_gene[0,0], '3D_data': assay_type, 'FDR': qval, 'R': valid_rho_cnn[0,0], 'NLL': valid_loss_cnn[0,0]}, ignore_index=True)
            df = df.append({'epoch': epoch, 'cell': cell_line, 'Method': 'Epi-CNN', 'Set': 'Expressed', 'valid_chr': valid_chr_str, 'test_chr': test_chr_str, 
                        'n_gene_test': n_gene[0,1], '3D_data': assay_type, 'FDR': qval, 'R': valid_rho_cnn[0,1], 'NLL': valid_loss_cnn[0,1]}, ignore_index=True)
            df = df.append({'epoch': epoch, 'cell': cell_line, 'Method': 'Epi-CNN', 'Set': 'Interacted', 'valid_chr': valid_chr_str, 'test_chr': test_chr_str, 
                        'n_gene_test': n_gene[0,2], '3D_data': assay_type, 'FDR': qval, 'R': valid_rho_cnn[0,2], 'NLL': valid_loss_cnn[0,2]}, ignore_index=True)

        df.to_csv(data_path+'/results/csv/cage_prediction/epi_models/R_NLL_epi_models_distal_reg_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.csv', sep="\t", index=False)
        

    ##### plot swarm plots #####
    g = sns.catplot(x="epoch", y="R",
                row='Set', hue="Method",
                data=df, kind="strip", palette={"Epi-GraphReg": "orange", "Epi-CNN": "deepskyblue"},
                height=3, aspect=3, sharey=False)
    g.set_xticklabels(rotation=90)
    plt.savefig('../figs/distal_reg/swarmplot_R_vs_epoch_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.pdf')


    g = sns.catplot(x="epoch", y="NLL",
                row='Set', hue="Method",
                data=df, kind="strip", palette={"Epi-GraphReg": "orange", "Epi-CNN": "deepskyblue"},
                height=3, aspect=3, sharey=False)
    g.set_xticklabels(rotation=90)
    plt.savefig('../figs/distal_reg/swarmplot_NLL_vs_epoch_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'.pdf')

    ##### plot scatter plots #####
    df = df_all_predictions
    df['pred_cage_epi_graphreg_log2'] = np.log2(df['pred_cage_epi_graphreg']+1)
    df['pred_cage_epi_cnn_log2'] = np.log2(df['pred_cage_epi_cnn']+1)
    df['true_cage_log2'] = np.log2(df['true_cage']+1)
    df['n_contact_log2'] = np.log2(df['n_contact'].astype(np.float32)+1)

    for i in range(1,1+64):

        if i == 1:
            epoch = 'best'
        else:
            epoch = str((i-1) * 10)

        df_expressed = df[df['true_cage']>=5].reset_index()
        df_expressed = df_expressed[df_expressed['epoch'] == 10 * (i-1)].reset_index()

        r_graphreg_ex = np.corrcoef(df_expressed['true_cage_log2'].values, df_expressed['pred_cage_epi_graphreg_log2'])[0,1]
        nll_graphreg_ex = df_expressed['nll_epi_graphreg'].values
        nll_graphreg_ex_mean = np.mean(nll_graphreg_ex)

        fig, ax = plt.subplots()
        g = sns.scatterplot(data=df_expressed, x="true_cage_log2", y="pred_cage_epi_graphreg_log2", hue='n_contact_log2', alpha=.5, palette=plt.cm.get_cmap('viridis_r'), ax=ax)
        norm = plt.Normalize(df_expressed['n_contact_log2'].min(), df_expressed['n_contact_log2'].max())
        sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=norm)
        sm.set_array([])
        ax.get_legend().remove()
        ax.figure.colorbar(sm, label="log2 (N + 1)")
        ax.set_title('Epi-GraphReg | {} | R = {:5.3f} | NLL = {:6.2f}'.format(cell_line, r_graphreg_ex, nll_graphreg_ex_mean))
        ax.set_xlabel('log2 (true + 1)')
        ax.set_ylabel('log2 (pred + 1)')
        plt.tight_layout()
        plt.savefig('../figs/distal_reg/scatterplots/scatterplot_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'_epoch_'+epoch+'.pdf')
        plt.close()

