from __future__ import division
import sys
sys.path.insert(0,'../train')

#from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from gat_layer import GraphAttention
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow import keras
import matplotlib.pyplot as plt
import time
from scipy.stats import spearmanr
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from scipy.stats import wilcoxon
import shap
from statannot import add_stat_annotation
import seaborn as sns
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import pyBigWig
from collections import Counter
# Needed for Illustrator
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = ['Arial','Helvetica']

##### Input 
data_path = '/media/labuser/STORAGE/GraphReg'   # data path
qval = .1                                       # 0.1, 0.01, 0.001
assay_type = 'HiChIP'                           # HiChIP, HiC, MicroC, HiCAR

if qval == 0.1:
    fdr = '1'
elif qval == 0.01:
    fdr = '01'
elif qval == 0.001:
    fdr = '001'

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
        X = tf.reshape(X_epi, [3*T, b, F])
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
        #Y = tf.gather(Y, idx, axis=1)

    else:
        X_epi = 0
        X = 0
        Y = 0
        adj = 0
        tss_idx = 0
        idx = 0
        pos = 0
    return data_exist, X, X_epi, Y, adj, idx, tss_idx, pos

def calculate_loss(cell_lines, gene_names_list, gene_tss_list, chr_list, batch_size, data_frame_tap_seq, n_enh, L):
    abc_all = np.array([])
    logFC_tap_seq_all = np.array([])
    e_graphreg_fa_all_deepshap = np.array([])
    e_graphreg_fa_all_saliency = np.array([])
    s_graphreg_fa_all_saliency = np.array([])
    e_cnn_fa_all_deepshap = np.array([])
    e_cnn_fa_all_saliency = np.array([])
    s_cnn_fa_all_saliency = np.array([])
    significant_tap_seq = np.array([], dtype=int)
    sp_abc = np.array([])
    sp_e_graphreg_deepshap = np.array([])
    sp_e_graphreg_saliency = np.array([])
    sp_s_graphreg_saliency = np.array([])
    sp_e_cnn_deepshap = np.array([])
    sp_e_cnn_saliency = np.array([])
    sp_s_cnn_saliency = np.array([])
    ap_abc = np.array([])
    auprc_abc = np.array([])
    ap_e_graphreg_deepshap = np.array([])
    ap_e_graphreg_saliency = np.array([])
    ap_s_graphreg_saliency = np.array([])
    ap_e_cnn_deepshap = np.array([])
    ap_e_cnn_saliency = np.array([])
    ap_s_cnn_saliency = np.array([])
    auprc_e_graphreg_deepshap = np.array([])
    auprc_e_graphreg_saliency = np.array([])
    auprc_s_graphreg_saliency = np.array([])
    auprc_e_cnn_deepshap = np.array([])
    auprc_e_cnn_saliency = np.array([])
    auprc_s_cnn_saliency = np.array([])
    D = np.array([])

    for num, cell_line in enumerate(cell_lines):
        for i, chrm in enumerate(chr_list):
            print(gene_names_list[i])
            file_name = data_path+'/data/tfrecords/tfr_epi_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'_'+chrm+'.tfr'
            iterator = dataset_iterator(file_name, batch_size)
            while True:
                data_exist, X, X_epi, Y, adj, idx, tss_idx, pos = read_tf_record_1shot(iterator)
                if data_exist:
                    if tf.reduce_sum(tf.gather(tss_idx, idx)) > 0:
                        if (pos[-1] < 1e15  and gene_tss_list[i] >= pos[0]+2000000 and gene_tss_list[i] < pos[0]+4000000):

                            print('start position: ', pos[0])

                            ############### TAP-seq analysis ###############

                            cut_tap_seq = data_frame_tap_seq.loc[data_frame_tap_seq['Gene'] == gene_names_list[i]]
                            d = np.minimum(np.abs(cut_tap_seq['Gene.TSS'] - cut_tap_seq['start']), np.abs(cut_tap_seq['Gene.TSS'] - cut_tap_seq['end'])).values
                            start_tap_seq = cut_tap_seq['start'].values
                            end_tap_seq = cut_tap_seq['end'].values
                            logFC_tap_seq = cut_tap_seq['Log-fold change'].values
                            abc_tap_seq = cut_tap_seq['ABC.Score'].values
                            significant_tap_seq_bool = cut_tap_seq['Significant'].values
                            significant_tap_seq_gene = np.array([])
                            for ii, gg in enumerate(significant_tap_seq_bool):
                                if (gg == True and logFC_tap_seq[ii]<0):
                                    significant_tap_seq_gene = np.append(significant_tap_seq_gene, 1)
                                elif (gg == True and logFC_tap_seq[ii]>0):
                                    significant_tap_seq_gene = np.append(significant_tap_seq_gene, 2)
                                else:
                                    significant_tap_seq_gene = np.append(significant_tap_seq_gene, 0)

                            print('len significant_tap_seq_gene', len(significant_tap_seq_gene))
                            print('num 0:', np.sum(significant_tap_seq_gene==0))
                            print('num 1:', np.sum(significant_tap_seq_gene==1))
                            print('num 2:', np.sum(significant_tap_seq_gene==2))
                            print('len abc_tap_seq:', len(abc_tap_seq))

                            start_idx = np.floor((start_tap_seq - pos[0])/100).astype('int64')
                            end_idx = np.ceil((end_tap_seq - pos[0])/100).astype('int64')
                            mid_pos = np.round((start_tap_seq + end_tap_seq)/2)
                            mid_idx = np.round((mid_pos - pos[0])/100).astype('int64')

                            expr_change_percent_vec = np.zeros(60000)
                            abc_vec = np.zeros(60000)
                            for k, idx in enumerate(mid_idx):
                                expr_change_percent_vec[idx] = logFC_tap_seq[k]
                                abc_vec[idx] = abc_tap_seq[k]

                            Y_true = np.zeros(len(significant_tap_seq_gene), dtype=int)
                            idx_sig = np.where(significant_tap_seq_gene==1)[0]
                            Y_true[idx_sig] = 1
                            if (len(Y_true) >= 10 and len(idx_sig) >= n_enh):
                                D = np.append(D,d)
                                significant_tap_seq = np.append(significant_tap_seq, significant_tap_seq_gene)
                                logFC_tap_seq_all = np.append(logFC_tap_seq_all, logFC_tap_seq)
                                abc_all = np.append(abc_all, abc_tap_seq)
                                sp_abc = np.append(sp_abc, spearmanr(abc_tap_seq, logFC_tap_seq)[0])

                                Y_pred_abc = abc_tap_seq

                                precision_abc, recall_abc, thresholds_abc = precision_recall_curve(Y_true, Y_pred_abc)
                                ap_abc = np.append(ap_abc, average_precision_score(Y_true, Y_pred_abc))
                                auprc_abc = np.append(auprc_abc, auc(recall_abc, precision_abc))
                                print('ABC AP score: ', average_precision_score(Y_true, Y_pred_abc))
                                print('ABC AUPRC score: ', auc(recall_abc, precision_abc))

                            ############# Feature Attribution of CNN #############

                            ##### Epi-CNN deepshap #####
                            saliency_method = 'deepshap'
                            shap_values_e_cnn = np.load(data_path+'/results/numpy/feature_attribution/Epi-CNN_tap_seq_'+saliency_method+'_'+cell_line+'_'+gene_names_list[i]+'.npy')
                            scores_cnn = K.reshape(shap_values_e_cnn, [60000,3])
                            scores_cnn = K.sum(scores_cnn, axis = 1).numpy()
                            print('Epi-CNN deepshap: ', scores_cnn.shape, np.min(scores_cnn), np.max(scores_cnn), np.mean(scores_cnn))

                            cnn_fa_gene = np.array([])
                            cnt = -1
                            for s_idx, e_idx in zip(start_idx, end_idx):
                                cnt = cnt + 1
                                mid_idx = int((s_idx+e_idx)/2)
                                cnn_fa_gene = np.append(cnn_fa_gene, np.max(scores_cnn[mid_idx-L:mid_idx+L+1])/np.max(scores_cnn))
                                #cnn_fa_gene = np.append(cnn_fa_gene, np.max(scores_cnn[s_idx:e_idx]))
                            #cnn_fa_gene = cnn_fa_gene/(np.max(cnn_fa_gene)+.1e-10)

                            Y_true = np.zeros(len(significant_tap_seq_gene), dtype=int)
                            idx_sig = np.where(significant_tap_seq_gene==1)[0]
                            Y_true[idx_sig] = 1
                            if (len(Y_true) >= 10 and len(idx_sig) >= n_enh):
                                e_cnn_fa_all_deepshap = np.append(e_cnn_fa_all_deepshap, cnn_fa_gene)
                                sp_e_cnn_deepshap = np.append(sp_e_cnn_deepshap, spearmanr(cnn_fa_gene, logFC_tap_seq)[0])
                                Y_pred_cnn = cnn_fa_gene
                                precision_cnn, recall_cnn, thresholds_cnn = precision_recall_curve(Y_true, Y_pred_cnn)
                                ap_e_cnn_deepshap = np.append(ap_e_cnn_deepshap, average_precision_score(Y_true, Y_pred_cnn))
                                auprc_e_cnn_deepshap = np.append(auprc_e_cnn_deepshap, auc(recall_cnn, precision_cnn))
                                print('Epi-CNN Deepshap AP score: ', average_precision_score(Y_true, Y_pred_cnn))
                                print('Epi-CNN Deepshap AUPRC score: ', auc(recall_cnn, precision_cnn))

                            ##### Epi-CNN saliency #####
                            saliency_method = 'saliency'
                            grads_cnn = np.load(data_path+'/results/numpy/feature_attribution/Epi-CNN_tap_seq_'+saliency_method+'_'+cell_line+'_'+gene_names_list[i]+'.npy')
                            scores_cnn = K.reshape(grads_cnn * X_epi, [60000,3])
                            scores_cnn = K.sum(scores_cnn, axis = 1).numpy()
                            print('Epi-CNN saliency: ', scores_cnn.shape, np.min(scores_cnn), np.max(scores_cnn), np.mean(scores_cnn))

                            cnn_fa_gene = np.array([])
                            cnt = -1
                            for s_idx, e_idx in zip(start_idx, end_idx):
                                cnt = cnt + 1
                                mid_idx = int((s_idx+e_idx)/2)
                                cnn_fa_gene = np.append(cnn_fa_gene, np.max(scores_cnn[mid_idx-L:mid_idx+L+1])/np.max(scores_cnn))
                                #cnn_fa_gene = np.append(cnn_fa_gene, np.max(scores_cnn[s_idx:e_idx]))
                            #cnn_fa_gene = cnn_fa_gene/(np.max(cnn_fa_gene)+.1e-10)

                            Y_true = np.zeros(len(significant_tap_seq_gene), dtype=int)
                            idx_sig = np.where(significant_tap_seq_gene==1)[0]
                            Y_true[idx_sig] = 1
                            if (len(Y_true) >= 10 and len(idx_sig) >= n_enh):
                                e_cnn_fa_all_saliency = np.append(e_cnn_fa_all_saliency, cnn_fa_gene)
                                sp_e_cnn_saliency = np.append(sp_e_cnn_saliency, spearmanr(cnn_fa_gene, logFC_tap_seq)[0])
                                Y_pred_cnn = cnn_fa_gene
                                precision_cnn, recall_cnn, thresholds_cnn = precision_recall_curve(Y_true, Y_pred_cnn)
                                ap_e_cnn_saliency = np.append(ap_e_cnn_saliency, average_precision_score(Y_true, Y_pred_cnn))
                                auprc_e_cnn_saliency = np.append(auprc_e_cnn_saliency, auc(recall_cnn, precision_cnn))
                                print('Epi-CNN Saliency AP score: ', average_precision_score(Y_true, Y_pred_cnn))
                                print('Epi-CNN Saliency AUPRC score: ', auc(recall_cnn, precision_cnn))

                            ##### Seq-CNN saliency #####
                            saliency_method = 'saliency'
                            grads_cnn = np.load(data_path+'/results/numpy/feature_attribution/Seq-CNN_tap_seq_'+saliency_method+'_'+cell_line+'_'+gene_names_list[i]+'.npy')
                            scores_cnn = K.reshape(grads_cnn, [60000,64])
                            scores_cnn = K.sum(scores_cnn, axis = 1).numpy()
                            print('Seq-CNN saliency: ', scores_cnn.shape, np.min(scores_cnn), np.max(scores_cnn), np.mean(scores_cnn))

                            cnn_fa_gene = np.array([])
                            cnt = -1
                            for s_idx, e_idx in zip(start_idx, end_idx):
                                cnt = cnt + 1
                                mid_idx = int((s_idx+e_idx)/2)
                                cnn_fa_gene = np.append(cnn_fa_gene, np.max(scores_cnn[mid_idx-L:mid_idx+L+1])/np.max(scores_cnn))
                                #cnn_fa_gene = np.append(cnn_fa_gene, np.max(scores_cnn[s_idx:e_idx]))
                            #cnn_fa_gene = cnn_fa_gene/(np.max(cnn_fa_gene)+.1e-10)

                            Y_true = np.zeros(len(significant_tap_seq_gene), dtype=int)
                            idx_sig = np.where(significant_tap_seq_gene==1)[0]
                            Y_true[idx_sig] = 1
                            if (len(Y_true) >= 10 and len(idx_sig) >= n_enh):
                                s_cnn_fa_all_saliency = np.append(s_cnn_fa_all_saliency, cnn_fa_gene)
                                sp_s_cnn_saliency = np.append(sp_s_cnn_saliency, spearmanr(cnn_fa_gene, logFC_tap_seq)[0])
                                Y_pred_cnn = cnn_fa_gene
                                precision_cnn, recall_cnn, thresholds_cnn = precision_recall_curve(Y_true, Y_pred_cnn)
                                ap_s_cnn_saliency = np.append(ap_s_cnn_saliency, average_precision_score(Y_true, Y_pred_cnn))
                                auprc_s_cnn_saliency = np.append(auprc_s_cnn_saliency, auc(recall_cnn, precision_cnn))
                                print('Seq-CNN Saliency AP score: ', average_precision_score(Y_true, Y_pred_cnn))
                                print('Seq-CNN Saliency AUPRC score: ', auc(recall_cnn, precision_cnn))


                            ############# Feature Attribution of GraphReg #############

                            ##### Epi-GraphReg deepshap #####
                            saliency_method = 'deepshap'
                            shap_values_e_graphreg = np.load(data_path+'/results/numpy/feature_attribution/Epi-GraphReg_tap_seq_'+saliency_method+'_'+cell_line+'_'+gene_names_list[i]+'.npy')
                            scores_graphreg = K.reshape(shap_values_e_graphreg, [60000,3])
                            scores_graphreg = K.sum(scores_graphreg, axis = 1).numpy()
                            print('Epi-GraphReg deepshap: ', scores_graphreg.shape, np.min(scores_graphreg), np.max(scores_graphreg), np.mean(scores_graphreg))

                            graphreg_fa_gene = np.array([])
                            cnt = -1
                            for s_idx, e_idx in zip(start_idx, end_idx):
                                cnt = cnt + 1
                                mid_idx = int((s_idx+e_idx)/2)
                                graphreg_fa_gene = np.append(graphreg_fa_gene, np.max(scores_graphreg[mid_idx-L:mid_idx+L+1])/np.max(scores_graphreg))
                                #graphreg_fa_gene = np.append(graphreg_fa_gene, np.max(scores_graphreg[s_idx:e_idx]))
                            #graphreg_fa_gene = graphreg_fa_gene/(np.max(graphreg_fa_gene)+.1e-10)

                            Y_true = np.zeros(len(significant_tap_seq_gene), dtype=int)
                            idx_sig = np.where(significant_tap_seq_gene==1)[0]
                            Y_true[idx_sig] = 1
                            if (len(Y_true) >= 10 and len(idx_sig) >= n_enh):
                                e_graphreg_fa_all_deepshap = np.append(e_graphreg_fa_all_deepshap, graphreg_fa_gene)
                                sp_e_graphreg_deepshap = np.append(sp_e_graphreg_deepshap, spearmanr(graphreg_fa_gene, logFC_tap_seq)[0])
                                Y_pred_graphreg = graphreg_fa_gene
                                precision_graphreg, recall_graphreg, thresholds_graphreg = precision_recall_curve(Y_true, Y_pred_graphreg)
                                ap_e_graphreg_deepshap = np.append(ap_e_graphreg_deepshap, average_precision_score(Y_true, Y_pred_graphreg))
                                auprc_e_graphreg_deepshap = np.append(auprc_e_graphreg_deepshap, auc(recall_graphreg, precision_graphreg))
                                print('Epi-GraphReg Deepshap AP score: ', average_precision_score(Y_true, Y_pred_graphreg))
                                print('Epi-GraphReg Deepshap AUPRC score: ', auc(recall_graphreg, precision_graphreg))

                            ##### Epi-GraphReg saliency #####
                            saliency_method = 'saliency'
                            shap_values_e_graphreg = np.load(data_path+'/results/numpy/feature_attribution/Epi-GraphReg_tap_seq_'+saliency_method+'_'+cell_line+'_'+gene_names_list[i]+'.npy')
                            scores_graphreg = K.reshape(shap_values_e_graphreg * X_epi, [60000,3])
                            scores_graphreg = K.sum(scores_graphreg, axis = 1).numpy()
                            print('Epi-GraphReg saliency: ', scores_graphreg.shape, np.min(scores_graphreg), np.max(scores_graphreg), np.mean(scores_graphreg))

                            graphreg_fa_gene = np.array([])
                            cnt = -1
                            for s_idx, e_idx in zip(start_idx, end_idx):
                                cnt = cnt + 1
                                mid_idx = int((s_idx+e_idx)/2)
                                graphreg_fa_gene = np.append(graphreg_fa_gene, np.max(scores_graphreg[mid_idx-L:mid_idx+L+1])/np.max(scores_graphreg))
                                #graphreg_fa_gene = np.append(graphreg_fa_gene, np.max(scores_graphreg[s_idx:e_idx]))
                            #graphreg_fa_gene = graphreg_fa_gene/(np.max(graphreg_fa_gene)+.1e-10)

                            Y_true = np.zeros(len(significant_tap_seq_gene), dtype=int)
                            idx_sig = np.where(significant_tap_seq_gene==1)[0]
                            Y_true[idx_sig] = 1
                            if (len(Y_true) >= 10 and len(idx_sig) >= n_enh):
                                e_graphreg_fa_all_saliency = np.append(e_graphreg_fa_all_saliency, graphreg_fa_gene)
                                sp_e_graphreg_saliency = np.append(sp_e_graphreg_saliency, spearmanr(graphreg_fa_gene, logFC_tap_seq)[0])
                                Y_pred_graphreg = graphreg_fa_gene
                                precision_graphreg, recall_graphreg, thresholds_graphreg = precision_recall_curve(Y_true, Y_pred_graphreg)
                                ap_e_graphreg_saliency = np.append(ap_e_graphreg_saliency, average_precision_score(Y_true, Y_pred_graphreg))
                                auprc_e_graphreg_saliency = np.append(auprc_e_graphreg_saliency, auc(recall_graphreg, precision_graphreg))
                                print('Epi-GraphReg Saliency AP score: ', average_precision_score(Y_true, Y_pred_graphreg))
                                print('Epi-GraphReg Saliency AUPRC score: ', auc(recall_graphreg, precision_graphreg))

                            ##### Seq-GraphReg saliency #####
                            saliency_method = 'saliency'
                            shap_values_s_graphreg = np.load(data_path+'/results/numpy/feature_attribution/Seq-GraphReg_tap_seq_'+saliency_method+'_'+cell_line+'_'+gene_names_list[i]+'.npy')
                            scores_graphreg = K.reshape(shap_values_s_graphreg, [60000,64])
                            scores_graphreg = K.sum(scores_graphreg, axis = 1).numpy()
                            print('Seq-GraphReg saliency: ', scores_graphreg.shape, np.min(scores_graphreg), np.max(scores_graphreg), np.mean(scores_graphreg))

                            graphreg_fa_gene = np.array([])
                            cnt = -1
                            for s_idx, e_idx in zip(start_idx, end_idx):
                                cnt = cnt + 1
                                mid_idx = int((s_idx+e_idx)/2)
                                graphreg_fa_gene = np.append(graphreg_fa_gene, np.max(scores_graphreg[mid_idx-L:mid_idx+L+1])/np.max(scores_graphreg))
                                #graphreg_fa_gene = np.append(graphreg_fa_gene, np.max(scores_graphreg[s_idx:e_idx]))
                            #graphreg_fa_gene = graphreg_fa_gene/(np.max(graphreg_fa_gene)+.1e-10)

                            Y_true = np.zeros(len(significant_tap_seq_gene), dtype=int)
                            idx_sig = np.where(significant_tap_seq_gene==1)[0]
                            Y_true[idx_sig] = 1
                            if (len(Y_true) >= 10 and len(idx_sig) >= n_enh):
                                s_graphreg_fa_all_saliency = np.append(s_graphreg_fa_all_saliency, graphreg_fa_gene)
                                sp_s_graphreg_saliency = np.append(sp_s_graphreg_saliency, spearmanr(graphreg_fa_gene, logFC_tap_seq)[0])
                                Y_pred_graphreg = graphreg_fa_gene
                                precision_graphreg, recall_graphreg, thresholds_graphreg = precision_recall_curve(Y_true, Y_pred_graphreg)
                                ap_s_graphreg_saliency = np.append(ap_s_graphreg_saliency, average_precision_score(Y_true, Y_pred_graphreg))
                                auprc_s_graphreg_saliency = np.append(auprc_s_graphreg_saliency, auc(recall_graphreg, precision_graphreg))
                                print('Seq-GraphReg Saliency AP score: ', average_precision_score(Y_true, Y_pred_graphreg))
                                print('Seq-GraphReg Saliency AUPRC score: ', auc(recall_graphreg, precision_graphreg))
                else:
                    break

    return (sp_abc, 
            sp_e_graphreg_deepshap, sp_e_graphreg_saliency, sp_s_graphreg_saliency,
            sp_e_cnn_deepshap, sp_e_cnn_saliency, sp_s_cnn_saliency,
            auprc_abc, 
            auprc_e_graphreg_deepshap, auprc_e_graphreg_saliency, auprc_s_graphreg_saliency,
            auprc_e_cnn_deepshap, auprc_e_cnn_saliency, auprc_s_cnn_saliency, 
            ap_abc, 
            ap_e_graphreg_deepshap, ap_e_graphreg_saliency, ap_s_graphreg_saliency,
            ap_e_cnn_deepshap, ap_e_cnn_saliency, ap_s_cnn_saliency, 
            abc_all, 
            e_graphreg_fa_all_deepshap, e_graphreg_fa_all_saliency, s_graphreg_fa_all_saliency,
            e_cnn_fa_all_deepshap, e_cnn_fa_all_saliency, s_cnn_fa_all_saliency, 
            logFC_tap_seq_all, significant_tap_seq, D)


#################### load model ####################
batch_size = 1
organism = 'human'            # human/mouse
cell_line = ['K562']          # K562/GM12878/mESC
dist = 0                      # minimum distance from TSS
n_enh = 1                     # number of enhancers >= n_enh
L = 40

chr_list = np.array([])
gene_tss_list = np.array([], dtype=np.int)
data_frame_tap_seq = pd.read_csv(data_path+'/data/csv/TAP_seq_enhancer_with_abc.csv')
data_frame_tap_seq = data_frame_tap_seq[data_frame_tap_seq['Distance to TSS'] > dist]
print(len(data_frame_tap_seq))
gene_names_list = np.unique(data_frame_tap_seq['Gene'].values)
for gene in gene_names_list:
    cut = data_frame_tap_seq.loc[data_frame_tap_seq['Gene'] == gene]
    chr_list = np.append(chr_list, cut['chr'].values[0])
    gene_tss_list = np.append(gene_tss_list, cut['Gene.TSS'].values[0])

# gene_names_list = ['IFITM1']
# chr_list = ['chr11']
# gene_tss_list = [313504]

print(len(gene_names_list), gene_names_list)
print(len(chr_list), chr_list)
print(len(gene_tss_list), gene_tss_list)


(sp_abc, 
    sp_e_graphreg_deepshap, sp_e_graphreg_saliency, sp_s_graphreg_saliency,
    sp_e_cnn_deepshap, sp_e_cnn_saliency, sp_s_cnn_saliency,
    auprc_abc, 
    auprc_e_graphreg_deepshap, auprc_e_graphreg_saliency, auprc_s_graphreg_saliency,
    auprc_e_cnn_deepshap, auprc_e_cnn_saliency, auprc_s_cnn_saliency, 
    ap_abc, 
    ap_e_graphreg_deepshap, ap_e_graphreg_saliency, ap_s_graphreg_saliency,
    ap_e_cnn_deepshap, ap_e_cnn_saliency, ap_s_cnn_saliency, 
    abc_all, 
    e_graphreg_fa_all_deepshap, e_graphreg_fa_all_saliency, s_graphreg_fa_all_saliency,
    e_cnn_fa_all_deepshap, e_cnn_fa_all_saliency, s_cnn_fa_all_saliency, 
    logFC_tap_seq_all, significant_tap_seq, D) = calculate_loss(cell_line, gene_names_list, gene_tss_list, chr_list, batch_size, data_frame_tap_seq, n_enh, L)
    

color_significant = []
for ii in significant_tap_seq:
    if ii == 0:
        color_significant.append('grey')
    elif ii == 1:
        color_significant.append('red')
    elif ii == 2:
        color_significant.append('blue')

idx_0 = np.where(significant_tap_seq==0)[0]
idx_1 = np.where(significant_tap_seq==1)[0]
idx_2 = np.where(significant_tap_seq==2)[0]

def box_plot(data, edge_color, fill_color):
    bp = ax.boxplot(data, patch_artist=True)

    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)


##### Box Plots #####
plt.close("all")
SP_list = [sp_abc, sp_e_graphreg_deepshap, sp_e_cnn_deepshap, sp_e_graphreg_saliency, sp_e_cnn_saliency, sp_s_graphreg_saliency, sp_s_cnn_saliency]
fig = plt.figure(1, figsize=(14, 12))
fig.tight_layout()
ax = fig.add_subplot(111)
bp = ax.boxplot(SP_list[0], positions=[1], widths = .4, patch_artist=True)
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], color='black')
for patch in bp['boxes']:
    patch.set(facecolor='lightgreen')   #deepskyblue
bp = ax.boxplot(SP_list[1:6:2], positions=[2,4,6], widths = .4, patch_artist=True)
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], color='black')
for patch in bp['boxes']:
    patch.set(facecolor='orange')   #deepskyblue
bp = ax.boxplot(SP_list[2:7:2], positions=[3,5,7], widths = .4, patch_artist=True)
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], color='black')
for patch in bp['boxes']:
    patch.set(facecolor='deepskyblue')   #deepskyblue
ax.set_xticklabels(['ABC', 'Epi-GraphReg Deepshap', 'Epi-GraphReg Saliency', 'Seq-GraphReg Saliency', 'Epi-CNN Deepshap', 'Epi-CNN Saliency', 'Seq-CNN Saliency'], rotation = 20, ha='right')
ax.set_ylabel("Spearman's Correlation", fontsize=40)
ax.set_title('Distribution of '+str(len(auprc_abc))+' Genes', fontsize=40)
ax.tick_params(axis='y', labelsize=30)
ax.tick_params(axis='x', labelsize=20)
ax.grid(axis='y')
fig.savefig('../figs/tap_seq/SP_boxplot.png')

plt.close("all")
aucpr_list = [auprc_abc, auprc_e_graphreg_deepshap, auprc_e_cnn_deepshap, auprc_e_graphreg_saliency, auprc_e_cnn_saliency, auprc_s_graphreg_saliency, auprc_s_cnn_saliency]
fig = plt.figure(1, figsize=(14, 12))
fig.tight_layout()
ax = fig.add_subplot(111)
bp = ax.boxplot(aucpr_list[0], positions=[1], widths = .4, patch_artist=True)
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], color='black')
for patch in bp['boxes']:
    patch.set(facecolor='lightgreen')   #deepskyblue
bp = ax.boxplot(aucpr_list[1:6:2], positions=[2,4,6], widths = .4, patch_artist=True)
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], color='black')
for patch in bp['boxes']:
    patch.set(facecolor='orange')   #deepskyblue
bp = ax.boxplot(aucpr_list[2:7:2], positions=[3,5,7], widths = .4, patch_artist=True)
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], color='black')
for patch in bp['boxes']:
    patch.set(facecolor='deepskyblue')   #deepskyblue
ax.set_xticklabels(['ABC', 'Epi-GraphReg Deepshap', 'Epi-GraphReg Saliency', 'Seq-GraphReg Saliency', 'Epi-CNN Deepshap', 'Epi-CNN Saliency', 'Seq-CNN Saliency'], rotation = 20, ha='right')
ax.set_ylabel('AUPRC', fontsize=40)
ax.set_title('Distribution of '+str(len(auprc_abc))+' Genes', fontsize=40)
ax.tick_params(axis='y', labelsize=30)
ax.tick_params(axis='x', labelsize=20)
ax.grid(axis='y')
fig.savefig('../figs/tap_seq/AUPRC_boxplot.png')

#### seaborn boxplot

df = pd.DataFrame(columns=['Method', 'FA', 'auPR', 'ID'])
N = len(auprc_abc)
for i in range(N):
    df = df.append({'Method': 'ABC', 'FA': 'N/A', 'auPR': auprc_abc[i], 'ID': i}, ignore_index=True)
for i in range(N):
    df = df.append({'Method': 'GraphReg', 'FA': 'Epi/DeepSHAP', 'auPR': auprc_e_graphreg_deepshap[i], 'ID': i}, ignore_index=True)
for i in range(N):
    df = df.append({'Method': 'CNN', 'FA': 'Epi/DeepSHAP', 'auPR': auprc_e_cnn_deepshap[i], 'ID': i}, ignore_index=True)
for i in range(N):
    df = df.append({'Method': 'GraphReg', 'FA': 'Epi/Saliency', 'auPR': auprc_e_graphreg_saliency[i], 'ID': i}, ignore_index=True)
for i in range(N):
    df = df.append({'Method': 'CNN', 'FA': 'Epi/Saliency', 'auPR': auprc_e_cnn_saliency[i], 'ID': i}, ignore_index=True)
for i in range(N):
    df = df.append({'Method': 'GraphReg', 'FA': 'Seq/Saliency', 'auPR': auprc_s_graphreg_saliency[i], 'ID': i}, ignore_index=True)
for i in range(N):
    df = df.append({'Method': 'CNN', 'FA': 'Seq/Saliency', 'auPR': auprc_s_cnn_saliency[i], 'ID': i}, ignore_index=True)

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(9, 5))
b=sns.boxplot(x='FA', y='auPR', data=df, hue='Method', palette={"GraphReg": "orange", "CNN": "deepskyblue", "ABC": "lightgreen"}, order = ['Epi/DeepSHAP', 'Epi/Saliency', 'Seq/Saliency', 'N/A'], ax=ax1)
ax1.yaxis.set_tick_params(labelsize=20)
ax1.xaxis.set_tick_params(labelsize=20)
ax1.set_title('TAP-seq | Distribution of '+str(len(auprc_abc))+' Genes', fontsize=20)
b.set_xlabel("",fontsize=20)
b.set_ylabel("auPR",fontsize=20)
loc, labels = plt.xticks()
b.set_xticklabels(labels, rotation=15)
plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
plt.setp(ax1.get_legend().get_texts(), fontsize='20')
plt.setp(ax1.get_legend().get_title(), fontsize='20')
plt.tight_layout()
fig.savefig('../figs/tap_seq/AUPRC_boxplot.pdf')

####

plt.close("all")
ap_list = [ap_abc, ap_e_graphreg_deepshap, ap_e_cnn_deepshap, ap_e_graphreg_saliency, ap_e_cnn_saliency, ap_s_graphreg_saliency, ap_s_cnn_saliency]
fig = plt.figure(1, figsize=(14, 12))
fig.tight_layout()
ax = fig.add_subplot(111)
bp = ax.boxplot(ap_list[0], positions=[1], widths = .4, patch_artist=True)
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], color='black')
for patch in bp['boxes']:
    patch.set(facecolor='lightgreen')   #deepskyblue
bp = ax.boxplot(ap_list[1:6:2], positions=[2,4,6], widths = .4, patch_artist=True)
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], color='black')
for patch in bp['boxes']:
    patch.set(facecolor='orange')   #deepskyblue
bp = ax.boxplot(ap_list[2:7:2], positions=[3,5,7], widths = .4, patch_artist=True)
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], color='black')
for patch in bp['boxes']:
    patch.set(facecolor='deepskyblue')   #deepskyblue
ax.set_xticklabels(['ABC', 'Epi-GraphReg Deepshap', 'Epi-GraphReg Saliency', 'Seq-GraphReg Saliency', 'Epi-CNN Deepshap', 'Epi-CNN Saliency', 'Seq-CNN Saliency'], rotation = 20, ha='right')
ax.set_ylabel('Average Precision', fontsize=40)
ax.set_title('Distribution of '+str(len(auprc_abc))+' Genes', fontsize=40)
ax.tick_params(axis='y', labelsize=30)
ax.tick_params(axis='x', labelsize=20)
ax.grid(axis='y')
fig.savefig('../figs/tap_seq/AP_boxplot.png')

###### Scatter Plots #####

plt.figure(figsize=(15,15))
plt.scatter(abc_all[idx_0], logFC_tap_seq_all[idx_0], color="grey", s=80, alpha=.5)
plt.scatter(abc_all[idx_2], logFC_tap_seq_all[idx_2], color="blue", s=80, alpha=.5)
plt.scatter(abc_all[idx_1], logFC_tap_seq_all[idx_1], color="red", s=80, alpha=.5)
plt.xlabel("ABC Score", fontsize=40)
plt.ylabel("Log Fold Change", fontsize=40)
#plt.xlim((0,14))
#plt.ylim((0,14))
plt.title('DE-G Pairs', fontsize=40)
props = dict(boxstyle='round', facecolor='white', alpha=1)
plt.text(2e-6,-1.5, r"$\rho$ = "+str(spearmanr(abc_all[idx_1], logFC_tap_seq_all[idx_1])[0].astype(np.float16))+ ' (Decrease)',
         horizontalalignment='left', verticalalignment='top', bbox=props, fontsize=30)
plt.tick_params(axis='x', labelsize=30)
plt.tick_params(axis='y', labelsize=30)
plt.xscale('log',basex=10) 
plt.xlim((0.000001,2))
plt.legend(['Not Significant','Increase','Decrease'], loc=3, fontsize=30)
plt.grid()
plt.savefig('../figs/tap_seq/ABC_vs_LogFC_scatter_plot_K562.png')

plt.figure(figsize=(15,15))
plt.scatter(e_graphreg_fa_all_deepshap[idx_0], logFC_tap_seq_all[idx_0], color="grey", s=80, alpha=.5)
plt.scatter(e_graphreg_fa_all_deepshap[idx_2], logFC_tap_seq_all[idx_2], color="blue", s=80, alpha=.5)
plt.scatter(e_graphreg_fa_all_deepshap[idx_1], logFC_tap_seq_all[idx_1], color="red", s=80, alpha=.5)
plt.xlabel("Epi-GraphReg Deepshap Score", fontsize=40)
plt.ylabel("Log Fold Change", fontsize=40)
#plt.xlim((0,14))
#plt.ylim((0,14))
plt.title('DE-G Pairs', fontsize=40)
props = dict(boxstyle='round', facecolor='white', alpha=1)
plt.text(2e-6,-1.5, r"$\rho$ = "+str(spearmanr(e_graphreg_fa_all_deepshap[idx_1], logFC_tap_seq_all[idx_1])[0].astype(np.float16))+ ' (Decrease)',
         horizontalalignment='left', verticalalignment='top', bbox=props, fontsize=30)
plt.tick_params(axis='x', labelsize=30)
plt.tick_params(axis='y', labelsize=30)
plt.xscale('log',basex=10) 
plt.xlim((0.000001,2))
plt.legend(['Not Significant','Increase','Decrease'], loc=3, fontsize=30)
plt.grid()
plt.savefig('../figs/tap_seq/Epi-GraphReg_deepshap_vs_LogFC_scatter_plot_K562.png')

plt.figure(figsize=(15,15))
plt.scatter(e_graphreg_fa_all_saliency[idx_0], logFC_tap_seq_all[idx_0], color="grey", s=80, alpha=.5)
plt.scatter(e_graphreg_fa_all_saliency[idx_2], logFC_tap_seq_all[idx_2], color="blue", s=80, alpha=.5)
plt.scatter(e_graphreg_fa_all_saliency[idx_1], logFC_tap_seq_all[idx_1], color="red", s=80, alpha=.5)
plt.xlabel("Epi-GraphReg Saliency Score", fontsize=40)
plt.ylabel("Log Fold Change", fontsize=40)
#plt.xlim((0,14))
#plt.ylim((0,14))
plt.title('DE-G Pairs', fontsize=40)
props = dict(boxstyle='round', facecolor='white', alpha=1)
plt.text(2e-6,-1.5, r"$\rho$ = "+str(spearmanr(e_graphreg_fa_all_saliency[idx_1], logFC_tap_seq_all[idx_1])[0].astype(np.float16))+ ' (Decrease)',
         horizontalalignment='left', verticalalignment='top', bbox=props, fontsize=30)
plt.tick_params(axis='x', labelsize=30)
plt.tick_params(axis='y', labelsize=30)
plt.xscale('log',basex=10) 
plt.xlim((0.000001,2))
plt.legend(['Not Significant','Increase','Decrease'], loc=3, fontsize=30)
plt.grid()
plt.savefig('../figs/tap_seq/Epi-GraphReg_saliency_vs_LogFC_scatter_plot_K562.png')

plt.figure(figsize=(15,15))
plt.scatter(s_graphreg_fa_all_saliency[idx_0], logFC_tap_seq_all[idx_0], color="grey", s=80, alpha=.5)
plt.scatter(s_graphreg_fa_all_saliency[idx_2], logFC_tap_seq_all[idx_2], color="blue", s=80, alpha=.5)
plt.scatter(s_graphreg_fa_all_saliency[idx_1], logFC_tap_seq_all[idx_1], color="red", s=80, alpha=.5)
plt.xlabel("Seq-GraphReg Saliency Score", fontsize=40)
plt.ylabel("Log Fold Change", fontsize=40)
#plt.xlim((0,14))
#plt.ylim((0,14))
plt.title('DE-G Pairs', fontsize=40)
props = dict(boxstyle='round', facecolor='white', alpha=1)
plt.text(2e-6,-1.5, r"$\rho$ = "+str(spearmanr(s_graphreg_fa_all_saliency[idx_1], logFC_tap_seq_all[idx_1])[0].astype(np.float16))+ ' (Decrease)',
         horizontalalignment='left', verticalalignment='top', bbox=props, fontsize=30)
plt.tick_params(axis='x', labelsize=30)
plt.tick_params(axis='y', labelsize=30)
plt.xscale('log',basex=10) 
plt.xlim((0.000001,2))
plt.legend(['Not Significant','Increase','Decrease'], loc=3, fontsize=30)
plt.grid()
plt.savefig('../figs/tap_seq/Seq-GraphReg_saliency_vs_LogFC_scatter_plot_K562.png')

plt.figure(figsize=(15,15))
plt.scatter(e_cnn_fa_all_deepshap[idx_0], logFC_tap_seq_all[idx_0], color="grey", s=80, alpha=.5)
plt.scatter(e_cnn_fa_all_deepshap[idx_2], logFC_tap_seq_all[idx_2], color="blue", s=80, alpha=.5)
plt.scatter(e_cnn_fa_all_deepshap[idx_1], logFC_tap_seq_all[idx_1], color="red", s=80, alpha=.5)
plt.xlabel("Epi-CNN Deepshap Score", fontsize=40)
plt.ylabel("Log Fold Change", fontsize=40)
#plt.xlim((0,14))
#plt.ylim((0,14))
plt.title('DE-G Pairs', fontsize=40)
props = dict(boxstyle='round', facecolor='white', alpha=1)
plt.text(2e-6,-1.5, r"$\rho$ = "+str(spearmanr(e_cnn_fa_all_deepshap[idx_1], logFC_tap_seq_all[idx_1])[0].astype(np.float16))+ ' (Decrease)',
         horizontalalignment='left', verticalalignment='top', bbox=props, fontsize=30)
plt.tick_params(axis='x', labelsize=30)
plt.tick_params(axis='y', labelsize=30)
plt.xscale('log',basex=10) 
plt.xlim((0.000001,2))
plt.legend(['Not Significant','Increase','Decrease'], loc=3, fontsize=30)
plt.grid()
plt.savefig('../figs/tap_seq/Epi-CNN_deepshap_vs_LogFC_scatter_plot_K562.png')

plt.figure(figsize=(15,15))
plt.scatter(e_cnn_fa_all_saliency[idx_0], logFC_tap_seq_all[idx_0], color="grey", s=80, alpha=.5)
plt.scatter(e_cnn_fa_all_saliency[idx_2], logFC_tap_seq_all[idx_2], color="blue", s=80, alpha=.5)
plt.scatter(e_cnn_fa_all_saliency[idx_1], logFC_tap_seq_all[idx_1], color="red", s=80, alpha=.5)
plt.xlabel("Epi-CNN Saliency Score", fontsize=40)
plt.ylabel("Log Fold Change", fontsize=40)
#plt.xlim((0,14))
#plt.ylim((0,14))
plt.title('DE-G Pairs', fontsize=40)
props = dict(boxstyle='round', facecolor='white', alpha=1)
plt.text(2e-6,-1.5, r"$\rho$ = "+str(spearmanr(e_cnn_fa_all_saliency[idx_1], logFC_tap_seq_all[idx_1])[0].astype(np.float16))+ ' (Decrease)',
         horizontalalignment='left', verticalalignment='top', bbox=props, fontsize=30)
plt.tick_params(axis='x', labelsize=30)
plt.tick_params(axis='y', labelsize=30)
plt.xscale('log',basex=10) 
plt.xlim((0.000001,2))
plt.legend(['Not Significant','Increase','Decrease'], loc=3, fontsize=30)
plt.grid()
plt.savefig('../figs/tap_seq/Epi-CNN_saliency_vs_LogFC_scatter_plot_K562.png')

plt.figure(figsize=(15,15))
plt.scatter(s_cnn_fa_all_saliency[idx_0], logFC_tap_seq_all[idx_0], color="grey", s=80, alpha=.5)
plt.scatter(s_cnn_fa_all_saliency[idx_2], logFC_tap_seq_all[idx_2], color="blue", s=80, alpha=.5)
plt.scatter(s_cnn_fa_all_saliency[idx_1], logFC_tap_seq_all[idx_1], color="red", s=80, alpha=.5)
plt.xlabel("Seq-CNN Saliency Score", fontsize=40)
plt.ylabel("Log Fold Change", fontsize=40)
#plt.xlim((0,14))
#plt.ylim((0,14))
plt.title('DE-G Pairs', fontsize=40)
props = dict(boxstyle='round', facecolor='white', alpha=1)
plt.text(2e-6,-1.5, r"$\rho$ = "+str(spearmanr(s_cnn_fa_all_saliency[idx_1], logFC_tap_seq_all[idx_1])[0].astype(np.float16))+ ' (Decrease)',
         horizontalalignment='left', verticalalignment='top', bbox=props, fontsize=30)
plt.tick_params(axis='x', labelsize=30)
plt.tick_params(axis='y', labelsize=30)
plt.xscale('log',basex=10) 
plt.xlim((0.000001,2))
plt.legend(['Not Significant','Increase','Decrease'], loc=3, fontsize=30)
plt.grid()
plt.savefig('../figs/tap_seq/Seq-CNN_saliency_vs_LogFC_scatter_plot_K562.png')

plt.figure(figsize=(15,15))
plt.scatter(D[idx_0], abc_all[idx_0], color="grey", s=80, alpha=.5)
plt.scatter(D[idx_2], abc_all[idx_2], color="blue", s=80, alpha=.5)
plt.scatter(D[idx_1], abc_all[idx_1], color="red", s=80, alpha=.5)
plt.title('DE-G Pairs', fontsize=40)
plt.xlabel("Distance (bp)", fontsize=40)
plt.ylabel("ABC Score", fontsize=40)
plt.tick_params(axis='x', labelsize=30)
plt.tick_params(axis='y', labelsize=30)
plt.xscale('log',basex=10) 
plt.yscale('log',basey=10)
plt.ylim((0.000001,2))
plt.xlim((5e2,4e6))
plt.legend(['Not Significant','Increase','Decrease'], loc=3, fontsize=30)
plt.grid()
plt.savefig('../figs/tap_seq/ABC_vs_Dist_scatter_plot_K562.png')

plt.figure(figsize=(15,15))
plt.scatter(D[idx_0], e_graphreg_fa_all_deepshap[idx_0], color="grey", s=80, alpha=.5)
plt.scatter(D[idx_2], e_graphreg_fa_all_deepshap[idx_2], color="blue", s=80, alpha=.5)
plt.scatter(D[idx_1], e_graphreg_fa_all_deepshap[idx_1], color="red", s=80, alpha=.5)
plt.title('DE-G Pairs', fontsize=40)
plt.xlabel("Distance (bp)", fontsize=40)
plt.ylabel("Epi-GraphReg Deepshap Score", fontsize=40)
plt.tick_params(axis='x', labelsize=30)
plt.tick_params(axis='y', labelsize=30)
plt.xscale('log',basex=10) 
plt.yscale('log',basey=10)
plt.ylim((0.000001,2))
plt.xlim((5e2,4e6))
plt.legend(['Not Significant','Increase','Decrease'], loc=3, fontsize=30)
plt.grid()
plt.savefig('../figs/tap_seq/Epi-GraphReg_deepshap_vs_Dist_scatter_plot_K562.png')

plt.figure(figsize=(15,15))
plt.scatter(D[idx_0], e_graphreg_fa_all_saliency[idx_0], color="grey", s=80, alpha=.5)
plt.scatter(D[idx_2], e_graphreg_fa_all_saliency[idx_2], color="blue", s=80, alpha=.5)
plt.scatter(D[idx_1], e_graphreg_fa_all_saliency[idx_1], color="red", s=80, alpha=.5)
plt.title('DE-G Pairs', fontsize=40)
plt.xlabel("Distance (bp)", fontsize=40)
plt.ylabel("Epi-GraphReg Saliency Score", fontsize=40)
plt.tick_params(axis='x', labelsize=30)
plt.tick_params(axis='y', labelsize=30)
plt.xscale('log',basex=10) 
plt.yscale('log',basey=10)
plt.ylim((0.000001,2))
plt.xlim((5e2,4e6))
plt.legend(['Not Significant','Increase','Decrease'], loc=3, fontsize=30)
plt.grid()
plt.savefig('../figs/tap_seq/Epi-GraphReg_saliency_vs_Dist_scatter_plot_K562.png')

plt.figure(figsize=(15,15))
plt.scatter(D[idx_0], s_graphreg_fa_all_saliency[idx_0], color="grey", s=80, alpha=.5)
plt.scatter(D[idx_2], s_graphreg_fa_all_saliency[idx_2], color="blue", s=80, alpha=.5)
plt.scatter(D[idx_1], s_graphreg_fa_all_saliency[idx_1], color="red", s=80, alpha=.5)
plt.title('DE-G Pairs', fontsize=40)
plt.xlabel("Distance (bp)", fontsize=40)
plt.ylabel("Seq-GraphReg Saliency Score", fontsize=40)
plt.tick_params(axis='x', labelsize=30)
plt.tick_params(axis='y', labelsize=30)
plt.xscale('log',basex=10) 
plt.yscale('log',basey=10)
plt.ylim((0.000001,2))
plt.xlim((5e2,4e6))
plt.legend(['Not Significant','Increase','Decrease'], loc=3, fontsize=30)
plt.grid()
plt.savefig('../figs/tap_seq/Seq-GraphReg_saliency_vs_Dist_scatter_plot_K562.png')

plt.figure(figsize=(15,15))
plt.scatter(D[idx_0], e_cnn_fa_all_deepshap[idx_0], color="grey", s=80, alpha=.5)
plt.scatter(D[idx_2], e_cnn_fa_all_deepshap[idx_2], color="blue", s=80, alpha=.5)
plt.scatter(D[idx_1], e_cnn_fa_all_deepshap[idx_1], color="red", s=80, alpha=.5)
plt.title('DE-G Pairs', fontsize=40)
plt.xlabel("Distance (bp)", fontsize=40)
plt.ylabel("Epi-CNN Deepshap Score", fontsize=40)
plt.tick_params(axis='x', labelsize=30)
plt.tick_params(axis='y', labelsize=30)
plt.xscale('log',basex=10) 
plt.yscale('log',basey=10)
plt.ylim((0.000001,2))
plt.xlim((5e2,4e6))
plt.legend(['Not Significant','Increase','Decrease'], loc=3, fontsize=30)
plt.grid()
plt.savefig('../figs/tap_seq/Epi-CNN_deepshap_vs_Dist_scatter_plot_K562.png')

plt.figure(figsize=(15,15))
plt.scatter(D[idx_0], e_cnn_fa_all_saliency[idx_0], color="grey", s=80, alpha=.5)
plt.scatter(D[idx_2], e_cnn_fa_all_saliency[idx_2], color="blue", s=80, alpha=.5)
plt.scatter(D[idx_1], e_cnn_fa_all_saliency[idx_1], color="red", s=80, alpha=.5)
plt.title('DE-G Pairs', fontsize=40)
plt.xlabel("Distance (bp)", fontsize=40)
plt.ylabel("Epi-CNN Saliency Score", fontsize=40)
plt.tick_params(axis='x', labelsize=30)
plt.tick_params(axis='y', labelsize=30)
plt.xscale('log',basex=10) 
plt.yscale('log',basey=10)
plt.ylim((0.000001,2))
plt.xlim((5e2,4e6))
plt.legend(['Not Significant','Increase','Decrease'], loc=3, fontsize=30)
plt.grid()
plt.savefig('../figs/tap_seq/Epi-CNN_saliency_vs_Dist_scatter_plot_K562.png')

plt.figure(figsize=(15,15))
plt.scatter(D[idx_0], s_cnn_fa_all_saliency[idx_0], color="grey", s=80, alpha=.5)
plt.scatter(D[idx_2], s_cnn_fa_all_saliency[idx_2], color="blue", s=80, alpha=.5)
plt.scatter(D[idx_1], s_cnn_fa_all_saliency[idx_1], color="red", s=80, alpha=.5)
plt.title('DE-G Pairs', fontsize=40)
plt.xlabel("Distance (bp)", fontsize=40)
plt.ylabel("Seq-CNN Saliency Score", fontsize=40)
plt.tick_params(axis='x', labelsize=30)
plt.tick_params(axis='y', labelsize=30)
plt.xscale('log',basex=10) 
plt.yscale('log',basey=10)
plt.ylim((0.000001,2))
plt.xlim((5e2,4e6))
plt.legend(['Not Significant','Increase','Decrease'], loc=3, fontsize=30)
plt.grid()
plt.savefig('../figs/tap_seq/Seq-CNN_saliency_vs_Dist_scatter_plot_K562.png')


####### Precision-Recall #######
Y_true = np.zeros(len(significant_tap_seq), dtype=int)
idx = np.where(significant_tap_seq==1)[0]
Y_true[idx] = 1

# ABC
Y_pred_abc = abc_all
precision_abc, recall_abc, thresholds_abc = precision_recall_curve(Y_true, Y_pred_abc)

average_precision_abc = average_precision_score(Y_true, Y_pred_abc)
auprc_abc_all = auc(recall_abc, precision_abc)

# Epi-models deepshap
Y_pred_gat = e_graphreg_fa_all_deepshap
Y_pred_cnn = e_cnn_fa_all_deepshap

precision_e_graphreg_deepshap, recall_e_graphreg_deepshap, thresholds_e_graphreg_deepshap = precision_recall_curve(Y_true, Y_pred_gat)
ap_e_graphreg_all_deepshap = average_precision_score(Y_true, Y_pred_gat)
auprc_e_graphreg_all_deepshap = auc(recall_e_graphreg_deepshap, precision_e_graphreg_deepshap)

precision_e_cnn_deepshap, recall_e_cnn_deepshap, thresholds_e_cnn_deepshap = precision_recall_curve(Y_true, Y_pred_cnn)
ap_e_cnn_all_deepshap = average_precision_score(Y_true, Y_pred_cnn)
auprc_e_cnn_all_deepshap = auc(recall_e_cnn_deepshap, precision_e_cnn_deepshap)

# Epi-models saliency
Y_pred_gat = e_graphreg_fa_all_saliency
Y_pred_cnn = e_cnn_fa_all_saliency

precision_e_graphreg_saliency, recall_e_graphreg_saliency, thresholds_e_graphreg_saliency = precision_recall_curve(Y_true, Y_pred_gat)
ap_e_graphreg_all_saliency = average_precision_score(Y_true, Y_pred_gat)
auprc_e_graphreg_all_saliency = auc(recall_e_graphreg_saliency, precision_e_graphreg_saliency)

precision_e_cnn_saliency, recall_e_cnn_saliency, thresholds_e_cnn_saliency = precision_recall_curve(Y_true, Y_pred_cnn)
ap_e_cnn_all_saliency = average_precision_score(Y_true, Y_pred_cnn)
auprc_e_cnn_all_saliency = auc(recall_e_cnn_saliency, precision_e_cnn_saliency)

# Seq-models saliency
Y_pred_gat = s_graphreg_fa_all_saliency
Y_pred_cnn = s_cnn_fa_all_saliency

precision_s_graphreg_saliency, recall_s_graphreg_saliency, thresholds_s_graphreg_saliency = precision_recall_curve(Y_true, Y_pred_gat)
ap_s_graphreg_all_saliency = average_precision_score(Y_true, Y_pred_gat)
auprc_s_graphreg_all_saliency = auc(recall_s_graphreg_saliency, precision_s_graphreg_saliency)

precision_s_cnn_saliency, recall_s_cnn_saliency, thresholds_s_cnn_saliency = precision_recall_curve(Y_true, Y_pred_cnn)
ap_s_cnn_all_saliency = average_precision_score(Y_true, Y_pred_cnn)
auprc_s_cnn_all_saliency = auc(recall_s_cnn_saliency, precision_s_cnn_saliency)

plt.figure(figsize=(10,10))
plt.plot(recall_abc, precision_abc, color='lightgreen', linewidth=3, label="ABC: AUC = "+str(auprc_abc_all.astype(np.float16)))
plt.plot(recall_e_graphreg_deepshap, precision_e_graphreg_deepshap, color='orange', linewidth=3, label="Epi-GraphReg (DeepSHAP): AUC = "+str(auprc_e_graphreg_all_deepshap.astype(np.float16)))
plt.plot(recall_e_graphreg_saliency, precision_e_graphreg_saliency, color='orange', linewidth=3, linestyle='--', label="Epi-GraphReg (Saliency): AUC = "+str(auprc_e_graphreg_all_saliency.astype(np.float16)))
plt.plot(recall_s_graphreg_saliency, precision_s_graphreg_saliency, color='orange', linewidth=3, linestyle=':', label="Seq-GraphReg (Saliency): AUC = "+str(auprc_s_graphreg_all_saliency.astype(np.float16)))

plt.plot(recall_e_cnn_deepshap, precision_e_cnn_deepshap, color='deepskyblue', linewidth=3, label="Epi-CNN (DeepSHAP): AUC = "+str(auprc_e_cnn_all_deepshap.astype(np.float16)))
plt.plot(recall_e_cnn_saliency, precision_e_cnn_saliency, color='deepskyblue', linewidth=3, linestyle='--', label="Epi-CNN (Saliency): AUC = "+str(auprc_e_cnn_all_saliency.astype(np.float16)))
plt.plot(recall_s_cnn_saliency, precision_s_cnn_saliency, color='deepskyblue', linewidth=3, linestyle=':', label="Seq-CNN (Saliency): AUC = "+str(auprc_s_cnn_all_saliency.astype(np.float16)))

plt.title('TAP-seq | DE-G Pairs ('+str(len(Y_true))+')', fontsize=40)
plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=40)
plt.xlabel("Recall", fontsize=40)
plt.ylabel("Precision", fontsize=40)
plt.tick_params(axis='x', labelsize=40)
plt.tick_params(axis='y', labelsize=40)
#plt.tight_layout()
plt.savefig('../figs/tap_seq/PR_curve.pdf', bbox_inches='tight')


##### SP ######
print('ABC SP for significants: ', spearmanr(abc_all[idx], logFC_tap_seq_all[idx])[0])
print('Epi-GraphReg (deepshap) SP for significants: ', spearmanr(e_graphreg_fa_all_deepshap[idx], logFC_tap_seq_all[idx])[0])
print('Epi-GraphReg (saliency) SP for significants: ', spearmanr(e_graphreg_fa_all_saliency[idx], logFC_tap_seq_all[idx])[0])
print('Seq-GraphReg (saliency) SP for significants: ', spearmanr(s_graphreg_fa_all_saliency[idx], logFC_tap_seq_all[idx])[0])
print('Epi-CNN (deepshap) SP for significants: ', spearmanr(e_cnn_fa_all_deepshap[idx], logFC_tap_seq_all[idx])[0])
print('Epi-CNN (saliency) SP for significants: ', spearmanr(e_cnn_fa_all_saliency[idx], logFC_tap_seq_all[idx])[0])
print('Seq-CNN (saliency) SP for significants: ', spearmanr(s_cnn_fa_all_saliency[idx], logFC_tap_seq_all[idx])[0])



