from __future__ import division
import sys
sys.path.insert(0,'../train')

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
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import pyBigWig
from collections import Counter

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

def log2(x):
  numerator = tf.math.log(x)
  denominator = tf.math.log(tf.constant(2.))
  return numerator / denominator

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

def calculate_loss(cell_lines, gene_names_list, gene_tss_list, chr_list, batch_size, saliency_method, write_bw, load_fa, organism, genome):
    L = 40
    abc_all = np.array([])
    expr_change_percent_all = np.array([])
    gat_fa_all = np.array([])
    cnn_fa_all = np.array([])
    significant_flowfish = np.array([], dtype=int)
    sp_abc = np.array([])
    sp_gat = np.array([])
    sp_cnn = np.array([])
    average_precision_abc = np.array([])
    auprc_abc = np.array([])
    average_precision_gat = np.array([])
    auprc_gat = np.array([])
    average_precision_cnn = np.array([])
    auprc_cnn = np.array([])
    x_background = np.zeros([1,60000,3]).astype('float32')
    #x_background = np.random.normal(0,1,size=[1000,60000,4]).astype('float32')
    #adj_background = np.zeros([1,1200,1200]).astype('float32')
    for num, cell_line in enumerate(cell_lines):
        for i, chrm in enumerate(chr_list):
            print(gene_names_list[i])
            file_name = data_path+'/data/tfrecords/tfr_epi_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'_'+chrm+'.tfr'
            iterator = dataset_iterator(file_name, batch_size)
            while True:
                data_exist, X, X_epi, Y, adj, adj_real, idx, tss_idx, pos = read_tf_record_1shot(iterator)
                if data_exist:
                    if tf.reduce_sum(tf.gather(tss_idx, idx)) > 0:
                        if (pos[0] > 0 and pos[-1] < 1e15  and gene_tss_list[i] >= pos[0]+2000000 and gene_tss_list[i] < pos[0]+4000000):

                            print('start position: ', pos[0])

                            ############### FlowFish analysis ###############

                            data_frame_flowfish = pd.read_csv(data_path+'/data/csv/FlowFISH_results.csv')  
                            cut_flowfish = data_frame_flowfish.loc[data_frame_flowfish['Gene'] == gene_names_list[i]]
                            #cut_flowfish = cut_flowfish[cut_flowfish['class'] != 'promoter']
                            #cut_flowfish = cut_flowfish[cut_flowfish['Reference'] == 'This study']
                            start_flowfish = cut_flowfish['start'].values
                            end_flowfish = cut_flowfish['end'].values
                            expr_change_percent_flowfish = cut_flowfish['Fraction.change.in.gene.expr'].values
                            abc_flowfish = cut_flowfish['ABC.Score'].values
                            significant_flowfish_bool = cut_flowfish['Significant'].values
                            significant_flowfish_gene = np.array([])
                            for ii, gg in enumerate(significant_flowfish_bool):
                                if gg == False:
                                    significant_flowfish_gene = np.append(significant_flowfish_gene, 0)
                                elif (gg == True and expr_change_percent_flowfish[ii]<0):
                                    significant_flowfish_gene = np.append(significant_flowfish_gene, 1)
                                elif (gg == True and expr_change_percent_flowfish[ii]>0):
                                    significant_flowfish_gene = np.append(significant_flowfish_gene, 2)

                            print('len significant_flowfish', len(significant_flowfish_gene))
                            print('num 0:', np.sum(significant_flowfish_gene==0))
                            print('num 1:', np.sum(significant_flowfish_gene==1))
                            print('num 2:', np.sum(significant_flowfish_gene==2))
                            print('len abc_flowfish:', len(abc_flowfish))

                            start_idx = np.floor((start_flowfish - pos[0])/100).astype('int64')
                            end_idx = np.ceil((end_flowfish - pos[0])/100).astype('int64')
                            mid_pos = np.round((start_flowfish + end_flowfish)/2)
                            mid_idx = np.round((mid_pos - pos[0])/100).astype('int64')

                            expr_change_percent_vec = np.zeros(60000)
                            abc_vec = np.zeros(60000)
                            for k, idx in enumerate(mid_idx):
                                expr_change_percent_vec[idx] = expr_change_percent_flowfish[k]
                                abc_vec[idx] = abc_flowfish[k]


                            Y_true = np.zeros(len(significant_flowfish_gene), dtype=int)
                            idx_sig = np.where(significant_flowfish_gene==1)[0]
                            Y_true[idx_sig] = 1
                            if (len(Y_true) >= 10 and len(idx_sig) >= 1):
                                significant_flowfish = np.append(significant_flowfish, significant_flowfish_gene)
                                expr_change_percent_all = np.append(expr_change_percent_all, expr_change_percent_flowfish)
                                abc_all = np.append(abc_all, abc_flowfish)
                                sp_abc = np.append(sp_abc, spearmanr(abc_flowfish, expr_change_percent_flowfish)[0])

                                Y_pred_abc = abc_flowfish

                                precision_abc, recall_abc, thresholds_abc = precision_recall_curve(Y_true, Y_pred_abc)
                                average_precision_abc = np.append(average_precision_abc, average_precision_score(Y_true, Y_pred_abc))
                                auprc_abc = np.append(auprc_abc, auc(recall_abc, precision_abc))
                                print('ABC AP score: ', average_precision_score(Y_true, Y_pred_abc))
                                print('ABC AUPRC score: ', auc(recall_abc, precision_abc))

                            ############# Feature Attribution of Epi-CNN #############

                            if saliency_method == 'deepshap':
                                if load_fa == False:
                                    explain_output_idx = np.floor((gene_tss_list[i]-pos[0])/5000).astype('int64').reshape([1,1])
                                    print('explain_output_idx: ', explain_output_idx)
                                    shap_values_cnn = 0
                                    for j in range(1,1+10):
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

                                        model_name_cnn = data_path+'/models/'+cell_line+'/Epi-CNN_'+cell_line+'_valid_chr_'+valid_chr_str+'_test_chr_'+test_chr_str+'.h5'
                                        model_cnn = tf.keras.models.load_model(model_name_cnn)
                                        model_cnn.trainable = False
                                        model_cnn._name = 'Epi-CNN'

                                        explainer_cnn = shap.GradientExplainer(model_cnn, x_background, batch_size=10)
                                        shap_values_cnn_, indexes_cnn = explainer_cnn.shap_values(X_epi.numpy(), ranked_outputs=explain_output_idx, output_rank_order="custom",nsamples=200)
                                        print(type(shap_values_cnn_[0]), shap_values_cnn_[0].shape)
                                        shap_values_cnn = shap_values_cnn + shap_values_cnn_[0]

                                    shap_values_cnn = shap_values_cnn/10
                                    np.save(data_path+'/results/numpy/feature_attribution/Epi-CNN_'+saliency_method+'_'+cell_line+'_'+gene_names_list[i]+'.npy', shap_values_cnn)
                                else:
                                    shap_values_cnn = np.load(data_path+'/results/numpy/feature_attribution/Epi-CNN_'+saliency_method+'_'+cell_line+'_'+gene_names_list[i]+'.npy')

                                scores_cnn = K.reshape(shap_values_cnn, [60000,3])
                                scores_cnn = K.sum(scores_cnn, axis = 1).numpy()
                                print('Epi-CNN: ', scores_cnn.shape, np.min(scores_cnn), np.max(scores_cnn), np.mean(scores_cnn))

                            elif saliency_method == 'saliency':
                                if load_fa == False:
                                    explain_output_idx_cnn = np.floor((gene_tss_list[i]-pos[0])/5000).astype('int64')
                                    print('explain_output_idx_cnn: ', explain_output_idx_cnn)
                                    grads_cnn = 0
                                    for j in range(1,1+10):
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

                                        model_name_cnn = data_path+'/models/'+cell_line+'/Epi-CNN_'+cell_line+'_valid_chr_'+valid_chr_str+'_test_chr_'+test_chr_str+'.h5'
                                        model_cnn = tf.keras.models.load_model(model_name_cnn)
                                        model_cnn.trainable = False
                                        model_cnn._name = 'Epi-CNN'

                                        with tf.GradientTape(persistent=True) as tape:
                                            inp = X_epi
                                            tape.watch(inp)
                                            preds = model_cnn(inp)
                                            target_cnn = preds[:, explain_output_idx_cnn-1]+preds[:, explain_output_idx_cnn]+preds[:, explain_output_idx_cnn+1]
                                            #target_cnn = preds[:, explain_output_idx_cnn]
                                        grads_cnn = grads_cnn + tape.gradient(target_cnn, inp)
                                    grads_cnn = grads_cnn/10
                                    np.save(data_path+'/results/numpy/feature_attribution/Epi-CNN_'+saliency_method+'_'+cell_line+'_'+gene_names_list[i]+'.npy', grads_cnn)
                                else:
                                    grads_cnn = np.load(data_path+'/results/numpy/feature_attribution/Epi-CNN_'+saliency_method+'_'+cell_line+'_'+gene_names_list[i]+'.npy')

                                scores_cnn = K.reshape(grads_cnn * X_epi, [60000,3])
                                scores_cnn = K.sum(scores_cnn, axis = 1).numpy()
                                print('Epi-CNN: ', scores_cnn.shape, np.min(scores_cnn), np.max(scores_cnn), np.mean(scores_cnn))

                            cnn_fa_gene = np.array([])
                            cnt = -1
                            for s_idx, e_idx in zip(start_idx, end_idx):
                                cnt = cnt + 1
                                mid_idx = int((s_idx+e_idx)/2)
                                cnn_fa_gene = np.append(cnn_fa_gene, np.max(scores_cnn[mid_idx-L:mid_idx+L+1])/np.max(scores_cnn))
                                #cnn_fa_gene = np.append(cnn_fa_gene, np.max(scores_cnn[s_idx:e_idx+1])/np.max(scores_cnn))

                            Y_true = np.zeros(len(significant_flowfish_gene), dtype=int)
                            idx_sig = np.where(significant_flowfish_gene==1)[0]
                            Y_true[idx_sig] = 1
                            if (len(Y_true) >= 10 and len(idx_sig) >= 1):
                                cnn_fa_all = np.append(cnn_fa_all, cnn_fa_gene)
                                sp_cnn = np.append(sp_cnn, spearmanr(cnn_fa_gene, expr_change_percent_flowfish)[0])

                                Y_pred_cnn = cnn_fa_gene

                                precision_cnn, recall_cnn, thresholds_cnn = precision_recall_curve(Y_true, Y_pred_cnn)
                                average_precision_cnn = np.append(average_precision_cnn, average_precision_score(Y_true, Y_pred_cnn))
                                auprc_cnn = np.append(auprc_cnn, auc(recall_cnn, precision_cnn))
                                print('Epi-CNN AP score: ', average_precision_score(Y_true, Y_pred_cnn))
                                print('Epi-CNN AUPRC score: ', auc(recall_cnn, precision_cnn))


                            ############# Feature Attribution of Epi-GraphReg #############

                            if saliency_method == 'deepshap':
                                if load_fa == False:
                                    explain_output_idx = np.floor((gene_tss_list[i]-pos[0])/5000).astype('int64').reshape([1,1])
                                    print('explain_output_idx: ', explain_output_idx)
                                    shap_values_gat = 0
                                    for j in range(1,1+10):
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

                                        model_name_gat = data_path+'/models/'+cell_line+'/Epi-GraphReg_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'_valid_chr_'+valid_chr_str+'_test_chr_'+test_chr_str+'.h5'
                                        model_gat = tf.keras.models.load_model(model_name_gat, custom_objects={'GraphAttention': GraphAttention})
                                        model_gat.trainable = False
                                        model_gat._name = 'Epi-GraphReg'
                                        model_gat_1 = tf.keras.Model(inputs=model_gat.inputs, outputs=model_gat.outputs[0])

                                        explainer_gat = shap.GradientExplainer(model_gat_1, [x_background, adj.numpy()], batch_size=1)
                                        shap_values_gat_, indexes_gat = explainer_gat.shap_values([X_epi.numpy(), adj.numpy()], nsamples=200, ranked_outputs=explain_output_idx, output_rank_order="custom")
                                        shap_values_gat = shap_values_gat + shap_values_gat_[0][0]

                                    shap_values_gat = shap_values_gat/10
                                    np.save(data_path+'/results/numpy/feature_attribution/Epi-GraphReg_'+saliency_method+'_'+cell_line+'_'+gene_names_list[i]+'.npy', shap_values_gat)
                                else:
                                    shap_values_gat = np.load(data_path+'/results/numpy/feature_attribution/Epi-GraphReg_'+saliency_method+'_'+cell_line+'_'+gene_names_list[i]+'.npy')

                                scores_gat = K.reshape(shap_values_gat, [60000,3])
                                scores_gat = K.sum(scores_gat, axis = 1).numpy()
                                print('Epi-GraphReg: ', scores_gat.shape, np.min(scores_gat), np.max(scores_gat), np.mean(scores_gat))

                            elif saliency_method == 'saliency':
                                if load_fa == False:
                                    explain_output_idx_gat = np.floor((gene_tss_list[i]-pos[0])/5000).astype('int64')
                                    print('explain_output_idx_gat: ', explain_output_idx_gat)
                                    grads_gat = 0
                                    for j in range(1,1+10):
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

                                        model_name_gat = data_path+'/models/'+cell_line+'/Epi-GraphReg_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'_valid_chr_'+valid_chr_str+'_test_chr_'+test_chr_str+'.h5'
                                        model_gat = tf.keras.models.load_model(model_name_gat, custom_objects={'GraphAttention': GraphAttention})
                                        model_gat.trainable = False
                                        model_gat._name = 'Epi-GraphReg'

                                        with tf.GradientTape(persistent=True) as tape:
                                            inp = X_epi
                                            tape.watch(inp)
                                            preds, _ = model_gat([inp, adj])
                                            target_gat = preds[:, explain_output_idx_gat-1]+preds[:, explain_output_idx_gat]+preds[:, explain_output_idx_gat+1]
                                            #target_gat = preds[:, explain_output_idx_gat]
                                        grads_gat = grads_gat + tape.gradient(target_gat, inp)
                                    grads_gat = grads_gat/10
                                    np.save(data_path+'/results/numpy/feature_attribution/Epi-GraphReg_'+saliency_method+'_'+cell_line+'_'+gene_names_list[i]+'.npy', grads_gat)
                                else:
                                    grads_gat = np.load(data_path+'/results/numpy/feature_attribution/Epi-GraphReg_'+saliency_method+'_'+cell_line+'_'+gene_names_list[i]+'.npy')

                                scores_gat = K.reshape(grads_gat * X_epi, [60000,3])
                                scores_gat = K.sum(scores_gat, axis = 1).numpy()
                                print('Epi-GraphReg: ', scores_gat.shape, np.min(scores_gat), np.max(scores_gat), np.mean(scores_gat))

                            gat_fa_gene = np.array([])
                            cnt = -1
                            for s_idx, e_idx in zip(start_idx, end_idx):
                                cnt = cnt + 1
                                mid_idx = int((s_idx+e_idx)/2)
                                gat_fa_gene = np.append(gat_fa_gene, np.max(scores_gat[mid_idx-L:mid_idx+L+1])/np.max(scores_gat))
                                #gat_fa_gene = np.append(gat_fa_gene, np.max(scores_gat[s_idx:e_idx+1])/np.max(scores_gat))

                            Y_true = np.zeros(len(significant_flowfish_gene), dtype=int)
                            idx_sig = np.where(significant_flowfish_gene==1)[0]
                            Y_true[idx_sig] = 1
                            if (len(Y_true) >= 10 and len(idx_sig) >= 1):
                                gat_fa_all = np.append(gat_fa_all, gat_fa_gene)
                                sp_gat = np.append(sp_gat, spearmanr(gat_fa_gene, expr_change_percent_flowfish)[0])

                                Y_pred_gat = gat_fa_gene

                                precision_gat, recall_gat, thresholds_gat = precision_recall_curve(Y_true, Y_pred_gat)
                                average_precision_gat = np.append(average_precision_gat, average_precision_score(Y_true, Y_pred_gat))
                                auprc_gat = np.append(auprc_gat, auc(recall_gat, precision_gat))
                                print('Epi-GraphReg AP score: ', average_precision_score(Y_true, Y_pred_gat))
                                print('Epi-GraphReg AUPRC score: ', auc(recall_gat, precision_gat))


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
                                bw_flowfish = pyBigWig.open(data_path+'/results/bigwig/feature_attribution/Epi-models/flowfish_'+cell_line+'_'+gene_names_list[i]+'.bw', "w")
                                bw_flowfish.addHeader(header)
                                bw_abc = pyBigWig.open(data_path+'/results/bigwig/feature_attribution/Epi-models/abc_'+cell_line+'_'+gene_names_list[i]+'.bw', "w")
                                bw_abc.addHeader(header)
                                bw_GraphReg = pyBigWig.open(data_path+'/results/bigwig/feature_attribution/Epi-models/Epi-GraphReg_'+saliency_method+'_'+cell_line+'_'+gene_names_list[i]+'.bw', "w")
                                bw_GraphReg.addHeader(header)
                                bw_CNN = pyBigWig.open(data_path+'/results/bigwig/feature_attribution/Epi-models/Epi-CNN_'+saliency_method+'_'+cell_line+'_'+gene_names_list[i]+'.bw', "w")
                                bw_CNN.addHeader(header)

                                chroms = np.array([chrm] * len(pos))
                                print(chrm, len(pos))
                                starts = pos.astype(np.int64)
                                ends = starts + 100
                                ends = ends.astype(np.int64)

                                bw_flowfish.addEntries(chroms, starts, ends=ends, values=expr_change_percent_vec)
                                bw_abc.addEntries(chroms, starts, ends=ends, values=abc_vec)
                                bw_GraphReg.addEntries(chroms, starts, ends=ends, values=scores_gat.ravel())
                                bw_CNN.addEntries(chroms, starts, ends=ends, values=scores_cnn.ravel())

                                bw_flowfish.close()
                                bw_abc.close()
                                bw_GraphReg.close()
                                bw_CNN.close()

                                cnt = -1
                                with open(data_path+'/results/bigwig/feature_attribution/Epi-models/candidates_'+cell_line+'_'+gene_names_list[i]+'.bed', "w") as bed_file:
                                        for s_idx, e_idx in zip(start_flowfish, end_flowfish):
                                            cnt = cnt+1
                                            line = chrm+'\t'+str(s_idx)+'\t'+str(e_idx)+'\t'+str(significant_flowfish_bool[cnt])
                                            bed_file.write(line)
                                            bed_file.write('\n')

                else:
                    #print('no data')
                    break

    print('len gat shap: ', len(gat_fa_all))
    return sp_abc, sp_gat, sp_cnn, auprc_abc, auprc_gat, auprc_cnn, average_precision_abc, average_precision_gat, average_precision_cnn, abc_all, gat_fa_all, cnn_fa_all, expr_change_percent_all, significant_flowfish

###################################### load model ######################################
batch_size = 1
organism = 'human'            # human/mouse
genome='hg19'                 # hg19/hg38
cell_line = ['K562']          # K562/GM12878/mESC/hESC
write_bw = True               # write the feature attribution scores to bigwig files
load_fa = False               # load feature attribution numpy files
saliency_method = 'saliency'  # 'saliency' or 'deepshap' 


chr_list = np.array([])
gene_tss_list = np.array([], dtype=np.int)
data_frame_flowfish = pd.read_csv(data_path+'/data/csv/FlowFISH_results.csv')  
data_frame_flowfish = data_frame_flowfish[data_frame_flowfish['chr'] != 'chrX']
#data_frame_flowfish = data_frame_flowfish[data_frame_flowfish['class'] != 'promoter']
#data_frame_flowfish = data_frame_flowfish[data_frame_flowfish['Reference'] == 'This study']
gene_names_list = np.unique(data_frame_flowfish['Gene'].values)
for gene in gene_names_list:
    cut = data_frame_flowfish.loc[data_frame_flowfish['Gene'] == gene]
    chr_list = np.append(chr_list, cut['chr'].values[0])
    gene_tss_list = np.append(gene_tss_list, cut['Gene.TSS'].values[0])

print(len(gene_names_list), gene_names_list)
print(len(chr_list), chr_list)
print(len(gene_tss_list), gene_tss_list)


sp_abc, sp_gat_shap, sp_cnn_shap, auprc_abc, auprc_gat, auprc_cnn, ap_abc, ap_gat, ap_cnn, abc_all, gat_fa_all, cnn_fa_all, expr_change_percent_all, significant_flowfish = calculate_loss(cell_line, gene_names_list, 
                    gene_tss_list, chr_list, batch_size, saliency_method, write_bw, load_fa, organism, genome)
print('sp_abc: ', sp_abc)
print('sp_gat_shap: ', sp_gat_shap)
print('sp_cnn_shap: ', sp_cnn_shap)

color_significant = []
for ii in significant_flowfish:
    if ii == 0:
        color_significant.append('grey')
    elif ii == 1:
        color_significant.append('red')
    elif ii == 2:
        color_significant.append('blue')

idx = np.where(significant_flowfish==1)[0]


plt.close("all")
SP_list = [sp_abc, sp_gat_shap]
fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)
ax.violinplot(SP_list)
ax.set_title('distribution of '+str(len(sp_abc))+' genes')
ax.set_ylabel('AUC_PR')
ax.set_xticklabels(['ABC', 'Epi-GraphReg'])
ax.grid()
fig.savefig('../figs/flowfish/SP_boxplot.png')

plt.close("all")
aucpr_list = [auprc_abc, auprc_gat, auprc_cnn]
fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)
ax.violinplot(aucpr_list, showmedians=True)
ax.set_title('Distribution of '+str(len(auprc_abc))+' genes', fontsize=20)
ax.set_ylabel('AUCPR', fontsize=20)
ax.set_xticklabels(['ABC', 'Epi-GraphReg', 'Epi-CNN'])
ax.tick_params(axis='y', labelsize=10)
ax.tick_params(axis='x', labelsize=20)
ax.grid()
fig.savefig('../figs/flowfish/AUCPR_boxplot.png')

plt.close("all")
aucpr_list = [ap_abc, ap_gat]
fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)
ax.violinplot(aucpr_list)
ax.set_title('Distribution of '+str(len(auprc_abc))+' genes')
ax.set_ylabel('Average Precision')
ax.set_xticklabels(['ABC', 'Epi-GraphReg'])
ax.grid()
fig.savefig('../figs/flowfish/AP_boxplot.png')


plt.figure(figsize=(15,15))
plt.scatter(abc_all, expr_change_percent_all, color=color_significant, marker="o",facecolors='none', s=80)
#for i, annot in enumerate(chr_pos):
#    plt.annotate(annot, (rho_gat[i], rho_1d[i]))
plt.xlabel("ABC score", fontsize=40)
plt.ylabel("Effect on gene expression (%)", fontsize=40)
#plt.xlim((0,14))
#plt.ylim((0,14))
plt.title('SP (significant decrease): '+str(spearmanr(abc_all[idx], expr_change_percent_all[idx])[0].astype(np.float16)), fontsize=20)
#plt.plot(np.linspace(0,1,100),np.linspace(0,1,100),color='red')
plt.tick_params(axis='x', labelsize=25)
plt.tick_params(axis='y', labelsize=25)
plt.xscale('log',basex=10) 
plt.xlim((0.000001,1))
plt.grid()
plt.savefig('../figs/flowfish/ABC_vs_Expr_Change_scatter_plot_K562.png')

plt.figure(figsize=(15,15))
plt.scatter(gat_fa_all, expr_change_percent_all, color=color_significant, marker="o", facecolors='none', s=80)
#for i, annot in enumerate(chr_pos):
#    plt.annotate(annot, (rho_gat[i], rho_1d[i]))
plt.xlabel("Epi-GraphReg saliency score", fontsize=40)
plt.ylabel("Effect on gene expression (%)", fontsize=40)
#plt.xlim((0,14))
#plt.ylim((0,14))
plt.title('SP (significant decrease): '+str(spearmanr(gat_fa_all[idx], expr_change_percent_all[idx])[0].astype(np.float16)), fontsize=20)
#plt.plot(np.linspace(0,1,100),np.linspace(0,1,100),color='red')
plt.tick_params(axis='x', labelsize=25)
plt.tick_params(axis='y', labelsize=25)
plt.xscale('log',basex=10)
plt.xlim((0.000001,200))
plt.grid()
plt.savefig('../figs/flowfish/Epi-GraphReg_saliency_vs_Expr_Change_scatter_plot_K562.png')

plt.figure(figsize=(15,15))
plt.scatter(cnn_fa_all, expr_change_percent_all, color=color_significant, marker="o", facecolors='none',s=80)
#for i, annot in enumerate(chr_pos):
#    plt.annotate(annot, (rho_gat[i], rho_1d[i]))
plt.xlabel("Epi-CNN saliency score", fontsize=40)
plt.ylabel("Effect on gene expression (%)", fontsize=40)
#plt.xlim((0,14))
#plt.ylim((0,14))
plt.title('SP (significant decrease): '+str(spearmanr(cnn_fa_all[idx], expr_change_percent_all[idx])[0].astype(np.float16)), fontsize=20)
#plt.plot(np.linspace(0,1,100),np.linspace(0,1,100),color='red')
plt.tick_params(axis='x', labelsize=25)
plt.tick_params(axis='y', labelsize=25)
plt.xscale('log',basex=10)
plt.xlim((0.000001,200))
plt.grid()
plt.savefig('../figs/flowfish/Epi-CNN_saliency_vs_Expr_Change_scatter_plot_K562.png')


####### Precision-Recall #######
Y_true = np.zeros(len(significant_flowfish), dtype=int)
idx = np.where(significant_flowfish==1)[0]
Y_true[idx] = 1
Y_pred_gat = gat_fa_all
Y_pred_cnn = cnn_fa_all
Y_pred_abc = abc_all

precision_gat, recall_gat, thresholds_gat = precision_recall_curve(Y_true, Y_pred_gat)
average_precision_gat = average_precision_score(Y_true, Y_pred_gat)
auprc_gat_all = auc(recall_gat, precision_gat)
print('Epi-GraphReg AP score: ', average_precision_gat)
print('Epi-GraphReg AUPRC score: ', auprc_gat_all)

precision_cnn, recall_cnn, thresholds_cnn = precision_recall_curve(Y_true, Y_pred_cnn)
average_precision_cnn = average_precision_score(Y_true, Y_pred_cnn)
auprc_cnn_all = auc(recall_cnn, precision_cnn)
print('Epi-CNN AP score: ', average_precision_cnn)
print('Epi-CNN AUPRC score: ', auprc_cnn_all)

precision_abc, recall_abc, thresholds_abc = precision_recall_curve(Y_true, Y_pred_abc)
average_precision_abc = average_precision_score(Y_true, Y_pred_abc)
auprc_abc_all = auc(recall_abc, precision_abc)
print('ABC AP score: ', average_precision_abc)
print('ABC AUPRC score: ', auprc_abc_all)

plt.figure(figsize=(15,15))
plt.plot(recall_gat, precision_gat, color='red', label="Epi-GraphReg: AUC = "+str(auprc_gat_all.astype(np.float16)))
plt.plot(recall_cnn, precision_cnn, color='green', label="Epi-CNN: AUC = "+str(auprc_cnn_all.astype(np.float16)))
plt.plot(recall_abc, precision_abc, color='blue', label="ABC: AUC = "+str(auprc_abc_all.astype(np.float16)))
plt.legend(loc="upper right", fontsize=40)
plt.xlabel("Recall", fontsize=40)
plt.ylabel("Precision", fontsize=40)
plt.tick_params(axis='x', labelsize=40)
plt.tick_params(axis='y', labelsize=40)
plt.savefig('../figs/flowfish/PR_curve.png')


##### SP ######
print('ABC SP for significants: ', spearmanr(abc_all[idx], expr_change_percent_all[idx])[0])
print('Epi-GraphReg SP for significants: ', spearmanr(gat_fa_all[idx], expr_change_percent_all[idx])[0])
print('Epi-CNN SP for significants: ', spearmanr(cnn_fa_all[idx], expr_change_percent_all[idx])[0])



