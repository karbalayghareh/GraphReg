from __future__ import division
import sys
from turtle import distance
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
import gc

data_path = '/media/labuser/STORAGE/GraphReg'   # data path

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
        #Y = tf.reshape(Y, [batch_size, 3*T, b])
        #Y = tf.reduce_sum(Y, axis=2)
        Y = tf.reshape(Y, [batch_size, 3*T])

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
    x_background = np.zeros([1,60000,3]).astype('float32')
    #x_background = np.random.normal(0,1,size=[1000,60000,4]).astype('float32')
    #adj_background = np.zeros([1,1200,1200]).astype('float32')
    
    for num, cell_line in enumerate(cell_lines):
        for i, chrm in enumerate(chr_list):
            enhancer_gene_score = open(data_path+'/results/csv/distal_reg_paper/gradients_per_gene/'+cell_line+'/grads_'+genome+'_FDR_'+fdr+'_'+gene_names_list[i]+'.tsv', "w")
            enhancer_gene_score.write('chr\tstart\tend\tname\tclass\tTargetGene\tTargetGeneTSS\tCellType\tH3K4me3\tH3K27ac\tDNase\tGrad_H3K4me3\tGrad_H3K27ac\tGrad_DNase\tDistanceToTSS')
            print(i, gene_names_list[i])
            file_name = data_path+'/data/tfrecords/distal_reg_paper/tfr_epi_RPGC_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'_'+chrm+'.tfr'
            iterator = dataset_iterator(file_name, batch_size)
            while True:
                data_exist, X, X_epi, Y, adj, idx, tss_idx, pos = read_tf_record_1shot(iterator)
                if data_exist:
                    if tf.reduce_sum(tf.gather(tss_idx, idx)) > 0:
                        if (gene_tss_list[i] >= pos[0]+2000000 and gene_tss_list[i] < pos[0]+4000000):

                            print('pos[0]: ', pos[0])

                            ############# Feature Attribution of Epi-GraphReg #############

                            if saliency_method == 'deepshap':
                                if load_fa == False:
                                    explain_output_idx = np.floor((gene_tss_list[i]-pos[0])/5000).astype('int64').reshape([1,1])
                                    print('explain_output_idx: ', explain_output_idx)
                                    shap_values_gat = 0
                                    for j in range(1,1+11):
                                        if j < 11:
                                            valid_chr_list = [j, j+10]
                                            test_chr_list = [j+1, j+11]
                                            valid_chr_str = ['chr'+str(c) for c in valid_chr_list]
                                            valid_chr_str = ','.join(valid_chr_str)
                                            test_chr_str = ['chr'+str(c) for c in test_chr_list]
                                            test_chr_str = ','.join(test_chr_str)
                                        elif j == 11:
                                            valid_chr_str = 'chr11,chr21'
                                            test_chr_str = 'chr1,chr22,chrX'

                                        model_name_gat = data_path+'/models/'+cell_line+'/distal_reg_paper/Epi-GraphReg_RPGC_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'_valid_'+valid_chr_str+'_test_'+test_chr_str+'.h5'
                                        model_gat = tf.keras.models.load_model(model_name_gat, custom_objects={'GraphAttention': GraphAttention})
                                        model_gat.trainable = False
                                        model_gat._name = 'Epi-GraphReg'
                                        model_gat_1 = tf.keras.Model(inputs=model_gat.inputs, outputs=model_gat.outputs[0])

                                        explainer_gat = shap.GradientExplainer(model_gat_1, [x_background, adj.numpy()], batch_size=1)

                                        if gene_tss_list[i]%5000 >= 4800:
                                            shap_values_gat_1, indexes_gat = explainer_gat.shap_values([X_epi.numpy(), adj.numpy()], nsamples=200, ranked_outputs=explain_output_idx, output_rank_order="custom")
                                            shap_values_gat_2, indexes_gat = explainer_gat.shap_values([X_epi.numpy(), adj.numpy()], nsamples=200, ranked_outputs=explain_output_idx+1, output_rank_order="custom")
                                            shap_values_gat = shap_values_gat_1[0][0] + shap_values_gat_2[0][0]
                                        elif gene_tss_list[i]%5000 <= 200:
                                            shap_values_gat_1, indexes_gat = explainer_gat.shap_values([X_epi.numpy(), adj.numpy()], nsamples=200, ranked_outputs=explain_output_idx, output_rank_order="custom")
                                            shap_values_gat_2, indexes_gat = explainer_gat.shap_values([X_epi.numpy(), adj.numpy()], nsamples=200, ranked_outputs=explain_output_idx-1, output_rank_order="custom")
                                            shap_values_gat = shap_values_gat_1[0][0] + shap_values_gat_2[0][0]
                                        else:
                                            shap_values_gat_1, indexes_gat = explainer_gat.shap_values([X_epi.numpy(), adj.numpy()], nsamples=200, ranked_outputs=explain_output_idx, output_rank_order="custom")
                                            shap_values_gat_total = shap_values_gat_1[0][0]

                                        shap_values_gat = shap_values_gat + shap_values_gat_total

                                    shap_values_gat = shap_values_gat/11
                                else:
                                    pass

                                scores_gat = K.reshape(shap_values_gat, [60000,3])
                                scores_gat = K.mean(scores_gat, axis = 1).numpy()
                                print('gat: ', scores_gat.shape, np.min(scores_gat), np.max(scores_gat), np.mean(scores_gat))

                            elif saliency_method == 'saliency':
                                if load_fa == False:
                                    explain_output_idx_gat = np.floor((gene_tss_list[i]-pos[0])/5000).astype('int64')
                                    print('explain_output_idx_gat: ', explain_output_idx_gat)
                                    grads_gat = 0
                                    for j in [5]: #range(1,1+11):
                                        if j < 11:
                                            valid_chr_list = [j, j+10]
                                            test_chr_list = [j+1, j+11]
                                            valid_chr_str = ['chr'+str(c) for c in valid_chr_list]
                                            valid_chr_str = ','.join(valid_chr_str)
                                            test_chr_str = ['chr'+str(c) for c in test_chr_list]
                                            test_chr_str = ','.join(test_chr_str)
                                        elif j == 11:
                                            valid_chr_str = 'chr11,chr21'
                                            test_chr_str = 'chr1,chr22,chrX'

                                        model_name_gat = data_path+'/models/'+cell_line+'/distal_reg_paper/Epi-GraphReg_RPGC_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'_valid_'+valid_chr_str+'_test_'+test_chr_str+'.h5'
                                        model_gat = tf.keras.models.load_model(model_name_gat, custom_objects={'GraphAttention': GraphAttention})
                                        model_gat.trainable = False
                                        model_gat._name = 'Epi-GraphReg'

                                        with tf.GradientTape(persistent=True) as tape:
                                            inp = X_epi
                                            tape.watch(inp)
                                            preds, _ = model_gat([inp, adj])
                                            if gene_tss_list[i]%5000 >= 4800:
                                                target_gat = preds[:, explain_output_idx_gat]+preds[:, explain_output_idx_gat+1]
                                            elif gene_tss_list[i]%5000 <= 200:
                                                target_gat = preds[:, explain_output_idx_gat-1]+preds[:, explain_output_idx_gat]
                                            else:
                                                target_gat = preds[:, explain_output_idx_gat]

                                        grads_gat = grads_gat + tape.gradient(target_gat, inp)
                                    #grads_gat = grads_gat/11
                                    grads_gat = grads_gat
                                else:
                                    pass

                                #scores_gat = K.reshape(grads_gat * X_epi, [60000,3])
                                #scores_gat = K.mean(scores_gat, axis = 1).numpy()
                                scores_gat = K.reshape(grads_gat, [60000,3]).numpy()
                                X_epi = K.reshape(X_epi, [60000,3]).numpy()
                                print('gat: ', scores_gat.shape, np.min(scores_gat), np.max(scores_gat), np.mean(scores_gat))


                            ########## Write bigwig saliency files #########
                            if write_bw == True and organism == 'human' and genome == 'hg19':
                                header = [("chr1", 249250621), ("chr2", 243199373), ("chr3", 198022430), ("chr4", 191154276), ("chr5", 180915260), ("chr6", 171115067),
                                            ("chr7", 159138663), ("chr8", 146364022), ("chr9", 141213431), ("chr10", 135534747), ("chr11", 135006516), ("chr12", 133851895),
                                            ("chr13", 115169878), ("chr14", 107349540), ("chr15", 102531392), ("chr16", 90354753), ("chr17", 81195210), ("chr18", 78077248),
                                            ("chr19", 59128983), ("chr20", 63025520), ("chr21", 48129895), ("chr22", 51304566), ("chrX", 155270560)]

                            if write_bw == True and organism == 'human' and genome == 'hg38':
                                header = [("chr1", 248956422), ("chr2", 242193529), ("chr3", 198295559), ("chr4", 190214555), ("chr5", 181538259), ("chr6", 170805979),
                                            ("chr7", 159345973), ("chr8", 145138636), ("chr9", 138394717), ("chr10", 133797422), ("chr11", 135086622), ("chr12", 133275309),
                                            ("chr13", 114364328), ("chr14", 107043718), ("chr15", 101991189), ("chr16", 90338345), ("chr17", 83257441), ("chr18", 80373285),
                                            ("chr19", 58617616), ("chr20", 64444167), ("chr21", 46709983), ("chr22", 50818468), ("chrX", 156040895)]

                            if write_bw == True and organism == 'mouse':
                                header = [("chr1", 195465000), ("chr2", 182105000), ("chr3", 160030000), ("chr4", 156500000), ("chr5", 151825000), ("chr6", 149730000),
                                        ("chr7", 145435000), ("chr8", 129395000), ("chr9", 124590000), ("chr10", 130685000), ("chr11", 122075000), ("chr12", 120120000),
                                        ("chr13", 120415000), ("chr14", 124895000), ("chr15", 104035000), ("chr16", 98200000), ("chr17", 94980000), ("chr18", 90695000),
                                        ("chr19", 61425000), ("chrX", 171025000)]

                            if write_bw == True:
                                bw_GraphReg = pyBigWig.open(data_path+'/results/bigwig/feature_attribution/Epi-models/Epi-GraphReg_'+saliency_method+'_'+cell_line+'_'+gene_names_list[i]+'.bw', "w")
                                bw_GraphReg.addHeader(header)

                                chroms = np.array([chrm] * len(pos))
                                print(chrm, len(pos))
                                starts = pos.astype(np.int64)
                                ends = starts + 100
                                ends = ends.astype(np.int64)

                                bw_GraphReg.addEntries(chroms, starts, ends=ends, values=scores_gat.ravel())
                                bw_GraphReg.close()

                            ########## Write Eenhancer-Gene scores to dataframe #########

                            for k in range(scores_gat.shape[0]):
                                name = chrm+':'+str(pos[k])+'-'+str(pos[k]+100)
                                distance_to_tss = np.abs((pos[k] + 50) - gene_tss_list[i])
                                line = '{}\t{}\t{}\t{}\t.\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(chrm, pos[k], pos[k]+100, name, gene_names_list[i], gene_tss_list[i], cell_line, X_epi[k,0], X_epi[k,1], X_epi[k,2], scores_gat[k,0], scores_gat[k,1], scores_gat[k,2], distance_to_tss)
                                enhancer_gene_score.write('\n')
                                enhancer_gene_score.write(line)

                            del model_gat, scores_gat, data_exist, X, X_epi, Y, adj, idx, tss_idx, pos, iterator
                            tf.keras.backend.clear_session()
                            gc.collect()
                            break
                else:
                    break

            enhancer_gene_score.close()

    return 'done'

###################################### load model ######################################
batch_size = 1
qval = .1                                       # 0.1, 0.01, 0.001
assay_type = 'HiC'                           # HiChIP, HiC, MicroC, HiCAR
organism = 'human'            # human/mouse
genome='hg38'                 # hg19/hg38
cell_line = ['GM12878']          # K562/GM12878/mESC/hESC
write_bw = False               # write the feature attribution scores to bigwig files
load_fa = False               # load feature attribution numpy files
saliency_method = 'saliency'  # 'saliency' or 'deepshap' 
crispr_dataset = 'none'      # 'fulco', 'gasperini', 'combined', 'none'

if qval == 0.1:
    fdr = '1'
elif qval == 0.01:
    fdr = '01'
elif qval == 0.001:
    fdr = '001'
elif qval == 0.5:
    fdr = '5'
elif qval == 0.9:
    fdr = '9'

if organism == 'human' and genome == 'hg19':
    filename_tss = data_path+'/data/tss/'+organism+'/'+genome+'/hg19_gencodev19_tss.bed'
elif organism == 'human' and genome == 'hg38':
    #filename_tss = data_path+'/data/tss/'+organism+'/'+genome+'/gencode.v38.annotation.gtf.tss.bed'
    filename_tss = data_path+'/data/tss/'+organism+'/distal_reg_paper/'+genome+'/RefSeqCurated.170308.bed.CollapsedGeneBounds.hg38.TSS500bp.bed'
elif organism == 'mouse':
    filename_tss = data_path+'/data/tss/'+organism+'/'+genome+'/mm10_gencode_vM9_tss.bed'

tss_dataframe = pd.read_csv(filename_tss, header=None, delimiter='\t')
tss_dataframe.columns = ["chr", "tss_1", "tss_2", "gene", "na", "strand"]

if crispr_dataset == 'fulco':
    crispr_data = pd.read_csv(data_path+'/data/csv/CRISPR_benchmarking_data/EPCrisprBenchmark_Fulco2019_K562_GRCh38.tsv', delimiter='\t')
    crispr_data = crispr_data[crispr_data['ValidConnection']=="TRUE"].reset_index(drop=True)
    gene_names_list = np.unique(crispr_data['measuredGeneSymbol'].values)
    print('Number of E-G pais in Fulco dataset: {}'.format(len(crispr_data)))
    print('Number of genes in Fulco dataset: {}'.format(len(gene_names_list)))
elif crispr_dataset == 'gasperini':
    crispr_data = pd.read_csv(data_path+'/data/csv/CRISPR_benchmarking_data/EPCrisprBenchmark_Gasperini2019_0.13gStd_0.8pwrAt15effect_GRCh38.tsv', delimiter='\t')
    crispr_data = crispr_data[crispr_data['ValidConnection']=="TRUE"].reset_index(drop=True)
    gene_names_list = np.unique(crispr_data['measuredGeneSymbol'].values)
    print('Number of E-G pais in Gasperini dataset: {}'.format(len(crispr_data)))
    print('Number of genes in Gasperini dataset: {}'.format(len(gene_names_list)))
elif crispr_dataset == 'combined':
    crispr_data = pd.read_csv(data_path+'/data/csv/CRISPR_benchmarking_data/EPCrisprBenchmark_ensemble_data_GRCh38.tsv', delimiter='\t')
    gene_names_list = np.unique(crispr_data['measuredGeneSymbol'].values)
    print('Number of E-G pais in combined dataset: {}'.format(len(crispr_data)))
    print('Number of genes in combined dataset: {}'.format(len(gene_names_list)))
elif crispr_dataset == 'none':
    tss_dataframe_sub = tss_dataframe[tss_dataframe['chr'].isin(['chr'+str(i) for i in range(1,23)]+['chrX'])].reset_index(drop=True)
    gene_names_list = tss_dataframe_sub['gene'].values
    chr_list = tss_dataframe_sub['chr'].values
    tss_list = tss_dataframe_sub['tss_1'].values.astype(np.int64)

if crispr_dataset != 'none':
    chr_list = []
    tss_list = []
    for g in gene_names_list:
        chr_list.append(crispr_data[crispr_data['measuredGeneSymbol']==g]['chrom'].values[0])
        tss_list.append(crispr_data[crispr_data['measuredGeneSymbol']==g]['startTSS'].values[0])

    df = pd.DataFrame({'chr': chr_list, 'tss': tss_list, 'gene': gene_names_list})
    if crispr_dataset == 'fulco':
        df.loc[3, 'tss'] = tss_dataframe[tss_dataframe['gene']=='C19orf43']['tss_1'].values[0]   # Replace TSS of C19orf43
    elif crispr_dataset == 'combined':
        df.loc[1181, 'tss'] = tss_dataframe[tss_dataframe['gene']=='C19orf43']['tss_1'].values[0]   # Replace TSS of C19orf43

    df = df.sort_values(by=['chr', 'tss']).reset_index(drop=True)

    gene_names_list = df['gene'].values
    chr_list = df['chr'].values
    tss_list = df['tss'].values.astype(np.int64)

calculate_loss(cell_line, gene_names_list, tss_list, chr_list, batch_size, saliency_method, write_bw, load_fa, organism, genome)
