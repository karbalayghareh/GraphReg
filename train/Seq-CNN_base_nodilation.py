from __future__ import division
from optparse import OptionParser
  
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow import keras
import matplotlib.pyplot as plt
import time
from scipy.stats import spearmanr
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

def main():
    usage = 'usage: %prog [options]'
    parser = OptionParser(usage)
    parser.add_option('-c', dest='cell_line',
        default='K562', type='str')
    parser.add_option('-o', dest='organism',
        default='human', type='str')
    parser.add_option('-v', dest='valid_chr',
        default='1,11', type='str')
    parser.add_option('-t', dest='test_chr',
        default='2,12', type='str')
    parser.add_option('-p', dest='data_path',
        default='/media/labuser/STORAGE/GraphReg', type='str')
    parser.add_option('-a', dest='assay_type',
        default='HiChIP', type='str')
    parser.add_option('-q', dest='qval',
        default=0.1, type='float')

    (options, args) = parser.parse_args()
    valid_chr_str = options.valid_chr.split(',')
    valid_chr = [int(valid_chr_str[i]) for i in range(len(valid_chr_str))]
    test_chr_str = options.test_chr.split(',')
    test_chr = [int(test_chr_str[i]) for i in range(len(test_chr_str))]

    data_path = options.data_path
    assay_type = options.assay_type
    qval = options.qval

    if qval == 0.1:
        fdr = '1'
    elif qval == 0.01:
        fdr = '01'
    elif qval == 0.001:
        fdr = '001'

    print('organism:', options.organism)
    print('cell type:', options.cell_line)
    print('valid chrs: ', valid_chr)
    print('test chrs: ', test_chr)
    print('data path: ', options.data_path)
    print('3D assay type: ', options.assay_type)
    print('HiCDCPlus FDR: ', options.qval)

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

        seq = tf.io.decode_raw(parsed_features['sequence'], tf.float64)
        seq = tf.cast(seq, tf.float32)

        return {'seq': seq, 'last_batch': last_batch, 'X_epi': X_epi, 'Y': Y, 'adj': adj, 'tss_idx': tss_idx}

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
            T = 400           # number of 5kb bins inside middle 2Mb region
            F = 4             # number of ACGT (4)
            seq = next_datum['seq']
            seq = tf.reshape(seq, [60, 100000, F])

            last_batch = next_datum['last_batch']
            tss_idx = next_datum['tss_idx']
            tss_idx = tf.reshape(tss_idx, [3*T])
            idx = tf.range(T, 2*T)

            Y = next_datum['Y']
            Y = tf.reshape(Y, [60, 1000])

            X_epi = next_datum['X_epi']
            X_epi = tf.reshape(X_epi, [60, 1000, 3])
            Y_h3k4me3 = X_epi[:,:,0]
            Y_h3k27ac = X_epi[:,:,1]
            Y_dnase = X_epi[:,:,2]

        else:
            seq = 0
            Y = 0
            Y_h3k4me3 = 0
            Y_h3k27ac = 0
            Y_dnase = 0
            tss_idx = 0
            idx = 0
        return data_exist, seq, Y, Y_h3k4me3, Y_h3k27ac, Y_dnase, idx, tss_idx


    def calculate_loss(model_cnn_base, chr_list, cell_lines, b_size):
        loss_h3k4me3_all = np.array([])
        loss_h3k27ac_all = np.array([])
        loss_dnase_all = np.array([])
        rho_h3k4me3_all = np.array([])
        rho_h3k27ac_all = np.array([])
        rho_dnase_all = np.array([])
        for num, cell_line in enumerate(cell_lines):
            for i in chr_list:
                print(' chr :', i)
                file_name = data_path+'/data/tfrecords/tfr_seq_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'_chr'+str(i)+'.tfr'
                iterator = dataset_iterator(file_name, batch_size=1)
                while True:
                    data_exist, seq, Y, Y_h3k4me3, Y_h3k27ac, Y_dnase, idx, tss_idx = read_tf_record_1shot(iterator)
                    if data_exist:
                        if tf.reduce_sum(tf.gather(tss_idx, idx)) > 0:
                            for jj in range(20,40,b_size):
                                seq_batch = seq[jj:jj+b_size]
                                Y_h3k4me3_batch = Y_h3k4me3[jj:jj+b_size]
                                Y_h3k27ac_batch = Y_h3k27ac[jj:jj+b_size]
                                Y_dnase_batch = Y_dnase[jj:jj+b_size]

                                Y_hat_h3k4me3, Y_hat_h3k27ac, Y_hat_dnase, _ = model_cnn_base(seq_batch)

                                loss_h3k4me3 = poisson_loss(Y_h3k4me3_batch, Y_hat_h3k4me3)
                                loss_h3k27ac = poisson_loss(Y_h3k27ac_batch, Y_hat_h3k27ac)
                                loss_dnase = poisson_loss(Y_dnase_batch, Y_hat_dnase)

                                loss_h3k4me3_all = np.append(loss_h3k4me3_all, loss_h3k4me3.numpy())
                                loss_h3k27ac_all = np.append(loss_h3k27ac_all, loss_h3k27ac.numpy())
                                loss_dnase_all = np.append(loss_dnase_all, loss_dnase.numpy())

                                e1 = np.random.normal(0,1e-6,size=len(Y_h3k4me3_batch.numpy().ravel()))
                                e2 = np.random.normal(0,1e-6,size=len(Y_h3k4me3_batch.numpy().ravel()))
                                rho_h3k4me3_all = np.append(rho_h3k4me3_all, np.corrcoef(np.log2(Y_h3k4me3_batch.numpy().ravel()+1)+e1, np.log2(Y_hat_h3k4me3.numpy().ravel()+1)+e2)[0,1])
                                rho_h3k27ac_all = np.append(rho_h3k27ac_all, np.corrcoef(np.log2(Y_h3k27ac_batch.numpy().ravel()+1)+e1, np.log2(Y_hat_h3k27ac.numpy().ravel()+1)+e2)[0,1])
                                rho_dnase_all = np.append(rho_dnase_all, np.corrcoef(np.log2(Y_dnase_batch.numpy().ravel()+1)+e1, np.log2(Y_hat_dnase.numpy().ravel()+1)+e2)[0,1])

                    else:
                        break

        print('len of test/valid batches: ', len(loss_h3k4me3_all))
        loss_h3k4me3 = np.mean(loss_h3k4me3_all)
        loss_h3k27ac = np.mean(loss_h3k27ac_all)
        loss_dnase = np.mean(loss_dnase_all)
        loss_valid = (1/3) * (loss_h3k4me3 + loss_h3k27ac + loss_dnase)

        rho_h3k4me3 = np.mean(rho_h3k4me3_all)
        rho_h3k27ac = np.mean(rho_h3k27ac_all)
        rho_dnase = np.mean(rho_dnase_all)

        return loss_valid, loss_h3k4me3, loss_h3k27ac, loss_dnase, rho_h3k4me3, rho_h3k27ac, rho_dnase

    # Parameters
    F = 4
    dropout_rate = 0.5       # dropout rate
    l2_reg = 0.0             # factor for l2 regularization
    re_load = False

    # Model definition

    if re_load:
        model_name = 'model_name.h5'
        model = tf.keras.models.load_model(model_name)
        model.summary()
    else:
        tf.keras.backend.clear_session()
        X_in = Input(shape=(100000,F))

        x = layers.Conv1D(256, 21, activation='relu', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(X_in)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool1D(2)(x)

        x = Dropout(dropout_rate)(x)
        x = layers.Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool1D(2)(x)

        x = Dropout(dropout_rate)(x)
        x = layers.Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool1D(5)(x)

        x = Dropout(dropout_rate)(x)
        x = layers.Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool1D(5)(x)

        x = Dropout(dropout_rate)(x)
        x = layers.Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(x)
        h = layers.BatchNormalization()(x)
        
        x1 = h
        x2 = h
        x3 = h

        for i in range(1,1+3):
            x1 = Dropout(dropout_rate)(x1)
            x1 = layers.Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(x1)
            x1 = layers.BatchNormalization()(x1)

        mu_h3k4me3 = layers.Conv1D(1, 3, activation='exponential', padding='same')(x1)
        mu_h3k4me3 = layers.Reshape([1000])(mu_h3k4me3)

        for i in range(1,1+3):
            x2 = Dropout(dropout_rate)(x2)
            x2 = layers.Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(x2)
            x2 = layers.BatchNormalization()(x2)

        mu_h3k27ac = layers.Conv1D(1, 3, activation='exponential', padding='same')(x2)
        mu_h3k27ac = layers.Reshape([1000])(mu_h3k27ac)

        for i in range(1,1+3):
            x3 = Dropout(dropout_rate)(x3)
            x3 = layers.Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(x3)
            x3 = layers.BatchNormalization()(x3)

        mu_dnase = layers.Conv1D(1, 3, activation='exponential', padding='same')(x3)
        mu_dnase = layers.Reshape([1000])(mu_dnase)

        # Build model
        model_cnn_base = Model(inputs=X_in, outputs=[mu_h3k4me3, mu_h3k27ac, mu_dnase, h])
        model_cnn_base._name = 'Seq-CNN_base_nodilution'
        model_cnn_base.summary()


    ########## training ##########

    cell_line = options.cell_line
    cell_lines = [cell_line]
    model_name_cnn_base = data_path+'/models/'+cell_line+'/Seq-CNN_base_nodilation_'+cell_line+'_valid_chr_'+options.valid_chr+'_test_chr_'+options.test_chr+'.h5'

    if options.organism == 'mouse':
        train_chr_list = [c for c in range(1,1+19)]
        valid_chr_list = valid_chr
        test_chr_list = test_chr
        vt = valid_chr_list + test_chr_list
        for j in range(len(vt)):
            train_chr_list.remove(vt[j])
    else:
        train_chr_list = [c for c in range(1,1+22)]
        valid_chr_list = valid_chr
        test_chr_list = test_chr
        vt = valid_chr_list + test_chr_list
        for j in range(len(vt)):
            train_chr_list.remove(vt[j])

    best_loss = 1e20
    max_early_stopping = 5
    n_epochs = 100
    opt = tf.keras.optimizers.Adam(learning_rate=.0002, decay=1e-6)
    b_size = 4
    t0 = time.time()
    for epoch in range(1,n_epochs+1):
        loss_h3k4me3_all = np.array([])
        loss_h3k27ac_all = np.array([])
        loss_dnase_all = np.array([])
        rho_h3k4me3_all = np.array([])
        rho_h3k27ac_all = np.array([])
        rho_dnase_all = np.array([])

        for num, cell_line in enumerate(cell_lines):
            for i in train_chr_list:
                print('train chr :', i)
                file_name_train = data_path+'/data/tfrecords/tfr_seq_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'_chr'+str(i)+'.tfr'
                iterator_train = dataset_iterator(file_name_train, batch_size=1)
                while True:
                    data_exist, seq, Y, Y_h3k4me3, Y_h3k27ac, Y_dnase, idx, tss_idx = read_tf_record_1shot(iterator_train)
                    if data_exist:
                        if tf.reduce_sum(tf.gather(tss_idx, idx)) > 0:
                            for jj in range(20,40,b_size):
                                seq_batch = seq[jj:jj+b_size]
                                Y_h3k4me3_batch = Y_h3k4me3[jj:jj+b_size]
                                Y_h3k27ac_batch = Y_h3k27ac[jj:jj+b_size]
                                Y_dnase_batch = Y_dnase[jj:jj+b_size]
                                with tf.GradientTape() as tape:
                                    Y_hat_h3k4me3, Y_hat_h3k27ac, Y_hat_dnase, _ = model_cnn_base(seq_batch)
                                    loss = (1/3) * (poisson_loss(Y_h3k4me3_batch, Y_hat_h3k4me3)
                                        + poisson_loss(Y_h3k27ac_batch, Y_hat_h3k27ac)
                                        + poisson_loss(Y_dnase_batch, Y_hat_dnase))

                                grads = tape.gradient(loss, model_cnn_base.trainable_variables)
                                opt.apply_gradients(zip(grads, model_cnn_base.trainable_variables))

                                loss_h3k4me3 = poisson_loss(Y_h3k4me3_batch, Y_hat_h3k4me3)
                                loss_h3k27ac = poisson_loss(Y_h3k27ac_batch, Y_hat_h3k27ac)
                                loss_dnase = poisson_loss(Y_dnase_batch, Y_hat_dnase)

                                loss_h3k4me3_all = np.append(loss_h3k4me3_all, loss_h3k4me3.numpy())
                                loss_h3k27ac_all = np.append(loss_h3k27ac_all, loss_h3k27ac.numpy())
                                loss_dnase_all = np.append(loss_dnase_all, loss_dnase.numpy())

                                e1 = np.random.normal(0,1e-6,size=len(Y_h3k4me3_batch.numpy().ravel()))
                                e2 = np.random.normal(0,1e-6,size=len(Y_h3k4me3_batch.numpy().ravel()))
                                rho_h3k4me3_all = np.append(rho_h3k4me3_all, np.corrcoef(np.log2(Y_h3k4me3_batch.numpy().ravel()+1)+e1,np.log2(Y_hat_h3k4me3.numpy().ravel()+1)+e2)[0,1])
                                rho_h3k27ac_all = np.append(rho_h3k27ac_all, np.corrcoef(np.log2(Y_h3k27ac_batch.numpy().ravel()+1)+e1,np.log2(Y_hat_h3k27ac.numpy().ravel()+1)+e2)[0,1])
                                rho_dnase_all = np.append(rho_dnase_all, np.corrcoef(np.log2(Y_dnase_batch.numpy().ravel()+1)+e1,np.log2(Y_hat_dnase.numpy().ravel()+1)+e2)[0,1])

                    else:
                        break
        if epoch == 1:
            print('len of train batches: ', len(loss_h3k4me3_all))
        loss_h3k4me3 = np.mean(loss_h3k4me3_all)
        loss_h3k27ac = np.mean(loss_h3k27ac_all)
        loss_dnase = np.mean(loss_dnase_all)
        loss_train = (1/3) * (loss_h3k4me3 + loss_h3k27ac + loss_dnase)

        rho_h3k4me3 = np.mean(rho_h3k4me3_all)
        rho_h3k27ac = np.mean(rho_h3k27ac_all)
        rho_dnase = np.mean(rho_dnase_all)

        print('train epoch: ', epoch, 'total loss: ', loss_train, 'loss h3k4me3: ', loss_h3k4me3, 'loss h3k27ac: ', loss_h3k27ac, 'loss dnase: ', loss_dnase, ', rho_h3k4me3: ', rho_h3k4me3, ', rho_h3k27ac: ', rho_h3k27ac, ', rho_dnase: ', rho_dnase, ', time passed: ', (time.time() - t0), ' sec')

        if epoch%1 == 0:
            valid_loss, valid_loss_h3k4me3, valid_loss_h3k27ac, valid_loss_dnase, valid_rho_h3k4me3, valid_rho_h3k27ac, valid_rho_dnase = calculate_loss(model_cnn_base, valid_chr_list, cell_lines, b_size)

        if valid_loss < best_loss:
            early_stopping_counter = 1
            best_loss = valid_loss
            model_cnn_base.save(model_name_cnn_base)
            print('valid epoch: ', epoch, 'total loss: ', valid_loss, 'loss h3k4me3: ', valid_loss_h3k4me3, 'loss h3k27ac: ', valid_loss_h3k27ac, 'loss dnase: ', valid_loss_dnase, ', rho_h3k4me3: ', valid_rho_h3k4me3, ', rho_h3k27ac: ', valid_rho_h3k27ac, ', rho_dnase: ', valid_rho_dnase, ', time passed: ', (time.time() - t0), ' sec')
            test_loss, test_loss_h3k4me3, test_loss_h3k27ac, test_loss_dnase, test_rho_h3k4me3, test_rho_h3k27ac, test_rho_dnase = calculate_loss(model_cnn_base, test_chr_list, cell_lines, b_size)
            print('test epoch: ', epoch, 'total loss: ', test_loss, 'loss h3k4me3: ', test_loss_h3k4me3, 'loss h3k27ac: ', test_loss_h3k27ac, 'loss dnase: ', test_loss_dnase, ', rho_h3k4me3: ', test_rho_h3k4me3, ', rho_h3k27ac: ', test_rho_h3k27ac, ', rho_dnase: ', test_rho_dnase, ', time passed: ', (time.time() - t0), ' sec')

        else:
            early_stopping_counter += 1
            if early_stopping_counter == max_early_stopping:
                break

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
