from __future__ import division
from optparse import OptionParser

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
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
import tensorflow_probability as tfp


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

    (options, args) = parser.parse_args()
    valid_chr_str = options.valid_chr.split(',')
    valid_chr = [int(valid_chr_str[i]) for i in range(len(valid_chr_str))]
    test_chr_str = options.test_chr.split(',')
    test_chr = [int(test_chr_str[i]) for i in range(len(test_chr_str))]
    print('valid chrs: ', valid_chr)
    print('test chrs: ', test_chr)

    def poisson_loss(y_true, mu_pred):
        nll = tf.reduce_mean(
            tf.math.lgamma(y_true + 1) + mu_pred - y_true * tf.math.log(mu_pred))
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
            T = 400       # number of 5kb bins inside middle 2Mb region 
            b = 50        # number of 100bp bins inside 5Kb region
            F = 4         # number of ACGT (4)
            seq = next_datum['seq']
            batch_size = tf.shape(seq)[0]
            seq = tf.reshape(seq, [60, 100000, F])
            adj = next_datum['adj']
            adj = tf.reshape(adj, [3*T, 3*T])
            adj = tf.reshape(adj, [batch_size, 3*T, 3*T])

            last_batch = next_datum['last_batch']
            tss_idx = next_datum['tss_idx']
            tss_idx = tf.reshape(tss_idx, [3*T])
            idx = tf.range(T, 2*T)

            Y = next_datum['Y']
            Y = tf.reshape(Y, [batch_size, 3*T, b])
            Y = tf.reduce_sum(Y, axis=-1)
            Y = tf.reshape(Y, [batch_size, 3*T])

            Y_epi = next_datum['X_epi']
            Y_epi = tf.reshape(Y_epi, [60, 1000, 3])
            Y_h3k4me3 = Y_epi[:,:,0]
            Y_h3k27ac = Y_epi[:,:,1]
            Y_dnase = Y_epi[:,:,2]

        else:
            seq = 0
            Y = 0
            Y_h3k4me3 = 0
            Y_h3k27ac = 0
            Y_dnase = 0
            adj = 0
            tss_idx = 0
            idx = 0
        return data_exist, seq, Y, Y_h3k4me3, Y_h3k27ac, Y_dnase, adj, idx, tss_idx


    def calculate_loss(model_gat, chr_list, cell_lines, batch_size, alpha):
        loss_cage_all = np.array([])
        loss_h3k4me3_all = np.array([])
        loss_h3k27ac_all = np.array([])
        loss_dnase_all = np.array([])
        rho_cage_all = np.array([])
        rho_h3k4me3_all = np.array([])
        rho_h3k27ac_all = np.array([])
        rho_dnase_all = np.array([])
        Y_hat_all = np.array([])
        Y_all = np.array([])
        for num, cell_line in enumerate(cell_lines):
            for i in chr_list:
                print(' chr :', i)
                file_name = '/media/labuser/STORAGE/GraphReg/data/tfrecords/tfr_seq_'+cell_line+'_chr'+str(i)+'.tfr'
                iterator = dataset_iterator(file_name, batch_size)
                while True:
                    data_exist, seq, Y, Y_h3k4me3, Y_h3k27ac, Y_dnase, adj, idx, tss_idx = read_tf_record_1shot(iterator)
                    if data_exist:
                        if tf.reduce_sum(tf.gather(tss_idx, idx)) > 0:
                            Y_hat_cage, Y_hat_h3k4me3, Y_hat_h3k27ac, Y_hat_dnase, _, _ = model_gat([seq, adj])
                            Y_hat_cage_idx = tf.gather(Y_hat_cage, idx, axis=1)
                            Y_idx = tf.gather(Y, idx, axis=1)

                            e1 = np.random.normal(0,1e-6,size=len(Y_idx.numpy().ravel()))
                            e2 = np.random.normal(0,1e-6,size=len(Y_idx.numpy().ravel()))
                            e3 = np.random.normal(0,1e-6,size=len(Y_h3k4me3.numpy().ravel()))
                            e4 = np.random.normal(0,1e-6,size=len(Y_h3k4me3.numpy().ravel()))

                            loss_cage = poisson_loss(Y_idx, Y_hat_cage_idx)
                            loss_h3k4me3 = poisson_loss(Y_h3k4me3, Y_hat_h3k4me3)
                            loss_h3k27ac = poisson_loss(Y_h3k27ac, Y_hat_h3k27ac)
                            loss_dnase = poisson_loss(Y_dnase, Y_hat_dnase)

                            loss_cage_all = np.append(loss_cage_all, loss_cage.numpy())
                            loss_h3k4me3_all = np.append(loss_h3k4me3_all, loss_h3k4me3.numpy())
                            loss_h3k27ac_all = np.append(loss_h3k27ac_all, loss_h3k27ac.numpy())
                            loss_dnase_all = np.append(loss_dnase_all, loss_dnase.numpy())

                            rho_cage_all = np.append(rho_cage_all, np.corrcoef(np.log2(Y_idx.numpy().ravel()+1)+e1,np.log2(Y_hat_cage_idx.numpy().ravel()+1)+e2)[0,1])
                            rho_h3k4me3_all = np.append(rho_h3k4me3_all, np.corrcoef(np.log2(Y_h3k4me3.numpy().ravel()+1)+e3,np.log2(Y_hat_h3k4me3.numpy().ravel()+1)+e4)[0,1])
                            rho_h3k27ac_all = np.append(rho_h3k27ac_all, np.corrcoef(np.log2(Y_h3k27ac.numpy().ravel()+1)+e3,np.log2(Y_hat_h3k27ac.numpy().ravel()+1)+e4)[0,1])
                            rho_dnase_all = np.append(rho_dnase_all, np.corrcoef(np.log2(Y_dnase.numpy().ravel()+1)+e3,np.log2(Y_hat_dnase.numpy().ravel()+1)+e4)[0,1])

                            Y_hat_all = np.append(Y_hat_all, Y_hat_cage_idx.numpy().ravel())
                            Y_all = np.append(Y_all, Y_idx.numpy().ravel())

                    else:
                        #print('no data')
                        break

        print('len of test/valid batches: ', len(rho_cage_all))
        loss_cage = np.mean(loss_cage_all)
        loss_h3k4me3 = np.mean(loss_h3k4me3_all)
        loss_h3k27ac = np.mean(loss_h3k27ac_all)
        loss_dnase = np.mean(loss_dnase_all)
        loss_valid = alpha * loss_cage + loss_h3k4me3 + loss_h3k27ac + loss_dnase
        loss_valid = alpha * loss_cage + (1-alpha) * (loss_h3k4me3 + loss_h3k27ac + loss_dnase)

        rho_cage = np.mean(rho_cage_all)
        rho_h3k4me3 = np.mean(rho_h3k4me3_all)
        rho_h3k27ac = np.mean(rho_h3k27ac_all)
        rho_dnase = np.mean(rho_dnase_all)

        #sp = spearmanr(Y_all, Y_hat_all)[0]
        return loss_valid, loss_cage, loss_h3k4me3, loss_h3k27ac, loss_dnase, rho_cage, rho_h3k4me3, rho_h3k27ac, rho_dnase

    # Parameters
    T = 400
    N = 3*T                         # window size
    F = 4                           # Original feature dimension
    F_ = 32                         # Output size of first GraphAttention layer
    n_attn_heads = 4                # Number of attention heads in first GAT layer
    dropout_rate = 0.5              # Dropout rate (between and inside GAT layers)
    l2_reg = 0.0                    # Factor for l2 regularization
    re_load = False

    # Model definition (as per Section 3.3 of the paper)

    if re_load:
        model_name = 'model_name.h5'
        model = tf.keras.models.load_model(model_name, custom_objects={'GraphAttention': GraphAttention})
        model.summary()
    else:
        tf.keras.backend.clear_session()
        X_in = Input(shape=(100000,4))
        A_in = Input(shape=(N,N))

        #x = Dropout(dropout_rate)(X_in)  
        x = layers.Conv1D(32, 21, activation='relu', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(X_in)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool1D(2)(x)

        x = layers.Conv1D(32, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool1D(2)(x)

        x = Dropout(dropout_rate)(x)
        x = layers.Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool1D(5)(x)

        x = Dropout(dropout_rate)(x)
        x = layers.Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool1D(5)(x)    

        x = Dropout(dropout_rate)(x)
        x = layers.Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(x)
        h = layers.BatchNormalization()(x)
        x = h

        for i in range(1,1+6):
            x = Dropout(dropout_rate)(x)
            x = layers.Conv1D(64, 3, activation='relu', dilation_rate=2**i, padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(x) + x
            x = layers.BatchNormalization()(x)

        mu_h3k4me3 = layers.Conv1D(1, 3, activation='exponential', padding='same')(x)
        mu_h3k4me3 = layers.Reshape([1000])(mu_h3k4me3)

        mu_h3k27ac = layers.Conv1D(1, 3, activation='exponential', padding='same')(x)
        mu_h3k27ac = layers.Reshape([1000])(mu_h3k27ac)

        mu_dnase = layers.Conv1D(1, 3, activation='exponential', padding='same')(x)
        mu_dnase = layers.Reshape([1000])(mu_dnase)

        x = Dropout(dropout_rate)(h)
        x = layers.Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool1D(5)(x)

        x = Dropout(dropout_rate)(x)
        x = layers.Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool1D(5)(x)

        x = Dropout(dropout_rate)(x)
        x = layers.Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool1D(2)(x)

        #x = Dropout(dropout_rate)(x)
        #x = layers.Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(x)
        #x = layers.BatchNormalization()(x)

        x = layers.Reshape([20,128,1])(x)
        x = K.permute_dimensions(x, (3, 0, 1, 2))
        x = layers.Reshape([1200,128])(x)

        att=[]
        for i in range(3):
            x, att_ = GraphAttention(F_,
                        attn_heads=n_attn_heads,
                        attn_heads_reduction='concat',
                        dropout_rate=dropout_rate,
                        activation='elu',
                        kernel_regularizer=l2(l2_reg),
                        attn_kernel_regularizer=l2(l2_reg))([x, A_in])
            x = layers.BatchNormalization()(x)
            att.append(att_)

        x = Dropout(dropout_rate)(x)
        x = layers.Conv1D(64, 1, activation='relu', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)

        mu_cage = layers.Conv1D(1, 1, activation='exponential', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(x)
        mu_cage = layers.Reshape([3*T])(mu_cage)

        # Build model
        model_gat = Model(inputs=[X_in, A_in], outputs=[mu_cage, mu_h3k4me3, mu_h3k27ac, mu_dnase, h, att])
        model_gat._name = 'Seq-GraphReg_e2e'
        model_gat.summary()
        #print(len(model_gat.trainable_variables))
        #keras.utils.plot_model(model, 'GAT.png', show_shapes = True)


    ########## training ##########

    cell_line = options.cell_line
    cell_lines = [cell_line]
    print('cell type:', cell_line)
    model_name_gat = '../models/'+cell_line+'/Seq-GraphReg_e2e_'+cell_line+'_valid_chr_'+options.valid_chr+'_fft.h5'

    if cell_line == 'mESC':
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

    best_loss = 10**20
    max_early_stopping = 10
    n_epochs = 100
    opt = tf.keras.optimizers.Adam(learning_rate=.0002, decay=1e-6)
    batch_size = 1
    t0 = time.time()

    lamda = 1
    tfd = tfp.distributions
    dist = tfd.Normal(loc=0, scale=3)
    kernel = dist.prob([-3, -2, -1, 0, 1, 2, 3])
    kernel = kernel/tf.reduce_max(kernel)
    print('smoothing kernel: ', kernel)
    kernel = tf.reshape(kernel, [-1,1,1])

    w = np.zeros(3000000-1)
    T = 60 * 150
    s = 0.2
    for i in range(3*T):
        if i<=T:
            w[i] = 1.
        else:
            w[i] = 1/(1+(i-T)**s)
    w = tf.constant(w, dtype=tf.float32)
    

    for epoch in range(1,n_epochs+1):
        alpha = np.max([.01, 1/(1+np.exp(-(epoch - 10)))])
        loss_cage_all = np.array([])
        loss_h3k4me3_all = np.array([])
        loss_h3k27ac_all = np.array([])
        loss_dnase_all = np.array([])
        rho_cage_all = np.array([])
        rho_h3k4me3_all = np.array([])
        rho_h3k27ac_all = np.array([])
        rho_dnase_all = np.array([])
        Y_hat_all = np.array([])
        Y_all = np.array([])
        for num, cell_line in enumerate(cell_lines):
            for i in train_chr_list:
                print('train chr :', i)
                file_name_train = '/media/labuser/STORAGE/GraphReg/data/tfrecords/tfr_seq_'+cell_line+'_chr'+str(i)+'.tfr'
                iterator_train = dataset_iterator(file_name_train, batch_size)
                while True:
                    data_exist, seq, Y, Y_h3k4me3, Y_h3k27ac, Y_dnase,  adj, idx, tss_idx = read_tf_record_1shot(iterator_train)
                    if data_exist:
                        if tf.reduce_sum(tf.gather(tss_idx, idx)) > 0:
                            with tf.GradientTape() as t2:
                                with tf.GradientTape(watch_accessed_variables=False) as t1:
                                      t1.watch(seq)
                                      Y_hat_cage, Y_hat_h3k4me3, Y_hat_h3k27ac, Y_hat_dnase, _, _ = model_gat([seq, adj])
                                      Y_hat_cage_idx = tf.gather(Y_hat_cage, idx, axis=1)
                                      Y_idx = tf.gather(Y, idx, axis=1)
                                      L_c = (alpha * poisson_loss(Y_idx, Y_hat_cage_idx)
                                            + (1-alpha) * (poisson_loss(Y_h3k4me3, Y_hat_h3k4me3)
                                            + poisson_loss(Y_h3k27ac, Y_hat_h3k27ac)
                                            + poisson_loss(Y_dnase, Y_hat_dnase)))

                                grad_input = t1.gradient(Y_hat_dnase, seq)
                                g = grad_input * seq
                                g = K.sum(g, axis=-1)
                                g = tf.reshape(g, [-1])
                                g = tf.abs(g)
                                g = tf.reshape(g, [1,-1,1])
                                g_s = tf.nn.conv1d(g, kernel, stride=1, padding='SAME')
                                g_s = tf.reshape(g_s, [-1])
                                g_s = tf.cast(g_s, dtype=tf.complex64)
                                m_dc = tf.abs(tf.signal.fft(g_s))[:len(g_s)//2]
                                m = m_dc[1:]
                                m_hat = m/K.sum(m)
                                L_p = 1 - K.sum(w * m_hat)
                                loss = L_c + lamda * L_p

                            grads = t2.gradient(loss, model_gat.trainable_variables)
                            opt.apply_gradients(zip(grads, model_gat.trainable_variables))

                            e1 = np.random.normal(0,1e-6,size=len(Y_idx.numpy().ravel()))
                            e2 = np.random.normal(0,1e-6,size=len(Y_idx.numpy().ravel()))
                            e3 = np.random.normal(0,1e-6,size=len(Y_h3k4me3.numpy().ravel()))
                            e4 = np.random.normal(0,1e-6,size=len(Y_h3k4me3.numpy().ravel()))

                            loss_cage = poisson_loss(Y, Y_hat_cage)
                            loss_h3k4me3 = poisson_loss(Y_h3k4me3, Y_hat_h3k4me3)
                            loss_h3k27ac = poisson_loss(Y_h3k27ac, Y_hat_h3k27ac)
                            loss_dnase = poisson_loss(Y_dnase, Y_hat_dnase)

                            loss_cage_all = np.append(loss_cage_all, loss_cage.numpy())
                            loss_h3k4me3_all = np.append(loss_h3k4me3_all, loss_h3k4me3.numpy())
                            loss_h3k27ac_all = np.append(loss_h3k27ac_all, loss_h3k27ac.numpy())
                            loss_dnase_all = np.append(loss_dnase_all, loss_dnase.numpy())

                            rho_cage_all = np.append(rho_cage_all, np.corrcoef(np.log2(Y_idx.numpy().ravel()+1)+e1,np.log2(Y_hat_cage_idx.numpy().ravel()+1)+e2)[0,1])
                            rho_h3k4me3_all = np.append(rho_h3k4me3_all, np.corrcoef(np.log2(Y_h3k4me3.numpy().ravel()+1)+e3,np.log2(Y_hat_h3k4me3.numpy().ravel()+1)+e4)[0,1])
                            rho_h3k27ac_all = np.append(rho_h3k27ac_all, np.corrcoef(np.log2(Y_h3k27ac.numpy().ravel()+1)+e3,np.log2(Y_hat_h3k27ac.numpy().ravel()+1)+e4)[0,1])
                            rho_dnase_all = np.append(rho_dnase_all, np.corrcoef(np.log2(Y_dnase.numpy().ravel()+1)+e3,np.log2(Y_hat_dnase.numpy().ravel()+1)+e4)[0,1])

                            #Y_hat_all = np.append(Y_hat_all, Y_hat_cage.numpy().ravel())
                            #Y_all = np.append(Y_all, Y_batch.numpy().ravel())

                    else:
                        #print('no data')
                        break
        if epoch == 1:
            print('len of test/valid batches: ', len(rho_cage_all))

        loss_cage = np.mean(loss_cage_all)
        loss_h3k4me3 = np.mean(loss_h3k4me3_all)
        loss_h3k27ac = np.mean(loss_h3k27ac_all)
        loss_dnase = np.mean(loss_dnase_all)
        loss_train = alpha * loss_cage + (1-alpha) * (loss_h3k4me3 + loss_h3k27ac + loss_dnase)

        rho_cage = np.mean(rho_cage_all)
        rho_h3k4me3 = np.mean(rho_h3k4me3_all)
        rho_h3k27ac = np.mean(rho_h3k27ac_all)
        rho_dnase = np.mean(rho_dnase_all)

        #sp = spearmanr(Y_all, Y_hat_all)[0]
        print('train epoch: ', epoch, 'total loss: ', loss_train, 'loss cage: ', loss_cage, 'loss h3k4me3: ', loss_h3k4me3, 'loss h3k27ac: ', loss_h3k27ac, 'loss dnase: ', loss_dnase, ', rho_cage: ', rho_cage, ', rho_h3k4me3: ', rho_h3k4me3, ', rho_h3k27ac: ', rho_h3k27ac, ', rho_dnase: ', rho_dnase, ', time passed: ', (time.time() - t0), ' sec')

        if epoch%1 == 0:
            valid_loss, valid_loss_cage, valid_loss_h3k4me3, valid_loss_h3k27ac, valid_loss_dnase, valid_rho_cage, valid_rho_h3k4me3, valid_rho_h3k27ac, valid_rho_dnase = calculate_loss(model_gat, valid_chr_list, cell_lines, batch_size, alpha)

        if valid_loss_cage < best_loss:
            early_stopping_counter = 1
            best_loss = valid_loss_cage
            model_gat.save(model_name_gat)
            print('valid epoch: ', epoch, 'total loss: ', valid_loss, 'loss cage: ', valid_loss_cage, 'loss h3k4me3: ', valid_loss_h3k4me3, 'loss h3k27ac: ', valid_loss_h3k27ac, 'loss dnase: ', valid_loss_dnase, ', rho_cage: ', valid_rho_cage, ', rho_h3k4me3: ', valid_rho_h3k4me3, ', rho_h3k27ac: ', valid_rho_h3k27ac, ', rho_dnase: ', valid_rho_dnase, ', time passed: ', (time.time() - t0), ' sec')
            test_loss, test_loss_cage, test_loss_h3k4me3, test_loss_h3k27ac, test_loss_dnase, test_rho_cage, test_rho_h3k4me3, test_rho_h3k27ac, test_rho_dnase = calculate_loss(model_gat, test_chr_list, cell_lines, batch_size, alpha)
            print('test epoch: ', epoch, 'total loss: ', test_loss, 'loss cage: ', test_loss_cage, 'loss h3k4me3: ', test_loss_h3k4me3, 'loss h3k27ac: ', test_loss_h3k27ac, 'loss dnase: ', test_loss_dnase, ', rho_cage: ', test_rho_cage, ', rho_h3k4me3: ', test_rho_h3k4me3, ', rho_h3k27ac: ', test_rho_h3k27ac, ', rho_dnase: ', test_rho_dnase, ', time passed: ', (time.time() - t0), ' sec')

        else:
            early_stopping_counter += 1
            if early_stopping_counter == max_early_stopping:
                break

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
