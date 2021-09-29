from __future__ import division
from optparse import OptionParser

#from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
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


def main():
    usage = 'usage: %prog [options]'
    parser = OptionParser(usage)
    parser.add_option('-c', dest='cell_line',
        default='K562', type='str')
    parser.add_option('-o', dest='organism',
        default='human', type='str')

    (options, args) = parser.parse_args()

    def negative_binomial_loss(y_true, mu_pred, r_pred):
        nll = tf.reduce_mean(
            tf.math.lgamma(r_pred)
            + tf.math.lgamma(y_true + 1)
            - tf.math.lgamma(r_pred + y_true)
            - r_pred * tf.math.log(r_pred/(mu_pred+r_pred))
            - y_true * tf.math.log(mu_pred/(mu_pred+r_pred)))
        return nll

    def poisson_loss(y_true, mu_pred):
        nll = tf.reduce_mean(
            tf.math.lgamma(y_true + 1) + mu_pred - y_true * tf.math.log(mu_pred))
        return nll

    def parse_proto(example_protos):
        features = {
            'last_batch': tf.io.FixedLenFeature([1], tf.int64),
            'adj': tf.io.FixedLenFeature([], tf.string),
            'adj_real': tf.io.FixedLenFeature([], tf.string),
            'tss_idx': tf.io.FixedLenFeature([], tf.string),
            'X_1d': tf.io.FixedLenFeature([], tf.string),
            'Y': tf.io.FixedLenFeature([], tf.string),
            'sequence': tf.io.FixedLenFeature([], tf.string)
        }
        parsed_features = tf.io.parse_example(example_protos, features=features)

        last_batch = parsed_features['last_batch']

        adj = tf.io.decode_raw(parsed_features['adj'], tf.float16)
        adj = tf.cast(adj, tf.float32)

        adj_real = tf.io.decode_raw(parsed_features['adj_real'], tf.float16)
        adj_real = tf.cast(adj_real, tf.float32)

        tss_idx = tf.io.decode_raw(parsed_features['tss_idx'], tf.float16)
        tss_idx = tf.cast(tss_idx, tf.float32)

        X_epi = tf.io.decode_raw(parsed_features['X_1d'], tf.float16)
        X_epi = tf.cast(X_epi, tf.float32)

        Y = tf.io.decode_raw(parsed_features['Y'], tf.float16)
        Y = tf.cast(Y, tf.float32)

        seq = tf.io.decode_raw(parsed_features['sequence'], tf.float64)
        seq = tf.cast(seq, tf.float32)

        return {'seq': seq, 'last_batch': last_batch, 'X_epi': X_epi, 'Y': Y, 'adj': adj, 'adj_real': adj_real, 'tss_idx': tss_idx}

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
            F = 4
            seq = next_datum['seq']
            batch_size = tf.shape(seq)[0]
            seq = tf.reshape(seq, [60, 100000, F])
            adj = next_datum['adj']
            adj = tf.reshape(adj, [3*T, 3*T])
            adj = tf.reshape(adj, [batch_size, 3*T, 3*T])
            #adj_real = next_datum['adj_real']
            #adj_real = tf.reshape(adj_real, [batch_size, 3*T, 3*T])
            #row_sum = tf.squeeze(tf.reduce_sum(adj, axis=-1))
            #print(row_max)

            last_batch = next_datum['last_batch']
            tss_idx = next_datum['tss_idx']
            tss_idx = tf.reshape(tss_idx, [3*T])

            #idx1 = tf.where(tf.math.logical_and(tf.math.logical_and(tss_idx > 0, row_sum > 1), Y1 >= 0))[:,0]
            #idx1 = tf.where(tss_idx > 0)[:,0]
            #print('idx1', idx1)
            if last_batch==0:
                #idx2 = tf.where(tf.math.logical_and(idx1>=T, idx1<2*T))[:,0]
                idx = tf.range(T, 2*T)
            else:
                #idx2 = tf.where(idx1>=T)[:,0]
                idx = tf.range(T, 2*T)

            #print('idx2', idx2)
            #idx = tf.gather(idx1, idx2)
            #print('idx', idx)

            Y = next_datum['Y']
            Y = tf.reshape(Y, [3*T, 50])
            Y = tf.reduce_sum(Y, axis=-1)
            Y = tf.reshape(Y, [1, 3*T])
            #Y = tf.gather(Y, idx, axis=1)

        else:
            seq = 0
            Y = 0
            adj = 0
            #adj_real = 0
            tss_idx = 0
            idx = 0
        return data_exist, seq, Y, adj, idx, tss_idx


    def calculate_loss(model_cnn, model_cnn_base, chr_list, cell_lines, batch_size):
        loss_cnn_all = np.array([])
        rho_cnn_all = np.array([])
        Y_hat_all = np.array([])
        Y_all = np.array([])
        for num, cell_line in enumerate(cell_lines):
            for i in chr_list:
                print(' chr :', i)
                file_name = '/home/labuser/Codes/basenji/data/tfrecords/tfr_seq_'+cell_line+'_chr'+str(i)+'.tfr'
                iterator = dataset_iterator(file_name, batch_size)
                while True:
                    data_exist, seq, Y, adj, idx, tss_idx = read_tf_record_1shot(iterator)
                    H = []
                    if data_exist:
                        if tf.reduce_sum(tf.gather(tss_idx, tf.range(400,800))) > 0:
                            for jj in range(0,60,10):
                                seq_batch = seq[jj:jj+10]
                                _,_,_, h = model_cnn_base(seq_batch)
                                H.append(h)
                            x_in_cnn = K.concatenate(H, axis = 0)
                            x_in_cnn = K.reshape(x_in_cnn, [1, 60000, 64])

                            Y_hat = model_cnn(x_in_cnn)
                            Y_hat_idx = tf.gather(Y_hat, idx, axis=1)
                            Y_idx = tf.gather(Y, idx, axis=1)

                            loss = poisson_loss(Y_idx, Y_hat_idx)
                            loss_cnn_all = np.append(loss_cnn_all, loss.numpy())
                            e1 = np.random.normal(0,1e-6,size=len(Y_idx.numpy().ravel()))
                            e2 = np.random.normal(0,1e-6,size=len(Y_idx.numpy().ravel()))
                            rho_cnn_all = np.append(rho_cnn_all, np.corrcoef(np.log2(Y_idx.numpy().ravel()+1)+e1, np.log2(Y_hat_idx.numpy().ravel()+1)+e2)[0,1])

                            Y_hat_all = np.append(Y_hat_all, Y_hat_idx.numpy().ravel())
                            Y_all = np.append(Y_all, Y_idx.numpy().ravel())
                    else:
                        #print('no data')
                        break

        print('len of test/valid Y: ', len(Y_all))
        valid_loss = np.mean(loss_cnn_all)
        rho = np.mean(rho_cnn_all)

        #sp = spearmanr(Y_all, Y_hat_all)[0]
        return valid_loss, rho

    # Parameters
    T = 400
    dropout_rate = 0.5              # Dropout rate 
    l2_reg = 0.0                    # Factor for l2 regularization
    re_load = False

    # Model definition

    if re_load:
        model_name = 'aaa.h5'
        model = tf.keras.models.load_model(model_name, custom_objects={'GraphAttention': GraphAttention})
        model.summary()
    else:
        tf.keras.backend.clear_session()
        X_in = Input(shape=(60000,64))

        x = layers.Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(X_in)
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
        x = layers.BatchNormalization()(x)

        for i in range(1,1+8):
            x = Dropout(dropout_rate)(x)
            x = layers.Conv1D(64, 3, activation='relu', dilation_rate=2**i, padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(x) + x
            x = layers.BatchNormalization()(x)

        #x = Dropout(dropout_rate)(x)
        mu_cage = layers.Conv1D(1, 1, activation='exponential', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(x)
        mu_cage = layers.Reshape([3*T])(mu_cage)

        # Build model
        model_cnn = Model(inputs=X_in, outputs=mu_cage)
        model_cnn._name = 'Seq-CNN'
        model_cnn.summary()
        #print(len(model_cnn.trainable_variables))
        #keras.utils.plot_model(model, 'GAT.png', show_shapes = True)


    ########## training ##########

    cell_line = options.cell_line
    cell_lines = [cell_line]
    print('cell type:', cell_line)

    model_name_cnn_base = '../models/'+cell_line+'/Seq-CNN_base_'+cell_line+'.h5'
    model_cnn_base = tf.keras.models.load_model(model_name_cnn_base)
    model_cnn_base._name = 'Seq-CNN_base'
    model_cnn_base.trainable = False
    model_cnn_base.summary()
    

    model_name_cnn = '../models/'+cell_line+'/Seq-CNN_'+cell_line+'.h5'

    if cell_line == 'mESC':
        train_chr_list = [1,2,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19]
        test_chr_list = [12]
        valid_chr_list =[3]
    else:
        train_chr_list = [1,2,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22]
        test_chr_list = [12]
        valid_chr_list =[3]

    best_loss = 10**20
    max_early_stopping = 10
    n_epochs = 100
    opt = tf.keras.optimizers.Adam(learning_rate=.0002, decay=1e-6)
    batch_size = 1
    t0 = time.time()
    for epoch in range(1,n_epochs+1):
        loss_cnn_all = np.array([])
        rho_cnn_all = np.array([])
        Y_hat_all = np.array([])
        Y_all = np.array([])
        for num, cell_line in enumerate(cell_lines):
            for i in train_chr_list:
                print('train chr :', i)
                file_name_train = '/home/labuser/Codes/basenji/data/tfrecords/tfr_seq_'+cell_line+'_chr'+str(i)+'.tfr'
                iterator_train = dataset_iterator(file_name_train, batch_size)
                while True:
                    data_exist, seq, Y, adj, idx, tss_idx = read_tf_record_1shot(iterator_train)
                    H = []
                    if data_exist:
                        if tf.reduce_sum(tf.gather(tss_idx, tf.range(400,800))) > 0:
                            for jj in range(0,60,10):
                                seq_batch = seq[jj:jj+10]
                                _,_,_, h = model_cnn_base(seq_batch)
                                H.append(h)
                            x_in_cnn = K.concatenate(H, axis = 0)
                            x_in_cnn = K.reshape(x_in_cnn, [1, 60000, 64])

                            with tf.GradientTape() as tape:
                                Y_hat = model_cnn(x_in_cnn)
                                Y_hat_idx = tf.gather(Y_hat, idx, axis=1)
                                Y_idx = tf.gather(Y, idx, axis=1)
                                loss = poisson_loss(Y_idx, Y_hat_idx)

                            grads = tape.gradient(loss, model_cnn.trainable_variables)
                            opt.apply_gradients(zip(grads, model_cnn.trainable_variables))

                            loss_cnn_all = np.append(loss_cnn_all, loss.numpy())
                            e1 = np.random.normal(0,1e-6,size=len(Y_idx.numpy().ravel()))
                            e2 = np.random.normal(0,1e-6,size=len(Y_idx.numpy().ravel()))
                            rho_cnn_all = np.append(rho_cnn_all, np.corrcoef(np.log2(Y_idx.numpy().ravel()+1)+e1,np.log2(Y_hat_idx.numpy().ravel()+1)+e2)[0,1])

                            Y_hat_all = np.append(Y_hat_all, Y_hat_idx.numpy().ravel())
                            Y_all = np.append(Y_all, Y_idx.numpy().ravel())
                    else:
                        #print('no data')
                        break
        if epoch == 1:
            print('len of train Y: ', len(Y_all))
        train_loss = np.mean(loss_cnn_all)
        rho = np.mean(rho_cnn_all)

        #sp = spearmanr(Y_all, Y_hat_all)[0]
        print('epoch: ', epoch, 'train loss: ', train_loss, ', train rho: ', rho, ', time passed: ', (time.time() - t0), ' sec')

        if epoch%1 == 0:
            valid_loss,  valid_rho = calculate_loss(model_cnn, model_cnn_base, valid_chr_list, cell_lines, batch_size)

        if valid_loss < best_loss:
            early_stopping_counter = 1
            best_loss = valid_loss
            model_cnn.save(model_name_cnn)
            print('epoch: ', epoch, 'valid loss: ', valid_loss, ', valid rho: ', valid_rho, ', time passed: ', (time.time() - t0), ' sec')
            test_loss, test_rho = calculate_loss(model_cnn, model_cnn_base, test_chr_list, cell_lines, batch_size)
            print('epoch: ', epoch, 'test loss: ', test_loss, ', test rho: ', test_rho, ', time passed: ', (time.time() - t0), ' sec')
            
        else:
            early_stopping_counter += 1
            if early_stopping_counter == max_early_stopping:
                break

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
