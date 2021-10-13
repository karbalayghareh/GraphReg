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
    parser.add_option('-n', dest='n_gat_layers',
        default=3, type='int')
    parser.add_option('-f', dest='fft',
        default=1, type='int')
    parser.add_option('-d', dest='dilated',
        default=1, type='int')

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
    print('number of GAT layers: ', options.n_gat_layers)
    print('CNN base fft: ', options.fft)
    print('CNN base dilated: ', options.dilated)

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
            T = 400       # number of 5kb bins inside middle 2Mb region 
            b = 50        # number of 100bp bins inside 5Kb region
            F = 4         # number of ACGT (4)
            seq = next_datum['seq']
            batch_size = tf.shape(seq)[0]
            seq = tf.reshape(seq, [60, 100000, F])
            adj = next_datum['adj']
            adj = tf.reshape(adj, [batch_size, 3*T, 3*T])

            last_batch = next_datum['last_batch']
            tss_idx = next_datum['tss_idx']
            tss_idx = tf.reshape(tss_idx, [3*T])
            idx = tf.range(T, 2*T)

            Y = next_datum['Y']
            Y = tf.reshape(Y, [3*T, b])
            Y = tf.reduce_sum(Y, axis=-1)
            Y = tf.reshape(Y, [1, 3*T])

        else:
            seq = 0
            Y = 0
            adj = 0
            tss_idx = 0
            idx = 0
        return data_exist, seq, Y, adj, idx, tss_idx

    def calculate_loss(model_gat, model_cnn_base, chr_list, cell_lines, batch_size):
        loss_gat_all = np.array([])
        rho_gat_all = np.array([])
        Y_hat_all = np.array([])
        Y_all = np.array([])
        for num, cell_line in enumerate(cell_lines):
            for i in chr_list:
                print(' chr :', i)
                file_name = data_path+'/data/tfrecords/tfr_seq_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'_chr'+str(i)+'.tfr'
                iterator = dataset_iterator(file_name, batch_size)
                while True:
                    data_exist, seq, Y, adj, idx, tss_idx = read_tf_record_1shot(iterator)
                    H = []
                    if data_exist:
                        if tf.reduce_sum(tf.gather(tss_idx, tf.range(400, 800))) > 0:
                            for jj in range(0,60,10):
                                seq_batch = seq[jj:jj+10]
                                _,_,_,h = model_cnn_base(seq_batch)
                                H.append(h)
                            x_in_gat = K.concatenate(H, axis = 0)
                            x_in_gat = K.reshape(x_in_gat, [1, 60000, 64])

                            Y_hat, _ = model_gat([x_in_gat, adj])
                            Y_hat_idx = tf.gather(Y_hat, idx, axis=1)
                            Y_idx = tf.gather(Y, idx, axis=1)

                            loss = poisson_loss(Y_idx, Y_hat_idx)
                            loss_gat_all = np.append(loss_gat_all, loss.numpy())
                            e1 = np.random.normal(0,1e-6,size=len(Y_idx.numpy().ravel()))
                            e2 = np.random.normal(0,1e-6,size=len(Y_idx.numpy().ravel()))
                            rho_gat_all = np.append(rho_gat_all, np.corrcoef(np.log2(Y_idx.numpy().ravel()+1)+e1,np.log2(Y_hat_idx.numpy().ravel()+1)+e2)[0,1])

                            Y_hat_all = np.append(Y_hat_all, Y_hat_idx.numpy().ravel())
                            Y_all = np.append(Y_all, Y_idx.numpy().ravel())
                    else:
                        break

        print('len of test/valid Y: ', len(Y_all))
        valid_loss = np.mean(loss_gat_all)
        rho = np.mean(rho_gat_all)

        return valid_loss, rho

    # Parameters
    T = 400
    b = 50
    N = 3*T                       # number of 5Kb bins inside 6Mb region
    F = 4                         # feature dimension
    F_ = 32                       # output size of GraphAttention layer
    n_attn_heads = 4              # number of attention heads in GAT layers
    dropout_rate = 0.5            # dropout rate
    l2_reg = 0.0                  # factor for l2 regularization
    re_load = False
    
    # Model definition 
    if re_load:
        model_name = 'model_name.h5'
        model = tf.keras.models.load_model(model_name, custom_objects={'GraphAttention': GraphAttention})
        model.summary()
    else:
        tf.keras.backend.clear_session()
        X_in = Input(shape=(60000,64))
        A_in = Input(shape=(N,N))

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

        att=[]
        for i in range(options.n_gat_layers):
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
        model_gat = Model(inputs=[X_in, A_in], outputs=[mu_cage, att])
        model_gat._name = 'Seq-GraphReg'
        model_gat.summary()
        #print(len(model_gat.trainable_variables))
        #keras.utils.plot_model(model, 'GAT.png', show_shapes = True)


    ########## training ##########

    cell_line = options.cell_line
    cell_lines = [cell_line]

    if (not options.fft) and options.dilated:
        model_name_cnn_base = data_path+'/models/'+cell_line+'/Seq-CNN_base_'+cell_line+'_valid_chr_'+options.valid_chr+'_test_chr_'+options.test_chr+'.h5'
        model_name_gat = data_path+'/models/'+cell_line+'/Seq-GraphReg_'+cell_line+'_'+options.assay_type+'_FDR_'+fdr+'_valid_chr_'+options.valid_chr+'_test_chr_'+options.test_chr+'.h5'
    elif (not options.fft) and (not options.dilated):
        model_name_cnn_base = data_path+'/models/'+cell_line+'/Seq-CNN_base_nodilation_'+cell_line+'_valid_chr_'+options.valid_chr+'_test_chr_'+options.test_chr+'.h5'
        model_name_gat = data_path+'/models/'+cell_line+'/Seq-GraphReg_nodilation_'+cell_line+'_'+options.assay_type+'_FDR_'+fdr+'_valid_chr_'+options.valid_chr+'_test_chr_'+options.test_chr+'.h5'
    elif options.fft and options.dilated:
        model_name_cnn_base = data_path+'/models/'+cell_line+'/Seq-CNN_base_fft_'+cell_line+'_valid_chr_'+options.valid_chr+'_test_chr_'+options.test_chr+'.h5'
        model_name_gat = data_path+'/models/'+cell_line+'/Seq-GraphReg_fft_'+cell_line+'_'+options.assay_type+'_FDR_'+fdr+'_valid_chr_'+options.valid_chr+'_test_chr_'+options.test_chr+'.h5'
    elif options.fft and (not options.dilated):
        model_name_cnn_base = data_path+'/models/'+cell_line+'/Seq-CNN_base_nodilation_fft_'+cell_line+'_valid_chr_'+options.valid_chr+'_test_chr_'+options.test_chr+'.h5'
        model_name_gat = data_path+'/models/'+cell_line+'/Seq-GraphReg_nodilation_fft_'+cell_line+'_'+options.assay_type+'_FDR_'+fdr+'_valid_chr_'+options.valid_chr+'_test_chr_'+options.test_chr+'.h5'

    model_cnn_base = tf.keras.models.load_model(model_name_cnn_base)
    model_cnn_base._name = 'Seq-CNN_base'
    model_cnn_base.trainable = False
    model_cnn_base.summary()

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
    batch_size = 1
    t0 = time.time()
    for epoch in range(1,n_epochs+1):
        loss_gat_all = np.array([])
        rho_gat_all = np.array([])
        Y_hat_all = np.array([])
        Y_all = np.array([])
        for num, cell_line in enumerate(cell_lines):
            for i in train_chr_list:
                print('train chr :', i)
                file_name_train =  data_path+'/data/tfrecords/tfr_seq_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'_chr'+str(i)+'.tfr'
                iterator_train = dataset_iterator(file_name_train, batch_size)
                while True:
                    data_exist, seq, Y, adj, idx, tss_idx = read_tf_record_1shot(iterator_train)
                    H = []
                    if data_exist:
                        if tf.reduce_sum(tf.gather(tss_idx, tf.range(400, 800))) > 0:
                            for jj in range(0,60,10):
                                seq_batch = seq[jj:jj+10]
                                _,_,_,h = model_cnn_base(seq_batch)
                                H.append(h)
                            x_in_gat = K.concatenate(H, axis = 0)
                            x_in_gat = K.reshape(x_in_gat, [1, 60000, 64])

                            with tf.GradientTape() as tape:
                                Y_hat, _ = model_gat([x_in_gat, adj])
                                Y_hat_idx = tf.gather(Y_hat, idx, axis=1)
                                Y_idx = tf.gather(Y, idx, axis=1)
                                loss = poisson_loss(Y_idx, Y_hat_idx)

                            grads = tape.gradient(loss, model_gat.trainable_variables)
                            opt.apply_gradients(zip(grads, model_gat.trainable_variables))

                            loss_gat_all = np.append(loss_gat_all, loss.numpy())
                            e1 = np.random.normal(0,1e-6,size=len(Y_idx.numpy().ravel()))
                            e2 = np.random.normal(0,1e-6,size=len(Y_idx.numpy().ravel()))
                            rho_gat_all = np.append(rho_gat_all, np.corrcoef(np.log2(Y_idx.numpy().ravel()+1)+e1,np.log2(Y_hat_idx.numpy().ravel()+1)+e2)[0,1])

                            Y_hat_all = np.append(Y_hat_all, Y_hat_idx.numpy().ravel())
                            Y_all = np.append(Y_all, Y_idx.numpy().ravel())
                    else:
                        break

        if epoch == 1:
            print('len of train Y: ', len(Y_all))
        train_loss = np.mean(loss_gat_all)
        rho = np.mean(rho_gat_all)

        print('epoch: ', epoch, 'train loss: ', train_loss, ', train rho: ', rho, ', time passed: ', (time.time() - t0), ' sec')

        if epoch%1 == 0:
            valid_loss, valid_rho = calculate_loss(model_gat, model_cnn_base, valid_chr_list, cell_lines, batch_size)

        if valid_loss < best_loss:
            early_stopping_counter = 1
            best_loss = valid_loss
            model_gat.save(model_name_gat)
            print('epoch: ', epoch, 'valid loss: ', valid_loss, ', valid rho: ', valid_rho, ', time passed: ', (time.time() - t0), ' sec')
            test_loss, test_rho = calculate_loss(model_gat, model_cnn_base, test_chr_list, cell_lines, batch_size)
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
