#!/usr/bin/env python
# Copyright 2017 Calico LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

# This code is copied from https://github.com/calico/basenji and modified.

# =========================================================================

from optparse import OptionParser
import collections
import os
import sys

import h5py
import numpy as np
import pdb
import pysam
import pandas as pd
import scipy.sparse


#from basenji_data import ModelSeq
ModelSeq = collections.namedtuple('ModelSeq', ['chr', 'start', 'end', 'label'])
from dna_io import dna_1hot


import tensorflow as tf

"""
Write TF Records for batches of model sequences.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options]'
  parser = OptionParser(usage)
  parser.add_option('-g', dest='genome_index',
      default=None, type='int', help='Genome index')
  parser.add_option('-s', dest='start_i',
      default=0, type='int',
      help='Sequence start index [Default: %default]')
  parser.add_option('-e', dest='end_i',
      default=None, type='int',
      help='Sequence end index [Default: %default]')
  parser.add_option('--te', dest='target_extend',
      default=None, type='int', help='Extend targets vector [Default: %default]')
  parser.add_option('--ts', dest='target_start',
      default=0, type='int', help='Write targets into vector starting at index [Default: %default')
  parser.add_option('-u', dest='umap_npy',
      help='Unmappable array numpy file')
  parser.add_option('--umap_set', dest='umap_set',
      default=None, type='float',
      help='Sequence distribution value to set unmappable positions to, eg 0.25.')
  (options, args) = parser.parse_args()

  ################################################################
  # Inputs

  organism = 'human'          # human/mouse
  res = '5kb'                 # 5kb/10kb
  cell_line = 'hESC'          # K562/GM12878/hESC/mESC
  genome='hg38'               # hg19/hg38/mm10
  model = 'seq'               # seq/epi
  assay_type = 'MicroC'        # HiC/HiChIP/MicroC/HiCAR
  qval = 0.1                    # 0.1/0.01/0.001
  data_path = '/media/labuser/STORAGE/GraphReg'

  if qval == 0.1:
    fdr = '1'
  elif qval == 0.01:
    fdr = '01'
  elif qval == 0.001:
    fdr = '001'

  if organism == 'human' and genome == 'hg19':
      chr_list = np.arange(1,1+22)
      fasta_file = data_path+'/data/genome/hg19.ml.fa'
  elif organism == 'human' and genome == 'hg38':
      chr_list = np.arange(1,1+22)
      fasta_file = data_path+'/data/genome/GRCh38.primary_assembly.genome.fa'
  elif organism == 'mouse':
      chr_list = np.arange(1,1+19)
      fasta_file = data_path+'/data/genome/mm10.fa'

  np.random.seed(0)

  q = np.zeros(3)
  for i in chr_list:
    print('chr ', i)
    chr_temp = 'chr'+str(i)
    seqs_bed_file = data_path+'/data/csv/seqs_bed/'+organism+'/'+genome+'/'+res+'/sequences_'+chr_temp+'.bed'
    tfr_file = data_path+'/data/tfrecords/tfr_'+model+'_'+cell_line+'_'+assay_type+'_FDR_'+fdr+'_'+chr_temp+'.tfr'

    ################################################################
    # read model sequences

    model_seqs = []
    for line in open(seqs_bed_file):
      a = line.split()
      model_seqs.append(ModelSeq(a[0],int(a[1]),int(a[2]),None))

    #if options.end_i is None:
    options.end_i = len(model_seqs)

    num_seqs = options.end_i - options.start_i
    print(num_seqs)
    ################################################################
    # determine sequence coverage files

    seqs_cov_files = [data_path+'/data/'+cell_line+'/seqs_cov/CAGE_cov_'+chr_temp+'.h5',
                      data_path+'/data/'+cell_line+'/seqs_cov/H3K4me3_cov_'+chr_temp+'.h5',
                      data_path+'/data/'+cell_line+'/seqs_cov/H3K27ac_cov_danwei_'+chr_temp+'.h5',
                      #data_path+'/data/'+cell_line+'/seqs_cov/H3K4me1_cov_FC_'+chr_temp+'.h5',
                      #data_path+'/data/'+cell_line+'/seqs_cov/H3K27me3_cov_FC_'+chr_temp+'.h5',
                      #data_path+'/data/'+cell_line+'/seqs_cov/CTCF_cov_FC_'+chr_temp+'.h5',
                      data_path+'/data/'+cell_line+'/seqs_cov/DNase_cov_'+chr_temp+'.h5'
                      ]
    seq_pool_len = h5py.File(seqs_cov_files[1], 'r')['seqs_cov'].shape[1]
    num_targets = len(seqs_cov_files)

    ################################################################
    # extend targets
    num_targets_tfr = num_targets
    if options.target_extend is not None:
      assert(options.target_extend >= num_targets_tfr)
      num_targets_tfr = options.target_extend

    # initialize targets
    targets = np.zeros((num_seqs, seq_pool_len, num_targets_tfr), dtype='float32')

    # read each target
    for ti in range(num_targets):
      seqs_cov_open = h5py.File(seqs_cov_files[ti], 'r')
      tii = options.target_start + ti
      tmp = seqs_cov_open['seqs_cov'][options.start_i:options.end_i,:]
      if ti > 0:
        if model == 'epi':
          tmp = np.log2(tmp+1)                  # log normalize
          if i == 1:
            q[ti-1] = np.max(tmp.ravel())
          x_max = q[ti-1]
          x_min = np.min(tmp.ravel())
          tmp = (tmp - x_min)/(x_max - x_min)   # in range [0, 1]

        print(ti, np.sort(tmp.ravel())[-200:])
        targets[:,:,ti] = tmp

      elif ti == 0:
        targets[:,:,ti] = tmp
        print(ti, np.sort(tmp.ravel())[-200:])

      seqs_cov_open.close()
      print('target shape: ', targets.shape)


    ################################################################
    # modify unmappable

    if options.umap_npy is not None and options.umap_set is not None:
      unmap_mask = np.load(options.umap_npy)

      for si in range(num_seqs):
        msi = options.start_i + si

        # determine unmappable null value
        seq_target_null = np.percentile(targets[si], q=[100*options.umap_set], axis=0)[0]

        # set unmappable positions to null
        targets[si,unmap_mask[msi,:],:] = np.minimum(targets[si,unmap_mask[msi,:],:], seq_target_null)

    ################################################################
    # write TFRecords

    # Graph from HiC
    hic_matrix_file = data_path+'/data/'+cell_line+'/hic/'+assay_type+'/'+assay_type+'_matrix_FDR_'+fdr+'_'+chr_temp+'.npz'
    sparse_matrix = scipy.sparse.load_npz(hic_matrix_file)
    hic_matrix = sparse_matrix.todense()
    print('hic_matrix shape: ', hic_matrix.shape)

    tss_bin_file = data_path+'/data/tss/'+organism+'/'+genome+'/tss_bins_'+chr_temp+'.npy'
    tss_bin = np.load(tss_bin_file, allow_pickle=True)
    print('num tss:', np.sum(tss_bin))

    bin_start_file = data_path+'/data/tss/'+organism+'/'+genome+'/bin_start_'+chr_temp+'.npy'
    bin_start = np.load(bin_start_file, allow_pickle=True)
    print('bin start:', bin_start)

    # open FASTA
    if model == 'seq':
      fasta_open = pysam.Fastafile(fasta_file)

    T = 400
    TT = T+T//2
    n_batch = 0

    # define options
    tf_opts = tf.io.TFRecordOptions(compression_type = 'ZLIB')
    with tf.io.TFRecordWriter(tfr_file, tf_opts) as writer:
      for si in range(TT,num_seqs-TT,T):
        n_batch = n_batch + 1
        hic_slice = hic_matrix[si-TT:si+TT,si-TT:si+TT]
        adj_real = np.copy(hic_slice)
        adj_real[adj_real>=1000] = 1000
        adj_real = np.log2(adj_real+1)
        adj_real = adj_real * (np.ones([3*T,3*T]) - np.eye(3*T))
        print('real_adj: ',adj_real)

        adj = np.copy(adj_real)
        adj[adj>0] = 1

        if np.abs(num_seqs-TT - si < T):
          last_batch = 1
        else:
          last_batch = 0

        tss_idx = tss_bin[si-TT:si+TT]
        bin_idx = bin_start[si-TT:si+TT]
        
        Y = targets[si-TT:si+TT,:,0]
        X_1d = targets[si-TT:si+TT,:,1:]

        X_1d = X_1d.astype(np.float16)
        adj = adj.astype(np.float16)
        adj_real = adj_real.astype(np.float16)
        Y = Y.astype(np.float16)
        bin_idx = bin_idx.astype(np.int64)
        tss_idx = tss_idx.astype(np.float16)

        # read FASTA
        if model == 'seq':
          seq_1hot = np.zeros([1,4])
          for msi in range(si-TT,si+TT):
            mseq = model_seqs[msi]
            seq_dna = fasta_open.fetch(mseq.chr, mseq.start, mseq.end)
            # one hot code
            seq_1hot = np.append(seq_1hot, dna_1hot(seq_dna), axis=0)

          seq_1hot = np.delete(seq_1hot, 0, axis=0)
          print('seq: ', np.shape(seq_1hot), seq_1hot)
        
        if model == 'seq':
            example = tf.train.Example(features=tf.train.Features(feature={
                'last_batch': _int_feature(last_batch),
                'sequence': _bytes_feature(seq_1hot.flatten().tostring()),
                'adj': _bytes_feature(adj.flatten().tostring()),
                #'adj_real': _bytes_feature(adj_real.flatten().tostring()),
                'X_1d': _bytes_feature(X_1d.flatten().tostring()),
                'tss_idx': _bytes_feature(tss_idx.flatten().tostring()),
                'bin_idx': _bytes_feature(bin_idx.flatten().tostring()),
                'Y': _bytes_feature(Y.flatten().tostring())}))
        elif model == 'epi':
            example = tf.train.Example(features=tf.train.Features(feature={
                'last_batch': _int_feature(last_batch),
                'adj': _bytes_feature(adj.flatten().tostring()),
                #'adj_real': _bytes_feature(adj_real.flatten().tostring()),
                'X_1d': _bytes_feature(X_1d.flatten().tostring()),
                'tss_idx': _bytes_feature(tss_idx.flatten().tostring()),
                'bin_idx': _bytes_feature(bin_idx.flatten().tostring()),
                'Y': _bytes_feature(Y.flatten().tostring())}))

        writer.write(example.SerializeToString())

      if model == 'seq':
        fasta_open.close()

      print('check symetric: ', check_symmetric(adj))
      print('number of batches: ', n_batch)
      print('q: ', q)
      
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def check_symmetric(a, tol=1e-4):
  return np.all(np.abs(a-a.transpose()) < tol)

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
