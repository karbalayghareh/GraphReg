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

import os
import sys
import collections

import h5py
import numpy as np
import qnorm
import pyBigWig
import intervaltree
ModelSeq = collections.namedtuple('ModelSeq', ['chr', 'start', 'end', 'label'])

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options]'
  parser = OptionParser(usage)
  parser.add_option('-b', dest='blacklist_bed',
      help='Set blacklist nucleotides to a baseline value.')
  parser.add_option('-c', dest='clip',
      default=50000, type='float',
      help='Clip values post-summary to a maximum [Default: %default]')
  parser.add_option('--crop', dest='crop_bp',
      default=0, type='int',
      help='Crop bp off each end [Default: %default]')
  parser.add_option('-s', dest='scale',
      default=1., type='float',
      help='Scale values by [Default: %default]')
  parser.add_option('--soft', dest='soft_clip',
      default=False, action='store_true',
      help='Soft clip values, applying sqrt to the execess above the threshold [Default: %default]')
  parser.add_option('-u', dest='sum_stat',
      default='sum',
      help='Summary statistic to compute in windows [Default: %default]')
  parser.add_option('-w',dest='pool_width',
      default=100, type='int',
      help='Average pooling width [Default: %default]')
  (options, args) = parser.parse_args()

  ################################################################
  # Inputs

  organism = 'human'
  res = '5kb'
  cell_line_1 = 'GM12878'
  cell_line_2 = 'K562'
  track = 'CAGE'
  genome = 'hg19'
  data_path = '/media/labuser/STORAGE/GraphReg'

  if organism == 'human':
      chr_list = np.arange(1,1+22)
  else:
      chr_list = np.arange(1,1+19)

  x_all_1 = np.array([])
  x_all_2 = np.array([])
  
  for i in chr_list:
    print(i)
    chr_temp = 'chr'+str(i)
    genome_cov_file_1 = data_path+'/data/'+cell_line_1+'/bigwig/'+track+'_'+cell_line_1+'.bw'
    genome_cov_file_2 = data_path+'/data/'+cell_line_2+'/bigwig/'+track+'_'+cell_line_2+'.bw'
    seqs_bed_file = data_path+'/data/csv/seqs_bed/'+organism+'/'+genome+'/'+res+'/sequences_'+chr_temp+'.bed'
    seqs_cov_file_1 = data_path+'/data/'+cell_line_1+'/seqs_cov_norm/'+track+'_cov_FC_'+chr_temp+'.h5'
    seqs_cov_file_2 = data_path+'/data/'+cell_line_2+'/seqs_cov_norm/'+track+'_cov_FC_'+chr_temp+'.h5'

    assert(options.crop_bp >= 0)

    # read model sequences
    model_seqs = []
    for line in open(seqs_bed_file):
      a = line.split()
      model_seqs.append(ModelSeq(a[0],int(a[1]),int(a[2]),None))

    # read blacklist regions
    black_chr_trees = read_blacklist(options.blacklist_bed)

    # compute dimensions
    num_seqs = len(model_seqs)
    seq_len_nt = model_seqs[0].end - model_seqs[0].start
    seq_len_nt -= 2*options.crop_bp
    target_length = seq_len_nt // options.pool_width
    assert(target_length > 0)

    # initialize sequences coverage file
    seqs_cov_numpy_1 = np.zeros([num_seqs, target_length])
    seqs_cov_numpy_2 = np.zeros([num_seqs, target_length])

    # open genome coverage file
    genome_cov_open_1 = CovFace(genome_cov_file_1)
    genome_cov_open_2 = CovFace(genome_cov_file_2)

    # for each model sequence
    for si in range(num_seqs):
      mseq = model_seqs[si]

      # read coverage
      seq_cov_nt_1 = genome_cov_open_1.read(mseq.chr, mseq.start, mseq.end)

      # determine baseline coverage
      baseline_cov_1 = np.percentile(seq_cov_nt_1, 10)
      baseline_cov_1 = np.nan_to_num(baseline_cov_1)

      # set blacklist to baseline
      if mseq.chr in black_chr_trees:
        for black_interval in black_chr_trees[mseq.chr][mseq.start:mseq.end]:
          # adjust for sequence indexes
          black_seq_start = black_interval.begin - mseq.start
          black_seq_end = black_interval.end - mseq.start
          seq_cov_nt_1[black_seq_start:black_seq_end] = baseline_cov_1

      # set NaN's to baseline
      nan_mask = np.isnan(seq_cov_nt_1)
      seq_cov_nt_1[nan_mask] = baseline_cov_1

      # crop
      if options.crop_bp:
        seq_cov_nt_1 = seq_cov_nt_1[options.crop_bp:-options.crop_bp]

      # sum pool
      seq_cov_1 = seq_cov_nt_1.reshape(target_length, options.pool_width)
      if options.sum_stat == 'sum':
        seq_cov_1 = seq_cov_1.sum(axis=1, dtype='float32')
      elif options.sum_stat in ['mean', 'avg']:
        seq_cov_1 = seq_cov_1.mean(axis=1, dtype='float32')
      elif options.sum_stat == 'median':
        seq_cov_1 = seq_cov_1.median(axis=1, dtype='float32')
      elif options.sum_stat == 'max':
        seq_cov_1 = seq_cov_1.max(axis=1)
      else:
        print('ERROR: Unrecognized summary statistic "%s".' % options.sum_stat,
              file=sys.stderr)
        exit(1)

      # clip
      if options.clip is not None:
        if options.soft_clip:
          clip_mask = (seq_cov_1 > options.clip)
          seq_cov_1[clip_mask] = options.clip + np.sqrt(seq_cov_1[clip_mask] - options.clip)
        else:
          seq_cov_1 = np.clip(seq_cov_1, 0, options.clip)

      # scale
      seq_cov_1 = options.scale * seq_cov_1

      # write
      seqs_cov_numpy_1[si,:] = seq_cov_1
    
    x_all_1 = np.append(x_all_1, seqs_cov_numpy_1.flatten())


    for si in range(num_seqs):
      mseq = model_seqs[si]

      # read coverage
      seq_cov_nt_2 = genome_cov_open_2.read(mseq.chr, mseq.start, mseq.end)

      # determine baseline coverage
      baseline_cov_2 = np.percentile(seq_cov_nt_2, 10)
      baseline_cov_2 = np.nan_to_num(baseline_cov_2)

      # set blacklist to baseline
      if mseq.chr in black_chr_trees:
        for black_interval in black_chr_trees[mseq.chr][mseq.start:mseq.end]:
          # adjust for sequence indexes
          black_seq_start = black_interval.begin - mseq.start
          black_seq_end = black_interval.end - mseq.start
          seq_cov_nt_2[black_seq_start:black_seq_end] = baseline_cov_2

      # set NaN's to baseline
      nan_mask = np.isnan(seq_cov_nt_2)
      seq_cov_nt_2[nan_mask] = baseline_cov_2

      # crop
      if options.crop_bp:
        seq_cov_nt_2 = seq_cov_nt_2[options.crop_bp:-options.crop_bp]

      # sum pool
      seq_cov_2 = seq_cov_nt_2.reshape(target_length, options.pool_width)
      if options.sum_stat == 'sum':
        seq_cov_2 = seq_cov_2.sum(axis=1, dtype='float32')
      elif options.sum_stat in ['mean', 'avg']:
        seq_cov_2 = seq_cov_2.mean(axis=1, dtype='float32')
      elif options.sum_stat == 'median':
        seq_cov_2 = seq_cov_2.median(axis=1, dtype='float32')
      elif options.sum_stat == 'max':
        seq_cov_2 = seq_cov_2.max(axis=1)
      else:
        print('ERROR: Unrecognized summary statistic "%s".' % options.sum_stat,
              file=sys.stderr)
        exit(1)

      # clip
      if options.clip is not None:
        if options.soft_clip:
          clip_mask = (seq_cov_2 > options.clip)
          seq_cov_2[clip_mask] = options.clip + np.sqrt(seq_cov_2[clip_mask] - options.clip)
        else:
          seq_cov_2 = np.clip(seq_cov_2, 0, options.clip)

      # scale
      seq_cov_2 = options.scale * seq_cov_2

      # write
      seqs_cov_numpy_2[si,:] = seq_cov_2
      #x_all_2 = np.append(x_all_2, seq_cov_2)
      #seqs_cov_open_2['seqs_cov'][si,:] = seq_cov_2.astype('float16')
    
    x_all_2 = np.append(x_all_2, seqs_cov_numpy_2.flatten())

    # close genome coverage file
    genome_cov_open_1.close()
    genome_cov_open_2.close()

  # DESEQ normalization
  x_all = np.vstack((x_all_1, x_all_2))
  x_all_log = np.log(x_all)
  x_all_mean = np.mean(x_all_log, axis=0)
  print('x_all_mean shape: ', x_all_mean.shape)
  #idx = np.where(~np.isneginf(x_all_mean))[0]
  idx = np.where(x_all_mean>=3)[0]
  print('x_all_mean_idx: ', len(idx))
  x_all_log_idx = x_all_log[:,idx]
  x_all_mean_idx = x_all_mean[idx]
  x_all_log_idx = x_all_log_idx - x_all_mean_idx
  x_medians_log = np.median(x_all_log_idx, axis=1).reshape([2,1])
  x_medians = np.exp(x_medians_log)
  print('x_medians: ', x_medians)

  for i in chr_list:
    print(i)
    chr_temp = 'chr'+str(i)
    genome_cov_file_1 = data_path+'/data/'+cell_line_1+'/bigwig/'+track+'_'+cell_line_1+'.bw'
    genome_cov_file_2 = data_path+'/data/'+cell_line_2+'/bigwig/'+track+'_'+cell_line_2+'.bw'
    seqs_bed_file = data_path+'/data/seqs_bed/'+organism+'/'+res+'/sequences_'+chr_temp+'.bed'
    seqs_cov_file_1 = data_path+'/data/'+cell_line_1+'/seqs_cov_norm/'+track+'_cov_FC_'+chr_temp+'.h5'
    seqs_cov_file_2 = data_path+'/data/'+cell_line_2+'/seqs_cov_norm/'+track+'_cov_FC_'+chr_temp+'.h5'

    assert(options.crop_bp >= 0)

    # read model sequences
    model_seqs = []
    for line in open(seqs_bed_file):
      a = line.split()
      model_seqs.append(ModelSeq(a[0],int(a[1]),int(a[2]),None))

    # read blacklist regions
    black_chr_trees = read_blacklist(options.blacklist_bed)

    # compute dimensions
    num_seqs = len(model_seqs)
    seq_len_nt = model_seqs[0].end - model_seqs[0].start
    seq_len_nt -= 2*options.crop_bp
    target_length = seq_len_nt // options.pool_width
    assert(target_length > 0)

    # initialize sequences coverage file
    seqs_cov_dir_1 = data_path+'/data/'+cell_line_1+'/seqs_cov_norm'
    if not os.path.isdir(seqs_cov_dir_1):
      os.mkdir(seqs_cov_dir_1)
    seqs_cov_open_1 = h5py.File(seqs_cov_file_1, 'w')
    seqs_cov_open_1.create_dataset('seqs_cov', shape=(num_seqs, target_length), dtype='float16')
    seqs_cov_numpy_1 = np.zeros([num_seqs, target_length])

    seqs_cov_dir_2 = data_path+'/data/'+cell_line_2+'/seqs_cov_norm'
    if not os.path.isdir(seqs_cov_dir_2):
      os.mkdir(seqs_cov_dir_2)
    seqs_cov_open_2 = h5py.File(seqs_cov_file_2, 'w')
    seqs_cov_open_2.create_dataset('seqs_cov', shape=(num_seqs, target_length), dtype='float16')
    seqs_cov_numpy_2 = np.zeros([num_seqs, target_length])

    # open genome coverage file
    genome_cov_open_1 = CovFace(genome_cov_file_1)
    genome_cov_open_2 = CovFace(genome_cov_file_2)

    # for each model sequence
    for si in range(num_seqs):
      mseq = model_seqs[si]

      # read coverage
      seq_cov_nt_1 = genome_cov_open_1.read(mseq.chr, mseq.start, mseq.end)

      # determine baseline coverage
      baseline_cov_1 = np.percentile(seq_cov_nt_1, 10)
      baseline_cov_1 = np.nan_to_num(baseline_cov_1)

      # set blacklist to baseline
      if mseq.chr in black_chr_trees:
        for black_interval in black_chr_trees[mseq.chr][mseq.start:mseq.end]:
          # adjust for sequence indexes
          black_seq_start = black_interval.begin - mseq.start
          black_seq_end = black_interval.end - mseq.start
          seq_cov_nt_1[black_seq_start:black_seq_end] = baseline_cov_1

      # set NaN's to baseline
      nan_mask = np.isnan(seq_cov_nt_1)
      seq_cov_nt_1[nan_mask] = baseline_cov_1

      # crop
      if options.crop_bp:
        seq_cov_nt_1 = seq_cov_nt_1[options.crop_bp:-options.crop_bp]

      # sum pool
      seq_cov_1 = seq_cov_nt_1.reshape(target_length, options.pool_width)
      if options.sum_stat == 'sum':
        seq_cov_1 = seq_cov_1.sum(axis=1, dtype='float32')
      elif options.sum_stat in ['mean', 'avg']:
        seq_cov_1 = seq_cov_1.mean(axis=1, dtype='float32')
      elif options.sum_stat == 'median':
        seq_cov_1 = seq_cov_1.median(axis=1, dtype='float32')
      elif options.sum_stat == 'max':
        seq_cov_1 = seq_cov_1.max(axis=1)
      else:
        print('ERROR: Unrecognized summary statistic "%s".' % options.sum_stat,
              file=sys.stderr)
        exit(1)

      # clip
      if options.clip is not None:
        if options.soft_clip:
          clip_mask = (seq_cov_1 > options.clip)
          seq_cov_1[clip_mask] = options.clip + np.sqrt(seq_cov_1[clip_mask] - options.clip)
        else:
          seq_cov_1 = np.clip(seq_cov_1, 0, options.clip)

      # scale
      seq_cov_1 = options.scale * seq_cov_1

      # write
      seq_cov_1 = seq_cov_1 / x_medians[0]
      seqs_cov_open_1['seqs_cov'][si,:] = seq_cov_1.astype('float16')
    
    for si in range(num_seqs):
      mseq = model_seqs[si]

      # read coverage
      seq_cov_nt_2 = genome_cov_open_2.read(mseq.chr, mseq.start, mseq.end)

      # determine baseline coverage
      baseline_cov_2 = np.percentile(seq_cov_nt_2, 10)
      baseline_cov_2 = np.nan_to_num(baseline_cov_2)

      # set blacklist to baseline
      if mseq.chr in black_chr_trees:
        for black_interval in black_chr_trees[mseq.chr][mseq.start:mseq.end]:
          # adjust for sequence indexes
          black_seq_start = black_interval.begin - mseq.start
          black_seq_end = black_interval.end - mseq.start
          seq_cov_nt_2[black_seq_start:black_seq_end] = baseline_cov_2

      # set NaN's to baseline
      nan_mask = np.isnan(seq_cov_nt_2)
      seq_cov_nt_2[nan_mask] = baseline_cov_2

      # crop
      if options.crop_bp:
        seq_cov_nt_2 = seq_cov_nt_2[options.crop_bp:-options.crop_bp]

      # sum pool
      seq_cov_2 = seq_cov_nt_2.reshape(target_length, options.pool_width)
      if options.sum_stat == 'sum':
        seq_cov_2 = seq_cov_2.sum(axis=1, dtype='float32')
      elif options.sum_stat in ['mean', 'avg']:
        seq_cov_2 = seq_cov_2.mean(axis=1, dtype='float32')
      elif options.sum_stat == 'median':
        seq_cov_2 = seq_cov_2.median(axis=1, dtype='float32')
      elif options.sum_stat == 'max':
        seq_cov_2 = seq_cov_2.max(axis=1)
      else:
        print('ERROR: Unrecognized summary statistic "%s".' % options.sum_stat,
              file=sys.stderr)
        exit(1)

      # clip
      if options.clip is not None:
        if options.soft_clip:
          clip_mask = (seq_cov_2 > options.clip)
          seq_cov_2[clip_mask] = options.clip + np.sqrt(seq_cov_2[clip_mask] - options.clip)
        else:
          seq_cov_2 = np.clip(seq_cov_2, 0, options.clip)

      # scale
      seq_cov_2 = options.scale * seq_cov_2

      # write
      seq_cov_2 = seq_cov_2 / x_medians[1]
      seqs_cov_open_2['seqs_cov'][si,:] = seq_cov_2.astype('float16')

    # close genome coverage file
    genome_cov_open_1.close()
    genome_cov_open_2.close()

    # close sequences coverage file
    seqs_cov_open_1.close()
    seqs_cov_open_2.close()


  # quantile normalization
  '''
  seqs_cov_numpy_1 = seqs_cov_numpy_1.flatten()
  seqs_cov_numpy_2 = seqs_cov_numpy_2.flatten()
  print('unnormalized '+cell_line_1+' '+track+' min, max, mean, median: ', np.min(seqs_cov_numpy_1), np.max(seqs_cov_numpy_1), np.mean(seqs_cov_numpy_1), np.median(seqs_cov_numpy_1))
  print('unnormalized '+cell_line_2+' '+track+' min, max, mean, median: ', np.min(seqs_cov_numpy_2), np.max(seqs_cov_numpy_2), np.mean(seqs_cov_numpy_2), np.median(seqs_cov_numpy_2))
  seqs_cov_numpy_all = np.vstack((seqs_cov_numpy_1, seqs_cov_numpy_2))
  seqs_cov_numpy_all = seqs_cov_numpy_all.transpose()
  print('seqs_cov_numpy_all shape: ', seqs_cov_numpy_all.shape)
  seqs_cov_numpy_all_qnorm = qnorm.quantile_normalize(seqs_cov_numpy_all, axis=1)
  seqs_cov_numpy_1_normalized = seqs_cov_numpy_all_qnorm[:,0]
  seqs_cov_numpy_2_normalized = seqs_cov_numpy_all_qnorm[:,1]
  print('quantile normalized '+cell_line_1+' '+track+' min, max, mean, median: ', np.min(seqs_cov_numpy_1_normalized), 
          np.max(seqs_cov_numpy_1_normalized), np.mean(seqs_cov_numpy_1_normalized), np.median(seqs_cov_numpy_1_normalized))
  print('quantile normalized '+cell_line_2+' '+track+' min, max, mean, median: ', np.min(seqs_cov_numpy_2_normalized), 
          np.max(seqs_cov_numpy_2_normalized), np.mean(seqs_cov_numpy_2_normalized), np.median(seqs_cov_numpy_2_normalized))
  seqs_cov_numpy_1_normalized = np.reshape(seqs_cov_numpy_1_normalized, [num_seqs, target_length])
  seqs_cov_numpy_2_normalized = np.reshape(seqs_cov_numpy_2_normalized, [num_seqs, target_length])
  

   # write
  for si in range(num_seqs):
    seqs_cov_open_1['seqs_cov'][si,:] = seqs_cov_numpy_1_normalized[si,:].astype('float16')
    seqs_cov_open_2['seqs_cov'][si,:] = seqs_cov_numpy_2_normalized[si,:].astype('float16')
  
  # close genome coverage file
  genome_cov_open_1.close()
  genome_cov_open_2.close()

  # close sequences coverage file
  seqs_cov_open_1.close()
  seqs_cov_open_2.close()
  '''


def read_blacklist(blacklist_bed, black_buffer=20):
  """Construct interval trees of blacklist
     regions for each chromosome."""
  black_chr_trees = {}

  if blacklist_bed is not None and os.path.isfile(blacklist_bed):
    for line in open(blacklist_bed):
      a = line.split()
      chrm = a[0]
      start = max(0, int(a[1]) - black_buffer)
      end = int(a[2]) + black_buffer

      if chrm not in black_chr_trees:
        black_chr_trees[chrm] = intervaltree.IntervalTree()

      black_chr_trees[chrm][start:end] = True

  return black_chr_trees


class CovFace:
  def __init__(self, cov_file):
    self.cov_file = cov_file
    self.bigwig = False

    cov_ext = os.path.splitext(self.cov_file)[1].lower()
    if cov_ext in ['.bw','.bigwig']:
      self.cov_open = pyBigWig.open(self.cov_file, 'r')
      self.bigwig = True
    elif cov_ext in ['.h5', '.hdf5', '.w5', '.wdf5']:
      self.cov_open = h5py.File(self.cov_file, 'r')
    else:
      print('Cannot identify coverage file extension "%s".' % cov_ext,
            file=sys.stderr)
      exit(1)

  def read(self, chrm, start, end):
    if self.bigwig:
      cov = self.cov_open.values(chrm, start, end, numpy=True).astype('float16')
    else:
      if chrm in self.cov_open:
        cov = self.cov_open[chrm][start:end]
      else:
        print("WARNING: %s doesn't see %s:%d-%d. Setting to all zeros." % \
          (self.cov_file, chrm, start, end), file=sys.stderr)
        cov = np.zeros(end-start, dtype='float16')
    return cov

  def close(self):
    self.cov_open.close()

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
