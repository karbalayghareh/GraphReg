import numpy as np
import pandas as pd
import time
import os
import scipy.sparse


##### write seqs #####
'''
T = 400
TT = T + T//2
organism = 'human'
genome='hg38'
for i in range(1,22+1):
    chr = 'chr'+str(i)
    filename_seqs = data_path+'/data/csv/seqs_bed/'+organism+'/'+genome+'/5kb/sequences_'+chr+'.bed'
    seq_dataframe = pd.DataFrame(data = [], columns = ["chr", "start", "end"])
    chrom = i

    if organism=='human' and genome=='hg19':
       chr_len = [249235000, 243185000, 197960000, 191040000, 180905000, 171050000, 159125000, 146300000, 141150000, 135520000, 134945000,
                       133840000, 115105000, 107285000, 102520000, 90290000, 81190000, 78015000, 59115000, 62965000, 48115000, 51240000]
    elif organism=='human' and genome=='hg38':
       chr_len = [248950000, 242185000, 198290000, 190205000, 181530000, 170800000, 159340000, 145130000, 138385000, 133790000, 135080000,
                       133270000, 114355000, 107035000, 101985000, 90330000, 83250000, 80365000, 58610000, 64435000, 46700000, 50810000]
    elif organism=='mouse':
       chr_len = [195465000, 182105000, 160030000, 156500000, 151825000, 149730000, 145435000, 129395000, 124590000, 130685000, 122075000, 
               120120000, 120415000, 124895000, 104035000, 98200000, 94980000, 90695000, 61425000]

    nodes_list = []
    for i in range(0, chr_len[chrom-1]+5000, 5000):
        nodes_list.append(i)
    nodes_list = np.array(nodes_list)
    left_padding = np.zeros(TT).astype(int)
    right_padding = np.zeros(TT).astype(int)
    nodes_list = np.append(left_padding, nodes_list)
    nodes_list = np.append(nodes_list, right_padding)
    seq_dataframe['start'] = nodes_list
    seq_dataframe['end'] = nodes_list + 5000
    seq_dataframe['chr'] = chr
    print(seq_dataframe)
    seq_dataframe.to_csv(filename_seqs, index = False, header = False, sep = '\t')
'''

##### extract hic from the output files of HiCDC+ #####

cell_line = 'hESC'           # GM12878/K562/hESC/mESC
organism = 'human'           # human/mouse
res = '5kb'                  # 5kb/10kb
genome = 'hg38'                # hg19/hg38/mm10
assay_type = 'HiCAR'        # HiC/HiChIP/MicroC/HiCAR
qval = 0.001                    # 0.1/0.01/0.001
data_path = '/media/labuser/STORAGE/GraphReg'

if qval == 0.1:
    fdr = '1'
elif qval == 0.01:
    fdr = '01'
elif qval == 0.001:
    fdr = '001'

if organism == 'mouse':
    N = 19
else:
    N = 22


for i in range(1,N+1):

    chr = 'chr'+str(i)
    filename_hic = data_path+'/data/'+cell_line+'/hic/'+assay_type+'/'+cell_line+'_'+assay_type+'_FDR_'+fdr+'_'+chr
    hic_dataframe = pd.read_csv(filename_hic, header=None, delimiter='\t')
    hic_dataframe.columns = ["chr", "start_i", "start_j", "qval", "count"]
    print(hic_dataframe)

    filename_seqs = data_path+'/data/csv/seqs_bed/'+organism+'/'+genome+'/'+res+'/sequences_'+chr+'.bed'
    seq_dataframe = pd.read_csv(filename_seqs, header=None, delimiter='\t')
    seq_dataframe.columns = ["chr", "start", "end"]
    print(seq_dataframe)
    nodes_list_all = seq_dataframe['start'].values
    n_nodes_all = len(nodes_list_all)
    print('number of all nodes (bins): ', n_nodes_all, nodes_list_all)


##### write the whole hic matrix for a chromosome as a sparse matrix #####
    hic = np.zeros([n_nodes_all,n_nodes_all])
    nodes_dict_all = {}
    for i in range(n_nodes_all):
        nodes_dict_all[nodes_list_all[i]] = i


    for i in range(n_nodes_all):
        #print(i)
        if nodes_list_all[i] > 0:
            cut = hic_dataframe.loc[hic_dataframe['start_i'] == nodes_list_all[i]]
            idx = cut['start_j'].values
            if len(idx) > 0:
                idx_col = np.zeros(len(idx), dtype=int)
                for j in range(len(idx)):
                    idx_col[j] = nodes_dict_all[idx[j]]
                hic[i,idx_col] = cut['count'].values
                print(cut['count'].values)

    hic_t = hic.transpose()
    hic_sym = hic + hic_t
    #row_max = np.max(hic_sym, axis=1)
    #print('hic_max: ', row_max)
    #hic_sym = hic_sym + np.diag(row_max)
    hic_sym = hic_sym.astype(np.float32)
    print(hic_sym[2000:2020,2000:2020])
    sparse_matrix = scipy.sparse.csr_matrix(hic_sym)
    scipy.sparse.save_npz(data_path+'/data/'+cell_line+'/hic/'+assay_type+'/'+assay_type+'_matrix_FDR_'+fdr+'_'+chr+'.npz', sparse_matrix)

