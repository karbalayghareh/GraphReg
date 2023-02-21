import numpy as np
import pandas as pd
import time
import os

resolution = '5kb'     # 5kb/10kb
organism = 'human'     # human/mouse
genome = 'hg38'        # hg38/hg19/mm10
thr = 0                # only keep the tss bins whose distance from bin borders are more than "thr" 
                       # (only applicable when want to consider the bins with 1 tss, otherwise thr = 0)
data_path = '/media/labuser/STORAGE/GraphReg'

if organism == 'mouse':
    chr_list = ['chr'+str(i) for i in range(1,20)] + ['chrX']
else:
    chr_list = ['chr'+str(i) for i in range(1,23)] + ['chrX']

n_tss_bins_all = 0
n_tss_bins_all_1 = 0
n_tss_bins_all_2 = 0
n_tss_bins_all_3 = 0
n_tss_bins_all_4 = 0
n_tss_bins_all_5 = 0
n_tss_bins_all_6 = 0

for chr in chr_list:

    if organism == 'human' and genome == 'hg19':
       filename_tss = data_path+'/data/tss/'+organism+'/'+genome+'/hg19_gencodev19_tss.bed'
    elif organism == 'human' and genome == 'hg38':
       #filename_tss = data_path+'/data/tss/'+organism+'/'+genome+'/gencode.v38.annotation.gtf.tss.bed'
       filename_tss = data_path+'/data/tss/'+organism+'/distal_regulation_group/'+genome+'/RefSeqCurated.170308.bed.CollapsedGeneBounds.hg38.TSS500bp.bed'
    elif organism == 'mouse':
        filename_tss = data_path+'/data/tss/'+organism+'/'+genome+'/mm10_gencode_vM9_tss.bed'

    tss_dataframe = pd.read_csv(filename_tss, header=None, delimiter='\t')
    #tss_dataframe.columns = ["chr", "tss_1", "tss_2", "ens", "gene", "strand", "type"]
    tss_dataframe.columns = ["chr", "tss_1", "tss_2", "gene", "na", "strand"]

    #protein_coding_tss = tss_dataframe.loc[(tss_dataframe['type'] == 'protein_coding') & (tss_dataframe['chr'] == chr)]
    protein_coding_tss = tss_dataframe[tss_dataframe['chr'] == chr]
    protein_coding_tss = protein_coding_tss.reset_index(drop=True)
    print(protein_coding_tss)
    n_tss = len(protein_coding_tss)
    
    filename_seqs = data_path+'/data/csv/seqs_bed/'+organism+'/'+genome+'/'+resolution+'/sequences_'+chr+'.bed'
    seq_dataframe = pd.read_csv(filename_seqs, header=None, delimiter='\t')
    seq_dataframe.columns = ["chr", "start", "end"]
    print(seq_dataframe)
    bin_start = seq_dataframe['start'].values
    bin_end = seq_dataframe['end'].values
    n_bin = len(bin_start)
    print('number of bins: ', n_bin)
    #np.save(data_path+'/data/tss/'+organism+'/'+genome+'/bin_start_'+chr, bin_start)
    np.save(data_path+'/data/tss/'+organism+'/distal_regulation_group/'+genome+'/bin_start_'+chr, bin_start)
    
    
    ### write tss
    tss = np.zeros(n_bin)
    gene_name = []
    for i in range(n_tss):
        idx_tss = seq_dataframe.loc[(seq_dataframe['start'] <= protein_coding_tss['tss_1'][i]) & (seq_dataframe['end'] > protein_coding_tss['tss_1'][i])].index
        if len(idx_tss)==0:
            print(protein_coding_tss['tss_1'][i])
        tss[idx_tss] = tss[idx_tss] + 1
    
    print('number of all tss bins:', np.sum(tss>0))
    print('number of bins with 1 tss:', np.sum(tss==1))
    print('number of bins with 2 tss:', np.sum(tss==2))
    print('number of bins with 3 tss:', np.sum(tss==3))
    print('number of bins with 4 tss:', np.sum(tss==4))
    print('number of bins with 5 tss:', np.sum(tss==5))


    n_tss_bins_all = n_tss_bins_all + np.sum(tss>0)
    n_tss_bins_all_1 = n_tss_bins_all_1 + np.sum(tss==1)
    n_tss_bins_all_2 = n_tss_bins_all_2 + np.sum(tss==2)
    n_tss_bins_all_3 = n_tss_bins_all_3 + np.sum(tss==3)
    n_tss_bins_all_4 = n_tss_bins_all_4 + np.sum(tss==4)
    n_tss_bins_all_5 = n_tss_bins_all_5 + np.sum(tss==5)
    n_tss_bins_all_6 = n_tss_bins_all_6 + np.sum(tss==6)
    
    print('number of tss: ', np.sum(tss).astype(np.int64))
    #np.save(data_path+'/data/tss/'+organism+'/'+genome+'/tss_bins_'+chr, tss)
    np.save(data_path+'/data/tss/'+organism+'/distal_regulation_group/'+genome+'/tss_bins_'+chr, tss)
    

    ### find gene names and their tss positions in the bins 
    pos_tss = np.zeros(n_bin).astype(np.int64)
    gene_name = np.array([""]*n_bin, dtype='|U16')
    for i in range(n_bin):
        if tss[i] >= 1:     # if want to choose only bins with one tss: tss[i] == 1
           pos_tss1 = protein_coding_tss.loc[(seq_dataframe['start'][i] <= protein_coding_tss['tss_1'] - thr) & (seq_dataframe['end'][i] > protein_coding_tss['tss_1'] + thr)]['tss_1'].values
           if len(pos_tss1)>0:
              pos_tss1 = pos_tss1[0]
              gene_name1 = protein_coding_tss.loc[(seq_dataframe['start'][i] <= protein_coding_tss['tss_1'] - thr) & (seq_dataframe['end'][i] > protein_coding_tss['tss_1'] + thr)]['gene'].values
              gene_names = gene_name1[0]
              for j in range(1,len(gene_name1)):
                  gene_names = gene_names + '+' + gene_name1[j]
              print('gene_names: ', gene_names)
    
              pos_tss[i] = pos_tss1
              gene_name[i] = gene_names
    
    print(len(pos_tss), pos_tss[0:800])
    print(len(gene_name), gene_name[:800])
    
    #np.save(data_path+'/data/tss/'+organism+'/'+genome+'/tss_pos_'+chr, pos_tss)
    np.save(data_path+'/data/tss/'+organism+'/distal_regulation_group/'+genome+'/tss_pos_'+chr, pos_tss)
    #np.save(data_path+'/data/tss/'+organism+'/'+genome+'/tss_gene_'+chr, gene_name)
    np.save(data_path+'/data/tss/'+organism+'/distal_regulation_group/'+genome+'/tss_gene_'+chr, gene_name)

print('number of all tss bins:', n_tss_bins_all)
print('number of bins with 1 tss:', n_tss_bins_all_1)
print('number of bins with 2 tss:', n_tss_bins_all_2)
print('number of bins with 3 tss:', n_tss_bins_all_3)
print('number of bins with 4 tss:', n_tss_bins_all_4)
print('number of bins with 5 tss:', n_tss_bins_all_5)
print('number of bins with 6 tss:', n_tss_bins_all_6)

#protein_coding_tss_all = tss_dataframe.loc[tss_dataframe['type'] == 'protein_coding']
#print('number of all tss: ', len(protein_coding_tss_all))

print('number of all tss: ', len(tss_dataframe))