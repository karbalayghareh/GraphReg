import pysam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_path = '/media/labuser/STORAGE/GraphReg'   # data path

fasta_open = pysam.Fastafile(data_path+'/data/genome/hg19.ml.fa')
peaks_fasta = open(data_path+'/results/fimo/peaks_H3K27ac_K562.fasta', "w")
peaks_bed = pd.read_csv(data_path+'/data/csv/K562_H3K27ac_peaks.bed', sep="\t", header=None)
peaks_bed = peaks_bed.sort_values(by=[0, 1]).reset_index(drop=True)
peaks_bed = peaks_bed[((peaks_bed[0] != 'chrX') & (peaks_bed[0] != 'chrY'))].reset_index(drop=True)

N = len(peaks_bed)
for i in range(N):
    chrm = peaks_bed[0].values[i]
    start = peaks_bed[1].values[i]
    end = peaks_bed[2].values[i]
    line = '>'+chrm+':'+str(start)+'-'+str(end)
    peaks_fasta.write(line)
    peaks_fasta.write('\n')
    seq_dna = fasta_open.fetch(chrm, start, end)
    peaks_fasta.write(seq_dna)
    peaks_fasta.write('\n')

peaks_fasta.close()


############### number of TFs in K562's H3K27ac peaks ############

# fimo_out_df = pd.read_csv(data_path+'/results/fimo/peaks_H3K27ac_K562/TF_positions.bed', sep="\t")
# fimo_out_df.columns = ['chr', 'start', 'end', 'TF', '-log10(pval)', 'strand']
# fimo_out_df = fimo_out_df.drop_duplicates(subset=['chr', 'start', 'end', 'TF']).reset_index(drop=True)
# fimo_out_df['TF'] = fimo_out_df['TF'].str.strip()
# fimo_out_df['chr'] = fimo_out_df['chr'].str.strip()
# fimo_out_df.loc[fimo_out_df['TF']=='(ARID3A)_(Mus_musculus)_(DBD_1.00)', 'TF'] = 'ARID3A'
# TFs = np.unique(fimo_out_df['TF'].values)
# print('TFs: ', TFs)
# fimo_out_df.to_csv(data_path+'/results/fimo/peaks_H3K27ac_K562/TF_positions_unique.bed', sep="\t", header=False, index=False)

fimo_out_df = pd.read_csv(data_path+'/results/fimo/peaks_H3K27ac_K562/TF_positions_unique.bed', sep="\t")
fimo_out_df.columns = ['chr', 'start', 'end', 'TF', '-log10(pval)', 'strand']
TFs = np.unique(fimo_out_df['TF'].values)
print('TFs: ', TFs)

N = len(TFs)
n_TF = np.zeros(N)
for i in range(N):
    n_TF[i] = len(fimo_out_df[fimo_out_df['TF']==TFs[i]])
n_TF = n_TF.astype(np.int64)
print('n_TF: ', n_TF)

df = pd.DataFrame(data=n_TF, index=list(TFs), columns=['Number'])
df = df.sort_values(by=['Number'])
plt.figure(figsize = (2,10))
ax = sns.heatmap(df, xticklabels=1, yticklabels=1, cmap="YlGnBu", annot=df, fmt="d")

