import numpy as np
import pandas as pd

data_path = '/media/labuser/STORAGE/GraphReg'   # data path

TAP_seq_enhancer_df = pd.read_csv(data_path+'/data/csv/TAP_seq_enhancer.csv')
TAP_seq_enhancer_df['ABC.Score'] = 0.0
ABC_K562_df = pd.read_csv(data_path+'/data/csv/ABC_preds_chr8_chr11_K562.txt', sep='\t')

for i in range(len(TAP_seq_enhancer_df)):
    print(i)
    tap_start = TAP_seq_enhancer_df.loc[i, 'start']
    tap_end = TAP_seq_enhancer_df.loc[i, 'end']
    tap_chr = TAP_seq_enhancer_df.loc[i, 'chr']
    tap_gene = TAP_seq_enhancer_df.loc[i, 'Gene']

    ABC_K562_df_sub = ABC_K562_df[(ABC_K562_df['chr'] == tap_chr) & (ABC_K562_df['TargetGene'] == tap_gene)]
    ABC_K562_df_sub = ABC_K562_df_sub[((tap_start >= ABC_K562_df_sub['start']) & (tap_start <= ABC_K562_df_sub['end'])) | ((tap_end >= ABC_K562_df_sub['start']) & (tap_end <= ABC_K562_df_sub['end'])) | ((tap_start <= ABC_K562_df_sub['start']) & (tap_end >= ABC_K562_df_sub['end']))]
    if len(ABC_K562_df_sub) >= 1 :
        TAP_seq_enhancer_df.loc[i, 'ABC.Score'] = np.mean(ABC_K562_df_sub['ABC_Score'].values)

TAP_seq_enhancer_df.to_csv(data_path+'/data/csv/TAP_seq_enhancer_with_abc.csv', sep=",", index=False)