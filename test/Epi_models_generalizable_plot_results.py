import numpy as np
from numpy.core.numeric import False_
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statannot import add_stat_annotation


##### check the effects of different 3D data and FDRs #####

data_path = '/media/labuser/STORAGE/GraphReg'   # data path
df = pd.DataFrame(columns=['Cell_train', 'Cell_test', 'Method', 'Set', 'valid_chr', 'test_chr', 'n_gene_test', '3D_data_train', '3D_data_test', 'FDR_train', 'FDR_test', 'R','NLL'])

cell_line_train_list = ['K562', 'GM12878']
cell_line_test_list = ['GM12878', 'K562']
fdr_dict = {'1': 0.1, '01': 0.01, '001': 0.001}

for c in range(1):
    cell_line_train = cell_line_train_list[c]
    cell_line_test = cell_line_test_list[c]

    if cell_line_test == 'GM12878' or cell_line_test == 'K562':
        genome='hg19'
        assay_type_test_list = ['HiC', 'HiChIP']
    elif cell_line_test == 'hESC':
        genome='hg38'
        assay_type_test_list = ['MicroC', 'HiCAR']

    if cell_line_train == 'GM12878' or cell_line_train == 'K562':
        assay_type_train_list = ['HiC', 'HiChIP']
    elif cell_line_train == 'hESC':
        assay_type_train_list = ['MicroC', 'HiCAR']

    for assay_type_train in assay_type_train_list:
        for assay_type_test in assay_type_test_list:
            for fdr_train in ['1', '01', '001']:
                for fdr_test in ['1', '01', '001']:

                    df1 = pd.read_csv(data_path+'/results/csv/cage_prediction/cell_to_cell/R_NLL_epi_models_'+cell_line_train+'_'+assay_type_train+'_FDR_'+fdr_train+'_to_'+cell_line_test+'_'+assay_type_test+'_FDR_'+fdr_test+'.csv', sep='\t')
                    df = df.append(df1, ignore_index=True)


g = sns.catplot(x="Set", y="NLL",
                hue="Method", row="FDR_train", col="FDR_test",
                data=df, kind="box",
                height=4, aspect=1, sharey=False)
#plt.savefig('../figs/Epi-models/boxplot_R_check_3D_and_fdr.pdf')

g = sns.catplot(x="Set", y="NLL",
                hue="Method",
                data=df, kind="box",
                height=4, aspect=1, sharey=False)