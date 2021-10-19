import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statannot import add_stat_annotation


##### check the effects of different 3D data and FDRs #####

data_path = '/media/labuser/STORAGE/GraphReg'   # data path
df = pd.DataFrame(columns=['Cell_train', 'Cell_test', 'Method', 'Set', 'valid_chr', 'test_chr', 'n_gene_test', '3D_data_train', '3D_data_test', 'FDR_train', 'FDR_test', 'R','NLL'])
df1 = pd.read_csv(data_path+'/results/csv/cage_prediction/cell_to_cell/R_NLL_epi_models_hESC_MicroC_FDR_1_to_GM12878_HiChIP_FDR_1.csv', sep='\t')
df = df.append(df1, ignore_index=True)

g = sns.catplot(x="Set", y="R",
                hue="Method",
                data=df, kind="box",
                height=4, aspect=1, sharey=False)
#plt.savefig('../figs/Epi-models/boxplot_R_check_3D_and_fdr.pdf')

g = sns.catplot(x="Set", y="NLL",
                hue="Method",
                data=df, kind="box",
                height=4, aspect=1, sharey=False)