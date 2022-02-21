from turtle import color
from numpy import size
from pyvis.network import Network
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch import stack


# read hic graph
df = pd.read_csv("/media/labuser/STORAGE/GraphReg/data/mESC/hic/HiChIP/mESC_HiChIP_FDR_001_chr1", sep="\t", names=["chr", "source", "target", "type", "weight"])
df = df[["source", "target", "type", "weight"]]
df = df.iloc[:20000]

# load pandas df as networkx graph
G = nx.from_pandas_edgelist(df, source = "source", target = "target", edge_attr = "weight")
#nx.draw_circular(G, node_size=10)
#nx.draw(G, node_size=10)
#plt.show()


# create vis network
net = Network(notebook=True)

# load the networkx graph
net.from_nx(G)

# show
#net.toggle_physics(True)
#net.barnes_hut()
net.force_atlas_2based()
#net.hrepulsion()
net.show("graph.html")


##### compare graphs

# K562
df1 = pd.read_csv("/media/labuser/STORAGE/GraphReg/results/csv/cage_prediction/epi_models/cage_predictions_epi_models_K562_HiChIP_FDR_1.csv", sep="\t")
df = df1[['n_contact']]
df['3D assay'] = 'HiChIP / FDR = 0.1'

df1 = pd.read_csv("/media/labuser/STORAGE/GraphReg/results/csv/cage_prediction/epi_models/cage_predictions_epi_models_K562_HiChIP_FDR_01.csv", sep="\t")
df1 = df1[['n_contact']]
df1['3D assay'] = 'HiChIP / FDR = 0.01'
df = df.append(df1)

df1 = pd.read_csv("/media/labuser/STORAGE/GraphReg/results/csv/cage_prediction/epi_models/cage_predictions_epi_models_K562_HiChIP_FDR_001.csv", sep="\t")
df1 = df1[['n_contact']]
df1['3D assay'] = 'HiChIP / FDR = 0.001'
df = df.append(df1)

df1 = pd.read_csv("/media/labuser/STORAGE/GraphReg/results/csv/cage_prediction/epi_models/cage_predictions_epi_models_K562_HiC_FDR_1.csv", sep="\t")
df1 = df1[['n_contact']]
df1['3D assay'] = 'HiC / FDR = 0.1'
df = df.append(df1)

df1 = pd.read_csv("/media/labuser/STORAGE/GraphReg/results/csv/cage_prediction/epi_models/cage_predictions_epi_models_K562_HiC_FDR_01.csv", sep="\t")
df1 = df1[['n_contact']]
df1['3D assay'] = 'HiC / FDR = 0.01'
df = df.append(df1)

df1 = pd.read_csv("/media/labuser/STORAGE/GraphReg/results/csv/cage_prediction/epi_models/cage_predictions_epi_models_K562_HiC_FDR_001.csv", sep="\t")
df1 = df1[['n_contact']]
df1['3D assay'] = 'HiC / FDR = 0.001'
df = df.append(df1)
df = df.reset_index(drop=True)
df = df.rename(columns = {'n_contact': 'Number of contacts', '3D assay': '3D assay / FDR'})

df_first_part = df[df['Number of contacts'] <=10]
sns.histplot(data=df_first_part, x="Number of contacts", hue="3D assay / FDR", log_scale = [False, True], binrange = [0, 10], binwidth = 1, multiple = 'dodge', shrink = .7, discrete=True)
plt.savefig('../figs/Epi-models/final/compare_graphs_K562_n_0-10.pdf')

df_seconf_part = df[df['Number of contacts'] >10]
sns.histplot(data=df_seconf_part, x="Number of contacts", hue="3D assay / FDR", log_scale = [False, True], binrange = [10, 200], binwidth = 20, multiple = 'dodge', shrink = .8)
plt.savefig('../figs/Epi-models/final/compare_graphs_K562.pdf')

# GM12878
df1 = pd.read_csv("/media/labuser/STORAGE/GraphReg/results/csv/cage_prediction/epi_models/cage_predictions_epi_models_GM12878_HiChIP_FDR_1.csv", sep="\t")
df = df1[['n_contact']]
df['3D assay'] = 'HiChIP / FDR = 0.1'

df1 = pd.read_csv("/media/labuser/STORAGE/GraphReg/results/csv/cage_prediction/epi_models/cage_predictions_epi_models_GM12878_HiChIP_FDR_01.csv", sep="\t")
df1 = df1[['n_contact']]
df1['3D assay'] = 'HiChIP / FDR = 0.01'
df = df.append(df1)

df1 = pd.read_csv("/media/labuser/STORAGE/GraphReg/results/csv/cage_prediction/epi_models/cage_predictions_epi_models_GM12878_HiChIP_FDR_001.csv", sep="\t")
df1 = df1[['n_contact']]
df1['3D assay'] = 'HiChIP / FDR = 0.001'
df = df.append(df1)

df1 = pd.read_csv("/media/labuser/STORAGE/GraphReg/results/csv/cage_prediction/epi_models/cage_predictions_epi_models_GM12878_HiC_FDR_1.csv", sep="\t")
df1 = df1[['n_contact']]
df1['3D assay'] = 'HiC / FDR = 0.1'
df = df.append(df1)

df1 = pd.read_csv("/media/labuser/STORAGE/GraphReg/results/csv/cage_prediction/epi_models/cage_predictions_epi_models_GM12878_HiC_FDR_01.csv", sep="\t")
df1 = df1[['n_contact']]
df1['3D assay'] = 'HiC / FDR = 0.01'
df = df.append(df1)

df1 = pd.read_csv("/media/labuser/STORAGE/GraphReg/results/csv/cage_prediction/epi_models/cage_predictions_epi_models_GM12878_HiC_FDR_001.csv", sep="\t")
df1 = df1[['n_contact']]
df1['3D assay'] = 'HiC / FDR = 0.001'
df = df.append(df1)
df = df.reset_index()
df = df.rename(columns = {'n_contact': 'Number of contacts', '3D assay': '3D assay / FDR'})

df_first_part = df[df['Number of contacts'] <=10]
sns.histplot(data=df_first_part, x="Number of contacts", hue="3D assay / FDR", log_scale = [False, True], binrange = [0, 10], binwidth = 1, multiple = 'dodge', shrink = .7, discrete=True)
plt.savefig('../figs/Epi-models/final/compare_graphs_GM12878_n_0-10.pdf')

df_seconf_part = df[df['Number of contacts'] >10]
sns.histplot(data=df_seconf_part, x="Number of contacts", hue="3D assay / FDR", log_scale = [False, True], binrange = [10, 200], binwidth = 20, multiple = 'dodge', shrink = .8)
plt.savefig('../figs/Epi-models/final/compare_graphs_GM12878.pdf')


# hESC
df1 = pd.read_csv("/media/labuser/STORAGE/GraphReg/results/csv/cage_prediction/epi_models/cage_predictions_epi_models_hESC_MicroC_FDR_1.csv", sep="\t")
df = df1[['n_contact']]
df['3D assay'] = 'MicroC / FDR = 0.1'

df1 = pd.read_csv("/media/labuser/STORAGE/GraphReg/results/csv/cage_prediction/epi_models/cage_predictions_epi_models_hESC_MicroC_FDR_01.csv", sep="\t")
df1 = df1[['n_contact']]
df1['3D assay'] = 'MicroC / FDR = 0.01'
df = df.append(df1)

df1 = pd.read_csv("/media/labuser/STORAGE/GraphReg/results/csv/cage_prediction/epi_models/cage_predictions_epi_models_hESC_MicroC_FDR_001.csv", sep="\t")
df1 = df1[['n_contact']]
df1['3D assay'] = 'MicroC / FDR = 0.001'
df = df.append(df1)

df1 = pd.read_csv("/media/labuser/STORAGE/GraphReg/results/csv/cage_prediction/epi_models/cage_predictions_epi_models_hESC_HiCAR_FDR_1.csv", sep="\t")
df1 = df1[['n_contact']]
df1['3D assay'] = 'HiCAR / FDR = 0.1'
df = df.append(df1)

df1 = pd.read_csv("/media/labuser/STORAGE/GraphReg/results/csv/cage_prediction/epi_models/cage_predictions_epi_models_hESC_HiCAR_FDR_01.csv", sep="\t")
df1 = df1[['n_contact']]
df1['3D assay'] = 'HiCAR / FDR = 0.01'
df = df.append(df1)

df1 = pd.read_csv("/media/labuser/STORAGE/GraphReg/results/csv/cage_prediction/epi_models/cage_predictions_epi_models_hESC_HiCAR_FDR_001.csv", sep="\t")
df1 = df1[['n_contact']]
df1['3D assay'] = 'HiCAR / FDR = 0.001'
df = df.append(df1)
df = df.reset_index()
df = df.rename(columns = {'n_contact': 'Number of contacts', '3D assay': '3D assay / FDR'})

df_first_part = df[df['Number of contacts'] <=10]
sns.histplot(data=df_first_part, x="Number of contacts", hue="3D assay / FDR", log_scale = [False, True], binrange = [0, 10], binwidth = 1, multiple = 'dodge', shrink = .7, discrete=True)
plt.savefig('../figs/Epi-models/final/compare_graphs_hESC_n_0-10.pdf')

df_seconf_part = df[df['Number of contacts'] >10]
sns.histplot(data=df_seconf_part, x="Number of contacts", hue="3D assay / FDR", log_scale = [False, True], binrange = [10, 200], binwidth = 20, multiple = 'dodge', shrink = .8)
plt.savefig('../figs/Epi-models/final/compare_graphs_hESC.pdf')

# mESC
df1 = pd.read_csv("/media/labuser/STORAGE/GraphReg/results/csv/cage_prediction/epi_models/cage_predictions_epi_models_mESC_HiChIP_FDR_1.csv", sep="\t")
df = df1[['n_contact']]
df['3D assay'] = 'HiChIP / FDR = 0.1'

df1 = pd.read_csv("/media/labuser/STORAGE/GraphReg/results/csv/cage_prediction/epi_models/cage_predictions_epi_models_mESC_HiChIP_FDR_01.csv", sep="\t")
df1 = df1[['n_contact']]
df1['3D assay'] = 'HiChIP / FDR = 0.01'
df = df.append(df1)

df1 = pd.read_csv("/media/labuser/STORAGE/GraphReg/results/csv/cage_prediction/epi_models/cage_predictions_epi_models_mESC_HiChIP_FDR_001.csv", sep="\t")
df1 = df1[['n_contact']]
df1['3D assay'] = 'HiChIP / FDR = 0.001'
df = df.append(df1)
df = df.append(df1)
df = df.reset_index()
df = df.rename(columns = {'n_contact': 'Number of contacts', '3D assay': '3D assay / FDR'})

df_first_part = df[df['Number of contacts'] <= 10]
sns.histplot(data=df_first_part, x="Number of contacts", hue="3D assay / FDR", log_scale = [False, True], binrange = [0, 10], binwidth = 1, multiple = 'dodge', shrink = .7, discrete=True)
plt.savefig('../figs/Epi-models/final/compare_graphs_mESC_n_0-10.pdf')

df_seconf_part = df[df['Number of contacts'] > 10]
sns.histplot(data=df_seconf_part, x="Number of contacts", hue="3D assay / FDR", log_scale = [False, True], binrange = [10, 125], binwidth = 15, multiple = 'dodge', shrink = .8)
plt.savefig('../figs/Epi-models/final/compare_graphs_mESC.pdf')
