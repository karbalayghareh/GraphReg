import numpy as np
import pandas as pd
import rpy2
from diffexpr.py_deseq import py_DESeq2

TF = 'ARID3A'
df_control_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/Control_rep1.tsv', sep='\t')
df_control_1 = df_control_1[df_control_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_control_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/Control_rep2.tsv', sep='\t')
df_control_2 = df_control_2[df_control_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'ATF3'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column

dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')

TF = 'BACH1'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')

TF = 'DLX1'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'FOXK2'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'GATA1'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'GATA2'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'HINFP'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'HMBOX1'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')

TF = 'HMGB2'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'HOXB4'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'HOXB9'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'HSF1'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')

TF = 'ILF2'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'JUND'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'KLF2'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'MAF1'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'MEIS2'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')

####################
TF = 'MITF'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'MXD3'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'NFATC1'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'NFYA'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep3.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep4.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'NFYB'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_3 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep3.tsv', sep='\t')
df_TF_3 = df_TF_3[df_TF_3['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_4 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep4.tsv', sep='\t')
df_TF_4 = df_TF_4[df_TF_4['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2', 'B_3', 'B_4'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)
df['B_3'] = df_TF_3['expected_count'].astype(np.int64)
df['B_4'] = df_TF_4['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([1234])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'NR2C2'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep3.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_3 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep4.tsv', sep='\t')
df_TF_3 = df_TF_3[df_TF_3['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2', 'B_3'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)
df['B_3'] = df_TF_3['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'NR2F2'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep3.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_3 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep4.tsv', sep='\t')
df_TF_3 = df_TF_3[df_TF_3['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2', 'B_3'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)
df['B_3'] = df_TF_3['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'NR4A1'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep3.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep4.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'NRF1'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep3.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_3 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep4.tsv', sep='\t')
df_TF_3 = df_TF_3[df_TF_3['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2', 'B_3'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)
df['B_3'] = df_TF_3['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'RELA'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep3.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep4.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'RFX5'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'RNF2'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep3.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep4.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'SIX5'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep3.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep4.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'SMAD5'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep3.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep4.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'SP1'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep3.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_3 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep4.tsv', sep='\t')
df_TF_3 = df_TF_3[df_TF_3['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2', 'B_3'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)
df['B_3'] = df_TF_3['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'SP2'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep3.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_3 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep4.tsv', sep='\t')
df_TF_3 = df_TF_3[df_TF_3['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2', 'B_3'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)
df['B_3'] = df_TF_3['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'SRF'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'STAT1'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'STAT2'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'STAT5A'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'STAT6'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'TEAD2'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'TEAD4'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'TFDP1'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'THAP1'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'TRIM28'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'UBTF'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')

TF = 'USF1'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'USF2'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'ZBTB33'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'ZNF143'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'ZNF384'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


TF = 'ZNF395'
df_TF_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep1.tsv', sep='\t')
df_TF_1 = df_TF_1[df_TF_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_TF_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/CRISPRi_K562_ENCODE/'+TF+'_rep2.tsv', sep='\t')
df_TF_2 = df_TF_2[df_TF_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'A_1', 'A_2', 'B_1', 'B_2'])
df['id'] = df_control_1['gene_id']
df['A_1'] = df_control_1['expected_count'].astype(np.int64)
df['A_2'] = df_control_2['expected_count'].astype(np.int64)
df['B_1'] = df_TF_1['expected_count'].astype(np.int64)
df['B_2'] = df_TF_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([AB])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','B','A'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('../results/csv/CRISPRi_K562_DESeq_results/'+TF+'_KO.tsv', sep='\t')


############################################ DESeq between GM12878 and K562 #########################################


df_GM_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/GM12878_RNA_seq_rep1.tsv', sep='\t')
df_GM_1 = df_GM_1[df_GM_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_GM_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/GM12878_RNA_seq_rep2.tsv', sep='\t')
df_GM_2 = df_GM_2[df_GM_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df_K_1 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/K562_RNA_seq_rep1.tsv', sep='\t')
df_K_1 = df_K_1[df_K_1['gene_id'].str.contains("ENSG")].reset_index(drop=True)
df_K_2 = pd.read_csv('/media/labuser/STORAGE/GraphReg/data/csv/K562_RNA_seq_rep2.tsv', sep='\t')
df_K_2 = df_K_2[df_K_2['gene_id'].str.contains("ENSG")].reset_index(drop=True)

df = pd.DataFrame(columns=['id', 'G_1', 'G_2', 'K_1', 'K_2'])
df['id'] = df_GM_1['gene_id']
df['G_1'] = df_GM_1['expected_count'].astype(np.int64)
df['G_2'] = df_GM_2['expected_count'].astype(np.int64)
df['K_1'] = df_K_1['expected_count'].astype(np.int64)
df['K_2'] = df_K_2['expected_count'].astype(np.int64)

sample_df = pd.DataFrame({'samplename': df.columns}) \
        .query('samplename != "id"')\
        .assign(sample = lambda d: d.samplename.str.extract('([GK])_', expand=False)) \
        .assign(replicate = lambda d: d.samplename.str.extract('_([123])', expand=False)) 
sample_df.index = sample_df.samplename


dds = py_DESeq2(count_matrix = df,
               design_matrix = sample_df,
               design_formula = '~ replicate + sample',
               gene_column = 'id') # <- telling DESeq2 this should be the gene ID column
    
dds.run_deseq() 
dds.get_deseq_result(contrast = ['sample','K','G'])
res = dds.deseq_result 
res.head()
res = res[res['pvalue']<=1]
res = res.sort_values(by=['pvalue'])
print(len(res))
res.to_csv('/media/labuser/STORAGE/GraphReg/results/csv/GM12878_K562_DESeq_resluts_RNA_seq.tsv', sep='\t')