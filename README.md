# GraphReg

**GraphReg** ([Chromatin interaction aware gene regulatory modeling with graph attention networks](https://www.biorxiv.org/content/10.1101/2021.03.31.437978v2.abstract)) is a deep learning based gene regulation model which integrates DNA sequence, 1D epigenomic data (such as chromatin accessability and histone modifications), and 3D chromatin conformation data (such as Hi-C, HiChIP, Micro-C, HiCAR) to predict gene expression in an informative way. **GraphReg** is a versatile model which can be used to answer interesting questions in regulatory genomics such as:

- How well we can predict expression of a gene by using the epigenomic features of its promoter and candidate enhancers and enhancer-promoter interactions? Can this model be used in unseen cell types to predict gene expression?

- What are the cis regulatory elements of the genes in each cell type? Which candidate enhancers are functional and play a role in gene regulation?

- Which transcription factor (TF) motifs are important for gene regulation? How do distal TF motifs regulate their target genes?


This repository contains all the codes for training **GraphReg** models and all the downstream analyses for gene expression prediction, enhancer validation, and discovering regulating TF motifs.

## Data preparation

### 1D data (epigenomic)
We need a coverage file `bigwig` for each epigenomic track. We have used two different approaches to generate `bigwig` files form alignment `BAM` files:

- `bam_cov.py` from [Basenji](https://github.com/calico/basenji).

- `bamCoverage` from [deepTools](https://deeptools.readthedocs.io/en/develop/content/tools/bamCoverage.html).

