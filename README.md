# GraphReg

**GraphReg** ([Chromatin interaction aware gene regulatory modeling with graph attention networks](https://www.biorxiv.org/content/10.1101/2021.03.31.437978v2.abstract)) is a deep learning based gene regulation model which integrates DNA sequence, 1D epigenomic data (such as chromatin accessability and histone modifications), and 3D chromatin conformation data (such as Hi-C, HiChIP, Micro-C, HiCAR) to predict gene expression in an informative way. **GraphReg** is a versatile model which can be used to answer interesting questions in regulatory genomics such as:

- How well we can predict expression of a gene by using the epigenomic features of its promoter and candidate enhancers and enhancer-promoter interactions? Can this model be used in unseen cell types to predict gene expression?

- What are the cis regulatory elements of the genes in each cell type? Which candidate enhancers are functional and play a role in gene regulation?

- Which transcription factor (TF) motifs are important for gene regulation? How do distal TF motifs regulate their target genes?


This repository contains all the codes for training **GraphReg** models and all the downstream analyses for gene expression prediction, enhancer validation, and discovering regulating TF motifs.

## Data preparation

### 1D data (epigenomic and CAGE)
We need a coverage file `bigwig` for each epigenomic track. We have used some useful functions from [Basenji](https://github.com/calico/basenji) for reading and writing the `bigwig` files, which can be found in [utils](https://github.com/karbalayghareh/GraphReg/tree/master/utils). 

 We can use two different approaches to generate `bigwig` files from alignment `BAM` files:

- [`bam_cov.py`](https://github.com/karbalayghareh/GraphReg/blob/master/utils/bam_cov.py) from Basenji. This works best when we want to work with each cell type individually. The coverage tracks from different cell types are not normalized by this method. In **Epi-GraphReg** if we are interested in cross-cell-type generalization, the coverage tracks should be normalized by other techniques such as DESeq, otherwise there would be batch effect between cell types due to sequencing depths, which would hurt the generalization performance. 

- [`bamCoverage`](https://deeptools.readthedocs.io/en/develop/content/tools/bamCoverage.html) from [deepTools](https://deeptools.readthedocs.io/en/develop/index.html). This is more suitable for cross-cell-type analyses, as they offer some normalization methods for `bigwig` files. In particular, we use 1x normalization or reads per genome coverage (RPGC), which normalizes the coverage in each bin by sequencing depth. We run `bamCoverage` with bin size 100 for epigenomic tracks and 5000 for CAGE-seq.

After generating the `bigwig` files, we use [data_read.py](https://github.com/karbalayghareh/GraphReg/blob/master/utils/data_read.py) to read the `bigwig` files and save the coverage signals in `hdf5` format. We use `pool_width=100` (to get the coverage in 100bp bins) for epigenomic tracks and `pool_width=5000` (to get the coverage in 5Kb bins) for CAGE. The reason of using 5Kb bins for CAGE is that we use 5Kb resolution of 3D assays and want to have corresponding bins. If we use `bam_cov.py` to generate `bigwig` files, we set `sum_stat='sum'` to sum all the base-pair coverage in each bin; otherwise, if we use `bamCoverage` to generate `bigwig` files, we set `sum_stat='max'` as the coverage per bin has already been computed per bin. 

### 3D data (chromatin conformation: Hi-C/HiChIP/Micro-C/HiCAR)
The chromatin conformation `fastq` data from various 3D assyas such as Hi-C, HiChIP, Micro-C, HiCAR could be aligned to any genome (using packages like [Juicer](https://github.com/aidenlab/juicer) or [HiC-Pro](https://github.com/nservant/HiC-Pro)) to get `.hic` files. **GraphReg** needs connecivity graphs for each chromosome. As these 3D data are very noisy, we need some statistical tools to get the significant interactions for the graphs, otherwise it would be very noisy. To this end, we use [HiCDCPlus](https://github.com/mervesa/HiCDCPlus) which gets the `.hic` files and returns the significance level (FDR) for each genomic interaction (of resolution 5Kb) based on a Negative Binomial model. We filter the inteactions and keep the ones with `FDR<=\alpha` to form the graphs and adjacency matrices. We have worked with three different values of `\alpha=0.1, 0.01, 0.001` and noticed that its ideal value depends on the 3D data. But, we recommend `\alpha=0.1` as a defalut and less stringent cutoff. 