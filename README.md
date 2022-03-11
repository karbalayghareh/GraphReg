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

After generating the `bigwig` files, we use [data_read.py](https://github.com/karbalayghareh/GraphReg/blob/master/utils/data_read.py) to read the `bigwig` files and save the coverage signals in `hdf5` format. We use `pool_width = 100` (to get the coverage in 100bp bins) for epigenomic tracks and `pool_width = 5000` (to get the coverage in 5Kb bins) for CAGE. The reason of using 5Kb bins for CAGE is that we use 5Kb resolution of 3D assays and want to have corresponding bins. If we use `bam_cov.py` to generate `bigwig` files, we set `sum_stat = 'sum'` to sum all the base-pair coverage in each bin; otherwise, if we use `bamCoverage` to generate `bigwig` files, we set `sum_stat = 'max'` as the coverage per bin has already been computed per bin. 

### 3D data (chromatin conformation: Hi-C/HiChIP/Micro-C/HiCAR)
The chromatin conformation `fastq` data from various 3D assyas such as Hi-C, HiChIP, Micro-C, HiCAR could be aligned to any genome (using packages like [Juicer](https://github.com/aidenlab/juicer) or [HiC-Pro](https://github.com/nservant/HiC-Pro)) to get `.hic` files. **GraphReg** needs connecivity graphs for each chromosome. As these 3D data are very noisy, we need some statistical tools to get the significant interactions for the graphs, otherwise it would be very noisy. To this end, we use [HiCDCPlus](https://github.com/mervesa/HiCDCPlus) which gets the `.hic` files and returns the significance level (FDR) for each genomic interaction (of resolution 5Kb) based on a Negative Binomial model. We filter the inteactions and keep the ones with `FDR <= alpha` to form the graphs and adjacency matrices. We have worked with three different values of `alpha = 0.1, 0.01, 0.001` and noticed that its ideal value depends on the 3D data. But, we recommend `alpha = 0.1` as a defalut and less stringent cutoff. 

The outputs of HiCDCPlus is given to [hic_to_graph.py](https://github.com/karbalayghareh/GraphReg/blob/master/utils/hic_to_graph.py) to generate the adjacency matrices for each chromosome, which are saved as sparce matrices. 

### TSS bins and positions
We need to have a `BED` file for TSS annotations. This file could be extracted from any gene annotation `GTF` files for any genome build. We have used GENCODE annotations which can be found [here](https://www.gencodegenes.org/). The TSS annotation `BED` file is given to [find_tss.py](https://github.com/karbalayghareh/GraphReg/blob/master/utils/find_tss.py) to compute the number of TSS's in each 5Kb bin. `find_tss.py` saves four outputs as numpy files: start position of each bin, number of TSS's in each bin, and the gane names (if existent) and their TSS positions in each bin. With 5Kb bins, the majority of them would have one TSS. However, there is a chance that a bin has 2 or 3 TSS's, in which case we save the first TSS position and all the genes (in the format `gene_name_1+gene_name_2`), because we want to keep track of all the genes appearing in each bin. 

### Write all data (1D, 3D, TSS) to TFRecords 
**GraphReg** has been implemented in TensorFlow. [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) is an efficient format to store and read data in TensorFlow. We use [data_write.py](https://github.com/karbalayghareh/GraphReg/blob/master/utils/data_write.py) to read (1) epigenomic coverage files (saved in `h5` format), (2) sparse adjacency matrices (saved in `npz` format), and (3) TSS files (saved in `np` format) and save them sequentially in TFRecords in each chromosome. We start from begining of each chromosome and write the epigenomic and CAGE coverages, adjacency matrices, and TSS annotations for the regions of 6Mb. Then we sweep the entire chromosome by steps of 2Mb. This way, there is no overlap for the middle 2Mb regions where we predict gene expression values. For each batch of 6Mb, the dimensions of data would be: `60,000` for each epigenomic track, `1200` for CAGE, and `1200 x 1200` for adjacency matrices. The predicted CAGE values in the middle `400` bins would appear in the loss function so that all the genes could see their distal enhancers up to 2Mb up- and downstream of their TSS. 

The TFRecord files are slightly different for **Epi-GraphReg** and **Seq-GraphReg** models: (1) TFRecords for **Seq-GraphReg** also contain one-hot-coded DNA sequences of the size `6,000,000 x 4`, as the DNA sequence is an input for these models, (2) The epigenomic signals for **Epi-GraphReg** undergo an extra log-normalization, via function `log2(x+1)`, to reduce their dynamic ranges, as they are inputs in  **Epi-GraphReg** models.

Now that we have generated TFRecord files, we are ready to train the models.

## Training GraphReg models

### Epi-GraphReg

Use [`Epi-GraphReg.py`](https://github.com/karbalayghareh/GraphReg/blob/master/train/Epi-GraphReg.py) to train the **Epi-GraphReg** models. You should specify the validation and test chromosomes. The remaining autosomal chromosomes are used for training. For example:
```
python Epi-GraphReg.py -c K562 -p $data_path -a HiChIP -g 1 -q 0.1 -v 3,13 -t 4,14
```
trains **Epi-GraphReg** on cell line K562, using graphs extracted from HiChIP with FDR (q-value) cutoff 0.1, in generalizble mode `-g 1`, with Chrs 3 and 13 as the  validation and Chrs 4 and 14 as the test chromosomes. `$data_path` is the directory where TFRecords have been stored. Training on generalizble mode means that the model uses the normalized epigenomic coverage tracks (for example the ones obtained from RPGC normalization) so that the trained model can be used in other cell types as well.

### Seq-GraphReg

The **Seq-GraphReg** models can be trained in two ways: (1) end to end, (2) separate. End to end training means that both epigenomic and CAGE data are predicted in a multi-task learning fashion. However, separate training means that we train two tasks (epigenomic and GACE) separately: we first use CNN layers to predict the epigenomic tracks from DNA sequence (similar to Basenji) and then feed the bottleneck representations to the graph attention layers to predict the CAGE values. So, which one should you use? It depends on the amount of GPU memory the users have access to. End to end training requires high GPU memory as it needs to load the entire 6Mb DNA sequence to the GPU memory. One advantage of an end to end model is the ability to do gradient-based feature attribution from output of any gene up to the base pair level, which consumes a lot of GPU memory. However, if a high GPU memory is not available, you can enjoy separate training strategy, where we can use smaller genomic regions of length 100Kb (instead of 6Mb) to predict the epigenomic data from DNA sequences as these are local features and no graph is used for this task. Then after predicting the entire 6Mb (60 mini batches of 100Kb), we concatenate their corrsponding bttleneck representations (with the size `60,000 x 64`, where 64 is the dimension of bottleneck representations) and feed that to the graph attention layers along with the corresponding graph of 6Mb region. 