# MoGCN
## What is it?
MoGCN, a multi-omics integration method based on graph convolutional network.<br>
![Image text](https://github.com/jiayee00/FYP-coding/blob/main/overall%20workflow.png) <br>
As shown in figure, inputs to the model are multi-omics expression matrices, including genomics, transcriptomics, proteomics, etc. MoGCN exploits the GCN model to incorporate and extend two multi-omics integration algorithms: Support vector machine-recursive feature elimination (SVM-RFE) based on feature matrix and similarity network fusion algorithm based on patient similarity network. <br>

## Requirements 
MoGCN is a Python scirpt tool, Python environment need:<br>
Python 3.6 or above <br>
Pytorch 1.4.0 or above <br>
snfpy 0.2.2 <br>


## Usage
The whole workflow is divided into three steps: <br>
* Use SVM-RFE to reduce the dimensionality of multi-omics data to obtain multi-omics feature matrix <br>
* Use SNF to construct patient similarity network <br>
* Input multi-omics feature matrix  and the patient similarity network to GCN <br>
* Conduct SMOTE before GCN classification <br>
The sample data is in the data folder, which contains the CNV, mRNA and RPPA data of BRCA. <br>
### Command Line Tool
```Python
python SVMRFE_run.py -p1 data/transcriptome.csv -p2 data/genome.csv -p3 data/proteome.csv -s 0 
python SNF.py -p data/transcriptome.csv data/genome.csv data/proteome.csv -m sqeuclidean
python GCN_run.py -fd result/latent_data_1300_target.csv -ad result/SNF_fused_matrix.csv -ld data/sample_classes.csv -ts data/test_sample.csv -m 1 -d gpu -p 20
```
The meaning of the parameters can be viewed through -h/--help <br>

### Data Format
* The input type of each omics data must be .csv, the rows represent samples, and the columns represent features (genes). In each expression matrix, the first column must be the samples, and the remaining columns are features. Samples in all omics data must be consistent. AE and SNF are unsupervised models and do not require sample labels.<br>
* GCN is a semi-supervised classification model, it requires sample label files (.csv format) during training. The first column of the label file is the sample name, the second column is the digitized sample label, the remaining columns are not necessary. <br>

## Contact
For any questions please contact Lee Jia Yee (Email: mandylee7311@gmail.com).

## License
MIT License

## Citation
Li X, Ma J, Leng L, Han M, Li M, He F and Zhu Y (2022) MoGCN: A Multi-Omics Integration Method Based on Graph Convolutional Network for Cancer Subtype Analysis. Front. Genet. 13:806842. doi: 10.3389/fgene.2022.806842. <br>

