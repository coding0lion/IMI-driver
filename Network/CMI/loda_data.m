function [dmfs_e dmfs_time sample_ID ID data, proSet]=loda_data(cancer)

load(['./data/',cancer,'/',cancer,'_clinic.mat']);%临床数据
load(['./data/',cancer,'/',cancer,'_gene_expression_RNAseq.mat'])%load基因表达数据
dmfs_e=patient_dmfs_e;
dmfs_time=patient_dmfs_time;
sample_ID=patient_ID;%临床数据样本id
ID=mRNA_patient_ID;%转录组样本id
data=mRNA_patient_data;
proSet=mRNA_gene;

end
