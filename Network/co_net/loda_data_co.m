function [gene,data]=loda_data_co(cancer)
load(['./data/',cancer,'/',cancer,'_gene_expression_RNAseq.mat'])%load基因表达数据
gene=mRNA_gene;
data=mRNA_patient_data;
end
