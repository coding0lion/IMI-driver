function [p]=mRNA_survival_analysis(analysis_gene,cancer)
clinic_path = ['../data/' cancer '/' cancer '_clinic.mat']
mRNA_path = ['../data/' cancer '/' cancer '_gene_expression_RNAseq.mat']

load(clinic_path)
load(mRNA_path)
clinic_temp_ID = patient_ID';
mRNA_temp_ID = mRNA_patient_ID;
%
[ind_ID,ind_mi]=mapping_sample(mRNA_temp_ID,clinic_temp_ID);
mRNA_id = mRNA_temp_ID(ind_ID);
clinic_id = clinic_temp_ID(ind_mi);
ma = mRNA_patient_data(:,ind_ID);
dmfs_e = patient_dmfs_e(ind_mi,:);
dmfs_time = patient_dmfs_time(ind_mi,:);
[ma ,phenotype]=mapping(dmfs_e,dmfs_time,clinic_id,mRNA_id,ma);
%分高表达和低表达组
gene_index=find(strcmp(analysis_gene,mRNA_gene)==1);
data=ma(gene_index,:);
data_M = median(data);
high = phenotype(data>=data_M,:);
low = phenotype(data<data_M,:);
note=[cancer,' of mRNA'];
p=logrank(high,low,note)
saveas(gcf, [note '.jpg']);
end