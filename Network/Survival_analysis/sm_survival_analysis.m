function [p high_num low_num]=sm_survival_analysis(analysis_gene,cancer)
clinic_path = ['../data/' cancer '/' cancer '_clinic.mat']
sm_path = ['../data/' cancer '/' cancer '_somatic_mutation.mat']

load(clinic_path)
load(sm_path)
clinic_temp_ID = patient_ID';
sm_temp_ID = sm_patient_ID;
%
[ind_ID,ind_mi]=mapping_sample(sm_temp_ID,clinic_temp_ID);
sm_id = sm_temp_ID(ind_ID);
clinic_id = clinic_temp_ID(ind_mi);
ma = sm_patient_data(:,ind_ID);
dmfs_e = patient_dmfs_e(ind_mi,:);
dmfs_time = patient_dmfs_time(ind_mi,:);
[ma ,phenotype]=mapping(dmfs_e,dmfs_time,clinic_id,sm_id,ma);
%分高表达和低表达组
gene_index=find(strcmp(analysis_gene,sm_gene)==1);
data=ma(gene_index,:);
if find(data==1)
    high = phenotype(data==1,:);
    low = phenotype(data==0,:);
    note=[cancer,' of sm'];
    p=logrank(high,low,note);
    saveas(gcf, [note '.jpg']);
    high_num = length(high);
    low_num = length(low);
else
    p = nan;
    high_num = nan;
    low_num = nan;
end
end