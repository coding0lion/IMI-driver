function [p]=met_survival_analysis(analysis_gene,cancer)
clinic_path = ['../data/' cancer '/' cancer '_clinic.mat']
met_path = ['../data/' cancer '/' cancer '_met.mat']

load(clinic_path)
load(met_path)
clinic_temp_ID = patient_ID';
met_temp_ID = met_patient_ID;
%
[ind_ID,ind_mi]=mapping_sample_met(met_temp_ID,clinic_temp_ID);
met_id = met_temp_ID(ind_ID);
clinic_id = clinic_temp_ID(ind_mi);
ma = met_patient_data(:,ind_ID);
dmfs_e = patient_dmfs_e(ind_mi,:);
dmfs_time = patient_dmfs_time(ind_mi,:);
[ma ,phenotype]=mapping_met(dmfs_e,dmfs_time,clinic_id,met_id,ma);
%分高表达和低表达组
gene_index=find(strcmp(analysis_gene,met_gene)==1);
p=0;
if gene_index
    data=ma(gene_index,:);
    data_M = median(data);
    high = phenotype(data>=data_M,:);
    low = phenotype(data<data_M,:);
    if ~isempty(high) && ~isempty(low)
        note=[cancer,' of met'];
        p=logrank(high,low,note);
        saveas(gcf, [note '.jpg']);
    else
        p=nan;
    end
else
    p=nan;
end
end