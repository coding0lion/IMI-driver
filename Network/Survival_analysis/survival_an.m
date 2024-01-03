%转录组生存分析
analysis_gene='RARB';
[v w]=xlsread("intogen_cancer.xlsx");
cancer_list=w(2:end,1);
p_val=zeros(length(cancer_list),3);
for i=1:length(cancer_list)
cancer=cancer_list{i};
%cancer='BRCA';
p_mRNA=mRNA_survival_analysis(analysis_gene,cancer)
p_val(i,1) = p_mRNA;
p_met=met_survival_analysis(analysis_gene,cancer)
p_val(i,2) = p_met;
[p_sm high_num low_num]=sm_survival_analysis(analysis_gene,cancer)
p_val(i,3) = p_sm;
num(i,1) = high_num;
num(i,2) = low_num;
end
xlswrite([analysis_gene '.xlsx'],p_val)
xlswrite([analysis_gene '_sm_num.xlsx'],num)
