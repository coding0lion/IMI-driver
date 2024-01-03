%根据癌症的miRNA，提取有用的对应关系
function [new_miR,new_ge]=new_miRNA_gene(miRNA_name,data)
% mach = readtable(".\miRNA_gene.txt",'ReadVariableNames',false);
load('miRNA_gene.mat');
m_miRNA_name=mach.Var1;
m_gene_name=mach.Var2;
% cm=readtable(".\change_name.csv",'ReadVariableNames',false);
% miRNA_name=cm.Var1;
start=0;
for i=1:length(miRNA_name)
    indx=find(strcmp(miRNA_name{i},m_miRNA_name));
    if indx
        new_ge(start+1:start+length(indx),1)=m_gene_name(indx);
        new_miR(start+1:start+length(indx),1)=miRNA_name(i);
        start=start+length(indx);
    end
end