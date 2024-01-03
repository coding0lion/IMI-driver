function [ind_ID,ind_mi]=mapping_sample(ID,miRNA_ID)
for i = 1:length(ID)
    ID_1{1,i}=ID{1,i}{1,1};
end
for j=1:length(miRNA_ID)
    miRNA_1{1,j}=miRNA_ID{1,j}{1,1};
end
[ c, ind_ID, ind_mi ]=intersect(ID_1,miRNA_1);
end