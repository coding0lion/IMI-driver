
ppi=readtable("pairs_g_g(800).txt",'ReadVariableNames',false);
pairs=[ppi.Var1,ppi.Var2];

load('BRCA_gene_expression_RNAseq.mat')
gene=mRNA_gene;
%%
load('cgc.mat')
cgc_gene=cgc.GeneSymbol;
cgc_lab=cgc.Tier;
know_gene=cgc_gene(find(cgc_lab==1));
know_lab=cgc_lab(find(cgc_lab==1));
%%
% [~,ngc_gene]=xlsread('BRCA.xlsx');
% ngc_gene=unique(ngc_gene);
% ngc_lab=ones(length(ngc_gene),1);
% know_gene=ngc_gene;
% know_lab=ngc_lab;
%%
gene=sortrows(gene,1);
%%
%打标签
for i=1:length(gene)
    index=find(strcmp(gene{i},know_gene)==1);
    if index
    gene_labs(i,1)=know_lab(index);
    else
    gene_labs(i,1)=-1;
    end
end
adj_m=create_adjacency_matrix(gene,pairs);%得到邻接矩阵
ex=ones(length(gene),1);
d=0.85;
r=Generank(adj_m,ex,d);%得到generank值

r_ppi=num2cell(r);
relt_ppi=[gene r_ppi];
[h_ppi,p_ppi]=rank_test(relt_ppi,gene_labs);%检验

relt_ppi=sortrows(relt_ppi,-2);%进行降序排列
rank=(1:length(relt_ppi))';
rank_ppi=[relt_ppi num2cell(rank)];
rank_ppi=sortrows(rank_ppi,1);%最终的序列特征
save(['./output/',cancer,'_rank(intogen)_ppi.mat'], 'h_ppi','p_ppi','rank_ppi','gene_labs','-v7.3')

[ii, jj] = find(adj_m); % row and col indices of connections 
y = accumarray(ii, jj-1 , [], @(x){sort(x.')}); % get all nodes connected to each node, 
node=[0:1:length(gene)-1]';

%存为邻接表
fid=fopen('sub_ppi.txt','wt');
for i=1:size(gene,1)%行
    b = node(i);
    fprintf(fid,'%.0f ',b);
   if i<=size(y,1)
    for j=1:size(y{i},2)%列
    a = y{i}(j);
%     a = cell2mat(a);
    fprintf(fid,'%.0f ',a);
    end
   end
    fprintf(fid,'\n');%加换行符
end