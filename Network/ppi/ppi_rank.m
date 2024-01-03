function ppi_rank(cancer,labs,benchmark)
load(['./data/',cancer,'/',cancer,'_gene_expression_RNAseq.mat'])
%gene_labs=xlsread(['./data/intogen_label/intogen_',cancer,'.xlsx']);
gene=mRNA_gene;
gene=sortrows(gene,1);

load('adj_m.mat')
ex=ones(length(gene),1);
d=0.85;
r=Generank(adj_m,ex,d);%得到generank值

r_ppi=num2cell(r);
relt_ppi=[gene r_ppi];

for j = 1:length(benchmark)
    gene_labs = labs(:,j);
    [h_ppi,p_ppi]=rank_test(relt_ppi,gene_labs);%检验
    relt_ppi=sortrows(relt_ppi,-2);%进行降序排列
    rank=(1:length(relt_ppi))';
    rank_ppi=[relt_ppi num2cell(rank)];
    rank_ppi=sortrows(rank_ppi,1);%最终的序列特征
    save(['./output/',cancer,'/',cancer,'_rank(',benchmark{j},')_ppi.mat'], 'h_ppi','p_ppi','rank_ppi','gene_labs','-v7.3')
end
end