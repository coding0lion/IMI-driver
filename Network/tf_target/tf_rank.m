function tf_rank(cancer,labs,benchmark)
%tf-target��ks��֤
load(['./data/',cancer,'/',cancer,'_gene_expression_RNAseq.mat'])
%gene_labs=xlsread(['./data/intogen_label/intogen_',cancer,'.xlsx']);
gene=mRNA_gene;
gene=sortrows(gene,1);

load('tf_net_adj_m.mat')
ex=ones(length(gene),1);
d=0.85;
r=Generank(adj_m,ex,d);%�õ�generankֵ


r_tf=num2cell(r);
relt_tf=[mRNA_gene r_tf];
for j = 1:length(benchmark)
    gene_labs = labs(:,j);
    [h_tf,p_tf]=rank_test(relt_tf,gene_labs);%����
    rank=(1:length(relt_tf))';
    rank_tf=[relt_tf num2cell(rank)];
    rank_tf=sortrows(rank_tf,1);%���յ���������
    save(['./output/',cancer,'/',cancer,'_rank(',benchmark{j},')_tf.mat'], 'h_tf','p_tf','rank_tf','gene_labs','-v7.3')
end
end