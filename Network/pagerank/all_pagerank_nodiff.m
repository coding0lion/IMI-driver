%rank_coTR为转录组基因共表达网络rank序列，rank_coME为甲基化共表达网络的rank序列
% clear;
% clc;
function all_pagerank(cancer)
load(['./data/',cancer,'/',cancer,'_gene_expression_RNAseq.mat'])
% cgc=readtable('tier1_2.txt');
% save('cgc.mat','cgc');
load('cgc.mat')%所有癌症相同的数据
cgc_gene=cgc.GeneSymbol;
cgc_lab=cgc.Tier;
know_gene=cgc_gene(find(cgc_lab==1));
know_lab=cgc_lab(find(cgc_lab==1));

gene=mRNA_gene;
gene=sortrows(gene,1);
%%
%打标签
for i=1:length(gene)
    index=find(strcmp(gene{i},know_gene)==1);
    if index
    gene_labs(i,1)=cgc_lab(index);
    else
    gene_labs(i,1)=-1;
    end
end
%设置边和差异表达的权重
w=0.6;
e=0.4;
%%
%转录组
load(['./output/',cancer,'/','co_net.mat'])
%load('co_net.mat')
d=0.85;
pair_co=pair_co;
[adj_m,weight]=create_adjacency_matrix_weight(gene,pair_co);
adj_m(logical(eye(size(adj_m))))=0; 
ex=ones(length(gene),1);
r=Generank(adj_m,ex,d);
r_co=num2cell(r);
relt_co=[gene r_co];
[h_coTR,p_coTR]=rank_test(relt_co,gene_labs);%检验
relt_co=sortrows(relt_co,-2);%进行降序排列
rank=(1:length(relt_co))';
rank_coTR=[relt_co num2cell(rank)];
rank_coTR=sortrows(rank_coTR,1);%最终的序列特征
%%
%甲基化基因共表达网络
load(['./output/',cancer,'/','Meco_net.mat'])
% load('Meco_net.mat')%output
load(['./data/',cancer,'/',cancer,'_met.mat'])
% load('BRCA_met.mat')%将病人和正常人分离 data
%[V,W]=xlsread(['./data/',cancer,'/',cancer,'_Dif_Methylation.xlsx']);
% save('BRCA_Dif_methylation','Dif_methylation');
%load('BRCA_Dif_methylation.mat')
%之前说在邻接矩阵中为mRNA的基因，但是还是所有甲基化基因效果更好一点
%但在邻接矩阵中为mRNA的基因加差异化的结果更好
%Dif_methylation=[W(2:end,1) num2cell(V)];
%Dif_methylation=sortrows(Dif_methylation,1);
%dmggene=Dif_methylation(:,1);
%甲基化共表达网络的ks验证
%删除甲基化数据比转录组数据多的基因对
methyl_gene=met_gene;
methyl_gene=intersect(methyl_gene,gene);
methyl_gene=sortrows(methyl_gene,1);
% 删除不在的基因对
location=0;
for j=1:2 %第一列和第二列都比一下
    for i=1:size(pair_co,1)
    %转录组中的基因如果不在methy1_gene中，删除
    index=find(strcmp(pair_co{i,j},methyl_gene)==1); %第i对中的基因在甲基化的index位置
    if isempty(index)
        loc=i;
        location=[location;i]; %记录删除位置
    end
    end
end
location(1,:)=[];
pair_co(location,:)=[];
% mRNAgene与差异甲基化基因对应
ex=ones(length(gene),1);
d=0.85;
Pair_co=pair_co;
[adj_m,weight]=create_adjacency_matrix_weight(gene,Pair_co);
adj_m(logical(eye(size(adj_m))))=0;
r=Generank(adj_m,ex,d);
r_co=num2cell(r);
relt_co=[mRNA_gene r_co];
[h_coME,p_coME]=rank_test(relt_co,gene_labs);%检验
relt_co=sortrows(relt_co,-2);%进行降序排列
rank=(1:length(relt_co))';
rank_coME=[relt_co num2cell(rank)];
rank_coME=sortrows(rank_coME,1);%最终的序列特征
%%
%基因依赖网络（cmi）
load(['./output/',cancer,'/','CMI_net.mat'])
d=0.85;
Pair_cmi=pair_cmi;
[adj_m,weight]=create_adjacency_matrix_weight(gene,Pair_cmi);
adj_m(logical(eye(size(adj_m))))=0;
ex=ones(length(gene),1);
r=Generank(adj_m,ex,d);
r_cmi=num2cell(r);
relt_cmi=[gene r_cmi];
[h_cmi,p_cmi]=rank_test(relt_cmi,gene_labs);%检验
relt_cmi=sortrows(relt_cmi,-2);%进行降序排列
rank=(1:length(relt_cmi))';
rank_cmi=[relt_cmi num2cell(rank)];
rank_cmi=sortrows(rank_cmi,1);%最终的序列特征%%
%%
%ceRNA网络
load(['./output/',cancer,'/','ceRNA_net.mat'])
d=0.85;
Pair_ceRNA=pair_ceRNA;
[adj_m,weight]=create_adjacency_matrix_weight(gene,Pair_ceRNA);
%想将初始值设为表达值乘权重之和
ex=ones(length(gene),1);
adj_m(logical(eye(size(adj_m))))=0;
r=Generank(adj_m,ex,d);
r_ceRNA=num2cell(r);
relt_ceRNA=[gene r_ceRNA];
[h_ceRNA,p_ceRNA]=rank_test(relt_ceRNA,gene_labs);%检验
relt_ceRNA=sortrows(relt_ceRNA,-2);%进行降序排列
rank=(1:length(relt_ceRNA))';
rank_ceRNA=[relt_ceRNA num2cell(rank)];
rank_ceRNA=sortrows(rank_ceRNA,1);%最终的序列特征
%%
load(['./data/',cancer,'/',cancer,'_somatic_mutation.mat']);%miRNA数据
data=zeros(length(mRNA_gene),size(sm_patient_data,2));
sm2mRNA_index=[];
for i=1:length(gene)
    index=find(strcmp(sm_gene(:,1),gene{i}));
    if index
    data(i,:)=sm_patient_data(index,:);
    end
end
A=data;
B =A';
C = A * B;
normA = sqrt(sum(A .^ 2, 2)); 
normB = sqrt(sum(B .^ 2, 1)); 
C = bsxfun(@rdivide, bsxfun(@rdivide, A * B, normA), normB);

C=C-diag(diag(C));%变为对角线为零的矩阵

%得到邻接矩阵
adj=C;
adj(isnan(adj))=0;
C=zeros(length(adj),length(adj));
adj_new=zeros(length(adj),length(adj));
n=10;%选取前n个最相关
for i=1:length(adj)
    temp_val=adj(:,i);
    [B,index] = sortrows(temp_val,-1);
    adj_new(index(B(1:n)~=0),i)=1;
    if index(B(1:n)~=0)
    C(index(B(1:n)~=0),i)=B(1:length(index(B(1:n)~=0)));
    end
end
ex=ones(length(gene),1);
d=0.85;
r=Generank(adj_new,ex,d);%得到generank值
r_sm=num2cell(r);
relt_sm=[gene r_sm];
[h_sm,p_sm]=rank_test(relt_sm,gene_labs);%检验
relt_sm=sortrows(relt_sm,-2);%进行降序排列
rank=(1:length(relt_sm))';
rank_sm=[relt_sm num2cell(rank)];
rank_sm=sortrows(rank_sm,1);%最终的序列特征

h_f=[h_coTR,h_coME,h_cmi,h_ceRNA,h_sm];
p_f=[p_coTR,p_coME,p_cmi,p_ceRNA,p_sm];
rank_f=[rank_coTR(:,3),rank_coME(:,3),rank_cmi(:,3),rank_ceRNA(:,3),rank_sm(:,3)];
rank_i=[rank_coTR(:,2),rank_coME(:,2),rank_cmi(:,2),rank_ceRNA(:,2),rank_sm(:,2)];
save(['./output/',cancer,'/',cancer,'_rank(cgc).mat'],'h_f','p_f','rank_f','gene_labs','rank_i','-v7.3'); 
%xlswrite('sy_rank.xlsx',rank_f,'Sheet1','A1');
