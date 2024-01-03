%tf-target的ks验证
tempdata=readtable('tf-targetdata.txt', 'ReadVariableNames',false);
pair_tf=[tempdata.Var1,tempdata.Var2];
load('BRCA_gene_expression_RNAseq.mat')
gene=mRNA_gene;
% load('cgc.mat')
% cgc_gene=cgc.GeneSymbol;
% cgc_lab=cgc.Tier;

[~,ngc_gene]=xlsread('BRCA.xlsx');
ngc_gene=unique(ngc_gene);
ngc_lab=ones(length(ngc_gene),1);

know_gene=ngc_gene;
know_lab=ngc_lab;

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
gene_labs(gene_labs==2)=-1;
ex=ones(length(mRNA_gene),1);
d=0.85;
%% 删除不在的基因对
location=[];
for j=1:2 %第一列和第二列都比一下
    for i=1:size(pair_tf,1)
    %转录组中的基因如果不在methy1_gene中，删除
    index=find(strcmp(pair_tf{i,j},gene)==1); %第i对中的基因在甲基化的index位置
    if isempty(index)
        loc=i;
        location=[location;i]; %记录删除位置
    end
    end
end
pair_tf(location,:)=[];
%%
Pair_tf=pair_tf(:,1:2);
Pair_tf=cellstr(Pair_tf);
adj_m=create_adjacency_matrix(mRNA_gene,Pair_tf);%得到邻接矩阵
%将对角线设为0
adj_m=adj_m-diag(diag(adj_m));%变为对角线为零的矩阵
r=Generank(adj_m,ex,d);%得到generank值

save('tf_net_adj_m.mat','adj_m')

r_tf=num2cell(r);
relt_tf=[mRNA_gene r_tf];
[h_tf,p_tf]=rank_test(relt_tf,gene_labs);%检验

rank=(1:length(relt_tf))';
rank_tf=[relt_tf num2cell(rank)];
rank_tf=sortrows(rank_tf,1);%最终的序列特征
save('ngc_rank_tf.mat', 'h_tf','p_tf','rank_tf','gene_labs','-v7.3')


% [ii, jj] = find(adj_m); % row and col indices of connections 
% y = accumarray(ii, jj-1 , [], @(x){sort(x.')}); % get all nodes connected to each node, 
% node=[0:1:length(gene)-1]';
% 
% %存为邻接表
% fid=fopen('sub_tf.txt','wt');
% for i=1:size(gene,1)%行
%     b = node(i);
%     fprintf(fid,'%.0f ',b);
%    if i<=size(y,1)
%     for j=1:size(y{i},2)%列
%     a = y{i}(j);
% %     a = cell2mat(a);
%     fprintf(fid,'%.0f ',a);
%     end
%    end
%     fprintf(fid,'\n');%加换行符
% end