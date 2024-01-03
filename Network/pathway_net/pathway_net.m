
load('BRCA_gene_expression_RNAseq.mat');

gene=sort(gene);

[data,str]=xlsread('pp.xlsx');
pathway=zeros(length(gene),size(str,1));
for i=1:length(gene)
    for j=1:size(str,1)
    if find(strcmp(gene{i},str(j,:))==1)
        pathway(i,j)=1;
    end
    end
end
%测试
AA=pathway(85,:);

BB =pathway(479,:);

dot(AA,BB)/(sqrt(sum(AA.*AA))*sqrt(sum(BB.*BB)));
%通过矩阵计算余弦相似度
%A（维度M×N）和B（N×P）。实际上，它们是向量的集合 - A中的行向量，B中的列向量。
%获得每对a和b的余弦相似度得分，其中a是来自矩阵A的向量（行），而b是向量列）从矩阵B开始。
A=pathway;
B =A';
C = A * B;
normA = sqrt(sum(A .^ 2, 2)); 
normB = sqrt(sum(B .^ 2, 1)); 
C = bsxfun(@rdivide, bsxfun(@rdivide, A * B, normA), normB);
C=C-diag(diag(C));%变为对角线为零的矩阵

%得到邻接矩阵
adj=C;
adj(isnan(adj))=0;
adj(adj>0)=1;

[col,row]=find(C>0);
%关系对
val=C(find(C>0));
pairs=[row,col,val];
unique(pairs)

[ii, jj] = find(adj); % row and col indices of connections 
y = accumarray(ii, jj-1 , [], @(x){sort(x.')}); % get all nodes connected to each node, 
node=[0:1:length(gene)-1]';

fid=fopen('sub_pathway.txt','wt');
for i=1:size(gene,1)%行
    b = node(i);
    fprintf(fid,'%.0f ',b);
    for j=1:size(y{i},2)%列
    a = y{i}(j);
%     a = cell2mat(a);
    fprintf(fid,'%.0f ',a);
    end
    fprintf(fid,'\n');%加换行符
end