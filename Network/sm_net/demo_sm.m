function demo_sm(cancer)
load(['./data/',cancer,'/',cancer,'_somatic_mutation.mat']);%miRNA数据
load(['./data/',cancer,'/',cancer,'_gene_expression_RNAseq.mat'])%load基因表达数据


data=zeros(length(mRNA_gene),size(sm_patient_data,2));
gene=sortrows(mRNA_gene,1);
sm2mRNA_index=[];
for i=1:length(gene)
    index=find(strcmp(sm_gene(:,1),gene{i}));
    if index
    data(i,:)=sm_patient_data(index,:);
    end
end
    
%通过矩阵计算余弦相似度
%A（维度M×N）和B（N×P）。实际上，它们是向量的集合 - A中的行向量，B中的列向量。
%获得每对a和b的余弦相似度得分，其中a是来自矩阵A的向量（行），而b是向量列）从矩阵B开始。
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
a_row_adj=reshape(adj,1,length(adj)*length(adj));
[val,index]=sort(a_row_adj,'descend');
%选取前5%；
%n=round(length(adj)*length(adj)/4*0.05);
%adj_new=zeros(20530,20530);
%adj_new(index(1:n))=1;
%adj_new=adj_new'+adj_new;

 n=10;%选取前n个最相关
 for i=1:length(adj)
 temp_val=adj(:,i);
 [B,index] = sortrows(temp_val,-1);
if index(B(1:n)~=0)
 adj_new(index(B(1:n)~=0),i)=B(B(1:n)~=0);
end
 end
    
%%
%保存为邻接表
[col,row]=find(adj_new>0);
%关系对
val=adj_new(find(adj_new>0));
pairs=[row,col,val];
unique(pairs);

[ii, jj] = find(adj_new); % row and col indices of connections 
y = accumarray(ii, jj-1 , [], @(x){sort(x.')}); % get all nodes connected to each node, 
node=[0:1:length(gene)-1]';

%存为邻接表
fid=fopen(['./output/',cancer,'/','sub_',cancer,'_sm.txt'],'wt');

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
end
