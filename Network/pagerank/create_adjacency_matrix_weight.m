%修改后，可以返回0/1邻接矩阵和有权重的邻接矩阵
function [adj_m,adj_m_weight]=create_adjacency_matrix(gene,Pair)
% n=length(gene);
% temp=zeros(n,n);
% tic
% for i=1:length(Pair)
% col=find(strcmp(Pair{i,1},gene)==1);
% row=find(strcmp(Pair{i,2},gene)==1);
% temp(col,row)=1;
% temp(row,col)=1;
% end
% toc
% pa=Pair(1:end,1:2);
% ge=unique(pa);
% set_set=setdiff(gene,ge);
% for j=1:length(set_set)
%     x=find(strcmp(set_set{j},gene)==1);
%     temp(x,x)=1;
% end
tic
a=Pair(:,1:2);
weight=Pair(:,3);
weight=cell2mat(weight);
%补为对称矩阵
aa=[a(:,2),a(:,1)];
haved=unique(a);
isolate=setdiff(gene,haved);%得到孤立点
isolate_pair=[isolate,isolate];
final_pair=[a;aa;isolate_pair];

[~,~,k]=unique(final_pair);
a=reshape(k,[],2);
b=unique(a);
c=find(b);
%生成数字的关系对
for i=1:length(b)
idx=find(a==b(i));
a(idx)=c(i);
end
e=accumarray(a,1);
%%
%可能会出现对应不全，倒置转置后维度不同（不过此方法速度很快）
% e1=e-diag(diag(e));%变为对角线为零的矩阵
% e=e+e1';%补全对称矩阵
%%
adj_m=e;
%权重矩阵
pair_weight=a(1:length(weight),:);
e_weight=zeros(length(gene),length(gene));
nn=(pair_weight(:,2)-1).*20530+pair_weight(:,1);
e_weight(nn)=weight;
e_weight=e_weight+e_weight';
adj_m_weight=e_weight;
toc
end