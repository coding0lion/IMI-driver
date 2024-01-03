%tf-target��ks��֤
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
%���ǩ
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
%% ɾ�����ڵĻ����
location=[];
for j=1:2 %��һ�к͵ڶ��ж���һ��
    for i=1:size(pair_tf,1)
    %ת¼���еĻ����������methy1_gene�У�ɾ��
    index=find(strcmp(pair_tf{i,j},gene)==1); %��i���еĻ����ڼ׻�����indexλ��
    if isempty(index)
        loc=i;
        location=[location;i]; %��¼ɾ��λ��
    end
    end
end
pair_tf(location,:)=[];
%%
Pair_tf=pair_tf(:,1:2);
Pair_tf=cellstr(Pair_tf);
adj_m=create_adjacency_matrix(mRNA_gene,Pair_tf);%�õ��ڽӾ���
%���Խ�����Ϊ0
adj_m=adj_m-diag(diag(adj_m));%��Ϊ�Խ���Ϊ��ľ���
r=Generank(adj_m,ex,d);%�õ�generankֵ

save('tf_net_adj_m.mat','adj_m')

r_tf=num2cell(r);
relt_tf=[mRNA_gene r_tf];
[h_tf,p_tf]=rank_test(relt_tf,gene_labs);%����

rank=(1:length(relt_tf))';
rank_tf=[relt_tf num2cell(rank)];
rank_tf=sortrows(rank_tf,1);%���յ���������
save('ngc_rank_tf.mat', 'h_tf','p_tf','rank_tf','gene_labs','-v7.3')


% [ii, jj] = find(adj_m); % row and col indices of connections 
% y = accumarray(ii, jj-1 , [], @(x){sort(x.')}); % get all nodes connected to each node, 
% node=[0:1:length(gene)-1]';
% 
% %��Ϊ�ڽӱ�
% fid=fopen('sub_tf.txt','wt');
% for i=1:size(gene,1)%��
%     b = node(i);
%     fprintf(fid,'%.0f ',b);
%    if i<=size(y,1)
%     for j=1:size(y{i},2)%��
%     a = y{i}(j);
% %     a = cell2mat(a);
%     fprintf(fid,'%.0f ',a);
%     end
%    end
%     fprintf(fid,'\n');%�ӻ��з�
% end