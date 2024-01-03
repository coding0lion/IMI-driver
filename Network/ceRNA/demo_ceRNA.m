function demo_ceRNA(cancer)
load(['./data/',cancer,'/',cancer,'_miRNA.mat']);%miRNA数据
load(['./data/',cancer,'/',cancer,'_gene_expression_RNAseq.mat'])%load基因表达数据


data=mRNA_patient_data;
gene=mRNA_gene;
ID=mRNA_patient_ID;
miRNA_data=miRNA_patient_data;
miRNA_ID=miRNA_patient_ID;

[mirna cm]=xlsread(['./data/',cancer,'/',cancer,'miRNAid.xlsx']);
cm=cm(2:end,1);
miRNA_name=cm;%转化类型后miRNA的名字


% load('change_name.mat');
%cm=readtable(".\change_name.csv",'ReadVariableNames',false);
% miRNA_name=cm.Var1;%转化类型后miRNA的名字
[new_miR,new_ge]=new_miRNA_gene(miRNA_name, miRNA_data);%得到根据miRNA和转录组数据的gene和miRNA对应关系
 ge_ulist=unique(new_ge);
 mi_ulist=unique(new_miR);
%%
%进行样本mapping
[ind_ID,ind_mi]=mapping_sample(ID,miRNA_ID);
miRNA_data=miRNA_data(:,ind_mi);
data=data(:,ind_ID);
%%
%所有癌症转录组对应的基因都是一样的，因此这里可以存为.mat文件
%求基因对应的miRNA集合
[g2mi]=miRNA_set(new_miR,new_ge);
%load('g2mi.mat');%基因对应的miRNA集合（根据TCGA,miRNA数据得到的集合）

%%
pair_nums=0;
U_len=length(mi_ulist);%全集长度,修改后为除去存在一半为NA的miRNA数目

%%
% cor_gene_name={};
% 
% for i=1:length(miRNA_name)
% [co_m_g,p_m_g]=corr(miRNA_data(i,:)',data'); %第i个miRNA和基因求相关系数
% %co的col表示与每个基因的相关系数
% %找出相关性为负值，且p值小于0.05的基因对
% %cor_gene_name基因集合
% pdco=(co_m_g<0)+(p_m_g<0.05);
% cor_gene_index=find(pdco>=2);
% cor_gene_name=[cor_gene_name;gene(find(pdco>=2))];%基因名
% cor_gene_name=unique(cor_gene_name);
% end
[mi2g]=miRNA_set(new_ge,new_miR);
[cor_gene_name]=miRNA_gene_corr(gene,miRNA_name,mi_ulist,mi2g,data,miRNA_data);


%%
%
candi_gene_num=length(cor_gene_name);
tic
for j=1:candi_gene_num-1
    %%
    indx_A=find(strcmp(cor_gene_name{j},ge_ulist)==1);
    if length(indx_A) %若A集合不为空则计算B基因对应的miRNA集合
    set_A=[];  
    set_A=g2mi{indx_A,1};%得到该基因对应的miRNA的集合
        for k=j+1:candi_gene_num
            %%
            indx_B=find(strcmp(cor_gene_name{k},ge_ulist)==1);
            if length(indx_B)%如果B基因对应的miRNA集合不为空，则求AB的交集
                set_B=[];
                set_B=g2mi{indx_B,1};%得到该基因对应的miRNA的集合
                inter_AB=intersect(set_A,set_B);%求AB的交集
                if length(inter_AB) %若交集不为空,则进行超几何分布检验
                    %p=1-hygecdf(交集数量-1,全集数量,集合A数量,集合B数量)；
                    p_hyp=1-hygecdf(length(inter_AB)-1,U_len,length(set_A),length(set_B));
                    if p_hyp<0.05
                        [co_g_g,p_g_g]=corr(data(j,:)',data(k,:)');
                        if co_g_g>0&&p_g_g<0.05
                            %记录基因对
                            pair_nums=pair_nums+1;
                            pair_ceRNA{pair_nums,1}=cor_gene_name{j};
                            pair_ceRNA{pair_nums,2}=cor_gene_name{k};
                            pair_ceRNA{pair_nums,3}=co_g_g;
                        end
                    end
                end
            end
        end
    end
end
 toc
save(['./output/',cancer,'/ceRNA_net.mat'],'pair_ceRNA','-v7.3'); 
end
