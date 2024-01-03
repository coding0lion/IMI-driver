function demo_cmi(cancer)
% load('TCGA_BRAC_mRNA.mat');
[dmfs_e, dmfs_time ,sample_ID ,ID ,data, probeSet]=loda_data(cancer);
%PPI网络数据
load('g_g.mat');

[ma, phenotype]=mapping(dmfs_e,dmfs_time,sample_ID,ID,data);
%进行测试的赋值
% ma=randn(30,length(phenotype));
% gene=probeSet(1:30,1);
% for i=1:1000
%     a=randi(50);
%     b=randi(70);
%     candidate_pair{i,1}=probeSet{a,1};
%     candidate_pair{i,2}=probeSet{b,1};
% end

%离散化，进行互信息计算
% phenotype(1:300,1)=0;
% phenotype(301:593,1)=1;
%%%%phenotype：n*1：n个样本的表型(0或者1)
%%%%%实际运行

gene=probeSet;
%根据ppi网络进行计算
candidate_pair(:,1)=g_g.X1;
candidate_pair(:,2)=g_g.X2;

permute_type=1;
p_cut=0.05;

tic;
%%构建基因依赖网络的主要函数GR_CMI.m
pair_cmi = GR_CMI(ma,gene,phenotype,permute_type,p_cut,candidate_pair);
%pair_cmi = GR_CMI(ma,gene,phenotype,permute_type,p_cut);%如果不依赖于ppi网络，则直接删去candidate_pair
toc;
save(['./output/',cancer,'/CMI_net.mat'],'pair_cmi','-v7.3'); %将结果存储在CMI_net这个文件中

end
