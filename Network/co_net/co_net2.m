%基因表达数据预计15mins
%load GSE2034_ma2;
%甲基化数据或者基因表达数据
%修改了一些错误的第二版本
[gene,data]=loda_data_co(cancer)
ma2=data;
ma2_geneName=gene;
zee=find(ma2(:,1)==0);
azee=find(all(ma2==0,2));
n=length(ma2_geneName);
index=(1:n)';   %n个基因的位置
tic
num=1;%用于记录非零的基因
for i=1:n
    [co,p]=corr(ma2(i,:)',ma2'); %第i个基因和其他所有基因求相关系数
    res=[index co' p'];
    res=sortrows(res,-2); %按照相关系数从大到小排序
    del=find(isnan(res(:,2))==1);%找到NAN的位置
    if length(del)==length(res)%如果这行全为零则跳过
        continue;
    end
    res(del,:)=[];
    %pair(((i-1)*5+1):i*5,1)=ma2_geneId(i,1);  %当前的基因的ID
    pair_gene(((num-1)*5+1):num*5,1)={ma2_geneName{i,1}}; 
    %pair(((i-1)*5+1):i*5,2)=ma2_geneId(res(2:6,1),1); %最相关的五个基因的ID(因为最相关的是自己，所以是第2个到第六个)
    pair_gene(((num-1)*5+1):num*5,2)={ma2_geneName{res(2:6,1),1}};
    pair_val(((num-1)*5+1):num*5,1:2)=res(2:6,2:3);  %%这5个基因与当前基因的相关系数和p-value 
    num=num+1;
end;
% xlswrite('co_net.xlsx',pair_gene,'Sheet1','A1');
% xlswrite('co_net.xlsx',pair_val,'Sheet1','C1');  %将共表达网络存储到excel文件中
pair_co=[pair_gene,num2cell(pair_val)];

save(['./output/',cancer,'/co_net.mat'], 'pair_co','-v7.3') 
toc;
