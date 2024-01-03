%mapping基因表达数据和临床数据
%输入参数说明
%dmfs_e临床数据的预后情况
%dmfs_time临床数据存活时间
%sample_ID临床数据中的样本ID
%ID基因表达数据的样本id
%ma基因表达数据的各基因的表达情况（就是data，老师代码里统一命名为了ma方便使用）
function [ma ,phenotype]=mapping_met(dmfs_e,dmfs_time,sample_ID,ID,ma)
%%
%进行将基因表达数与临床数据的样本进行匹配
ID=reshape(ID,length(ID),1);%基因表达数据中的样本的ID
%%
%标记样本数据的预后情况（四种情况）1825为五年
for k=1:length(sample_ID)
    if dmfs_e(k,1)==0&&dmfs_time(k,1)<1825%预后未知的情况
        yu(k,1)=2;
    elseif dmfs_e(k,1)==1&&dmfs_time(k,1)<1825
        yu(k,1)=1;
    elseif dmfs_e(k,1)==0&&dmfs_time(k,1)>=1825
        yu(k,1)=0;
    else
        yu(k,1)=0; 
    end
end
unkonw=find(yu==2);%记录预后未知的样本的索引
bad=find(yu==1);
good=find(yu==0);
%%
%去除不合格的数据，两类
sample_ID(unkonw)={0};
cont=zeros(length(ID),1);%与基因表达数据中样本数长度一致
phenotype(:,1) = dmfs_time;
phenotype(:,2) = cont;
for i=1:length(ID)
    for j=1:length(sample_ID)%临床数据中样本的ID长度
        if  strcmp(ID{i,1},sample_ID{1,j})
            cont(i,1)=j;
            if find(j==bad)
                phenotype(i,2)=1;%标记预后情况
            end
        end
    end
end
del_sample=find(cont==0);
cont(del_sample)=[];
phenotype(del_sample,:)=[];%删除不合格数据后的预后情况
%%
ma(:,del_sample)=[];
end