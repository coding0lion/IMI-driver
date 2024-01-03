function pair = GR_CMI(ma,gene,phenotype,permute_type,p_cut,candidate_pair)
%%%%%%%%%%该程序用于计算基因A与表型的相关性对基因B的依赖性，所用的方法是条件互信息
%%%%%%几个输入参数如下：
%%%%%%ma:m*n:转录组矩阵数据，一行为一个基因，一列为一个样本
%%%%%%gene:m*1：每一行基因对应的gene symbol
%%%%%%phenotype：n*1：n个样本的表型(0或者1)
%%%%%permute_type:基因依赖关系对的时候p值的计算方法。1 (直接采样得到背景分布，然后估计出每对基因的基因依赖值的p值) or 2 (对每对基因，使用permutation的方法来计算当前这一对基因对的p值).
%%%%%当permute_type=1时，计算会比较快（不是那么准确），当permute_type=2时，计算会比较慢（更准确）
%%%%%p_cut:p-value的阈值，用于决定当前的基因关系对是否显著
%%%%candidate_pair：会计算的候选基因关系对，当该参数为默认值时（不对该参数进行赋值），程序会计算前面的m个基因的所有可能的关系对
%%%对于candidate_pair，个人建议以PPI关系对作为输入，这样可以减少计算量，同时基于PPI关系对，会一定程度避免假阳性结果
%%%即只有在PPI中有交互的关系对（A-B）才计算A对B以及B对A的依赖关系

%%输出：Pair: num*4，num个显著的基因依赖关系，第一列和第二列为基因对，第三列为CMI值，第四列为p值
%%表明的是第二列基因对表型的互信息是以来对第一列的基因

%%同时程序也会产生一个CMI_net.mat文件，改文件也存储了Pair变量。

if (nargin<5)
    error('There should be at least five input parameters');
end

n=length(gene);

if (nargin==5)
    cou=0;
    for i=1:n
        for j=(i+1):n
            cou=cou+1;
            candidate_pair{cou,1}=gene{i,1};
            candidate_pair{cou,2}=gene{j,1};
            loc(cou,1)=i;
            loc(cou,2)=j;
        end
    end
else
    num=length(candidate_pair(:,1));
    flag=ones(num,1);
    loc=zeros(num,2);
    index=(1:length(gene))';
    for i=1:num
        lo=strcmpi(gene,candidate_pair{i,1});
        if (sum(lo)==1)
            loc(i,1)=index(lo,1);
            lo=strcmpi(gene,candidate_pair{i,2});
            if (sum(lo)==1)
                loc(i,2)=index(lo,1);
            else
                flag(i,1)=0;
            end
        else
            flag(i,1)=0;
        end
    end
    candidate_pair=candidate_pair(flag==1,:);
    loc=loc(flag==1,:);
end


path(path,'.\MI_tool');

if (permute_type==1)
    w=length(phenotype);
    s=fix(0.35*w);
    e=w-s+1;
    n=length(gene);
    
    OV_CMI=zeros(100000,1);
    for i=1:100000
        index=randperm(n)';
        xx=(ma(index(1,1),:))';
        yy=(ma(index(2,1),:))';
        
        yy=yy-median(yy);
        yy(yy<=0)=0;
        yy(yy>0)=1;
        
        tri=[xx yy phenotype];
        tri=sortrows(tri,1);
        
        
        h1=mutualInformation(tri(1:s,2),tri(1:s,3)); %?°35%????????
        h2=mutualInformation(tri(e:w,2),tri(e:w,3)); %?ó35%????????
        
        h=h2-h1;  %CMI
        OV_CMI(i,1)=h;
    end
    pd = fitdist(OV_CMI,'Normal');
    
    n=length(candidate_pair);
    cou=0;
    for i=1:n
        xx=(ma(loc(i,1),:))';
        yy=(ma(loc(i,2),:))';
        yy=yy-median(yy);
        yy(yy<=0)=0;
        yy(yy>0)=1;
        
        tri=[xx yy phenotype];
        tri=sortrows(tri,1);
        
        h1=mutualInformation(tri(1:s,2),tri(1:s,3)); %?°35%????????
        h2=mutualInformation(tri(e:w,2),tri(e:w,3)); %?ó35%????????
        
        h=h2-h1;
        
        p=(1-cdf(pd,abs(h)))*2;
        if (p<=p_cut)
            cou=cou+1;
            pair{cou,1}=candidate_pair{i,1};
            pair{cou,2}=candidate_pair{i,2};
            pair{cou,3}=h;
            pair{cou,4}=p;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%
        xx=(ma(loc(i,2),:))';
        yy=(ma(loc(i,1),:))';
        yy=yy-median(yy);
        yy(yy<=0)=0;
        yy(yy>0)=1;
        
        tri=[xx yy phenotype];
        tri=sortrows(tri,1);
        
        h1=mutualInformation(tri(1:s,2),tri(1:s,3)); %?°35%????????
        h2=mutualInformation(tri(e:w,2),tri(e:w,3)); %?ó35%????????
        
        h=h2-h1;
        
        p=(1-cdf(pd,abs(h)))*2;
        if (p<=p_cut)
            cou=cou+1;
            pair{cou,1}=candidate_pair{i,2};
            pair{cou,2}=candidate_pair{i,1};
            pair{cou,3}=h;
            pair{cou,4}=p;
        end
    end
else
    w=length(phenotype);
    s=fix(0.35*w);
    e=w-s+1;
    n=length(candidate_pair);
    cou=0;
    for i=1:n
        xx=(ma(loc(i,1),:))';
        yy=(ma(loc(i,2),:))';
        yy=yy-median(yy);
        yy(yy<=0)=0;
        yy(yy>0)=1;
        
        tri=[xx yy phenotype];
        tri=sortrows(tri,1);
        
        h1=mutualInformation(tri(1:s,2),tri(1:s,3)); %?°35%????????
        h2=mutualInformation(tri(e:w,2),tri(e:w,3)); %?ó35%????????
        
        h=h2-h1;
        
        null_cou=0;
        for j=1:1000
            lo=randperm(w)';
            tri=[xx(lo,1) yy phenotype];
            tri=sortrows(tri,1);
            
            h1=mutualInformation(tri(1:s,2),tri(1:s,3)); %?°35%????????
            h2=mutualInformation(tri(e:w,2),tri(e:w,3)); %?ó35%????????
            
            null_h=h1-h2;
            if (abs(null_h)>=abs(h))
                null_cou=null_cou+1;
            end
        end
        
        p=null_cou/1000;
        
        if (p<=p_cut)
            cou=cou+1;
            pair{cou,1}=candidate_pair{i,1};
            pair{cou,2}=candidate_pair{i,2};
            pair{cou,3}=h;
            pair{cou,4}=p;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%
        xx=(ma(loc(i,2),:))';
        yy=(ma(loc(i,1),:))';
        yy=yy-median(yy);
        yy(yy<=0)=0;
        yy(yy>0)=1;
        
        tri=[xx yy phenotype];
        tri=sortrows(tri,1);
        
        h1=mutualInformation(tri(1:s,2),tri(1:s,3)); %?°35%????????
        h2=mutualInformation(tri(e:w,2),tri(e:w,3)); %?ó35%????????
        
        h=h2-h1;
        
        null_cou=0;
        for j=1:1000
            lo=randperm(w)';
            tri=[xx(lo,1) yy phenotype];
            tri=sortrows(tri,1);
            
            h1=mutualInformation(tri(1:s,2),tri(1:s,3)); %?°35%????????
            h2=mutualInformation(tri(e:w,2),tri(e:w,3)); %?ó35%????????
            
            null_h=h1-h2;
            if (abs(null_h)>=abs(h))
                null_cou=null_cou+1;
            end
        end
        
        p=null_cou/1000;
        
        if (p<=p_cut)
            cou=cou+1;
            pair{cou,1}=candidate_pair{i,2};
            pair{cou,2}=candidate_pair{i,1};
            pair{cou,3}=h;
            pair{cou,4}=p;
        end
    end
end

end



