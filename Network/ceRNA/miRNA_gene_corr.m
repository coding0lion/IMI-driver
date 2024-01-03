function [cor_gene_name]=miRNA_gene_corr(gene,miRNA_name,mi_ulist,mi2g,data,miRNA_data)
%miRNA_name已经删除了NA超过一半的
tic
cor_gene_name={};
n=length(mi_ulist);
    for k=1:n
        index_mi=find(strcmp(mi_ulist{k},miRNA_name)==1);
        mi_data=miRNA_data(index_mi,:)';%提取出miRNA数据
        del_index=find(mi_data==-1);%删除NA的数据
        mi_data(del_index,:)=[];
        if ~isempty(mi_data)
        if ~isempty(mi2g{k,1})
            index_ge=[];
            for j=1:length(mi2g{k,1})
                index_ge=[index_ge;find(strcmp(mi2g{k,1}(j),gene)==1)];%miRNA对应的基因在转录组数据上的索引
            end
            gene_data=data(index_ge,:)';%提取miRNA对应基因的转录组数据
            gene_data(del_index,:)=[];%删除miRNA为NA对应位置的数据
            [co_m_g,p_m_g]=corr(mi_data,gene_data,'Type','Spearman');%计算相关性
            pdco=(co_m_g<0)+(p_m_g<0.05);%负相关且p小于0.05
            cor_gene_index=find(pdco>=2);
            cor_gene_name=[cor_gene_name;gene(find(pdco>=2))];%基因名
            cor_gene_name=unique(cor_gene_name);
        end
        end
    end
toc     
end  