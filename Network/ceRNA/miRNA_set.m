%求gene对应的miRAN的集合
function [g2mi]=miRNA_set(new_miR,new_ge)
    ge_ulist=unique(new_ge);
    g2mi{length(ge_ulist),1}={};
    tic
    for ii=1:length(ge_ulist)
        g2mi{ii}= unique(new_miR(find(strcmp(ge_ulist{ii},new_ge)==1),1)');
    end
    toc
    %save('g2mi.mat','g2mi');
end