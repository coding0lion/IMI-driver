function [h,p]=rank_test(relt,labs)
nn=[relt ,num2cell(labs)];
nn=sortrows(nn,-2);%½øĞĞ½µĞòÅÅÁĞ
rank=(1:length(relt))';
top=rank(rank(cell2mat(nn(:,3))~=-1),1);
[h,p] = kstest2(rank,top,'Tail','smaller');
end