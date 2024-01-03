%PAAD 体细胞突变数据
clear
fid = fopen('PAAD_mc3_gene_level.txt');
nCols = 178; %sample数+1
format = repmat ('%q', [1 nCols]);
format=['%s' format];
format=[format '%*[^\n]'];
ID = textscan (fid, format,1,'HeaderLines',0);
c = textscan (fid, format,'HeaderLines',1);
fclose(fid);
gene=c(1,1);
dataed=c(1,[2:nCols]);
ID=ID(1,[2:nCols]);
 %   for (n=1:1:size(h,2))
 %    dataed(:,n)=reshape(h{n}, 2238,1);
 %   end
save('PAAD_somatic_mutation.mat', 'gene', 'dataed', 'ID','-v7.3')