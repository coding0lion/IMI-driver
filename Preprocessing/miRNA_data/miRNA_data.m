%PAAD mRNAÊý¾Ý
clear
fid = fopen('miRNA_HiSeq_gene');
nCols = 183; %%sampleÊý+1
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
save('PAAD_miRNA.mat', 'gene', 'dataed', 'ID','-v7')