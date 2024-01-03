%PAAD 转录组表达数据
clear
fid = fopen('HiSeqV2_PANCAN');
nCols = 184; %sample数+1
format = repmat ('%q', [1 nCols]);
format=['%s' format];
format=[format '%*[^\n]'];
ID = textscan (fid, format,1,'HeaderLines',0);
c = textscan (fid, format,'HeaderLines',1);
fclose(fid);
ID=ID(1,[2:nCols]);
gene=c{1,1};
j=1;
for i=2:nCols
dataed(:,j)=c{1,i};
j=j+1;
end
[row,col]=size(dataed);
b=str2num(char(dataed));
data=reshape(b,row,col);
save('PAAD_gene_expression_RNAseq.mat', 'gene', 'data', 'ID','-v7')