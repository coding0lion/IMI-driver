%PAAD 临床数据
clear
fid = fopen('phenotype_Curated_survival_data.txt');
nCols = 11; %需要修改
format = repmat ('%q', [1 nCols]);
format=['%s' format];
format=[format '%*[^\n]'];
headk = textscan (fid, format,1,'HeaderLines',0);
c = textscan (fid, format,'HeaderLines',1);
head(1,1)=headk(1,3);
head(1,2)=headk(1,4);
fclose(fid);


sample_ID=c{1,1};
PATIENT_ID=c{1,2};
a=c{1,3}(:,1);
for i=1:length(c{1,1})
 x{i,1}=str2num(a{i});
end
for j=1:length(x)
dmfs_e(j,1)=x{j}(1);
end
b=c{1,4}(:,1);
for i=1:length(c{1,1})
 x{i,1}=str2num(b{i});
end
for j=1:length(x)
dmfs_time(j,1)=x{j}(1);
end
save('PAAD_clinic.mat', 'sample_ID','PATIENT_ID', 'head', 'dmfs_e','dmfs_time','-v7')