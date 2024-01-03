[v w]=xlsread("cancer.xlsx");
cancer_list=w(:,1);
addpath(genpath('output'));
for i=1:length(v)
cancer=cancer_list{i};
netname='co';
load(['output/',cancer,'/',netname,'_net'])
pairs2adjtable(cancer,netname,pair_co);

netname='Meco';
load(['output/',cancer,'/',netname,'_net'])
pairs2adjtable(cancer,netname,pair_Meco);

netname='ceRNA';
load(['output/',cancer,'/',netname,'_net'])
pairs2adjtable(cancer,netname,pair_ceRNA);

netname='CMI';
load(['output/',cancer,'/',netname,'_net'])
pairs2adjtable(cancer,netname,pair_cmi);
end
