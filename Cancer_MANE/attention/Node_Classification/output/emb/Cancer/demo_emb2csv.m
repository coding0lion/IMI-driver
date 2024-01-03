[~,word]=xlsread("cancer_29.xlsx");
for i = 1:length(word) 
cancer = word{i};
emb2csv(cancer);
end
