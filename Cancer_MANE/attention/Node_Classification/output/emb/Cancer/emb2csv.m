function emb2csv(cancer)
data=load(['Embedding_concatenated5_epoch_50_' cancer  '.txt']);
data=sortrows(data,1);
gapd=size(data,1)/20530;
if (gapd~=1)
    new_data=[];
    for i=1:gapd:size(data,1)
        temp=[];
        temp=reshape(data(i:i+gapd-1,:),1,size(data,2)*gapd);
        new_data=[new_data;temp];
    end
    isnan_index=find(isnan(new_data(1,:)));
    new_data(:,isnan_index)=[];
    gene_index = new_data(:,1);
    new_data(:,1)=[];
    new_data=new_data(:,gapd:end);
    new_data = [gene_index new_data];
else
    new_data = data;
end
save_path = ['../xgboost/Embedding_5_64_50_specific_mut_intogen_' cancer  '.csv'];
writetable(table(new_data),save_path);
end
