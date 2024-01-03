%已做对应，对应为排序后的转录组基因
function diff_transcription(cancer)
load(['./data/',cancer,'/',cancer,'_gene_expression_RNAseq.mat'])
normal_AVG=mean(mRNA_normal_data,2); %对行求均值
patient_AVG=mean(mRNA_patient_data,2); %对行求均值
Dif_transcription=normal_AVG-patient_AVG;
relt=[mRNA_gene num2cell(Dif_transcription)];
[gene m]=sort(mRNA_gene);
Dif_transcription=Dif_transcription(m);
save(['./output/',cancer,'/',cancer,'_Dif_transcription.mat'], 'Dif_transcription','gene','-v7.3')
end