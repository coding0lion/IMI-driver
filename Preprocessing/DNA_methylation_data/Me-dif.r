## Mythelationd difference
library(ChAMP)
library(data.table)
library(downloader)
library(writexl)
library(R.utils)
#download data
for(i in 1:29){
cancerlist<-c('ACC','BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC',
       'ESCA', 'HNSC', 'KICH', 'KIRC', 'KIRP',
        'LGG', 'LIHC', 'LUAD', 'LUSC', 'PAAD',
       'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'STAD', 'TGCT',
       'THCA', 'THYM', 'UCEC', 'UCS', 'UVM')
cancer<-cancerlist[i]
url<-paste("https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.",cancer,".sampleMap%2F",cancer,"_clinicalMatrix",sep = "")
download.file(url,destfile =paste("/out/data/",cancer,"clinicalMatrix",sep = ""),mode='wb',method = "curl")
data_samp<-read.delim(paste("/out/data/",cancer,"clinicalMatrix",sep = ""))
data_beta<-fread(file=paste("/out/data/",cancer,"Methylation450",sep = ""))
data_samp<-data_samp[,c("sampleID","sample_type")]
data_samp$sampletype<-ifelse(data_samp$sample_type=="Primary Tumor","Tumor","Normal")
data_samp<-data_samp[,c(1,3)]
ID<-intersect(data_samp$sampleID,colnames(data_beta))
pdata<-data_samp[data_samp$sampleID %in% ID,]
names(pdata)<-c("sample_name","sample_group")
pdata$sample_group<-ifelse(pdata$sample_group=="Tumor","T","C")
data_beta<-as.data.frame(data_beta)
row.names(data_beta)<-data_beta[,1]
data_beta<-data_beta[,-1]
data_beta<-data_beta[,ID] 
data_beta[is.na(data_beta)==TRUE] <- 0.0001  
sum(is.na(data_beta))
data_order<-as.matrix(data_beta)
myload<-champ.filter(beta=data_order,pd=pdata)
save(myload,file=paste("/out/myload/",cancer,"_methload.rds",sep = ""))
rm(data_beta) 
rm(data_samp)
rm(myload)
rm(pdata)
load(paste("/out/myload/",cancer,"_methload.rds",sep = ""))
myDMP<-champ.DMP(beta=myload$beta,pheno = myload$pd$sample_group)
result<-myDMP[[1]][1]
result$sample<-row.names(result)
Methylationid<-fread(file = "F:/university/research/drivergene/pancancer/co-MEt/probeMap_illuminaMethyl450_hg19_GPL16304_TCGAlegacy")#导入xena上的mapping数据，该数据不变
Methylationpr<-fread(file = "F:/university/research/drivergene/pancancer/co-MEt/GPL18809-31921.txt")#导入知道promoter信息的数据
oo="Dif_Methylation.xlsx"
#调用函数
path="/Methylation/"
setwd(path)
source("Me_gene.r")
Me_gene(cancer,result,oo)
rm(myload)
rm(myDMP)
}
