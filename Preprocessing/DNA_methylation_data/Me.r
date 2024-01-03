
library(downloader)
library(data.table)
library(writexl)
library(R.utils)
#Download
cancerlist<-c("BLCA","GBM","HNSC","KIRC","KIRP","LIHC","LUAD","LUSC","OV","PAAD","PRAD","READ","STAD","THCA","THYM","ACC","CHOL","CESC","COAD","UCEC","ESCA","KICH","DLBC","LGG","SKCM","MESO","UVM","PCPG","SARC","TGCT","UCS")
for(i in 1:31){
cancer<-cancerlist[i]
#Read other files
Methylationid<-fread(file = "/home/probeMap_illuminaMethyl450_hg19_GPL16304_TCGAlegacy")#xena mapping data
Methylationpr<-fread(file = "/home/GPL18809-31921.txt")
url<-paste("https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.",cancer,".sampleMap%2FHumanMethylation450.gz",sep = "")
download.file(url,destfile =paste("/home/hjm/Methydata/data/",cancer,"Methylation450.gz",sep = ""),mode='wb',method = "curl")
gunzip(paste("/home/hjm/Methydata/data/",cancer,"Methylation450.gz",sep = ""))
Methylation<-fread(file=paste("/home/hjm/Methydata/data/",cancer,"Methylation450",sep = ""))
oo="Me.xlsx"
path="/home/=Methydata/"
setwd(path)
source("Me_gene.r")
Me_gene(cancer,Methylation,k)
}
