
Me_gene<-function(cancer,Methylation,k){
  a<-Methylationpr$ID 
  b<-Methylationpr$REGULATORY_FEATURE_GROUP 
  c<-as.data.frame(cbind(a,b)) 
  c<-c[grepl("Promoter",c$b),] 
  Menewid<-Methylationid[Methylationid$`#id` %in% c[,1] ,] 
  Menewid=Menewid[Menewid$gene!="."] 
  Mid<-Menewid$`#id` 
  Methylation<-Methylation[Methylation$sample %in% Mid ,]
  k<-(strsplit(Menewid$gene,','))
  end<-0
  for(i in 1:dim(Menewid)[1]){
    end<-end+length(k[[i]])
  } 
  pair<-as.data.frame(matrix(0,end,2))
  m<-1
  for(i in 1:dim(Menewid)[1]){
    for(j in m:(m+length(k[[i]])-1)){
      pair[j,1]<-Menewid[i,1]
      pair[j,2]<-k[[i]][j-m+1]
    }
    m<-m+length(k[[i]])
  }
  colnames(pair)[1]<-'sample'
  data<-merge(pair,Methylation,'sample') 
  daaaa<-aggregate(data,by=list(gene=data$V2),median) 
  daaaa1<-daaaa[,-c(2,3)]
  write_xlsx(daaaa1,path=paste("/out/",cancer,"_",oo,sep = ""),format_headers =FALSE)
}