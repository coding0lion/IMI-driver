######Heatmap
library(ComplexHeatmap)
features_45=c(0.383769028,0.759958945,0.683648549,0.298561151,0.992490237,0.525316456)
features_special=c(0.133541125,0.744341608,0.534961989,0.371247829,0.882105358,0.092317548)
features_consistent=c(0.268403695,0.824855901,0.632716816,0.332733813,0.972864724,0.254470426)
features_dif=c(0.033435406,0.563470466,0.433574998,0.38545026,0.704317337,0.037330805)
features_all=c(0.426084611,0.881338867,0.706252689,0.345757876,0.992234733,0.556603692)
data<-rbind(features_consistent,features_45,features_dif,features_special,features_all)
colnames(data)<-c('Mcc','Auc','F1','Sensitivity','Specificity',	'Precision')
rownames(data)<-c('Common network','Common multi-omics','Variation analysis data','Specific network','All features')
annotation_row =as.data.frame(c('Common features','Common features','Specific features','Specific features','All features'))
rownames(annotation_row)<-rownames(data)
colnames(annotation_row)<-"classify"
for(i in 1:6){
  data[,i]=(data[,i]-min(data[,i]))/(max(data[,i])-min(data[,i]))
}
library(pheatmap) 
p<-pheatmap(data,cluster_rows = F, cluster_cols = F,angle_col = 45,display_numbers=T,
            color = colorRampPalette(c('#E6E8FA','#BF9663'))(50)
            ,border_color = "white",cellwidth = 40, cellheight = 35,
            ,gaps_row = c(2, 4),fontsize_number = 10,annotation_row =annotation_row,
            annotation_colors=list(classify=c("Common features"="#2D5873","Specific features"="#7BA696","All features"="#BFBA9F"))
            ,rect_gp = gpar(col = "white", lwd = 2,lty=2),number_color = "black",annotation_names_row = FALSE)        

### ks-test(p-value)
library(readxl)
library(ggplot2)
common<-read_xlsx('/data/ks_p_common(-log10).xlsx')
common<-as.data.frame(common)
ggplot(common, aes(network, logP)) +
  geom_count()
specific<-read_xlsx('/data/ks_p_specific(-log10).xlsx')
colnames(specific)<-c("cancer","RCN","RCMN","GDN","ceRNA","RCSMN")
c<-specific[,2]
c$network<-colnames(specific)[2]
colnames(c)[1]<-"-logP"
for (i in 3:6){
  a<-specific[,i]
  a$network<-colnames(specific)[i]
  colnames(a)[1]<-"-logP"
  c<-rbind(c,a)
}
common<-common[,c(2,1)]
colnames(common)<-colnames(c)
c<-rbind(c,common)
parts<-c("PPI","TRN","GPSN","RCSMN","RCMN","RCN","ceRNA","GDN")
ggplot(c,aes(x=factor(network,levels=parts),y=c$`-logP`,color=network)) +
  geom_point(alpha=0,show.legend = FALSE)+
  scale_color_manual(values=c("#253659", "#BFBA9F", "#7BA696","#BF9663","#2D5873","#497373","#D95F43","#400101","#400101"))+
  geom_jitter(width = 0.15,
              height =0, size =3, shape =16, 
              stroke = 0.3,
              alpha=0.7,show.legend = FALSE)+
  geom_hline(yintercept=1.3,lty=5,color="#9F5F9F",size=0.6)+
  ylab("-logP")+
  xlab("Network")+
  theme_classic() 

#### box-plot
library(ggplot2)
namep<-c("p1","p2","p3","p4","p5","p6","p7","p8","p9")
index=c('MCC','AUC','Precision','F1') 
benchmarklist=c('2020Rule','benchmark','CGCpointMut','CGC','CTAT','HCD','MouseMut','oncoGene','IntOGen')
benchmarklist2=c('Rule2020','benchmark','CGCpointMut','CGC','CTAT','HCD','MouseMut','oncoGene','ourmethod')
for(nowindex in 1:4){
  for(NNN in 1:8){
    nowbenchmark=benchmarklist[NNN]
    url=paste('/data/cgc_result',nowbenchmark,'cgc_result.csv',sep = "")
    data<-read.csv(url)
    colnames(data)[1]<-"Algorithms"
    data[data$Algorithms %in% benchmarklist2,]$Algorithms="IMI-Driver"
    if("driverRWH" %in% data$Algorithms){
      data[data$Algorithms=="driverRWH",]$Algorithms="DriverRWH"}
    methods<-c("MaxMIF","MutPanning","WITER","DriverML","DNsum","DNmax","DriverRWH")
    methods<-c("IMI-Driver",methods)
    title=paste(nowbenchmark," ( ",index[nowindex]," )",sep = "")
    title1<-paste("Multi-method comparison on ",nowbenchmark," dataset",sep = "")
    assign(namep[NNN],ggplot(data,aes(x=factor(Algorithms,levels=methods),y=data[,index[nowindex]],color=Algorithms),cex.axis=1.5)+
             geom_boxplot(size = 0.8, width = 0.8, alpha = 0)+
             scale_color_manual(values=c("#011604", "#BFBA9F", "#7BA696","#2D5873","#497373","#BF9663","#400101","#80001E"))+
             geom_jitter(position = position_jitter(0.4), alpha = 0.2, size=1.5)+
             theme_classic()+
             ylab(index[nowindex])+
             ggtitle(title1)+
             labs(x="Algorithms")+
             theme(plot.title = element_text(hjust = 0.5))+
             theme(legend.position = "none"))}
  #intogen
  nowbenchmark="IntOGen"
  url=paste('F:/university/research/drivergene/pancancer/8基准/',nowbenchmark,'cgc_result.csv',sep = "")
  data1<-read.csv(url)
  colnames(data1)[1]<-"Algorithms"
  data1[data1$Algorithms %in% benchmarklist2,]$Algorithms="IMI-Driver"
  methods<-c("MaxMIF","MutPanning","WITER","DriverML","DNsum","DNmax","DriverRWH")
  methods<-c("IMI-Driver",methods)
  title=paste(nowbenchmark," ( ",index[nowindex]," )",sep = "")
  title1<-paste("Multi-method comparison on ",nowbenchmark," dataset",sep = "")
  assign("p9",ggplot(data1,aes(x=factor(Algorithms,levels=methods),y=data1[,index[nowindex]],color=Algorithms),cex.axis=1.5)+
           geom_boxplot(size = 0.8, width = 0.8, alpha = 0)+
           scale_color_manual(values=c("#011604", "#BFBA9F", "#7BA696","#2D5873","#497373","#BF9663","#400101","#80001E"))+
           geom_jitter(position = position_jitter(0.4), alpha = 0.2, size=1.5)+
           theme_classic()+
           ylab(index[nowindex])+
           ggtitle(title1)+
           labs(x="Algorithms")+
           theme(plot.title = element_text(hjust = 0.5))+
           theme(legend.position = "none"))
  mmm<-plot_grid(p1,p2,p3,p4,p5,p6,p7,p8,p9,ncol=3)
  pdf(paste("/outdata/",nowindex,".pdf",sep = ""),width = 18,height = 12)
  mmm
  print(mmm) 
  dev.off()
}
###about 0,1 matrix
library(readxl)
DNmax<-read_xlsx('data/0_1/DNmax_STRINGv10_01.xlsx')
DNsum<-read_xlsx('data/0_1/DNsum_STRINGv10_01.xlsx')
DriverML<-read_xlsx('data/0_1/DriverML_01.xlsx')
driverRWH<-read_xlsx('data/0_1/driverRWH_01.xlsx')
MaxMIF<-read_xlsx('data/0_1/MaxMIF_STRINGv10_01.xlsx')
MutPanning<-read_xlsx('data/0_1/MutPanning_01.xlsx')
WITER<-read_xlsx('data/0_1/WITER_01.xlsx')
our<-read_xlsx('data/0_1/mut2_0_1.xlsx')
intogendata<-read_xlsx('intogen union.xlsx')
#intogen gene list
intogenname<-unique(intogendata$SYMBOL)
#cgc gene list
genename<-read_excel('gene.xlsx',col_names = FALSE)
colnames(genename)=c('gene','label')
genename$gene<-gsub("'", '', genename$gene)
raw_label = read_excel('benchmark.xlsx')
raw_label=as.data.frame(raw_label)
rownames(raw_label)<-genename$gene
engenename=rownames(raw_label[raw_label==1,])
data=our[our$sum>=5,]
data1<-data[,c(1,19)]
method=c("DNmax","DNsum","DriverML","driverRWH","MaxMIF","MutPanning","WITER")
for(i in 1:length(method)){
  s=get(method[i])
  s1=s[,c(1,19)]
  data1<-merge(data1,s1,"gene",all.x=TRUE)
}
rownames(data1)<-data1[,1]
data1<-data1[,-1]
colnames(data1)<-c("IMI-Driver",method)
data1[is.na(data1)]=0
data2=data1
data2$drivergene="non-driver"
endgenename<-union(engenename,intogenname)
data2[rownames(data2) %in% endgenename,]$drivergene="Known driver"
library(pheatmap) 
annotation_row=as.data.frame(data2[,c('drivergene')])
rownames(annotation_row)<-rownames(data2)
colnames(annotation_row)<-c('Known_driver')
drivercolor <- c("#666666","white")
names(drivercolor) <- c("Known driver","non-driver") 
ann_colors <- list(Known_driver=drivercolor)
pheatmap(data1,angle_col = 45,fontsize =6,color = colorRampPalette(c('white','#BF9663'))(50),fontsize_row=5,
         cluster_rows = F, cluster_cols = F,
         fontsize_number = 10,number_color ="brown",border_color = 'white',cellwidth = 13, cellheight = 8,annotation_row =annotation_row,
         annotation_colors = ann_colors,annotation_legend = FALSE
)
dev.off()




