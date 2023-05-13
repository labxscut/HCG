mutation1<-read.table("D:/preSolve2/hot_p/DataSet/New_mutation_ex.txt",header = T)
library(pheatmap)
Type=c(rep("CIN",322),rep("GS",24),rep("MSI",69),rep("EBV",38))
rownames(mutation1) <- mutation1[,1]
mutation1 <- mutation1[,-1]
names(Type)=colnames(mutation1)
Type=as.data.frame(Type)
annotation_row = data.frame(rep(c('mutaion') , 110))
rownames(annotation_row) <- rownames(mutation1)
colnames(annotation_row) <- 'mutation'
ann_colors = list(
  Type = c(CIN = "#3dc03a", GS = "#f03013", MSI = "#13d3f0", EBV = "#0682ee")
  \#mutation = c(mutation='green')
)
pheatmap(mutation1,
         annotation_row = annotation_row,
         annotation_col = Type,
         color = colorRampPalette(c("#f4f6f9","#db0000"))(50),
         annotation_colors = ann_colors,
         \## scale = 'row',
         legend_breaks = c(0,1), legend_labels = c("0","1"),
         cluster_rows = F,
         cluster_cols = F,
         border = T,
         show_rownames=F,
         labels_col = rep('',length(Type[,1])),
         gaps_col = c(322,346,415),
         \## fontsize = 10,
)