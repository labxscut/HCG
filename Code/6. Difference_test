library(readr)
F_tot <- read_table("D:/preSolve2/diff_check/New_pat_esca.txt")
F_tot <- as.data.frame(F_tot)
rownames(F_tot) <- F_tot[,1]
F_tot <- F_tot[,-1]
N = 77 ;
T_cna <- F_tot[,c(112:142)]
T_cna <- cbind(F_tot[,1],T_cna)
colnames(T_cna)[1] <- "SUBTYPE"
cna_name <- colnames(T_cna[,c(2:32)])
Tc1 = T_cna
for (i in 1 : N) {
  if(Tc1[i,1] == 1)
    Tc1[i,1] = 1
  else Tc1[i,1] = 2
}
library(dplyr)
p1val = array(0,dim=c(31,2))
p1val = as.data.frame(p1val)
cnt = 0
for(gene in cna_name){
  p = t.test( Tc1[,gene]~Tc1$SUBTYPE )$p.value
  cnt = cnt + 1 ;
  p1val[cnt,1] = gene
  p1val[cnt,2] = p
}
colnames(p1val)[1] <- "Gene"
colnames(p1val)[2] <- "P"
P_cna <- arrange(p1val , P)
P_cna <- as.data.frame(P_cna)
write.table(P_cna,file = "D:/preSolve2/diff_check/cna_CIN_esca.txt",sep='\t')
\####################################
Tc2 = T_cna
for (i in 1 : N) {
  if(Tc2[i,1] == 2)
    Tc2[i,1] = 1
  else Tc2[i,1] = 2
}
library(dplyr)
p2val = array(0,dim=c(31,2))
p2val = as.data.frame(p2val)
cnt = 0
for(gene in cna_name){
  p = t.test( Tc2[,gene]~Tc2$SUBTYPE )$p.value
  cnt = cnt + 1 ;
  p2val[cnt,1] = gene
  p2val[cnt,2] = p
}
colnames(p2val)[1] <- "Gene"
colnames(p2val)[2] <- "P"
P_cna <- arrange(p2val , P)
P_cna <- as.data.frame(P_cna)
write.table(P_cna,file = "D:/preSolve2/diff_check/cna_GS_esca.txt",sep='\t')
\######################################
Tc3 = T_cna
for (i in 1 : N) {
  if(Tc3[i,1] == 3)
    Tc3[i,1] = 1
  else Tc3[i,1] = 2
}
library(dplyr)
p3val = array(0,dim=c(31,2))
p3val = as.data.frame(p3val)
cnt = 0
for(gene in cna_name){
  p = t.test( Tc3[,gene]~Tc3$SUBTYPE )$p.value
  cnt = cnt + 1 ;
  p3val[cnt,1] = gene
  p3val[cnt,2] = p
}
colnames(p3val)[1] <- "Gene"
colnames(p3val)[2] <- "P"
P_cna <- arrange(p3val , P)
P_cna <- as.data.frame(P_cna)
write.table(P_cna,file = "D:/preSolve2/diff_check/cna_MSI_esca.txt",sep='\t')
\################################
Tc4 = T_cna
for (i in 1 : N) {
  if(Tc4[i,1] == 4)
    Tc4[i,1] = 1
  else Tc4[i,1] = 2
}
library(dplyr)
p4val = array(0,dim=c(31,2))
p4val = as.data.frame(p4val)
cnt = 0
for(gene in cna_name){
  p = t.test( Tc4[,gene]~Tc4$SUBTYPE )$p.value
  cnt = cnt + 1 ;
  p4val[cnt,1] = gene
  p4val[cnt,2] = p
}
colnames(p4val)[1] <- "Gene"
colnames(p4val)[2] <- "P"
P_cna <- arrange(p4val , P)
P_cna <- as.data.frame(P_cna)
write.table(P_cna,file = "D:/preSolve2/diff_check/cna_EBV_esca.txt",sep='\t')

library(readr);
Datar <- read_table("D:/preSolve2/mely_EBV.txt",col_names = c("id","Gene","value")) ;
Datar <- Datar[,-1];
Datar <- Datar[-1,];
for(i in 1:nrow(Datar[,1])){
  x = Datar[i,1];
  x = as.character(x);
  y = gsub('["]','',x);
  Datar[i,1] = y;
}
tmp <- as.vector(Datar[,2]);
Tp <- as.numeric(unlist(tmp));
v <- p.adjust( Tp , method = "BH" )
sz = 0 ;
for(i in 1:length(v)) {
  if(v[i] < 1e-5)
    sz = i ;
}
v <- v[c(1:sz)]
write.table(cbind(Datar[(1:sz),1],v),file = "D:/preSolve2/mely_EBV_p.txt",     sep='    ',col.names = FALSE,quote=FALSE)