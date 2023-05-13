library(survival)
library(survminer)
library(ggplot2)
library(readr)
library(dplyr)
library(extrafont)
library(extrafontdb)
P_val <- read.table("D:/preSolve2/cox/patient_32.txt",header = T)
P_val <- arrange(P_val , SUBTYPE)
patient_information <- P_val
for(i in 1:351) {
  if(patient_information[i,"AGE"] > 64)
    patient_information[i,"AGE"] = ">=65"
  else patient_information[i,"AGE"] = "<65"
}
patient_information <- within(patient_information,{
  SUBTYPE <- factor(SUBTYPE,labels = c("CIN","GS","MSI","EBV"))
  SEX <- factor(SEX,labels = c("female","male"))
})
patient_information$OS_MONTHS=as.numeric(patient_information$OS_MONTHS)
patient_information$OS_STATUS=as.numeric(patient_information$OS_STATUS)
cox_pict1 <- coxph(Surv(OS_MONTHS , OS_STATUS) ~ SUBTYPE + AGE + SEX , data = patient_information)
ggforest(cox_pict1,
         main = 'Ⅱ-HC(A)',
         fontsize = 0.8
)