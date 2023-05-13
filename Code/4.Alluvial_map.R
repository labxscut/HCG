survival<- read.csv('C:\\Users\\project\\Desktop\\label_SMOTE_train_valid.csv')
library(ggalluvial)
subtype <- survival[,2:3]
subtype[,3] <- survival[,4]
colnames(subtype)[2:3] <- c('ORIGION_SUBTYPE','FOUR_SUBTYPE')
rownames(subtype) <- c(1:dim(subtype)[1])
aggregate <- matrix(0,16,4)
aggregate[,1] <- c(rep('1',4),rep('2',4),rep('3',4),rep('4',4))
aggregate[,2] <- rep(c('1','2','3','4'),4)
for (i in 1:16) {
  aggregate[i,3] <- dim(subset(subtype, ORIGION_SUBTYPE==aggregate[i,1] & FOUR_SUBTYPE==aggregate[i,2]))[1]
}
aggregate[,4] <- rep(c('No change','Change','Change','Change','Change','No change','Change','Change',
                       'Change','Change','No change','Change','Change','Change','Change','No change'))
colnames(aggregate) <- c('ORIGION_SUBTYPE','FOUR_SUBTYPE','Frequency','Status')
aggregate <- as.data.frame(aggregate)
aggregate[,1] <- as.factor(aggregate[,1])
aggregate[,2] <- as.factor(aggregate[,2])
aggregate[,4] <- as.factor(aggregate[,4])
aggregate[,1] <- rep(c('CIN','CIN','CIN','CIN','GS','GS','GS','GS','MSI','MSI','MSI','MSI','EBV','EBV','EBV','EBV'))
aggregate[,2] <- rep(c('CIN','GS','MSI','EBV','CIN','GS','MSI','EBV','CIN','GS','MSI','EBV','CIN','GS','MSI','EBV'))
write.csv(aggregate,"C:\\Users\\project\\Desktop\\123.csv")
aggregate<- read.csv('C:\\Users\\project\\Desktop\\123.csv')
library(ggalluvial)
is_alluvia_form(as.data.frame(aggregate), axes = 1:3, silent = TRUE)
pic<-ggplot(data = aggregate,
            aes(axis1 =ORIGION_SUBTYPE, axis2 = FOUR_SUBTYPE, y = Frequency)) +
  geom_alluvium(aes(fill = Status), width = 1/12) +
  geom_stratum(width = 1/12) +
  geom_label(stat = "stratum", aes(label = after_stat(stratum))) +
  scale_x_continuous(breaks = 1:2, labels = c("Original Subtype", "4 Classfier")) +
  scale_fill_brewer(type = "qual", palette = "Set1") +
  ggtitle("Original Subtype vs 4 Classfier") +
  #coord_flip() +
  theme_minimal()
ggsave("alluv_4.jpg", pic , width = 12, height = 8, dpi = 800)


survival<- read.csv('C:\\Users\\project\\Desktop\\data\\label_train_valid.csv')
library(ggalluvial)
subtype <- survival[,2:3]
subtype[,3] <- survival[,5]
colnames(subtype)[2:3] <- c('ORIGION_SUBTYPE','FOUR_SUBTYPE')
rownames(subtype) <- c(1:dim(subtype)[1])

aggregate <- matrix(0,16,4)
aggregate[,1] <- c(rep('1',4),rep('2',4),rep('3',4),rep('4',4))
aggregate[,2] <- rep(c('1','2','3','4'),4)
for (i in 1:16) {
  aggregate[i,3] <- dim(subset(subtype, ORIGION_SUBTYPE==aggregate[i,1] & FOUR_SUBTYPE==aggregate[i,2]))[1]
}
aggregate[,4] <- rep(c('No change','Change','Change','Change','Change','No change','Change','Change',
                       'Change','Change','No change','Change','Change','Change','Change','No change'))
colnames(aggregate) <- c('ORIGION_SUBTYPE','FOUR_SUBTYPE','Frequency','Status')
aggregate <- as.data.frame(aggregate)
aggregate[,1] <- as.factor(aggregate[,1])
aggregate[,2] <- as.factor(aggregate[,2])
aggregate[,4] <- as.factor(aggregate[,4])

aggregate[,1] <- rep(c('CIN','CIN','CIN','CIN','GS','GS','GS','GS','MSI','MSI','MSI','MSI','EBV','EBV','EBV','EBV'))
aggregate[,2] <- rep(c('CIN','GS','MSI','EBV','CIN','GS','MSI','EBV','CIN','GS','MSI','EBV','CIN','GS','MSI','EBV'))
write.csv(aggregate,"C:\\Users\\project\\Desktop\\123.csv")

aggregate<- read.csv('C:\\Users\\project\\Desktop\\123.csv')
is_alluvia_form(as.data.frame(aggregate), axes = 1:3, silent = TRUE)
pic<-ggplot(data = aggregate,
            aes(axis1 =ORIGION_SUBTYPE, axis2 = FOUR_SUBTYPE, y = Frequency)) +
  geom_alluvium(aes(fill = Status), width = 1/12) +
  geom_stratum(width = 1/12) +
  geom_label(stat = "stratum", aes(label = after_stat(stratum))) +
  scale_x_continuous(breaks = 1:2, labels = c("Original Subtype", "2*2*2 Classfier")) +
  scale_fill_brewer(type = "qual", palette = "Set1") +
  ggtitle("Original Subtype vs 2*2*2 Classfier") +
  #coord_flip() +
  theme_minimal()
ggsave("alluv_222.jpg", pic , width = 12, height = 8, dpi = 800)


survival<- read.csv('C:\\Users\\project\\Desktop\\label_train_valid.csv')
library(ggalluvial)
subtype <- survival[,2:3]
subtype[,3] <- survival[,5]
colnames(subtype)[2:3] <- c('ORIGION_SUBTYPE','FOUR_SUBTYPE')
rownames(subtype) <- c(1:dim(subtype)[1])

aggregate <- matrix(0,16,4)
aggregate[,1] <- c(rep('1',4),rep('2',4),rep('3',4),rep('4',4))
aggregate[,2] <- rep(c('1','2','3','4'),4)
for (i in 1:16) {
  aggregate[i,3] <- dim(subset(subtype, ORIGION_SUBTYPE==aggregate[i,1] & FOUR_SUBTYPE==aggregate[i,2]))[1]
}
aggregate[,4] <- rep(c('No change','Change','Change','Change','Change','No change','Change','Change',
                       'Change','Change','No change','Change','Change','Change','Change','No change'))
colnames(aggregate) <- c('ORIGION_SUBTYPE','FOUR_SUBTYPE','Frequency','Status')
aggregate <- as.data.frame(aggregate)
aggregate[,1] <- as.factor(aggregate[,1])
aggregate[,2] <- as.factor(aggregate[,2])
aggregate[,4] <- as.factor(aggregate[,4])

aggregate[,1] <- rep(c('CIN','CIN','CIN','CIN','GS','GS','GS','GS','MSI','MSI','MSI','MSI','EBV','EBV','EBV','EBV'))
aggregate[,2] <- rep(c('CIN','GS','MSI','EBV','CIN','GS','MSI','EBV','CIN','GS','MSI','EBV','CIN','GS','MSI','EBV'))
write.csv(aggregate,"C:\\Users\\project\\Desktop\\123.csv")

aggregate<- read.csv('C:\\Users\\project\\Desktop\\123.csv')
library(ggalluvial)
is_alluvia_form(as.data.frame(aggregate), axes = 1:3, silent = TRUE)
pic<-ggplot(data = aggregate,
            aes(axis1 =ORIGION_SUBTYPE, axis2 = FOUR_SUBTYPE, y = Frequency)) +
  geom_alluvium(aes(fill = Status), width = 1/12) +
  geom_stratum(width = 1/12) +
  geom_label(stat = "stratum", aes(label = after_stat(stratum))) +
  scale_x_continuous(breaks = 1:2, labels = c("Original Subtype", "3+2 Classfier")) +
  scale_fill_brewer(type = "qual", palette = "Set1") +
  ggtitle("Original Subtype vs 3+2 Classfier") +
  #coord_flip() +
  theme_minimal()
pic
ggsave("alluv_32.jpg", pic , width = 12, height = 8, dpi = 800)