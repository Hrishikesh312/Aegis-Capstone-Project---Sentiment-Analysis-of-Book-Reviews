################################################################################
# Capstone Project - Sentiment Analysis of Book Reviews
# Module-3 : Analysis of Results
#   This module provides analysis of results of implementation of 
#   Machine Learning model
################################################################################


#Part 1 -
#Include the required libraries   

rm(list = ls())

library(tm)
library(dplyr)
library(ggplot2)
library(stringr)
library(caret)
library(e1071)
library(syuzhet) 
library(qdapDictionaries)
library(knitr)
library(naivebayes)
library(lattice)

set.seed(10000)

par(mfrow=c(2,2))


################################################################################

#Part 2 
#Read the csv files into a dataframe and perform initial processing
df0 <- read.csv("./Analysis of Results/Module2_Prediction Summary_Type1.csv", stringsAsFactors = F)

df2 <- read.csv("./Analysis of Results/Module2_Prediction Summary_Type2.csv", stringsAsFactors = F)
df0 <- rbind(df0, df2)

cases  <- unique(df0$Case)

results <- matrix(nrow = length(cases), ncol = 14)
results[,] <- 0
colnames(results) <- c("Total Test Data", "Actual Negative", "Actual Negative %", "Predicted & Actual Negative", "Actual Neutral", "Actual Neutral %", "Predicted & Actual Neutral", "Actual Positive", "Actual Positive %", "Predicted & Actual Positive", "Prediction Accuracy - Negative", "Prediction Accuracy - Neutral", "Prediction Accuracy - Positive", "Prediction Accuracy - Overall")

for (i in 1:length(cases))
{
  dftemp <- df0[df0$Case==cases[i],]
  results[i,1] <- sum(dftemp$Freq)
  results[i,2] <- sum(dftemp[dftemp$Actual==-1,]$Freq)
  results[i,3] <- sum(dftemp[dftemp$Actual==-1,]$Freq) / sum(dftemp$Freq) * 100
  results[i,4] <- sum(dftemp[dftemp$Actual==-1 & dftemp$Predictions==-1,]$Freq)
  results[i,5] <- sum(dftemp[dftemp$Actual==0,]$Freq)
  results[i,6] <- sum(dftemp[dftemp$Actual==0,]$Freq) / sum(dftemp$Freq) * 100
  results[i,7] <- sum(dftemp[dftemp$Actual==0 & dftemp$Predictions==0,]$Freq)
  results[i,8] <- sum(dftemp[dftemp$Actual==1,]$Freq)
  results[i,9] <- sum(dftemp[dftemp$Actual==1,]$Freq) / sum(dftemp$Freq) * 100
  results[i,10] <- sum(dftemp[dftemp$Actual==1 & dftemp$Predictions==1,]$Freq)
  results[i,11] <- sum(dftemp[dftemp$Actual==-1 & dftemp$Predictions==-1,]$Freq) / sum(dftemp[dftemp$Actual==-1,]$Freq) * 100
  results[i,12] <- sum(dftemp[dftemp$Actual==0 & dftemp$Predictions==0,]$Freq) / sum(dftemp[dftemp$Actual==0,]$Freq) * 100
  results[i,13] <- sum(dftemp[dftemp$Actual==1 & dftemp$Predictions==1,]$Freq) / sum(dftemp[dftemp$Actual==1,]$Freq) * 100
  results[i,14] <- sum(results[i,c(4,7,10)]) / results[i,1] * 100
}

dfout <- as.data.frame(results, stringsAsFactors = FALSE)
dfout <- cbind(cases, dfout)
colnames(dfout)[1] <- "Case"

dfout2 <- dfout


################################################################################

#Part 3

par(mfrow=c(2,3))

casesType1 <- c("1", "2", "3a", "3b", "3c", "3d")
dfout <- dfout[dfout$Case %in% casesType1, ] 

dftemp <- dfout[, c(1, 4, 7, 10, 12, 13, 14, 15)]

pieLabels <- rep("", 3)
for (i in 1:nrow(dftemp))
{
  pieData <- c(dftemp[i,2], dftemp[i,3], dftemp[i,4])
  pieLabels[1] <- paste("Neg ",round(dftemp[i,2],2),"%", " Accuracy ", round(dftemp[i,5],2), "%", sep="")
  pieLabels[2] <- paste("Neutral ",round(dftemp[i,3],2),"%", " Accuracy ", round(dftemp[i,6],2), "%", sep="")
  pieLabels[3] <- paste("Pos ",round(dftemp[i,4],2),"%", " Accuracy ", round(dftemp[i,7],2), "%", sep="")
  pie(pieData,labels = pieLabels, col=rainbow(length(pieLabels)), main=paste("Case",i, "Overall Accuracy = ", round(dftemp[i,8],2), "%"))
}

################################################################################

#Part 4

par(mfrow=c(2,2))

dfout <- dfout[order(dfout$`Actual Positive %`), ]
plot(dfout$`Actual Positive %`, dfout$`Prediction Accuracy - Positive`, type="o" , xlab="Positive %", ylab="Accuracy Positive %", ylim=c(70,100), col=2)
plot(dfout$`Actual Positive %`, dfout$`Prediction Accuracy - Overall`, type="o", xlab="Positive %", ylab="Accuracy Overall %", ylim=c(70,100), col=3)

dfout <- dfout[order(dfout$`Actual Positive`), ]
plot(dfout$`Actual Positive`, dfout$`Prediction Accuracy - Positive`, type="o", xlab="Positive Count", ylab="Accuracy Positive %", ylim=c(70,100), col=4)
plot(dfout$`Actual Positive`, dfout$`Prediction Accuracy - Overall`, type="o", xlab="Positive Count", ylab="Accuracy Overall %", ylim=c(70,100), col=6)

dfout <- dfout[order(dfout$`Actual Negative %`), ]
plot(dfout$`Actual Negative %`, dfout$`Prediction Accuracy - Negative`, type="o" , xlab="Negative %", ylab="Accuracy Negative %", col=2)
plot(dfout$`Actual Negative %`, dfout$`Prediction Accuracy - Overall`, type="o", xlab="Negative %", ylab="Accuracy Overall %", col=3)

dfout <- dfout[order(dfout$`Actual Negative`), ]
plot(dfout$`Actual Negative`, dfout$`Prediction Accuracy - Negative`, type="o", xlab="Negative Count", ylab="Accuracy Negative %", col=4)
plot(dfout$`Actual Negative`, dfout$`Prediction Accuracy - Overall`, type="o", xlab="Negative Count", ylab="Accuracy Overall %", col=6)

dfout <- dfout[order(dfout$`Actual Neutral %`), ]
plot(dfout$`Actual Neutral %`, dfout$`Prediction Accuracy - Neutral`, type="o" , xlab="Neutral %", ylab="Accuracy Neutral %", col=2)
plot(dfout$`Actual Neutral %`, dfout$`Prediction Accuracy - Overall`, type="o", xlab="Neutral %", ylab="Accuracy Overall %", col=3)

dfout <- dfout[order(dfout$`Actual Neutral`), ]
plot(dfout$`Actual Neutral`, dfout$`Prediction Accuracy - Neutral`, type="o", xlab="Neutral Count", ylab="Accuracy Neutral %", col=4)
plot(dfout$`Actual Neutral`, dfout$`Prediction Accuracy - Overall`, type="o", xlab="Neutral Count", ylab="Accuracy Overall %", col=6)

dfout <- dfout[order(dfout$`Total Test Data`), ]
plot(dfout$`Total Test Data`, dfout$`Prediction Accuracy - Overall` , type="o", xlab="Total Test Data", ylab="Accuracy Overall %", col=2)
plot(dfout$`Total Test Data`, dfout$`Prediction Accuracy - Negative`, type="o", xlab="Total Test Data", ylab="Accuracy Negative %", col=3)
plot(dfout$`Total Test Data`, dfout$`Prediction Accuracy - Neutral`, type="o", xlab="Total Test Data", ylab="Accuracy Neutral %", col=4)
plot(dfout$`Total Test Data`, dfout$`Prediction Accuracy - Positive`, type="o", xlab="Total Test Data", ylab="Accuracy Positive %", col=6)


################################################################################

#Part 5

casesType2 <- c("3a", "3b", "3c", "3d", "3a2", "3b2", "3c2", "3d2")
dfout2 <- dfout2[dfout2$Case %in% casesType2, ] 

par(mfrow=c(2,2))

#### Case 3a
dftemp <- data.frame(nrow=4, ncol=2)
dftemp[1:4,1] <- c(1:4)
for (j in 1:4)
{
  dftemp[j, 2] <- dfout2[dfout2$Case=="3a", j+11]
}
colnames(dftemp) <- c("Parameter", "Value")

plot(dftemp$Parameter, dftemp$Value, ylim=c(0,100), axes = F, xlab = NA, ylab = NA, type="o", col=1)
box()
axis(side = 1, lwd = 0, line = -.4, at=seq(1,4,by=1))
axis(side = 2, lwd = 0, line = -.4, las = 1, at=seq(0,100,by=10))
mtext(side = 1, "1=Neg, 2=Neutral, 3=Pos, 4=Overall", line = 2)
mtext(side = 2, "Accuracy %", line = 2)
mtext(side = 3, "Accuracy % for Case 3a", line = 2)
mtext(side = 3, "Black=Original  Blue=New", line = 1)

dftemp <- data.frame(nrow=4, ncol=2)
dftemp[1:4,1] <- c(1:4)
for (j in 1:4)
{
  dftemp[j, 2] <- dfout2[dfout2$Case=="3a2", j+11]
}
colnames(dftemp) <- c("Parameter", "Value")

points(dftemp$Parameter, dftemp$Value, type="o", col=4)
####

#### Case 3b
dftemp <- data.frame(nrow=4, ncol=2)
dftemp[1:4,1] <- c(1:4)
for (j in 1:4)
{
  dftemp[j, 2] <- dfout2[dfout2$Case=="3b", j+11]
}
colnames(dftemp) <- c("Parameter", "Value")

plot(dftemp$Parameter, dftemp$Value, ylim=c(0,100), axes = F, xlab = NA, ylab = NA, type="o", col=1)
box()
axis(side = 1, lwd = 0, line = -.4, at=seq(1,4,by=1))
axis(side = 2, lwd = 0, line = -.4, las = 1, at=seq(0,100,by=10))
mtext(side = 1, "1=Neg, 2=Neutral, 3=Pos, 4=Overall", line = 2)
mtext(side = 2, "Accuracy %", line = 2)
mtext(side = 3, "Accuracy % for Case 3b", line = 2)
mtext(side = 3, "Black=Original  Blue=New", line = 1)

dftemp <- data.frame(nrow=4, ncol=2)
dftemp[1:4,1] <- c(1:4)
for (j in 1:4)
{
  dftemp[j, 2] <- dfout2[dfout2$Case=="3b2", j+11]
}
colnames(dftemp) <- c("Parameter", "Value")

points(dftemp$Parameter, dftemp$Value, type="o", col=4)
####

#### Case 3c
dftemp <- data.frame(nrow=4, ncol=2)
dftemp[1:4,1] <- c(1:4)
for (j in 1:4)
{
  dftemp[j, 2] <- dfout2[dfout2$Case=="3c", j+11]
}
colnames(dftemp) <- c("Parameter", "Value")

plot(dftemp$Parameter, dftemp$Value, ylim=c(0,100), axes = F, xlab = NA, ylab = NA, type="o", col=1)
box()
axis(side = 1, lwd = 0, line = -.4, at=seq(1,4,by=1))
axis(side = 2, lwd = 0, line = -.4, las = 1, at=seq(0,100,by=10))
mtext(side = 1, "1=Neg, 2=Neutral, 3=Pos, 4=Overall", line = 2)
mtext(side = 2, "Accuracy %", line = 2)
mtext(side = 3, "Accuracy % for Case 3c", line = 2)
mtext(side = 3, "Black=Original  Blue=New", line = 1)

dftemp <- data.frame(nrow=4, ncol=2)
dftemp[1:4,1] <- c(1:4)
for (j in 1:4)
{
  dftemp[j, 2] <- dfout2[dfout2$Case=="3c2", j+11]
}
colnames(dftemp) <- c("Parameter", "Value")

points(dftemp$Parameter, dftemp$Value, type="o", col=4)
####

#### Case 3d
dftemp <- data.frame(nrow=4, ncol=2)
dftemp[1:4,1] <- c(1:4)
for (j in 1:4)
{
  dftemp[j, 2] <- dfout2[dfout2$Case=="3d", j+11]
}
colnames(dftemp) <- c("Parameter", "Value")

plot(dftemp$Parameter, dftemp$Value, ylim=c(0,100), axes = F, xlab = NA, ylab = NA, type="o", col=1)
box()
axis(side = 1, lwd = 0, line = -.4, at=seq(1,4,by=1))
axis(side = 2, lwd = 0, line = -.4, las = 1, at=seq(0,100,by=10))
mtext(side = 1, "1=Neg, 2=Neutral, 3=Pos, 4=Overall", line = 2)
mtext(side = 2, "Accuracy %", line = 2)
mtext(side = 3, "Accuracy % for Case 3d", line = 2)
mtext(side = 3, "Black=Original  Blue=New", line = 1)

dftemp <- data.frame(nrow=4, ncol=2)
dftemp[1:4,1] <- c(1:4)
for (j in 1:4)
{
  dftemp[j, 2] <- dfout2[dfout2$Case=="3d2", j+11]
}
colnames(dftemp) <- c("Parameter", "Value")

points(dftemp$Parameter, dftemp$Value, type="o", col=4)
####

