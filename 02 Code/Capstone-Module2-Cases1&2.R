################################################################################
# Capstone Project - Sentiment Analysis of Book Reviews
# Module-2 : Machine Learning Model
#  This module labels the sentiment values in desired number of levels, 
#  generates DTM and implements Naïve Bayes classification algorithm
################################################################################


#Part 1 -
#Include the required libraries   

rm(list = ls())

library(tm)
library(RTextTools)
library(dplyr)
library(ggplot2)
library(stringr)
library(caret)
library(e1071)
library(syuzhet) 
library(qdapDictionaries)
library(knitr)
library(naivebayes)

set.seed(10000)

par(mfrow=c(1,1))

################################################################################


#Part 2 -
#Read the csv file with sentiment extracted from each review in Module 1

#Read the csv files into a dataframe
df0 <- read.csv("Module1_Review Data with sentiments.csv", stringsAsFactors = F)
colnames(df0)[1] <- "Id"

m <- nrow(df0)

maxScore <- max(df0$Score)

df0$ReviewText <- gsub("zzyyzzyy-","zzyyzzyy",df0$ReviewText)

par(mfrow=c(2,3))
meanSentiment <- c(1:maxScore)
sdSentiment <- c(1:maxScore)
medianSentiment <- c(1:maxScore)
for (i in 1:maxScore)
{ 
  meanSentiment[i] <- mean(df0[df0$Score==i,]$ReviewSentiment)
  sdSentiment[i] <- sd(df0[df0$Score==i,]$ReviewSentiment)
  medianSentiment[i] <- median(df0[df0$Score==i,]$ReviewSentiment)
  barplot(table(df0[df0$Score==i,]$ReviewSentiment), main=paste("Review Score = ",i), col=i+1, xlab="Sentiment Score", ylab="Frequency")
}
#par(mfrow=c(1,1))
plot(1:maxScore, meanSentiment, type = "o", col="blue", cex=2, ylab="Avg Sentiment Score", xlab="Review Score", main="Avg Sentiment", ylim=c(0,2))

sentimentLevels <- c(1:m)
for (i in 1:m)
{
  if (df0$ReviewSentiment[i] > 0) 
  { sentimentLevels[i] <- 1 }
  if (df0$ReviewSentiment[i] == 0) 
  { sentimentLevels[i] <- 0 }
  if (df0$ReviewSentiment[i] < 0) 
  { sentimentLevels[i] <- -1 }
}
df0 <- cbind(df0, sentimentLevels)
colnames(df0)[7] <- "SentimentLevel"

df0$Score <- as.factor(df0$Score)
df0$ReviewSentiment <- as.factor(df0$ReviewSentiment)
df0$ScoreLevel <- as.factor(df0$ScoreLevel)
df0$SentimentLevel <- as.factor(df0$SentimentLevel)


################################################################################

#Part 3 -
#Generate DTM

rc <- Corpus(VectorSource(df0$ReviewText))

#Data transformation: lowercasing removing numbers, removing white spacing
rc <- tm_map(rc, tolower)
rc <- tm_map(rc, stripWhitespace)
rc <- tm_map(rc, removeNumbers)
rc <- tm_map(rc, removePunctuation, preserve_intra_word_dashes=TRUE)

#Remove stopwords
my_stopwords <- stopwords('english')
#retained_words <- c("not", "isn't", "aren't", "wasn't", "weren't", "doesn't", "don't", "didn't", "wont", "shant", "wouldn't", "shouldn't", "hasn't", "haven't", "hadn't")
#my_stopwords <- my_stopwords[!my_stopwords %in% retained_words]
rc <- tm_map(rc, removeWords, my_stopwords)

dtm <- DocumentTermMatrix(rc, control = list(weighting = weightTfIdf))

dim(dtm)
inspect(dtm)

rc[[1]]$content

#Find most frequent terms in training corpus and regenerate DTM
frequentTerms_dtm <- findFreqTerms(dtm, 50)
dtm <- DocumentTermMatrix(rc, control=list(dictionary = frequentTerms_dtm))

dtm <- as.matrix(dtm)


################################################################################

#Part 4 -
#Reassign frequencies of terms in DTM according to their sentiments

positive_sentiment_word_list <- scan("Positive_Sentiment_Word_List.txt", what="character", comment.char = ";")
negative_sentiment_word_list <- scan("Negative_Sentiment_Word_List.txt", what="character", comment.char = ";")

# Function to extract sentiment using two methods
#   Method 1 - Using get_nrc_sentiment function
#   Method 2 - using positive and negative word list

fn_getSentiment <- function(x)
{
  #Method 1
  y1 <- get_nrc_sentiment(x)$positive - get_nrc_sentiment(x)$negative
  
  #Method 2 
  y21 <- ifelse (x %in% positive_sentiment_word_list, 1, 0)
  y22 <- ifelse (x %in% negative_sentiment_word_list, 1, 0)
  y2 <- y21 - y22 
  
  #Merge two results
  if (y1 == 0 & y2 == 0) 
  { y <- 0 }
  if (y1 != 0 & y2 == 0) 
  { y <- y1 }
  if (y1 == 0 & y2 != 0) 
  { y <- y2 }
  if (y1 > 0 & y2 > 0) 
  { y <- mean(y1, y2) }
  if (y1 < 0 & y2 < 0) 
  { y <- mean(y1, y2) }
  if (y1 > 0 & y2 < 0) 
  { y <- 0 }
  if (y1 < 0 & y2 > 0) 
  { y <- 0 }
  
  return (y)
}

for (j in 1:ncol(dtm))
{
  print (j)
  currentTermNegative <- "F"
  if ( nchar(colnames(dtm)[j]) > 8 )
  {
    if (substr(colnames(dtm)[j],1,8) == "zzyyzzyy")
    {
      currentTerm <- substr(colnames(dtm)[j], 9, nchar(colnames(dtm)[j]))
      currentTermNegative <- "T"
    }
    else
    {
      currentTerm <- colnames(dtm)[j]
    }
  }
  else
  {
    currentTerm <- colnames(dtm)[j]
  }
  wordSentiment <- fn_getSentiment(currentTerm)
  if (currentTermNegative == "T")
  {
    wordSentiment <- -1*wordSentiment
  }
  dtm[,j] <- (wordSentiment*10 + 20) * dtm[,j]
}


################################################################################

#Part 5 -
#Append DTM to df0 

dtmTrainSize <- ncol(dtm)

df0 <- cbind(df0, dtm)

for (j in 1:ncol(df0))
{
  df0[,j] <- as.factor(df0[,j])
}

write.csv(df0, file = "Module2_Review Data appended with DTM.csv")

################################################################################


#Part 6 -
#Divide the entire review data into Training and Validation sets

df0 <- df0[sample(nrow(df0)),]

trn <- as.integer(0.8 * nrow(df0))
vld <- nrow(df0) - trn

m <- trn

#df0b <- df0

################################################################################


#Part 7 -
#Run Naive Bayes classification

#Training NB classifier
nbc <- naive_bayes(df0[1:trn, c(-1,-3,-4,-5,-6,-7)], df0[1:trn, 7], data=df0[1:trn, ], laplace = 1) 

#Testing NB classifier
nbc_pred <- predict(nbc, df0[trn+1 : vld, c(-1,-3,-4,-5,-6,-7)]) 

t1 <- table("Predictions"= nbc_pred,  "Actual" = df0$SentimentLevel[trn+1 : vld])
t1

t1DF <- as.data.frame(t1)
write.csv(t1DF, file = "Module2_Prediction Summary.csv")

predictionAccuracy_overall <- sum(t1DF$Freq[t1DF$Predictions==t1DF$Actual]) / sum(t1DF$Freq)
predictionAccuracy_negative <- sum(t1DF$Freq[t1DF$Predictions==-1 & t1DF$Actual==-1]) / sum(t1DF$Freq[t1DF$Actual==-1])
predictionAccuracy_neutral <- sum(t1DF$Freq[t1DF$Predictions==0 & t1DF$Actual==0]) / sum(t1DF$Freq[t1DF$Actual==0])
predictionAccuracy_positive <- sum(t1DF$Freq[t1DF$Predictions==1 & t1DF$Actual==1]) / sum(t1DF$Freq[t1DF$Actual==1])

predictionColumn <- c(1:nrow(df0))
predictionColumn[1:trn] <- 999
predictionColumn[trn+1 : vld] <- nbc_pred

df0 <- cbind(predictionColumn, df0)
colnames(df0)[1] <- "Prediction"
df0$Prediction <- as.factor(df0$Prediction)

write.csv(df0[trn+1 : vld, ], file = "Module2_Prediction.csv")

################################################################################
