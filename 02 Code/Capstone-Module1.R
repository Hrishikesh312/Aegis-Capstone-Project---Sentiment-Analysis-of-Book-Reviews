################################################################################
# Capstone Project - Sentiment Analysis of Book Reviews
# Module-1 : Sentiment Extraction
#   This module is for data exploration, pre-processing, handing NOT's 
#   and extracting overall sentiment 
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

set.seed(10000)

par(mfrow=c(1,1))

################################################################################


#Part 2 -
#Read the review data into a data frame and perform initial processing 
#Remove non-required columns 
#Add a column for length of review text

#Read the csv files into a dataframe
dftemp <- read.csv2("Fillian_Flynn-Gone_Girl.csv", sep="\t", header = FALSE, stringsAsFactors = F)
df0 <- dftemp

dftemp <- read.csv2("Donna-Tartt-The-Goldfinch.csv", sep="\t", header = FALSE, stringsAsFactors = F)
df0 <- rbind(df0, dftemp)

dftemp <- read.csv2("Andy-Weir-The-Martian.csv", sep="\t", header = FALSE, stringsAsFactors = F)
df0 <- rbind(df0, dftemp)

dftemp <- read.csv2("Laura-Hillenbrand-Unbroken.csv", sep="\t", header = FALSE, stringsAsFactors = F)
df0 <- rbind(df0, dftemp)


#Remove the columns not required and retain only the score and review text. 
df0 <- df0[,-c(2,3)]
colnames(df0) <- c("Score","ReviewText")
m <- nrow(df0)

#Remove the html tags in the review text
df0$ReviewText <- as.character(df0$ReviewText)
df0$ReviewText <- gsub("<.*?>", " ", df0$ReviewText)

#Make the scores numeric
df0$Score <- as.numeric(df0$Score)

#Determine length of the review text and add that as a column to the dataframe
df0 <- cbind(df0, nchar(df0$ReviewText))
colnames(df0)[3] <- "ReviewLength"

#Add levels for the scores in alignment with the expected sentiment
ScoreLevels <- df0$Score
for (i in 1:m)
{
  if (df0$Score[i] < 3) 
  { 
    ScoreLevels[i] <- 1 
  }
  else 
  { 
    if (df0$Score[i] == 3)
    {
      ScoreLevels[i] <- 2
    }
    else
    {
      ScoreLevels[i] <- 3
    }
  }
}

df0 <- cbind(df0, factor(ScoreLevels, levels=c(1, 2, 3), labels=c("Negative", "Neutral", "Positive")))
colnames(df0)[4] <- "ScoreLevel"

################################################################################


#Part 3 -
#Analyze review-data and filter the data

# Distribution of Reviews
barplot(table(as.factor(df0$Score)), main="Distribution of Reviews", ylim=c(0,20000))

#Filter less frequently occuring very long reviews
hist(df0$ReviewLength, xlab = "Review Length", main = "Histogram for Review Length", col = "light blue")

#Check Review lengths by scores
par(mfrow=c(2,3))
maxScore <- max(df0$Score)
meanReviewLength <- c(1:maxScore)
sdReviewLength <- c(1:maxScore)
medianReviewLength <- c(1:maxScore)
for (i in 1:maxScore)
{ 
  meanReviewLength[i] <- mean(df0[df0$Score==i,]$ReviewLength)
  sdReviewLength[i] <- sd(df0[df0$Score==i,]$ReviewLength)
  medianReviewLength[i] <- median(df0[df0$Score==i,]$ReviewLength)
  barplot(table(df0[df0$Score==i,]$ReviewLength), main=i, col=i+1)
}

par(mfrow=c(1,1))

df2 <- df0[df0$ReviewLength > 1000, ]
df0 <- df0[df0$ReviewLength <= 1000, ]

hist(df0$ReviewLength)

df2 <- rbind(df2, df0[df0$ReviewLength > 60, ])
df0 <- df0[df0$ReviewLength <= 60, ]

hist(df0$ReviewLength)

m <- nrow(df0)

#Review Lengths by rating
boxplot(df0$ReviewLength ~ df0$Score)

################################################################################


#Part 4 -

#Part 4a
#Perform Text Mining using functions in tm package

rc <- Corpus(VectorSource(df0$ReviewText))

#Data transformation: lowercasing removing numbers, removing white spacing
rc <- tm_map(rc, tolower)
rc <- tm_map(rc, stripWhitespace)
rc <- tm_map(rc, removeNumbers)

#Convert negative contractions like isn't and doesn't to their expanded forms
data(contractions)
for (i in 1:m)
{
  df0$ReviewText[i] <- rc[[i]]$content
}

for (i in 1:nrow(contractions))
{
  df0$ReviewText <- gsub(contractions[i,1], contractions[i,2], df0$ReviewText)
}

rc <- Corpus(VectorSource(df0$ReviewText))
rc <- tm_map(rc, tolower)
rc <- tm_map(rc, stripWhitespace)
rc <- tm_map(rc, removeNumbers)


#Part 4b
#Sentiment Extraction - for each review determine the overall sentiment level
#This also includes the code for handling the NOTs

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

# Code to reversing the sentiments when preceeded by negating words 
negativeVerbs <- c("fail", "fails", "failed", "failing", "lack", "lacks", "lacked", "lacking", "devoid")

reviewSentiment <- c(1:m)
reviewSentiment[] <- 0
reviewSentiment <- as.integer(reviewSentiment)

s <- m
#s <- 5
i <- 1
for (i in 1:s)
{
  wordsInReview <- strsplit(rc[[i]]$content, " ")[[1]]
  #a <- "it is neither a good book nor a wonderful book"
  #wordsInReview <- strsplit(a, " ")[[1]]
  ignoreWord <- c(1:length(wordsInReview))
  ignoreWord[] <- 0
  wordSentiment <- c(1:length(wordsInReview))
  wordSentiment[] <- 0
  notCount <- 1
  for (j in 1:length(wordsInReview))
  {
    if (ignoreWord[j]==0)
    {
      wordSentiment[j] <- fn_getSentiment(wordsInReview[j])
      case <- 0
      if (j != length(wordsInReview) & wordsInReview[j] == "not" & ! wordsInReview[j+1] %in% c("only", "just"))
      { case <- 1 }
      if (j != length(wordsInReview) & wordsInReview[j] %in% negativeVerbs)
      { case <- 1}
      if (case != 0)
      { 
        stopNot <- 0
        k <- 0
        while(stopNot < 1)
        {
          k <- k + 1
          if (j+k <= length(wordsInReview) & wordsInReview[j+k] != "not")
          {
            ignoreWord[j+k] <- 1
            if ( fn_getSentiment(wordsInReview[j+k]) != 0)
            {
              stopNot <- 1
              wordSentiment[j] <- wordSentiment[j] + ((-1)^notCount) * fn_getSentiment(wordsInReview[j+k])
              if ((notCount %% 2) != 0) 
                { wordsInReview[j+k] <- paste("zzyyzzyy-", wordsInReview[j+k], sep="") }
              if (j+k+2 <= length(wordsInReview) & wordsInReview[j+k+1] %in% c("and", "or") & fn_getSentiment(wordsInReview[j+k+2]) != 0)
              {
                if ((notCount %% 2) != 0) 
                  { wordsInReview[j+k] <- paste("zzyyzzyy-", wordsInReview[j+k], sep="") }
                wordSentiment[j] <- wordSentiment[j] + ((-1)^notCount) * fn_getSentiment(wordsInReview[j+k+2])
                ignoreWord[j+k+1] <- 1
                ignoreWord[j+k+2] <- 1
                k <- k + 2
              }
              notCount <- 1
            }
          }
          else
          {
            stopNot <- 1
            if ( (j+k <= length(wordsInReview)) & (wordsInReview[j+k] == "not") )
            {
              stopMultipleNot <- 0
              if ( any ( wordsInReview[(j+1):(j+k-1)] %in% c("and", "or", "but") ) == TRUE | any ( grepl("[[:punct:]]", wordsInReview[(j+1):(j+k-1)]) ) == TRUE )
              { 
                stopMultipleNot <- 1
              }
              else
              { 
                notCount <- notCount + 1
              }
            }
          }
        }
      }
      reviewSentiment[i] <- reviewSentiment[i] + wordSentiment[j]   
    }
  }
  df0$ReviewText[i] <- combine_words(wordsInReview, sep=" ", and="")
}
reviewSentiment[1:5]

df0 <- cbind(df0, reviewSentiment)
colnames(df0)[5] <- "ReviewSentiment"

df1 <- df0[1:s, ]
#df1 <- df1[, -c(2,3,4)]
df1 <- df1[with(df1, order(Score, ReviewSentiment)), ]
#df1$Score <- as.factor(df1$Score)
#df1$ReviewSentiment <- as.factor(df1$ReviewSentiment)


#Part 4c
#Analyze oouput of Sentiment Extraction
#Plot the distribution of Sentiment scores for every value of review score in the original data

par(mfrow=c(2,3))
meanSentiment <- c(1:maxScore)
sdSentiment <- c(1:maxScore)
medianSentiment <- c(1:maxScore)
modeSentiment <- c(1:maxScore)
for (i in 1:maxScore)
{ 
  meanSentiment[i] <- mean(df1[df1$Score==i,]$ReviewSentiment)
  sdSentiment[i] <- sd(df1[df1$Score==i,]$ReviewSentiment)
  medianSentiment[i] <- median(df1[df1$Score==i,]$ReviewSentiment)
  modeSentiment[i] <- mode(df1[df1$Score==i,]$ReviewSentiment)
  barplot(table(df1[df1$Score==i,]$ReviewSentiment), main=i, col=i+1, ylim=c(0,5000))
}
plot(1:maxScore, meanSentiment, type = "o", col="red")

write.csv(df1, file = "Module1_Review Data with sentiments.csv")

################################################################################
