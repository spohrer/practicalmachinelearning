# analysis 
setwd("~/Documents/online-learning/data-science-practical-machine-learning-2014/code")

testing <- read.csv("pml-testing.csv")
training <- read.csv("pml-training.csv", na.strings=c("","NA"))

# convert all character colummns to numeric, except classe


dim(training)
dim(testing)



str(training)

require(Hmisc)
describe(training)




# first cut at a model

# training2 <- training[,8:ncol(training)]
# dim(training2)

# test df <- training2[1:50, 1:50]; i = 1
# out <- data.frame( )
# fixdataframe <- function(df){
#     for( i in 1:ncol(df)){
#         temp <- df[,i]
#         if(class(temp) == "character") { temp <- as.numeric(temp)}
#         if( i == 1) {out <- as.data.frame(temp) } else { out <- cbind(out, temp) }
#     }
#     return(out)
# }


# let's try getting rid of the summary statistics columns first
# 
 idx<-apply(training, 2, function(x) sum(is.na(x)))
 describe(idx)

# ANSWER: this shows that there were 60 columns that had values for every observation,
# and 100 columns with 19216 NA's in that column
# so create a column index of which columns to keep

keepthesecolumns <- ifelse(idx == 0, TRUE, FALSE)

# new training set without the columns of summary statistics
training2 <- training[,keepthesecolumns]
testing2 <- testing[, keepthesecolumns]
dim(training2)
dim(testing2)
str(training2)
str(testing2)

# now get rid of the first 7 fields
training3 <- training2[,8:ncol(training2)]
testing3 <- testing2[,8:ncol(testing2)]
dim(training3)
dim(testing3)

# now turn the class variable 'classe' into a factor
training3$classe <- factor(as.character( training3$classe) )

# since randomforest takes a long time,
# let's train against smaller subsets

TEST = TRUE
if(TEST) {
index <- sample(nrow(training),size=500, replace= FALSE )
training3 <- training3[ index, ]
dim(training3)
}


# actually, the data should be split into a training set, 
# a testing set, and saving the 20 obs as a validation set

 
library(caret)

# create trainSet, testSet, validationSet
inTrain <- createDataPartition( y=training3$classe, p=0.7,  list=FALSE)
trainSet <- training3[inTrain, ]
testSet <- training3[-inTrain, ]
validationSet <- testing3

dim(trainSet)
dim(testSet)
dim(validationSet)

# from Applied Predictive Modeling, page 436
#fiveStats <- function(...) { c(twoClassSummary(...), defaultSummary(...))}
#ctrl <- trainControl(method="cv", classProbs= FALSE, 
#                     summaryFunction=fiveStats, verboseIter=TRUE )

set.seed(12345)
ctrl <- trainControl(method="oob", number=5)
modelFit <- train(classe~., data=trainSet, method="rf", 
                  trControl=ctrl, ntree=500, tuneLength=5, metric="Accuracy")

modelFit

#getTree(modelFit$finalModel, k=2)

pred <- predict(modelFit, trainSet)
print("Model accuracy based on training set")
table(pred, trainSet$classe)

pred <- predict(modelFit, testSet)
print("Model accuracy based on held out testing set")
table(pred, testSet$classe)


# confusion matrix

cm <- confusionMatrix(data= pred,  # predicted results
                reference= testSet$classe  )    # true results
cm
myAccuracy <- cm$overall["Accuracy"]


# look at variable importance

varImp(modelFit)
varlist <- varImp(modelFit)

vli <- varlist$importance[ order(varlist$importance$Overall, decreasing=TRUE), ,drop=FALSE]
numShown <- 10
myBars <- as.matrix(vli[1:numShown,])
myNames <- rownames(vli)[1:numShown]
# par(las=1)
op <- par(no.readonly = TRUE) # the whole list of settable par's.
par(mai=c(1,2,1,1)+0.1 ) # expand left margin)

barplot(myBars,
        horiz=TRUE,
        beside = TRUE, 
        names.arg = myNames,
        cex.names=0.7,
        las=1,  # asix labels always horizontal
        xlab="Relative Importance",
        main=paste0("Top ", as.character(numShown), " Variables ranked by Importance"))

par(op)

testpred <- predict(modelFit, testing3)
print("Predictions made on validation set")
testpred

answers <- as.character(testpred)

# submission

# answers = rep("A", 20) ie as a character vector

pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

# this will write out twenty separate files, one for each answer
# into your working directory

pml_write_files(answers)


