#### *c-spohrer* {.author}

#### *06/18/2014* {.date}

**SYNOPSIS**

We apply random forests as a machine learning tool to measurements of
arm motion while a subject performs bicep curls to classify whether they
have performed the exercise correctly.

**DATA SET**

The data set consists of measurements from 6 participants as to how well
they performed dumbell exercises. The acceleromters that recorded the
measurements were attached to variuous parts of their arms. The
particpants performed the Bicep Curl in 1 correct method and in 4
specific incorrect methods, resulting in 5 classes that will be
predicted from the measurements.

The correct class is Class A, the incorrect, but distinct classes are B,
C, D and E.

There are 159 independent variables, of which nearly 100 variables are
descriptive statistics of the other fields, and so will be eliminated
from the model.

**APPROACH**

We will use the caret pacakge and the R statistical language to perform
our analysis and to model our predictions. The data is provided as one
file for training, and a smaller file of 20 records for validation and
submission for grading purposes.

For our purposes, we split the training file into a training set of 70%
of the records, and a testing set with the remaining 30% of the
observations, leaving the 20 records in the seperate testing file as our
validation set.

**PRE-PROCESSING**

The classification variable is ‘classe’ and consists of the values: A,
B, C, D, E; with A = correct performance of the exercise, and the others
correspond to intentionally performing the exercise wrong.

The first 7 independent variables are descriptive of the observation,
and so will be eliminated. They are: \* raw\_timestamp\_part\_1 \*
raw\_timestamp\_part\_2 \* cvtd\_timestamp \* new\_window \* num\_window

As mentioned above, many variables are descriptive statistics of the raw
measurements, and so they will be eliminated as well. These variablers
start with names such as: \* kurtosis \* skewness \* max \* min \* var
\* avg \* stddev \* total

Eliminating these variables also had the added benefit of eliminated any
problems with missing values.

**EXPERIMENT**

The following analysis, code and results hightlight the steps taken to
apply the random forest machine learning technique to the data set. The
data is prepared, split into training and testing sets, the algorithm
applied, classification results reported, and the model applied to a
held-out set of twenty observations.

``` {.r}
# analysis 
# setwd("~/Documents/online-learning/data-science-practical-machine-learning-2014/code")

# read in the data sets
testing <- read.csv("pml-testing.csv")
training <- read.csv("pml-training.csv", na.strings=c("","NA"))

dim(training)
```

    ## [1] 19622   160

``` {.r}
dim(testing)
```

    ## [1]  20 160

``` {.r}
# take a closer look at the data
# str(training)
# require(Hmisc)
# describe(training)




# let's try getting rid of the summary statistics columns first
# determine which columns have a lot of NA's'
 
 idx<-apply(training, 2, function(x) sum(is.na(x)))

# describe(idx)

# RESULTS: this shows that there were 60 columns that had values for every observation,
# and 100 columns with 19216 NA's in that column
# so create a column index of which columns to keep

keepthesecolumns <- ifelse(idx == 0, TRUE, FALSE)

# new training set without the columns of summary statistics

training2 <- training[,keepthesecolumns]
testing2 <- testing[, keepthesecolumns]
dim(training2)
```

    ## [1] 19622    60

``` {.r}
dim(testing2)
```

    ## [1] 20 60

``` {.r}
#str(training2)
#str(testing2)

# now get rid of the first 7 fields - since these were descriptive as well
training3 <- training2[,8:ncol(training2)]
testing3 <- testing2[,8:ncol(testing2)]
dim(training3)
```

    ## [1] 19622    53

``` {.r}
dim(testing3)
```

    ## [1] 20 53

``` {.r}
# now turn the class variable 'classe' into a factor
training3$classe <- factor(as.character( training3$classe) )

# since randomforest takes a long time,
# let's train against smaller subsets for testing purposes

TEST =  FALSE # TRUE # FALSE
if(TEST) {
index <- sample(nrow(training),size=500, replace= FALSE )
training3 <- training3[ index, ]
dim(training3)
}


# Split into a training set, and a testing set, 
# and save the 20 obs as a validation set

 
library(caret)
```

    ## Loading required package: lattice
    ## Loading required package: ggplot2

``` {.r}
# create trainSet, testSet, validationSet
inTrain <- createDataPartition( y=training3$classe, p=0.7,  list=FALSE)
trainSet <- training3[inTrain, ]
testSet <- training3[-inTrain, ]
validationSet <- testing3

dim(trainSet)
```

    ## [1] 13737    53

``` {.r}
dim(testSet)
```

    ## [1] 5885   53

``` {.r}
dim(validationSet)
```

    ## [1] 20 53

``` {.r}
set.seed(12345)

# use out-of-bag sampling instead of cross-validation to find the
# best choice of paramters, ie. mtry
ctrl <- trainControl(method="oob", number=5)
modelFit <- train(classe~., data=trainSet, method="rf", 
                  trControl=ctrl, ntree=500, tuneLength=5, metric="Accuracy")
```

    ## Loading required package: randomForest
    ## randomForest 4.6-7
    ## Type rfNews() to see new features/changes/bug fixes.

``` {.r}
modelFit
```

    ## Random Forest 
    ## 
    ## 13737 samples
    ##    52 predictors
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Out of Bag Resampling 
    ## 
    ## Summary of sample sizes:  
    ## 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy  Kappa
    ##   2     1         1    
    ##   10    1         1    
    ##   30    1         1    
    ##   40    1         1    
    ##   50    1         1    
    ## 
    ## Accuracy was used to select the optimal model using  the largest value.
    ## The final value used for the model was mtry = 14.

``` {.r}
pred <- predict(modelFit, trainSet)
print("Model accuracy based on training set")
```

    ## [1] "Model accuracy based on training set"

``` {.r}
table(pred, trainSet$classe)
```

    ##     
    ## pred    A    B    C    D    E
    ##    A 3906    0    0    0    0
    ##    B    0 2658    0    0    0
    ##    C    0    0 2396    0    0
    ##    D    0    0    0 2252    0
    ##    E    0    0    0    0 2525

``` {.r}
pred <- predict(modelFit, testSet)
print("Model accuracy based on held out testing set")
```

    ## [1] "Model accuracy based on held out testing set"

``` {.r}
table(pred, testSet$classe)
```

    ##     
    ## pred    A    B    C    D    E
    ##    A 1672    6    0    0    0
    ##    B    0 1133    4    0    0
    ##    C    1    0 1021    4    4
    ##    D    0    0    1  959    2
    ##    E    1    0    0    1 1076

``` {.r}
# confusion matrix

cm <- confusionMatrix(data= pred,  # predicted results
                reference= testSet$classe  )    # true results
cm
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1672    6    0    0    0
    ##          B    0 1133    4    0    0
    ##          C    1    0 1021    4    4
    ##          D    0    0    1  959    2
    ##          E    1    0    0    1 1076
    ## 
    ## Overall Statistics
    ##                                         
    ##                Accuracy : 0.996         
    ##                  95% CI : (0.994, 0.997)
    ##     No Information Rate : 0.284         
    ##     P-Value [Acc > NIR] : <2e-16        
    ##                                         
    ##                   Kappa : 0.995         
    ##  Mcnemar's Test P-Value : NA            
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity             0.999    0.995    0.995    0.995    0.994
    ## Specificity             0.999    0.999    0.998    0.999    1.000
    ## Pos Pred Value          0.996    0.996    0.991    0.997    0.998
    ## Neg Pred Value          1.000    0.999    0.999    0.999    0.999
    ## Prevalence              0.284    0.194    0.174    0.164    0.184
    ## Detection Rate          0.284    0.193    0.173    0.163    0.183
    ## Detection Prevalence    0.285    0.193    0.175    0.163    0.183
    ## Balanced Accuracy       0.999    0.997    0.997    0.997    0.997

``` {.r}
myAccuracy <- cm$overall["Accuracy"] * 100
myAccuracy <- as.character(round(myAccuracy, 2))


# look at variable importance

# Report on the variables with the most importance
varImp(modelFit)
```

    ## rf variable importance
    ## 
    ##   only 20 most important variables shown (out of 52)
    ## 
    ##                      Overall
    ## roll_belt              100.0
    ## yaw_belt                64.3
    ## pitch_forearm           61.1
    ## magnet_dumbbell_z       52.9
    ## pitch_belt              49.9
    ## magnet_dumbbell_y       48.9
    ## roll_forearm            41.6
    ## roll_dumbbell           24.6
    ## magnet_dumbbell_x       24.5
    ## accel_dumbbell_y        23.4
    ## accel_belt_z            23.0
    ## magnet_belt_z           21.6
    ## magnet_belt_y           20.3
    ## accel_forearm_x         19.0
    ## accel_dumbbell_z        17.6
    ## magnet_forearm_z        16.2
    ## roll_arm                15.3
    ## gyros_belt_z            14.8
    ## total_accel_dumbbell    13.9
    ## yaw_arm                 13.6

``` {.r}
varlist <- varImp(modelFit)

# plot those variables with the  most importance
vli <- varlist$importance[ order(varlist$importance$Overall, decreasing=TRUE), ,drop=FALSE]
numShown <- 10
myBars <- as.matrix(vli[1:numShown,])
myNames <- rownames(vli)[1:numShown]

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
```

![plot of chunk
unnamed-chunk-1](data:image/png;base64,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 "plot of chunk unnamed-chunk-1")

``` {.r}
par(op)

testpred <- predict(modelFit, testing3)
print("Predictions made on validation set")
```

    ## [1] "Predictions made on validation set"

``` {.r}
testpred
```

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E

``` {.r}
# write out the files for submission

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
```

**CONCLUSION**

We have shown that the random forest algorithm can be applied to this
data with excellent results. The classification accuracy is greater then
99.59% for predicting all 5 classes, thus showing we can differentiate
between a bicep curl performed correctly, or performed in 4 other
incorrect variations.

**REFERENCE:**

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H.
Qualitative Activity Recognition of Weight Lifting Exercises.
Proceedings of 4th International Conference in Cooperation with SIGCHI
(Augmented Human ’13) . Stuttgart, Germany: ACM SIGCHI, 2013.

Read more:
[http://groupware.les.inf.puc-rio.br/har\#ixzz350q5jpWJ](http://groupware.les.inf.puc-rio.br/har#ixzz350q5jpWJ)
