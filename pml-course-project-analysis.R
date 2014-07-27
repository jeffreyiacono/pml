# Practical Machine Learning: Course Project
# see: https://class.coursera.org/predmachlearn-003/human_grading/view/courses/972148/assessments/4/submissions
#
# Assumes that the pml-training.csv and pml-test.csv are in the same directory
# as this script.

install.packages("caret")
library(caret)

# read training data in, normalize bad values
plmTrain <- read.csv("pml-training.csv", header = TRUE, na.strings = c("NA","", "#DIV/0!"))
# remove any cols that are missing data
plmTrain <- plmTrain[, names(plmTrain)[sapply(plmTrain, function(x) { !any(is.na(x)) })]]
# remove non-numerics
plmTrain <- plmTrain[, -c(1, 2, 3, 4, 5, 6, 7)]

# create training / test
inTrain <- createDataPartition(y = plmTrain$classe, p = 0.7, list = FALSE)
training <- plmTrain[inTrain, ]
testing <- plmTrain[-inTrain, ]

# inspect for low variation
nearZeroVar(training[, -53], saveMetrics = TRUE)

# inspect for high correlation
M <- abs(cor(training[, -53]))
diag(M) <- 0
which(M > 0.95, arr.ind = TRUE)

# due to correlation between features, perform principle components
preProc <- preProcess(training[, -53], method = "pca", threshold = 0.95)
trainPC <- predict(preProc, training[,-53])

# train the model
modelFit <- train(training$classe ~ ., method = "rf", data = trainPC)

# test
testPC <- predict(preProc, testing[, -53])
confusionMatrix(testing$classe, predict(modelFit, testPC))

# generate predictions
plmTest <- read.csv("pml-testing.csv", header = TRUE)
plmTestPC <- predict(preProc, plmTest[, names(training[, -53])])
predict(modelFit, plmTestPC)
