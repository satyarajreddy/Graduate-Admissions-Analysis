rm(list = ls())
# options(repos = c(CRAN = "http://cran.rstudio.com"))
setwd("C:/Users/Satyaraj/Downloads/Big Data Analytics CIS8695 Term Project")

library(tidyverse)
library(ggplot2)

grad.df <- read.csv("Admission_Predict_Ver1.1.csv")
attach(grad.df)
names(grad.df) <- c("serial", "gre", "toefl","uni_rating","sop","lor","gpa","research","admit")
head(grad.df, 5)


ggplot(grad.df,aes(gre,color=factor(research)))+geom_density(size=2)+ggtitle("GRE vs Research Distribution")

#We can see from above density plot that students with research experience are more likely to have a higher GRE score.

ggplot(grad.df,aes(gre,admit))+geom_point()+geom_smooth()+ggtitle("GRE vs Admit Chance")



ggplot(grad.df,aes(toefl,admit))+geom_point()+geom_smooth()+ggtitle("TOEFL vs Admit Chance")

boxplot(grad.df$gre,col="#0099FF",
        horizontal=TRUE,xlab="GRE",main="Boxplot for GRE")

hist(grad.df$gre,col="#0099FF",
     xlab="GRE",
     ylab="Frequency",
     main="Histogram for GRE",
     labels=TRUE)

#We can see from above plots that the median of GRE is around 318. Also, one should reach 325 if he/she wants to be top 25%.

ggplot(grad.df,aes(gpa,admit))+geom_point(color="#339966")+facet_grid(research~.)

# I want to research the correlation between GPA and admission rate. I find useful to divide it into two groups- students with research experience and students who do not. When compared to students wit research expience, the other group of students barely get 9+ gpa. If you did research before and you gpa is higher than 9, we can be pretty sure you have a BIG chance to be admitted.

ggplot(data = grad.df) + 
  geom_point(mapping = aes(x=gre, y =toefl, color = admit)) + 
  facet_wrap(~ research)+
  xlab("GRE Score")+
  ylab("TOEFL Score")
library(caret)

grad1.df<-grad.df[complete.cases(grad.df),]
grad1.df<-grad1.df[,-1]
# get 80% index numbers of original dataset
validation.index <- createDataPartition(grad1.df$admit, p=0.80, list=FALSE)
# choose 20% data as test dataset
validation.data <- grad1.df[-validation.index,]
# 80% choose 80% data as train dataset
train.data <- grad1.df[validation.index,]

#Random Forest

library(randomForest)

rf<-randomForest(admit~.,data=train.data,importance=TRUE)
print(rf)

importance(rf)

varImpPlot(rf)

# We can see that GPA, GRE and TOEFL are most important factors in grad school application

pred_rf <- predict(rf, validation.data)
confusionMatrix(pred_rf, as.factor(validation.data$admit))
rmse=RMSE(pred_rf,validation.data$admit)
r2=R2(pred_rf, validation.data$admit, form = "traditional")
cat("RMSE:", rmse,"\nR^2:", r2)



# Linear Regression Model


lm_model1<-lm(train.data$admit~gre+toefl+lor+gpa+research,train.data)
summary(lm_model1)

pred_lm=predict(lm_model1,validation.data)

rmse=RMSE(pred_lm,validation.data$admit)
r2=R2(pred_lm, validation.data$admit, form = "traditional")
cat("RMSE:", rmse,"\nR^2:", r2)

summary(pred_lm)

########################################Data Imputation################################
#there is one outlier in LOR and chance of admit.the lor outlier is 1 and the chance of admit is 0.34.

#to impute the outlier. I will replace LOR with mode
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}
modeLOR <- getmode(grad.df$lor)
print(modeLOR)

grad.df$lor[grad.df$lor ==1] <- 3

#replacing chance of admit with median which is 0.72
grad.df$admit[grad.df$admit==0.34] <- 0.72

#converting "research" to categorical variable
grad.df$research[grad.df$research ==1] <- "Yes"
grad.df$research[grad.df$research ==0] <- "No"

grad.df$research <- as.factor(grad.df$research) #2 is yes, 1 is no

#creating a new categorical variable using the existing variable

grad.df$chance_low_high[grad.df$admit <= 0.50 ] <- "low"
grad.df$chance_low_high[grad.df$admit > 0.50 ] <- "high"

grad.df$chance_low_high <- as.factor(grad.df$chance_low_high)


#removing unwanted columns
grad.df$serial <- NULL
grad.df$admit <- NULL

str(grad.df)

##KNN

set.seed(30)
new <- createDataPartition(y = grad.df$chance_low_high,p = 0.65,list = FALSE) #creating train and test dataset
grad_train <-grad.df[new,]
grad_test <- grad.df[-new,]
con <- trainControl(method = "repeatedcv", number = 2, repeats = 5)
knn <- train(chance_low_high ~ ., data = grad_train,
             method ="knn", trControl = con, preProcess = c("center","scale"))
predict <- predict(knn,newdata = grad_test) 
head(predict)

confusionMatrix(predict, grad_test$chance_low_high)

#using caret package we have accuracy of 94% with no false negatives. 

#############different k-values##################################
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 2)
set.seed(3333)
knn_fit <- train(chance_low_high ~., data = grad_train, method = "knn",trControl=trctrl,preProcess = c("center", "scale"),tuneLength = 10)
knn_fit

df <- knn_fit$results
ggplot(data = df)+
  geom_line(mapping = aes(x = df$k, y = df$Accuracy))+
  xlab("K values")+
  ylab("Accuracy")

#Building Ensemble Methods
set.seed(1)

str(grad.df)


#Spliting training set into two parts based on outcome: 75% and 25%
index <- createDataPartition(grad.df$chance_low_high, p=0.75, list=FALSE)
trainSet_ensemble <- grad.df[ index,]
testSet_ensemble <- grad.df[-index,]

#Defining the training controls for multiple models
fitControl <- trainControl(
  method = "cv",
  number = 5,
  savePredictions = 'final',
  classProbs = T)

#Defining the predictors and outcome
predictors<-c("gre", "toefl", "gpa")
outcomeName<-'chance_low_high'

#Training the Logistic regression model
model_lr<-train(trainSet_ensemble[,predictors],trainSet_ensemble[,outcomeName],method='glm',trControl=fitControl,tuneLength=3)

#Predicting using logistic regressions model
testSet_ensemble$pred_lr<-predict(object = model_lr,testSet_ensemble[,predictors])

#Checking the accuracy of the Logistic regression model
confusionMatrix(testSet_ensemble$chance_low_high,testSet_ensemble$pred_lr) #the accuracy is 93%, slightly higher than the previous two models


#Training the knn model
model_knn<-train(trainSet_ensemble[,predictors],trainSet_ensemble[,outcomeName],method='knn',trControl=fitControl,tuneLength=3)

#Predicting using knn model
testSet_ensemble$pred_knn<-predict(object = model_knn,testSet_ensemble[,predictors])

#Checking the accuracy of the knn model
confusionMatrix(testSet_ensemble$chance_low_high,testSet_ensemble$pred_knn)

#averaging the predictions from each model. Since we are predicting whether the chance of admit is high or low, we are averaging the probabilities 
#Predicting the probabilities
testSet_ensemble$pred_knn_prob<-predict(object = model_knn,testSet_ensemble[,predictors],type='prob')
testSet_ensemble$pred_lr_prob<-predict(object = model_lr,testSet_ensemble[,predictors],type='prob')

#Taking average of predictions
testSet_ensemble$pred_avg<-(testSet_ensemble$pred_knn_prob$high+testSet_ensemble$pred_lr_prob$high)/2



#Splitting into binary classes at 0.5
testSet_ensemble$pred_avg<-as.factor(ifelse(testSet_ensemble$pred_avg>0.5,'X1','X0'))

ensemble.averaging<-confusionMatrix(testSet_ensemble$chance_low_high,testSet_ensemble$pred_avg)
#Neural Network


library(NeuralNetTools)
library(neuralnet)
library(nnet)
library(caret)
grad_nn.df <- read.csv("Admission_Predict_Ver1.1.csv")
grad_nn.df = subset(grad_nn.df, select = -c(1) )
set.seed(2)
training<-sample(row.names(grad_nn.df), dim(grad_nn.df)[1]*0.6)
validation<-setdiff(row.names(grad_nn.df), training)

trainData <- grad_nn.df[training,]
validData <- grad_nn.df[validation,]

nn2<-nnet(Chance.of.Admit~.,size=5,data=trainData)

plotnet(nn2)
neuralweights(nn2)
olden(nn2)


grad_nn.df <- read.csv("Admission_Predict_Ver1.1.csv")
grad_nn.df = subset(grad_nn.df, select = -c(1) )
normalize=function(x)
{
  return ((x-min(x))/(max(x)-min(x)))
}
admission_norm=as.data.frame(lapply(grad_nn.df,normalize))
summary(admission_norm)
pairs.panels(admission_norm)
admission_train=admission_norm[1:299,]
admission_test=admission_norm[300:500,]
admission_ann=neuralnet(Chance.of.Admit~.,data=admission_norm,hidden=c(3,2))
plot(admission_ann,rep="best")

model_results = compute(admission_ann, admission_test)
predicted_admit = model_results$net.result
cor(predicted_admit, admission_test$Chance.of.Admit)