misClass =function(pred.class,true.class,produceOutput=FALSE){
  confusion.mat = table(pred.class,true.class)
  if(produceOutput){
    return(1-sum(diag(confusion.mat))/sum(confusion.mat))
  } else{
    print('misclass')
    print(1-sum(diag(confusion.mat))/sum(confusion.mat))
    print('confusion mat')
    print(confusion.mat)
} } 

load('Ytrain.Rdata')
load('Yvalidate.Rdata')
load('Xtrain.Rdata')
load('Xvalidate.Rdata')
load('Xtest.Rdata')

nTrain    = nrow(Xtrain)
nValidate = nrow(Xvalidate)
nTest     = nrow(Xtest)

require(randomForest)
#####
# Imputation
#    We are going to do a supervised imputation Xtrain and Xvalidate at the same time
#    and then impute Xtest with medians/modes
#    Note: the decision to impute both Xtrain and Xvalidate at the same
#          time isn't a simple one.  Arguably, we shouldn't use the validation
#          data at all until we want to compare methods to each other.
#          However, as every method gets to use the imputed data, it should
#          still be a fair comparison and a better imputation (due to using
#          more data)
#####

#Impute Xtrain and Xvalidate with rfimpute
XtrainAndValidate        = rbind(Xtrain,Xvalidate)
YtrainAndValidate        = as.factor(c(Ytrain,Yvalidate))
XtrainAndValidateImputed = rfImpute(XtrainAndValidate,YtrainAndValidate)

#(note: first column is the supervisor)
XtrainImputed    = XtrainAndValidateImputed[1:nTrain,-1]
XvalidateImputed = XtrainAndValidateImputed[(nTrain+1):(nTrain+nValidate),-1]

#Impute Xtest with roughFix
XtestImputed     = na.roughfix(Xtest)

########################
# Get predictions for each considered method
########################
YvalidateHat = list()
YtestHat     = list()

#####
# Find important features and get logistic lasso predictions
#####
require(glmnet)
#I used ~. instead of ~.-1 to encode an intercept in the feature matrix
XtrainImputedMM    = model.matrix(~.,XtrainImputed)
XvalidateImputedMM = model.matrix(~.,XvalidateImputed)
XtestImputedMM     = model.matrix(~.,XtestImputed)

#Note that we are rescaling categorical features
out.glmnet  = cv.glmnet(x=XtrainImputedMM,y=Ytrain,
                        family='binomial',intercept=FALSE)

#check to see if CV curve has a good solution
plot(out.glmnet)

#these drop commands are due to glmnet reporting results as a matrix
#  (note: this is for extension to G > 2 with 'multinomial')
YvalidateHat[['glmnet']] = drop(predict(out.glmnet,XvalidateImputedMM,
                                        s='lambda.min',type='class'))
YtestHat[['glmnet']]     = drop(predict(out.glmnet,XtestImputedMM,
                                        s='lambda.min',type='class'))

#We can still do relaxed lasso using logistic regression, drop intercept entry
(S_lambdamin = predict(out.glmnet,s='lambda.1se',type='nonzero')[,1])
#manually add column of ones for intercept:
XtrainImputedMM_int    = XtrainImputedMM[,S_lambdamin]
XvalidateImputedMM_int = XvalidateImputedMM[,S_lambdamin]
XtestImputedMM_int     = XtestImputedMM[,S_lambdamin]
out.refit   = glm.fit(x=XtrainImputedMM_int,y=factor(Ytrain),
                      family=binomial(link = "logit"))

probValidateHat = exp(XvalidateImputedMM_int %*%out.refit$coefficients)/
                    (1+exp(XvalidateImputedMM_int %*%out.refit$coefficients))
Yhat = rep('fraud',nValidate)
Yhat[probValidateHat > 0.5] = 'no fraud'#glm codes it in alphabetical order
YvalidateHat[['refit']] = Yhat
probTestHat     = exp(XtestImputedMM_int %*%out.refit$coefficients)/
                    (1+exp(XtestImputedMM_int %*%out.refit$coefficients))
Yhat = rep('fraud',nValidate)
Yhat[probTestHat > 0.5] = 'no fraud'#glm codes it in alphabetical order
YtestHat[['refit']] = Yhat


#####
# Find randomForest predictions
#####
require(randomForest)
out.rf = randomForest(x=XtrainImputed,y=as.factor(Ytrain),ntree=200)

#check to see if OOB error rate has stabilized
plot(out.rf)

probValidateHat = predict(out.rf, XvalidateImputed, type='prob')
probTestHat     = predict(out.rf, XtestImputed, type='prob')
threshGrid = (3:7)/10
for(thresh in threshGrid){
  methodName = paste(c('rf_',as.character(thresh)),collapse='')
  tmp = rep('no fraud',nValidate)
  tmp[probValidateHat[,1] > thresh] = 'fraud'
  YvalidateHat[[methodName]] = tmp
  tmp = rep('no fraud',nTest)
  tmp[probTestHat[,1] > thresh] = 'fraud'
  YtestHat[[methodName]]     = tmp 
}

#####
# Find randomForest predictions: nominal -> interval
#####
require(randomForest)
XtrainImputed_ord    = XtrainImputed
XvalidateImputed_ord = XvalidateImputed
XtestImputed_ord     = XtestImputed

XtrainImputed_ord$V255    = as.numeric(XtrainImputed_ord$V255)
XtrainImputed_ord$V256    = as.numeric(XtrainImputed_ord$V256)
XvalidateImputed_ord$V255 = as.numeric(XvalidateImputed_ord$V255)
XvalidateImputed_ord$V256 = as.numeric(XvalidateImputed_ord$V256)
XtestImputed_ord$V255     = as.numeric(XtestImputed_ord$V255)
XtestImputed_ord$V256     = as.numeric(XtestImputed_ord$V256)

out.rf_ord = randomForest(x=XtrainImputed_ord,y=as.factor(Ytrain),ntree=200)

#check to see if OOB error rate has stabilized
plot(out.rf_ord)

YvalidateHat[['rf_ord']] = predict(out.rf_ord, XvalidateImputed_ord, type='class')
YtestHat[['rf_ord']]     = predict(out.rf_ord, XtestImputed_ord, type='class')

#############
# Get misclassification rates and output predictions
############# 
lapply(YvalidateHat,misClass,Yvalidate,produceOutput = TRUE)
#Here, I would choose the threshold 0.3
#Let's look at what the test estimate of the risk would be
load('Ytest.Rdata')
lapply(YtestHat,misClass,Ytest,produceOutput = TRUE)



#Note: Once we settle on to a method, we should combine the training and 
#      validation data together to train a classifier to predict the test 
#      data 

#############
# Lastly, you should consider applying boosting to this data set.  Note
#   that GBM has functionality for automatically accounting for missing values
#############

