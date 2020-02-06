require(dplyr)
require(gbm)
load(file="2014flights.Rdata")
df <- df %>% filter(CANCELLED==0)
#df<-df[,-c(4,6,8,11,12,13,20,24,25,26,27,28,29)]
df<-df[,-c(4,6,8,9,11,12,13,15,16,20,24,28,29)]
df<-na.omit(df)
format(object.size(df),units='MiB')
Y<- df$ARR_DELAY
X<- df[-13]


#splitting datat into training and testing set
smp_size <- floor(0.7 * nrow(df))

set.seed(123)
train_ind <- sample(seq_len(nrow(df)), size = smp_size)

X_train <- X[train_ind, ]
X_test <- X[-train_ind, ]

Y_train <- Y[train_ind]
Y_test <- Y[-train_ind]

out.boost<- gbm(Y_train~.,data=X_train,
              distribution="gaussian",
              n.trees=1000,
              shrinkage=0.01,
              interaction.depth=3,
              bag.fraction = 0.5,
              n.minobsinnode = 10,
              cv.folds = 3,
              keep.data=TRUE,
              verbose=TRUE,
              n.cores=8)
boost.sum<-summary(out.boost)
f.predict<- predict(out.boost,X_test)
print(mean((Y_test-f.predict)**2))
most.influential = which(names(X_train)%in%boost.sum[1:1,1])
plot(out.boost,i.var=most.influential)
