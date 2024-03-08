
###
### This is suggested solution for the highnote freemium exercise which asks you
### to create a predictive model of adoption for the highnote freemium dataset.
### This script does the following:
###  0) sets up the environment
###  1) imports the freemium dataset from a text file
###  2) creates another version of the data with all missing values recoded to their mean
###  3) computes descriptive statistics and plots
###  4) estimates a tree and logistic regression model
###     for each of these models it computes predictions, a confusion matrix, and lift
###     of those observations in the top decile.
###




###############################################################################
### setup
###############################################################################

# setup environment, for plots
if (!require(reshape2)) {install.packages("reshape2"); library(reshape2)}
if (!require(gplots)) {install.packages("gplots"); library(gplots)}
if (!require(ggplot2)) {install.packages("ggplot2"); library(ggplot2)}
# setup environment, make sure this library has been installed
if (!require(tree)) {install.packages("tree"); library(tree)}
# setup environment (if you want to use fancy tree plots)
if (!require(rpart)) {install.packages("rpart"); library(rpart)}
if (!require(rattle)) {install.packages("rattle"); library(rattle)}
if (!require(rpart.plot)) {install.packages("rpart.plot"); library(rpart.plot)}
if (!require(RColorBrewer)) {install.packages("RColorBrewer"); library(RColorBrewer)}
if (!require(party)) {install.packages("party"); library(party)}
if (!require(partykit)) {install.packages("partykit"); library(partykit)}
# a better scatterplot matrix routine
if (!require(car)) {install.packages("car"); library(car)}
# better summary tables
if (!require(psych)) {install.packages("psych"); library(psych)}
# tools for visualizing models
if (!require(ROCR)) {install.packages("ROCR"); library(ROCR)}  # ROC curve for tree or logistic

# import dataset from file (!!change the directory to where your data is stored!!)
setwd("~/Documents/class/marketing analytics/cases/freemium/data")
freemium=read.csv("High Note data csv.csv")




###############################################################################
### prepare the dataset for analysis
###############################################################################

# compute the number of observations in the freemium dataset
nobs=nrow(freemium)

# set the random number seed so the samples will be the same if regenerated
set.seed(1248765792)

# prepare new values using a uniform random number, each record in freemium has 
# a corresponding uniform random value which will be used to decide if the observation
# is assigned to the training, validation or prediction sample
randvalue=runif(nobs)
trainsample=randvalue<.6
validsample=(randvalue>=.6 & randvalue<.9)
predsample=(randvalue>=.9)
plotsample=sample(1:nrow(freemium),300)

# copy the dataset to one that has recoded values for all missing values
# this adds the columns age_Missing, male_Missing, good_country_Missing, shouts_Missing
# if the corresponding column of age, male, good_country, or shouts is NA
# if the values is missing then the Missing variable is set to 1 and 0 otherwise
# and the value in the original column is replaced with the average value
rfreemium=freemium

# create columns to code if the variable is missing
rfreemium$age_Missing=as.numeric(is.na(freemium$age))
rfreemium$age[rfreemium$age_Missing==1]=mean(freemium$age,na.rm=T)
rfreemium$male_Missing=as.numeric(is.na(freemium$male))
rfreemium$male[rfreemium$male_Missing==1]=mean(freemium$male,na.rm=T)
rfreemium$good_country_Missing=as.numeric(is.na(freemium$good_country))
rfreemium$good_country[rfreemium$good_country_Missing==1]=mean(freemium$good_country,na.rm=T)
rfreemium$shouts_Missing=as.numeric(is.na(freemium$shouts))
rfreemium$shouts[rfreemium$shouts_Missing==1]=mean(freemium$shouts,na.rm=T)
rfreemium$avg_friend_age_Missing=as.numeric(is.na(freemium$avg_friend_age))
rfreemium$avg_friend_age[rfreemium$avg_friend_age_Missing==1]=mean(freemium$avg_friend_age,na.rm=T)
rfreemium$avg_friend_male_Missing=as.numeric(is.na(freemium$avg_friend_male))
rfreemium$avg_friend_male[rfreemium$avg_friend_male_Missing==1]=mean(freemium$avg_friend_male,na.rm=T)
# since there are not too many missing observations for friend_cnt, subscriber_friend_cnt,
# friend_country_cnt, and tenure these then the missing values for these are simply set to the mean
rfreemium$friend_cnt[is.na(rfreemium$friend_cnt)]=mean(freemium$friend_cnt,na.rm=T)
rfreemium$subscriber_friend_cnt[is.na(rfreemium$subscriber_friend_cnt)]=mean(freemium$subscriber_friend_cnt,na.rm=T)
rfreemium$friend_country_cnt[is.na(rfreemium$friend_country_cnt)]=mean(freemium$friend_country_cnt,na.rm=T)
rfreemium$tenure[is.na(rfreemium$tenure)]=mean(freemium$tenure,na.rm=T)

# create a list with the variables that will be used in the analysis
varlist=c("age","male","friend_cnt","subscriber_friend_cnt","avg_friend_age","avg_friend_male","friend_country_cnt",
          "songsListened","playlists","posts","shouts","lovedTracks","tenure","good_country")
# also create a list for the recoded values
rvarlist=c("age","age_Missing","male","male_Missing","friend_cnt","subscriber_friend_cnt","avg_friend_age","avg_friend_age_Missing",
           "avg_friend_male","avg_friend_male_Missing","friend_country_cnt","songsListened","playlists","posts",
           "shouts","shouts_Missing","lovedTracks","tenure","good_country","good_country_Missing")
crvarlist=c("adopter",rvarlist)

##
# create a dummy variable for subscriber friend count change
##
# Note that the delta1_subscriber_friend_cnt variable represents changes in the number of
# subscriber friends in both directions.  If a user's subscribe friend count increases, the
# delta1_subscriber_friend_cnt is positive.  If a user's subscriber friend count decreases, the
# delta1_subscriber_friend_cnt is negative.  E.g., delta1_subscriber_friend_cnt=1 implies that the 
# user obtained one more friend who is a subscriber.  It could be that this user made a new friend
# who is a subscriber, or that an existing friend switched from being a non-subscriber to a
# subscriber.
#
# Similarly, delta1_subscriber_friend_cnt=-1 implies that the user lost a friend who is a subscriber,
# or that one of the user's friend swtiched from being a subscriber to a non-subscriber. For
# delta1_subscriber_friend_cnt=1, we generate dum_delta1_subsfrcnt=1; for
# delta1_subscriber_friend_cnt=-1, we generate dum_delta1_subscfrcnt=0.
#
# The impact of increasing subscriber friend and decreasing subscriber friend is asymmmetric. An
# additional subscriber friend might have a positive influence on the user.  However, the attribution of 
# a subscriber firned usually does not have a negative impact on the user's adoption decision. We
# hence transform the delta1_subscriber_friend_cnt variable to be a dummary variable,
# dum_delta1_subscfrnt.  If the delta1_subscriber_friend_cnt is positive, dum_delta1_usbsfrcnt
# equals to 1. Ohterwise, dum_delta1_subsfrcnt equals to 0. And we use the dum_delta1_usbsfrcnt
# as our explanatory variable instead of delta1_subscriber_friend_cnt, because
# dum_delta1_subsfrcnt is a more accurate variable that captures the actual influence.
rfreemium$dum_delta1_subsfrcnt=(rfreemium$delta1_subscriber_friend_cnt>0)+1




###############################################################################
### understanding the data with descriptive statistics and graphics
###############################################################################

# number of observations
sum(trainsample)
sum(validsample)
sum(predsample)

# let's take a look at just one observation
print(freemium[1,])
# same observation but just a few values
print(freemium[1,varlist])  

# describe the data using only the training data
summary(freemium[trainsample,varlist])

# use the describe function in the psych package to generate nicer tables
describe(freemium[,varlist],fast=TRUE)
# describe the freemium data for adopters and non-adopters
describeBy(freemium[,varlist],group=freemium$adopter,fast=TRUE)

# do the same thing with the recoded data
describe(rfreemium[,rvarlist],fast=TRUE)
describeBy(rfreemium[,rvarlist],group=rfreemium$adopter,fast=TRUE)

# do the same thing with the recoded data (but just for the training data)
describe(rfreemium[trainsample,rvarlist],fast=TRUE)
describeBy(rfreemium[trainsample,rvarlist],group=rfreemium$adopter[trainsample],fast=TRUE)

# boxplots
par(mfrow=c(3,4),mar=c(5,5,1,1))
boxplot(age~adopter,data=freemium[plotsample,],xlab="adopter",ylab="age")
boxplot(friend_cnt~adopter,data=freemium[plotsample,],xlab="adopter",ylab="friend_cnt")
boxplot(subscriber_friend_cnt~adopter,data=freemium[plotsample,],xlab="adopter",ylab="subscriber_friend_cnt")
boxplot(avg_friend_age~adopter,data=freemium[plotsample,],xlab="adopter",ylab="avg_friend_age")
boxplot(avg_friend_male~adopter,data=freemium[plotsample,],xlab="adopter",ylab="avg_friend_male")
boxplot(friend_country_cnt~adopter,data=freemium[plotsample,],xlab="adopter",ylab="friend_country_cnt")
boxplot(songsListened~adopter,data=freemium[plotsample,],xlab="adopter",ylab="songsListened")
boxplot(playlists~adopter,data=freemium[plotsample,],xlab="adopter",ylab="playlists")
boxplot(posts~adopter,data=freemium[plotsample,],xlab="adopter",ylab="posts")
boxplot(shouts~adopter,data=freemium[plotsample,],xlab="adopter",ylab="shouts")
boxplot(lovedTracks~adopter,data=freemium[plotsample,],xlab="adopter",ylab="lovedTracks")
boxplot(tenure~adopter,data=freemium[plotsample,],xlab="adopter",ylab="tenure")

# cross tabs to understand relationships across discrete values
xtabs(~male+adopter,data=freemium)
xtabs(~good_country+adopter,data=freemium)

# compute correlation matrix (using only complete sets of observations)
cormat=cor(freemium[,varlist],use="pairwise.complete.obs")
# print out in nicer way
round(cormat,digits=2)
# here is a better visualization of the correlation matrix using a heatmap
qplot(x=Var1,y=Var2,data=melt(cormat),fill=value,geom="tile")+
  scale_fill_gradient2(limits=c(-1, 1)) + theme(axis.text.x = element_text(angle = 90, hjust = 1))

# nicer scatterplot matrix
par(mfrow=c(1,1),mar=c(5,4,4,1))
scatterplotMatrix(~age+tenure+friend_cnt|adopter,data=freemium[plotsample,])  # small plot to see easier, the "|" is used to print separate plots by adopter
scatterplotMatrix(~age+friend_cnt+subscriber_friend_cnt+avg_friend_age+avg_friend_male+friend_country_cnt|adopter,data=freemium[plotsample,])
scatterplotMatrix(~songsListened+playlists+posts+shouts+lovedTracks+tenure|adopter,data=freemium[plotsample,])




###############################################################################
### estimate a tree model with all variables (simple structure)
###############################################################################

# estimate a model with all the variables
#ctree1 = tree(adopter~., data=rfreemium[trainsample,crvarlist],mindev=.005)
#summary(ctree1)
#plot(ctree1)
#text(ctree1,cex=.5)

# estimate a model with all the variables
ctree1 = rpart(adopter~., data=rfreemium[trainsample,crvarlist], control=rpart.control(cp=0.005))  # ! try different values of cp like .001, .002, and .005
summary(ctree1)
plot(ctree1)
text(ctree1)
prp(ctree1)
fancyRpartPlot(ctree1)

# predict probability (for training sample)
padopter = predict(ctree1,newdata=rfreemium[trainsample,crvarlist],type='vector')
cadopter = (padopter>.25)+0    # notice that we use a cutoff of 25% because it is harder to predict adopters
trueadopter = freemium$adopter[trainsample]
(results = xtabs(~cadopter+trueadopter) )  # confusion matrix (columns have truth, rows have predictions)
(accuracy = (results[1,1]+results[2,2])/sum(results) )  # how many correct guesses along the diagonal
(truepos = results[2,2]/(results[1,2]+results[2,2]))  # how many correct "adopter" guesses
(precision = results[2,2]/(results[2,1]+results[2,2])) # proportion of correct positive guesses 
(trueneg = results[1,1]/(results[2,1]+results[1,1]))  # how many correct "non-adopter" guesses

# predict probability (for validation sample)
padopter = predict(ctree1,newdata=rfreemium[validsample,crvarlist],type='vector')
cadopter = (padopter>.25)+0    # notice that we use a cutoff of 25% because it is harder to predict adopters
trueadopter = freemium$adopter[validsample]
(results = xtabs(~cadopter+trueadopter) )  # confusion matrix (columns have truth, rows have predictions)
(accuracy = (results[1,1]+results[2,2])/sum(results) )  # how many correct guesses along the diagonal
(truepos = results[2,2]/(results[1,2]+results[2,2]))  # how many correct "adopter" guesses
(precision = results[2,2]/(results[2,1]+results[2,2])) # proportion of correct positive guesses 
(trueneg = results[1,1]/(results[2,1]+results[1,1]))  # how many correct "non-adopter" guesses

# compute the predictions for the 10% of most likely adopterers (for validation sample)
topadopter = as.vector(padopter>=as.numeric(quantile(padopter,probs=.9)))
( baseconv=sum(trueadopter==1)/length(trueadopter) )  # what proportion would we have expected purely due to chance
( actconv=sum(trueadopter[topadopter])/sum(topadopter))  # what proportion did we actually predict
( lift=actconv/baseconv )  # what is the ratio of how many we got to what we expected

# compute the predictions for each decline of most likely adopterers (for validation sample)
vprob=seq(.9,.1,-.1)  # define 90th to 10th percentiles
vlift1=rep(0,length(vprob))  # save results to vector
for (i in 1:length(vprob)) {
  topadopter = as.vector(padopter>=as.numeric(quantile(padopter,probs=vprob[i])))  # compute indices of topadopters
  ( baseconv=sum(trueadopter==1)/length(trueadopter) )  # what proportion would we have expected purely due to chance
  ( actconv=sum(trueadopter[topadopter])/sum(topadopter))  # what proportion did we actually predict
  ( vlift1[i]=actconv/baseconv )  # what is the ratio of how many we got to what we expected
}
plot(vlift1,axes=F,xlab="Percentile",ylab="Lift")   # plot the lift
axis(2)  # overlay y axis
axis(1,at=1:length(vprob),labels=vprob)  # overlay x axis, but use vprob as labels

# compute ROC and AUC
rocpred1 = prediction(padopter,freemium$adopter[validsample])  # compute predictions using "prediction"
rocperf1 = performance(rocpred1, measure = "tpr", x.measure = "fpr")
plot(rocperf1, col=rainbow(10)); abline(a=0, b= 1)
auc1.tmp = performance(rocpred1,"auc")  # compute area under curve
(auc1 = as.numeric(auc1.tmp@y.values))

# predict probability (for prediction sample)
padopter = predict(ctree1,newdata=rfreemium[predsample,crvarlist],type='vector')
cadopter = as.vector((padopter>.25)+0)  # classify the predictions as adopters or not
trueadopter = freemium$adopter[predsample]
(results = xtabs(~cadopter+trueadopter))  # confusion matrix
(accuracy = (results[1,1]+results[2,2])/sum(results) )  # how many correct guesses along the diagonal
(truepos = results[2,2]/(results[1,2]+results[2,2]))  # how many correct "adopter" guesses
(precision = results[2,2]/(results[2,1]+results[2,2])) # proportion of correct positive guesses 
(trueneg = results[1,1]/(results[2,1]+results[1,1]))  # how many correct "non-adopter" guesses




###############################################################################
### estimate a tree model with all variables (more complex)
###############################################################################

# estimate a model with all the variables
#ctree2 = tree(adopter~., data=rfreemium[trainsample,crvarlist],mindev=.001)
#summary(ctree2)
#plot(ctree2)
#text(ctree2,cex=.5)

# redo this with rpart instead of tree
ctree2 = rpart(adopter~., data=rfreemium[trainsample,crvarlist], control=rpart.control(cp=0.001))
summary(ctree2)
plot(ctree2)
text(ctree2)   # standard tree plot
prp(ctree2)    # plot as a tree
fancyRpartPlot(ctree2)   # prettier plot

# predict probability (for validation sample)
padopter = predict(ctree2,newdata=rfreemium[trainsample,crvarlist],type='vector')
cadopter = (padopter>.25)+0    # notice that we use a cutoff of 25% because it is harder to predict adopters
trueadopter = freemium$adopter[trainsample]
(results = xtabs(~cadopter+trueadopter) )  # confusion matrix (columns have truth, rows have predictions)
(accuracy = (results[1,1]+results[2,2])/sum(results) )  # how many correct guesses along the diagonal
(truepos = results[2,2]/(results[1,2]+results[2,2]))  # how many correct "adopter" guesses
(precision = results[2,2]/(results[2,1]+results[2,2])) # proportion of correct positive guesses 
(trueneg = results[1,1]/(results[2,1]+results[1,1]))  # how many correct "non-adopter" guesses

# predict probability (for validation sample)
padopter = predict(ctree2,newdata=rfreemium[validsample,crvarlist],type='vector')
cadopter = (padopter>.25)+0    # notice that we use a cutoff of 25% because it is harder to predict adopters
trueadopter = freemium$adopter[validsample]
(results = xtabs(~cadopter+trueadopter) )  # confusion matrix (columns have truth, rows have predictions)
(accuracy = (results[1,1]+results[2,2])/sum(results) )  # how many correct guesses along the diagonal
(truepos = results[2,2]/(results[1,2]+results[2,2]))  # how many correct "adopter" guesses
(precision = results[2,2]/(results[2,1]+results[2,2])) # proportion of correct positive guesses 
(trueneg = results[1,1]/(results[2,1]+results[1,1]))  # how many correct "non-adopter" guesses

# compute the predictions for the 10% of most likely adopters (for validation sample)
topadopter = as.vector(padopter>=as.numeric(quantile(padopter,probs=.9)))
( baseconv=sum(trueadopter==1)/length(trueadopter) )  # what proportion would we have expected purely due to chance
( actconv=sum(trueadopter[topadopter])/sum(topadopter))  # what proportion did we actually predict
( lift=actconv/baseconv )  # what is the ratio of how many we got to what we expected

# compute the predictions for each decline of most likely adopterers (for validation sample)
vprob=seq(.9,.1,-.1)  # define 90th to 10th percentiles
vlift2=rep(0,length(vprob))  # save results to vector
for (i in 1:length(vprob)) {
  topadopter = as.vector(padopter>=as.numeric(quantile(padopter,probs=vprob[i])))  # compute indices of topadopters
  ( baseconv=sum(trueadopter==1)/length(trueadopter) )  # what proportion would we have expected purely due to chance
  ( actconv=sum(trueadopter[topadopter])/sum(topadopter))  # what proportion did we actually predict
  ( vlift2[i]=actconv/baseconv )  # what is the ratio of how many we got to what we expected
}
plot(vlift2,axes=F,xlab="Percentile",ylab="Lift")   # plot the lift
axis(2)  # overlay y axis
axis(1,at=1:length(vprob),labels=vprob)  # overlay x axis, but use vprob as labels

# compute ROC and AUC
rocpred2 = prediction(padopter,freemium$adopter[validsample])  # compute predictions using "prediction"
rocperf2 = performance(rocpred2, measure = "tpr", x.measure = "fpr")
plot(rocperf2, col=rainbow(10)); abline(a=0, b= 1)
auc2.tmp = performance(rocpred2,"auc")  # compute area under curve
(auc2 = as.numeric(auc2.tmp@y.values))

# predict probability (for prediction sample)
padopter = predict(ctree2,newdata=rfreemium[predsample,crvarlist],type='vector')
cadopter = as.vector((padopter>.25)+0)  # classify the predictions as adopters or not
trueadopter = freemium$adopter[predsample]
(results = xtabs(~cadopter+trueadopter))  # confusion matrix
(accuracy = (results[1,1]+results[2,2])/sum(results) )  # how many correct guesses along the diagonal
(truepos = results[2,2]/(results[1,2]+results[2,2]))  # how many correct "adopter" guesses
(precision = results[2,2]/(results[2,1]+results[2,2])) # proportion of correct positive guesses 
(trueneg = results[1,1]/(results[2,1]+results[1,1]))  # how many correct "non-adopter" guesses




###############################################################################
### additional exploratory analysis
###############################################################################

# determine fraction of users who shout, lovetracks, playlist, post
perc.users=colSums(rfreemium[trainsample,c("shouts","lovedTracks","playlists","posts")]!=0)/sum(trainsample)
barplot(perc.users,xlab="Activity",ylab="Percent Engaged",main="User initiated actions",col="lightblue")

# check by adopters and non-adopters
perc.adopt=colSums(rfreemium[trainsample & rfreemium$adopter==1,c("shouts","lovedTracks","playlists","posts")]!=0)/sum(rfreemium$adopter[trainsample]==1)
perc.free=colSums(rfreemium[trainsample & rfreemium$adopter==0,c("shouts","lovedTracks","playlists","posts")]!=0)/sum(rfreemium$adopter[trainsample]==0)
barplot(rbind(perc.free,perc.adopt),beside=TRUE,col=c("blue","red"),xlab="Activity",ylab="Percent Engaged",main="User initiated actions")
legend("topright",c("Free users","Adopters"),pch=15,col=c("blue","red"),bty="n")

# count number that shout, love and create playlist (overall and for adopters)
perc.heavyuser=sum(
  rfreemium$shouts[trainsample]>0 &
  rfreemium$lovedTracks[trainsample]>0 &
  rfreemium$playlists[trainsample]>0)/sum(trainsample)
perc.adoptuser=sum(
  rfreemium$shouts[trainsample & rfreemium$adopter==1]>0 &
  rfreemium$lovedTracks[trainsample & rfreemium$adopter==1]>0 &
  rfreemium$playlists[trainsample & rfreemium$adopter==1]>0)/sum(rfreemium$adopter[trainsample]==1)
perc.heavyfree=sum(
  rfreemium$shouts[trainsample & rfreemium$adopter==0]>0 &
    rfreemium$lovedTracks[trainsample & rfreemium$adopter==0]>0 &
    rfreemium$playlists[trainsample & rfreemium$adopter==0]>0)/sum(rfreemium$adopter[trainsample]==0)
cbind(perc.heavyuser,perc.heavyfree,perc.adoptuser)  # print totals




###############################################################################
### compare models
###############################################################################

# plot all ROC curves together
plot(rocperf1,col="blue"); abline(a=0,b=1)
plot(rocperf2,add=TRUE,col="green")
legend("bottomright",c("Simple Tree","Complex Tree"),pch=15,col=c("blue","green"),bty="n")

# plot lift
plot(vlift1,axes=F,xlab="Percentile",ylab="Lift",col="blue")   # plot the lift
lines(vlift1,col="blue")
axis(2)  # overlay y axis
axis(1,at=1:length(vprob),labels=vprob)  # overlay x axis, but use vprob as labels
points(vlift2,col="green"); lines(vlift2,col="green")
legend("topright",c("Simple Tree","Complex Tree"),pch=15,col=c("blue","green"),bty="n")




###############################################################################
### estimate a tree model for sample of variables (use as a benchmark to compare with next model)
###############################################################################

# estimate a baseline model with a sample of variables
svarlist=c("adopter","good_country","friend_cnt","subscriber_friend_cnt","songsListened","lovedTracks","posts","playlists","shouts")
ctree3 = rpart(adopter~., data=rfreemium[trainsample,svarlist], control=rpart.control(cp=0.001))
fancyRpartPlot(ctree3)   # prettier plot

# predict probability (for validation sample)
padopter = predict(ctree3,newdata=rfreemium[validsample,svarlist],type='vector')
cadopter = (padopter>.25)+0    # notice that we use a cutoff of 25% because it is harder to predict adopters
trueadopter = freemium$adopter[validsample]
(results = xtabs(~cadopter+trueadopter) )  # confusion matrix (columns have truth, rows have predictions)
(accuracy = (results[1,1]+results[2,2])/sum(results) )  # how many correct guesses along the diagonal
(truepos = results[2,2]/(results[1,2]+results[2,2]))  # how many correct "adopter" guesses
(precision = results[2,2]/(results[2,1]+results[2,2])) # proportion of correct positive guesses 
(trueneg = results[1,1]/(results[2,1]+results[1,1]))  # how many correct "non-adopter" guesses

# compute the predictions for the 10% of most likely adopterers (for validation sample)
topadopter = as.vector(padopter>=as.numeric(quantile(padopter,probs=.9)))
( baseconv=sum(trueadopter==1)/length(trueadopter) )  # what proportion would we have expected purely due to chance
( actconv=sum(trueadopter[topadopter])/sum(topadopter))  # what proportion did we actually predict
( lift=actconv/baseconv )  # what is the ratio of how many we got to what we expected




###############################################################################
### estimate a tree model with a sample of variables (consider the change from the previous period)
###############################################################################

# estimate a baseline model with a sample of variables
svarlist2=c("adopter","delta1_good_country","delta1_friend_cnt","dum_delta1_subsfrcnt","delta1_songsListened",
            "delta1_lovedTracks","delta1_posts","delta1_playlists","delta1_shouts")
ctree4 = rpart(adopter~., data=rfreemium[trainsample,svarlist2], control=rpart.control(cp=0.001))
fancyRpartPlot(ctree4)   # prettier plot

# predict probability (for validation sample)
padopter = predict(ctree4,newdata=rfreemium[validsample,svarlist2],type='vector')
cadopter = (padopter>.25)+0    # notice that we use a cutoff of 25% because it is harder to predict adopters
trueadopter = freemium$adopter[validsample]
(results = xtabs(~cadopter+trueadopter) )  # confusion matrix (columns have truth, rows have predictions)
(accuracy = (results[1,1]+results[2,2])/sum(results) )  # how many correct guesses along the diagonal
(truepos = results[2,2]/(results[1,2]+results[2,2]))  # how many correct "adopter" guesses
(precision = results[2,2]/(results[2,1]+results[2,2])) # proportion of correct positive guesses 
(trueneg = results[1,1]/(results[2,1]+results[1,1]))  # how many correct "non-adopter" guesses

# compute the predictions for the 10% of most likely adopterers (for validation sample)
topadopter = as.vector(padopter>=as.numeric(quantile(padopter,probs=.9)))
( baseconv=sum(trueadopter==1)/length(trueadopter) )  # what proportion would we have expected purely due to chance
( actconv=sum(trueadopter[topadopter])/sum(topadopter))  # what proportion did we actually predict
( lift=actconv/baseconv )  # what is the ratio of how many we got to what we expected

