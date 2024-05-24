install.packages("corrplot")
install.packages("rpart.plot")
install.packages("caret")
install.packages("accuracy")
install.packages("C50")
install.packages("gridExtra")
install.packages("gains")
library(rpart.plot)
library(corrplot)
library(caret)
library(forecast)
library(ggplot2)
library(dplyr)
library(rpart)
library(C50)
library(gridExtra)
library(gains)
library(e1071)

north_point_dta<-read.csv("North-Point List.csv")

north_point_dta

#printing head of data
head_north<-head(north_point_dta)
print(head_north)

# data type
str(north_point_dta)


summary(north_point_dta)

#null values 
sum(is.na(north_point_dta))

#no null values

#removing Sequence number
north_point_dta <- north_point_dta[, !(names(north_point_dta) %in% c("sequence_number"))]
head(north_point_dta)
####################Exploratory Data Analysis#############

##########univariate Distribution plots
numeric_variables<-c('Freq','last_update_days_ago','X1st_update_days_ago','Spending')
categoric_varibales<-c('US',"Web.order",'Address_is_res','Purchase',"Gender.male")
par(mfrow = c(2, 2)) 
#numeric variables distribution
for (feature in numeric_variables) {
  
  boxplot(north_point_dta[[feature]], main = feature, xlab = feature, 
          ylab = "Frequency", 
          col = c("red", "lightblue", "green", "orange", "purple"))
}
par(mfrow=c(1,1))

######categorical distribution

par(mfrow = c(3, 2)) 
#numeric variables distribution
for (feature in categoric_varibales) {
  
  barplot(table(north_point_dta[[feature]]), main = feature, xlab = "Category", 
          ylab = "Frequency", 

                    col = c("red", "lightblue", "green", "orange", "purple"))
}
par(mfrow=c(1,1))

#sources distribution
source <- c("source_a", "source_c", "source_b", "source_d", "source_e", "source_m", "source_o", "source_h", "source_r", "source_s", "source_t", "source_u", "source_p", "source_x", "source_w")

source_counts <- data.frame(source = source, count = 0)

for (s in source) {
  source_counts$count[source_counts$source == s] <- sum(north_point_dta[[s]] == 1)
  source_counts$purchase[source_counts$source == s] <- sum(north_point_dta[[s]] == 1 & north_point_dta$Purchase == 1)
  source_counts$spend[source_counts$source==s]<-sum(north_point_dta$Spending[north_point_dta[[s]] == 1])
}
print(source_counts)
#total count
total_count_source <- sum(source_counts$count)
total_count_purchase<- sum(source_counts$purchase)
total_count_spend<- sum(source_counts$spend)

###Source Distribution
sorted_source_counts <- source_counts %>% arrange(desc(count))

ggplot(sorted_source_counts, aes(x = reorder(source, count), y = count)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Count of Sources", x = "Source", y = "Count")
print(sorted_source_counts)
print(total_count_source)
print(total_count_purchase)

############################

head(north_point_dta)

# Attribute vs Target Target are First ->purchase ,second->spending

##########Target as Purchase
# spending vs purchase
boxplot(Spending ~ Purchase, data = north_point_dta,main = "Spending for the Purchase done", xlab = "Purchase", ylab = "Spending")

average_spending<- sum(north_point_dta$Spending)/1000
print(average_spending)

#Gender VS Purchase
ggplot(north_point_dta, aes(x = as.factor(Gender.male), fill = as.factor(Purchase))) +
  geom_bar(position = "stack") +
  labs(title = "bar graph for count of purchase  by Gender",
       x = "Gender", 
       y = "Count of purchase")
 
########source vs Purchase
ggplot(sorted_source_counts, aes(x = reorder(source, count))) +
  geom_col(aes(y = count, fill = "Count"), position = "dodge") +
  geom_col(aes(y = purchase, fill = "Purchase"), position = "dodge") +
  labs(title = "Count and Purchase by Source",
       x = "Source",
       y = "Count / Purchase") 

#US vs purchasers
ggplot(north_point_dta, aes(x = as.factor(US), fill= as.factor(Purchase))) +
  geom_bar() +
  labs(title = "Purchase vs US",
       x = "US",
       y = "Count") 
#weborder vs purchaser
ggplot(north_point_dta, aes(x = as.factor(Web.order), fill= as.factor(Purchase))) +
  geom_bar() +
  labs(title = paste("Purchase vs Web.order"),
       x = "Web.order",
       y = "Count") 



########Target as Spending

######spending as Target

#3 Freq vs spending

ggplot(north_point_dta,aes(x=Freq,y=Spending))+
  geom_point()+
  labs(title = "scatter plot for freq and spend",
       x = "freq", 
       y = "spending") 

#web order
ggplot(north_point_dta,aes(x=as.factor(Web.order),y=Spending))+
  geom_boxplot()+
  labs(title = "box plot for Web.order and spend",
       x = "Web.order", 
       y = "spending")

#source and spend
ggplot(source_counts, aes(x = source, y =  spend)) +
  geom_bar(stat="identity",fill="lightblue") +
  labs(title = "Total Spend by Source", x = "Source", y = "Total Spend") 
sorted_source_counts <- source_counts %>% arrange(desc(spend))
sorted_source_counts
#total spend
print(total_count_spend)
####profit estimation for rest 180,000 and mail costs 2$ and 0.053 response rate average spend of 202.888
estimated_profit<-((180000 * 202.888)* (0.053))-(2 * 180000)

print(estimated_profit)


#Correlation plot for numeric columns
columns_to_correlate<-c(numeric_variables, categoric_varibales)
  
cor_matrix<-cor(north_point_dta[,  columns_to_correlate], use = "complete.obs")
  corrplot(cor_matrix, method = "number")
  pairs.panels(cor_matrix[,columns_to_correlate])
  
# chi_square
  #US
  Chi_US = table(north_point_dta$US, north_point_dta$Purchase)
  chisq.test(Chi_US)
  #
  Chi_Gender_male = table(north_point_dta$Gender.male, north_point_dta$Purchase)
  chisq.test(Chi_Gender_male)
  
  Chi_Address_res = table(north_point_dta$Address_is_res, north_point_dta$Purchase)
  chisq.test(Chi_Address_res)
  #########CHi square Source E Hypothesis rejected
  Chi_Source_E = table(north_point_dta$source_e, north_point_dta$Purchase)
  chisq.test(Chi_Source_E)
  #######Sourcea
  Chi_Source_a = table(north_point_dta$source_a, north_point_dta$Purchase)
  chisq.test(Chi_Source_a)
  Chi_Source_c = table(north_point_dta$source_c, north_point_dta$Purchase)
  chisq.test(Chi_Source_c)
  Chi_Source_b = table(north_point_dta$source_b, north_point_dta$Purchase)
  chisq.test(Chi_Source_b)
  Chi_Source_m = table(north_point_dta$source_m, north_point_dta$Purchase)
  chisq.test(Chi_Source_m)
  Chi_Source_o = table(north_point_dta$source_o, north_point_dta$Purchase)
  chisq.test(Chi_Source_o)
  Chi_Source_h = table(north_point_dta$source_h, north_point_dta$Purchase)
  chisq.test(Chi_Source_h)
  Chi_Source_r = table(north_point_dta$source_r, north_point_dta$Purchase)
  chisq.test(Chi_Source_r)
  Chi_Source_s = table(north_point_dta$source_s, north_point_dta$Purchase)
  chisq.test(Chi_Source_s)
  Chi_Source_t = table(north_point_dta$source_t, north_point_dta$Purchase)
  chisq.test(Chi_Source_t)

  Chi_Source_u = table(north_point_dta$source_u, north_point_dta$Purchase)
  chisq.test(Chi_Source_u)
  Chi_Source_p = table(north_point_dta$source_p, north_point_dta$Purchase)
  chisq.test(Chi_Source_p)
  Chi_Source_x = table(north_point_dta$source_x, north_point_dta$Purchase)
  chisq.test(Chi_Source_x)
  Chi_Source_w = table(north_point_dta$source_m, north_point_dta$Purchase)
  chisq.test(Chi_Source_w)
  

  
  # source vs purchase
#factoring data
#needed to factor the target variables beacause it is categorcial model.
  north_point_dta <- north_point_dta %>%
    mutate(
      Purchase = factor(Purchase, levels=c(0,1), labels=c("No","Yes"))
    )
  head(north_point_dta)
  
#needed to create new set 
  
  NP_data<-north_point_dta
 
  

# partition
  set.seed(149)
  
  train_indx <- sample(rownames(NP_data),size = nrow(NP_data)*0.4)
  validation_indx<-sample(setdiff(rownames(NP_data), train_indx), size =nrow(NP_data)*0.35)
  holdout_indx <- setdiff(row.names(NP_data), union(train_indx,validation_indx))

  NP_train_data <- NP_data[train_indx,]
  NP_validation_data <- NP_data[validation_indx, ]
  NP_holdout_data <- NP_data[holdout_indx,]
  
  dim(NP_train_data)
  dim(NP_validation_data)
  dim(NP_holdout_data)
  
  prop.table(table(NP_train_data$Purchase))[2]*100
  prop.table(table(NP_validation_data$Purchase))[2]*100
  prop.table(table(NP_holdout_data$Purchase))[2]*100

  
###################naive bayees model(benchmark model)##############

  naiveBModel <- naiveBayes(Purchase ~ ., data = NP_train_data[,-which(names(NP_train_data) %in% c('Spending'))])
  
  naive_prediction<- predict(naiveBModel, newdata = NP_validation_data)
  
  summary(naiveBModel)

  confusion_matrix_naive <- confusionMatrix(naive_prediction, NP_validation_data$Purchase ,positive = "Yes")
  print(confusion_matrix_naive)
  
####################Logistic regression################# 
  
  head(north_point_dta)
  
  trControl <- caret::trainControl(method="cv", number=5, allowParallel=TRUE)
  Log_Model <- caret::train(Purchase ~ ., data=NP_train_data[,-c(24)], trControl=trControl,
                            method="glm", family="binomial")
  
  Log_Model
  Log_prediction<-predict(Log_Model,NP_validation_data)
  summary(Log_Model)
  #validation 
  confusion_matrix_Log <- confusionMatrix(Log_prediction, NP_validation_data$Purchase,positive = "Yes")
  print(confusion_matrix_Log)
  
  #sensititvity 0.79 and accuracy 79%
  
  ######TO find best model in logistic regression
  #############Forward Step-wise logistic regression
  
  fit_null <- glm(Purchase~1, data = NP_train_data[,-c(24)], family = "binomial")
  fit_full <- glm(Purchase ~ ., data = NP_train_data[,-c(24)], family = "binomial")
  forward_log <- step(fit_null, scope=list(lower=fit_null, upper=fit_full), direction = "forward")
  summary(forward_log)
  predict_forward <- predict(forward_log, NP_validation_data[ ,-c(23,24)], type = "response")
  predict_class <- ifelse(predict_forward > 0.5, "Yes", "No")
  confusionMatrix(factor(predict_class,levels=c("Yes","No")),factor(NP_validation_data$Purchase,levels = c("Yes","No")),positive = "Yes")
  
  ## Backward Step-wise logistic regression
  
  backward_log <- step(fit_full, direction = "backward")
  predict_backward <- predict(backward_log, NP_validation_data[ ,-c(23,24)], type = "response" )
  predict_class2 <- ifelse(predict_backward > 0.5, "Yes", "No")
  confusionMatrix(factor(predict_class2,levels=c("Yes","No")),factor(NP_validation_data$Purchase,levels = c("Yes","No")),positive="Yes")


###############################regression algorithms spending as target#############################3

  NP_train_data_p <-  NP_train_data[NP_train_data$Purchase=="Yes",]
  NP_validation_data_p <- NP_validation_data[NP_validation_data$Purchase=="Yes", ]
 
  dim(NP_train_data_p)
  dim(NP_validation_data_p)
  
  summary(NP_train_data_p$Spending)
  
  summary(NP_validation_data_p$Spending)
  
  
  hist(NP_train_data_p$Spending, main = "Spending distribution when purchased")
 
  
  car::vif(lm(Spending ~ last_update_days_ago + X1st_update_days_ago + Freq, data=NP_train_data_p))
  
  # removed 'X1st_update_days_ago','Purchase' beacuse ist update days ago has multi collinearity 
  #and since we only have purchasers data which we dont need the column for analysis.
  NP_train_data_p=NP_train_data_p[,-which(names(NP_train_data_p) %in% c('X1st_update_days_ago','Purchase'))]
  dim(NP_train_data_p)
  head(NP_train_data_p)
  #########################multiple linear regresssion
  
  linear_model<-lm( Spending ~ .,data = NP_train_data_p)
  
  summary(linear_model)
  
  linear_Prediction<-predict(linear_model,NP_validation_data_p)
  cor(linear_Prediction,NP_validation_data_p$Spending)
  accuracy(linear_Prediction,NP_validation_data_p$Spending)
  
  ## stepwise selection
  backWard_selection <- step(linear_model, direction = "backward")
  backWard_selection_prediction<-predict(backWard_selection,NP_validation_data_p)
  cor(backWard_selection_prediction,NP_validation_data_p$Spending)
  accuracy(backWard_selection_prediction,NP_validation_data_p$Spending)
  

  
  #forword
  model_null <- lm(Spending~1, data=NP_train_data_p)
  model_full <- lm(Spending~., data=NP_train_data_p)
  forward_step_reg<-step(model_null, scope=list(lower=model_null, upper=model_full), direction = "forward")
  
 # forward_selection<-step(linear_model,direction="forward")
  summary(forward_step_reg)
  forWard_selection_prediction<-predict(forward_step_reg,NP_validation_data_p)
  cor(forWard_selection_prediction,NP_validation_data_p$Spending)
  accuracy(forWard_selection_prediction,NP_validation_data_p$Spending)
  

  

  
  ###################regression tree####################
  reg_tree<-rpart(Spending ~ ., data = NP_train_data_p)
  reg_tree
  
  summary(reg_tree)
  
  rpart.plot(reg_tree)
  #predict with tree
  tree_predictions<-predict(reg_tree,newdata = NP_validation_data_p)
  cor(tree_predictions,NP_validation_data_p$Spending)
  
  accuracy(tree_predictions,NP_validation_data_p$Spending)


  ###############################Validation with holdout#############################
  ###########################predict the best classification algorithm with new Holdout data ####################
  head(NP_holdout_data)
  #4a.predicted probability of purchase
  NP_holdout_data$predicted_purchaser_prob<-predict(forward_log,newdata = NP_holdout_data,  type = "response")
  head(NP_holdout_data)
  
  #4.b   best prediction algorithm
  NP_holdout_data$predicted_spend<-predict(linear_model,newdata = NP_holdout_data)
  head(NP_holdout_data)
  
  #4c Adjusted probability
  NP_holdout_data$adjusted_prob_purchaser<-NP_holdout_data$predicted_purchaser_prob * 0.1065 ### which is purchase rate before equal sampling
    head(NP_holdout_data)
  
  #4d
  NP_holdout_data$Adj_predict_spend<-NP_holdout_data$predicted_spend * NP_holdout_data$predicted_purchaser_prob
  head(NP_holdout_data)
  
  #4e gain chart

  
  # cumulative gains
  Spending <- NP_holdout_data$Spending
  gain <- gains(Spending, NP_holdout_data$predicted_spend)
  
  # cumulative gains chart
  lift_chart <- data.frame(
    ncases = c(0, gain$cume.obs),
    cumSpending = c(0, gain$cume.pct.of.total * sum(Spending))
  )
  gainc <- ggplot(lift_chart, aes(x = ncases, y = cumSpending)) +
    geom_line() +
    geom_line(data = data.frame(ncases = c(0, nrow(NP_holdout_data)), cumSpending = c(0, sum(Spending))),
              color = "gray", linetype = 2) + # adds baseline
    labs(x = "# Cases", y = "Cumulative Spending", title = "Cumulative gains chart") +
    scale_y_continuous(labels = scales::comma) 
  
  # decile-wise lift chart
  Decile_wise <- data.frame(
    percentile = gain$depth,
    meanResponse = gain$mean.resp / mean(Spending)
  )
  gain2 <- ggplot(Decile_wise, aes(x = percentile, y = meanResponse)) +
    geom_bar(stat = "identity") +
    labs(x = "Percentile", y = "Decile mean / global mean", title = "Decile-wise lift chart")

  grid.arrange(gainc, gain2, ncol = 2)
  
  
