train <- read.csv("/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/data/train_has_dest_r.csv")
test <- read.csv("/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/data/test_has_dest_r.csv")
summary(train)

train$date_account_created_month <- factor(train$date_account_created_month, levels = 1: 12, 
                                           labels = c("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"))
train$date_first_active_month <- factor(train$date_first_active_month, levels = 1: 12, 
                                           labels = c("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"))
test$date_account_created_month <- factor(test$date_account_created_month, levels = 1: 12, 
                                           labels = c("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"))
test$date_first_active_month <- factor(test$date_first_active_month, levels = 1: 12, 
                                          labels = c("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"))

test$has_destination <- factor(test$has_destination, levels = c(0, 1), labels = c('No', 'Yes'))

time_df <- data.frame(row.names = 'run time')

start <- Sys.time()
full_lr <- glm(has_destination ~ ., data = train, family = binomial)
end <- Sys.time()
time_df$LR_Full <- end - start
prob_train <- predict(full_lr, type = 'response')
prob_test <- predict(full_lr, newdata = test, type = 'response')
write.csv(prob_train, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result_has_dest/Prob_Train_LR_Full.csv", row.names = FALSE)
write.csv(prob_test, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result_has_dest/Prob_Test_LR_Full.csv", row.names = FALSE)

summary_full_lr <-  summary(full_lr)
summary_full_lr
write.csv(summary_full_lr$coef, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result_has_dest/LR_Full_Coef.csv")

LR_Full_Coef <- exp(coef(full_lr))
LR_Full_Coef
LR_Full_CI <- exp(confint(full_lr))
LR_Full_CI
write.csv(LR_Full_CI, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result_has_dest/LR_Full_CI.csv")


lr0 <- glm(has_destination ~ 1, data = train, family = binomial)
step_lr <- step(lr0, direction = "forward", scope = formula(full_lr), data = train)
step_lr
start <- Sys.time()
step_lr <- glm(formula = has_destination ~ age_bkt + signup_method + signup_flow + 
                 affiliate_channel + first_browser + first_affiliate_tracked + 
                 date_first_active_month + date_account_created_year + first_os + 
                 language + first_device + affiliate_provider + signup_app + 
                 gender + date_first_active_dayofyear + date_account_created_days_to_next_holiday, 
               family = binomial, data = train)
end <- Sys.time()
time_df$LR_Step <- end - start
prob_train <- predict(step_lr, type = 'response')
prob_test <- predict(step_lr, newdata = test, type = 'response')
write.csv(prob_train, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result_has_dest/Prob_Train_LR_Step.csv", row.names = FALSE)
write.csv(prob_test, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result_has_dest/Prob_Test_LR_Step.csv", row.names = FALSE)

summary_step_lr <-  summary(step_lr)
summary_step_lr
write.csv(summary_step_lr$coef, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result_has_dest/LR_Step_Coef.csv")

LR_Step_Coef <- exp(coef(step_lr))
LR_Step_Coef
LR_Step_CI <- exp(confint(step_lr))
LR_Step_CI
write.csv(LR_Step_CI, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result_has_dest/LR_Step_CI.csv")

threshold <- sum(test$has_destination == 'Yes')/length(test$has_destination)
predict_test <- ifelse(prob_test > threshold, 'Yes', 'No')
table <- table(test$has_destination, predict_test)
error_test <- round(mean(predict_test != test$has_destination),3)

library(rpart)
library(rpart.plot)			# For Enhanced tree plots via PRP()
set.seed(2019)
options(digits = 3)

# default cp = 0.01. Set cp = 0 to guarantee no pruning in order to complete phrase 1: Grow tree to max.
start <- Sys.time()
cart_full <- rpart(has_destination ~ ., data = train, method = 'class', control = rpart.control(minsplit = 2, cp = 0))
end <- Sys.time()
time_df$Cart_Full <- end - start
prob_train <- predict(cart_full, type='prob')
prob_test <- predict(cart_full, type='prob', newdata = test)
write.csv(prob_train, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result_has_dest/Prob_Train_Cart_Full.csv", row.names = FALSE)
write.csv(prob_test, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result_has_dest/Prob_Test_Cart_Full.csv", row.names = FALSE)

# print(cart_full)
# printcp(cart_full, digits = 3)
# plotcp(cart_full)

cp.opt <- cart_full$cptable[which.min(cart_full$cptable[,"xerror"]),"CP"]
start <- Sys.time()
cart_prune <- prune(cart_full, cp = cp.opt)
end <- Sys.time()
time_df$Cart_Prune <- end - start + time_df$Cart_Full
prob_train <- predict(cart_prune, type='prob')
prob_test <- predict(cart_prune, type='prob', newdata = test)
write.csv(prob_train, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result_has_dest/Prob_Train_Cart_Prune.csv", row.names = FALSE)
write.csv(prob_test, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result_has_dest/Prob_Test_Cart_Prune.csv", row.names = FALSE)

# print(cart_prune)
# printcp(cart_prune, digits = 3)
# prp(cart_prune, type=2, cex = 0.4, extra=104, nn=T, fallen.leaves=T, branch.lty=3, nn.box.col = 'light blue', min.auto.cex = 0.7, nn.cex = 0.6, split.cex = 1.1, shadow.col="grey")
# cart_prune$variable.importance

library(randomForest)
start <- Sys.time()
randomForest <- randomForest(has_destination ~ ., data = train, method = 'class', randomForest = TRUE, ntree = 100)
end <- Sys.time()
time_df$Random_Forest <- end - start
prob_train <- predict(randomForest, type='prob')
prob_test <- predict(randomForest, type='prob', newdata = test)
write.csv(prob_train, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result_has_dest/Prob_Train_Random_Forest.csv", row.names = FALSE)
write.csv(prob_test, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result_has_dest/Prob_Test_Random_Forest.csv", row.names = FALSE)

importance(randomForest)
varImpPlot(randomForest)

library(earth)
start <- Sys.time()
mars <- earth(has_destination~., nfold=10,  data=train, glm=list(family=binomial), degree=2)
end <- Sys.time()
time_df$Mars <- end - start
prob_train <- predict(mars, type='prob')
prob_test <- predict(mars, type='prob', newdata = test)
write.csv(prob_train, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result_has_dest/Prob_Train_Mars.csv", row.names = FALSE)
write.csv(prob_test, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result_has_dest/Prob_Test_Mars.csv", row.names = FALSE)

summary_mars <- summary(mars)
plotd(mars)
ev <- evimp (mars)
ev

# time_df$Mars <- time_df$Mars * 60
# time_df$Random_Forest <- time_df$Random_Forest * 60
# write.csv(time_df, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result_has_dest/Run_Time_R.csv", row.names = FALSE)