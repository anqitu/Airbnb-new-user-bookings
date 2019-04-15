users <- read.csv("/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/data/users_dest_r.csv")
summary(users)

users$date_account_created_month <- factor(users$date_account_created_month, levels = 1: 12, 
                                           labels = c("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"))


# Try Train vs Test set split.
library(caTools)
# Generate a random number sequence that can be reproduced to check results thru the seed number.
set.seed(2019)
# Randomly split data into two sets in predefined ratio while preserving relative ratios of different categories of Y in both sets.
attach((users))
split <- sample.split(Y = date_account_created_month, SplitRatio = 0.7)
# Get training and test data
train <- subset(users, split == T)
test <- subset(users, split == F)
summary(train)
summary(test)
write.csv(train, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/dest_train.csv", row.names = FALSE)
write.csv(test, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/dest_test.csv", row.names = FALSE)

time_df <- data.frame(row.names = 'run time')

library(nnet)
start <- Sys.time()
full_lr <- multinom(country_destination ~ ., data = train, MaxNWts =10000000)
end <- Sys.time()
time_df$LR_Full <- end - start
prob_train <- predict(full_lr, type = 'prob')
prob_test <- predict(full_lr, newdata = test, type = 'prob')
write.csv(prob_train, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/Dest_Prob_Train_LR_Full.csv", row.names = FALSE)
write.csv(prob_test, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/Dest_Prob_Test_LR_Full.csv", row.names = FALSE)

# start <- Sys.time()
# summary_full_lr = summary(full_lr)
# end <- Sys.time()
# time_df$summary_full_lr <- end - start

# start <- Sys.time()
# LR_Full_OR.CI <- exp(confint(full_lr))
# end <- Sys.time()
# time_df$LR_Full_OR.CI <- end - start
# write.csv(OR.CI, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/LR_Full_ORCI.csv", row.names = FALSE)
# LR_Full_OR.CI

# lr0 <- multinom(country_destination ~ 1, data = train,  MaxNWts =10000000) # null model with no X.
# step_lr <- step(lr0, direction = "forward", scope = formula(full_lr), data = train)
start <- Sys.time()
# step_lr <- multinom(formula = country_destination ~ affiliate_channel +
#                       age_bkt + date_first_active_month + language + gender +
#                       first_device, data = train, MaxNWts = 1e+07)
# step_lr <- multinom(formula = country_destination ~ age_bkt + action_count_ajax_refresh_subtotal +
#                       language + action_detail_count_message_thread + signup_flow +
#                       date_account_created_dayofyear + gender + os_secs_elapsed_Android +
#                       action_count_requested + action_count_dashboard, data = train, MaxNWts = 1e+07)
step_lr <- multinom(formula = country_destination ~ language + age_bkt + 
           signup_app + gender + date_account_created_dayofyear + first_device + 
           first_os + first_affiliate_tracked, data = train, MaxNWts = 1e+07)
end <- Sys.time()
time_df$LR_Step <- end - start
prob_train <- predict(step_lr, type = 'prob')
prob_test <- predict(step_lr, newdata = test, type = 'prob')
write.csv(prob_train, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/Dest_Prob_Train_LR_Step.csv", row.names = FALSE)
write.csv(prob_test, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/Dest_Prob_Test_LR_Step.csv", row.names = FALSE)

# log odds
OR <- exp(coef(step_lr))
OR
write.csv(OR, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/dest_lr_step_or.csv")

# OR.CI <- exp(confint(step_lr))
# OR.CI
# write.csv(OR.CI, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/dest_lr_step_orci.csv")

# summary_step_lr = summary(step_lr)
# multinom function does not include p-value calculation for the regression coefficients, so we calculate p-values using Wald Z tests
# z <- summary(step_lr)$coefficients/summary(step_lr)$standard.errors
# pvalue <- (1 - pnorm(abs(z), 0, 1))*2  # 2-tailed test p-values
# pvalue


library(rpart)
library(rpart.plot)			# For Enhanced tree plots via PRP()
set.seed(2019)
options(digits = 3)

# default cp = 0.01. Set cp = 0 to guarantee no pruning in order to complete phrase 1: Grow tree to max.
start <- Sys.time()
cart_full <- rpart(country_destination ~ ., data = train, method = 'class', control = rpart.control(minsplit = 2, cp = 0))
end <- Sys.time()
time_df$Cart_Full <- end - start
prob_train <- predict(cart_full, type='prob')
prob_test <- predict(cart_full, type='prob', newdata = test)
write.csv(prob_train, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/Dest_Prob_Train_Cart_Full.csv", row.names = FALSE)
write.csv(prob_test, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/Dest_Prob_Test_Cart_Full.csv", row.names = FALSE)
cart_full$variable.importance

# print(cart_full)
# printcp(cart_full, digits = 3)
# plotcp(cart_full)

cp.opt <- cart_full$cptable[which.min(cart_full$cptable[,"xerror"]),"CP"]
start <- Sys.time()
cart_prune <- prune(cart_full, cp = cp.opt)
# cart_prune <- prune(cart_full, cp = 2.71e-04)
end <- Sys.time()
time_df$Cart_Prune <- end - start + time_df$Cart_Full
prob_train <- predict(cart_prune, type='prob')
prob_test <- predict(cart_prune, type='prob', newdata = test)
write.csv(prob_train, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/Dest_Prob_Train_Cart_Prune.csv", row.names = FALSE)
write.csv(prob_test, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/Dest_Prob_Test_Cart_Prune.csv", row.names = FALSE)

print(cart_prune)
printcp(cart_prune, digits = 3)
prp(cart_prune, type=2, cex = 0.5, extra=104, nn=T, fallen.leaves=T, branch.lty=3, nn.box.col = 'light blue', min.auto.cex = 0.3, nn.cex = 0.5, split.cex = 1.5, shadow.col="grey")
cart_prune$variable.importance
write.csv(cart_prune$variable.importance, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/dest_cart_prune_vi.csv")

library(randomForest)
start <- Sys.time()
randomForest <- randomForest(country_destination ~ ., data = train, method = 'class', randomForest = TRUE, ntree = 100)
end <- Sys.time()
time_df$Random_Forest <- end - start
prob_train <- predict(randomForest, type='prob')
prob_test <- predict(randomForest, type='prob', newdata = test)
write.csv(prob_train, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/Dest_Prob_Train_Random_Forest.csv", row.names = FALSE)
write.csv(prob_test, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/Dest_Prob_Test_Random_Forest.csv", row.names = FALSE)

importance(randomForest)
varImpPlot(randomForest)
randomForest$importance

write.csv(randomForest$importance, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/dest_forest_vi.csv")

library(earth)
start <- Sys.time()
mars <- earth(country_destination~., nfold=10,  data=train, glm=list(family=binomial), degree=2)
end <- Sys.time()
time_df$Mars <- end - start
prob_train <- predict(mars, type='response')
prob_test <- predict(mars, type='response', newdata = test)

write.csv(prob_train, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/Dest_Prob_Train_Mars.csv", row.names = FALSE)
write.csv(prob_test, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/Dest_Prob_Test_Mars.csv", row.names = FALSE)

summary_mars <- summary(mars)
summary_mars$coefficients
# summary_mars

evimp(mars)
plotmo(mars, nresponse="ES")
write.csv(evimp(mars), file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/dest_mars_vi.csv")
write.csv(summary_mars$coefficients, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/dest_mars_coef.csv")


time_df$Mars <- time_df$Mars * 60
# time_df$Random_Forest <- time_df$Random_Forest * 60
write.csv(time_df, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/Run_Time_Dest.csv", row.names = FALSE)

