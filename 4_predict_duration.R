users <- read.csv("/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/data/users_duration_r.csv")
summary(users)

users$date_account_created_month <- factor(users$date_account_created_month, levels = 1: 12, 
                                           labels = c("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"))

users$booking_account_diff <- ifelse(users$booking_account_diff <= 2, 'Yes', 'No')
users$booking_account_diff <- as.factor(users$booking_account_diff)
summary(users)

# Try Train vs Test set split.
library(caTools)
# Generate a random number sequence that can be reproduced to check results thru the seed number.
set.seed(2019)
# Randomly split data into two sets in predefined ratio while preserving relative ratios of different categories of Y in both sets.
attach((users))
split <- sample.split(Y = booking_account_diff, SplitRatio = 0.7)
# Get training and test data
train <- subset(users, split == T)
test <- subset(users, split == F)
write.csv(train, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/duration_train.csv", row.names = FALSE)
write.csv(test, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/duration_test.csv", row.names = FALSE)

time_df <- data.frame(row.names = 'run time')

library(nnet)
start <- Sys.time()
full_lr <- glm(booking_account_diff ~ ., family = binomial, data = train)
# full_lr <- multinom(booking_account_diff ~ ., data = train, MaxNWts =10000000)
end <- Sys.time()
time_df$LR_Full <- end - start
# prob_train <- predict(full_lr)
# prob_test <- predict(full_lr, newdata = test)
# write.csv(prob_train, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/Duration_Prob_Train_LR_Full.csv", row.names = FALSE)
# write.csv(prob_test, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/Duration_Prob_Test_LR_Full.csv", row.names = FALSE)

prob_train <- predict(full_lr, type = 'response')
threshold <- sum(train$booking_account_diff == 'Yes')/length(train$booking_account_diff)
prob_train <- ifelse(prob_train > threshold, "Yes", 'No')
table(train$booking_account_diff, prob_train)
1 - round(mean(train$booking_account_diff != prob_train),3)

prob_test <- predict(full_lr, type = 'response', newdata = test)
prob_test <- ifelse(prob_test > threshold, "Yes", 'No')
table(test$booking_account_diff, prob_test)
1 - round(mean(test$booking_account_diff != prob_test),3)
write.csv(prob_train, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/Duration_Prob_Train_LR_Full.csv", row.names = FALSE)
write.csv(prob_test, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/Duration_Prob_Test_LR_Full.csv", row.names = FALSE)


# summary_full_lr = summary(full_lr)
# LR_Full_OR.CI <- exp(confint(full_lr))
# write.csv(LR_Full_OR.CI, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/Duration_LR_Full_ORCI.csv")
# LR_Full_OR.CI
#
# # multinom function does not include p-value calculation for the regression coefficients, so we calculate p-values using Wald Z tests
# z <- summary_full_lr$coefficients/summary_full_lr$standard.errors
# pvalue <- (1 - pnorm(abs(z), 0, 1))*2  # 2-tailed test p-values
# pvalue


# lr0 <- multinom(booking_account_diff ~ 1, family = binomial, data = train) # null model with no X.
# lr0 <- glm(booking_account_diff ~ 1, family = binomial, data = train) # null model with no X.
# step_lr <- step(lr0, direction = "forward", scope = formula(full_lr), data = train)
start <- Sys.time()
# step_lr <- multinom(formula = booking_account_diff ~ action_count_requested +
#                       action_detail_count_update_listing + total_secs_elapsed_sum +
#                       actions_secs_sum_view_search_results + actions_secs_sum_p3 +
#                       first_browser + actions_secs_sum_update + total_secs_elapsed_max +
#                       gender + action_count_dashboard + actions_secs_sum_show +
#                       action_count_personalize + affiliate_channel + actions_secs_sum_header_userpic +
#                       age_bkt + actions_secs_sum_search_results + actions_secs_sum_index +
#                       actions_secs_sum_social_connections + action_detail_count_message_thread +
#                       device_secs_elapsed_Others + date_account_created_dayofweek +
#                       action_count_edit + actions_secs_sum_personalize + first_os +
#                       date_account_created_dayofyear + actions_secs_sum_reviews +
#                       total_secs_elapsed_median + actions_secs_sum_active, data = train,
#                     MaxNWts = 1e+07)

# step_lr <- glm(formula = booking_account_diff ~ action_count_requested +
#       action_detail_count_update_listing + total_secs_elapsed_sum +
#       actions_secs_sum_view_search_results + actions_secs_sum_p3 +
#       first_browser + actions_secs_sum_update + total_secs_elapsed_max +
#       gender + action_count_dashboard + actions_secs_sum_show +
#       action_count_personalize + affiliate_channel + actions_secs_sum_header_userpic +
#       age_bkt + actions_secs_sum_search_results + actions_secs_sum_index +
#       action_detail_count_message_thread + actions_secs_sum_social_connections +
#       device_secs_elapsed_Others + date_account_created_dayofweek +
#       action_count_edit + actions_secs_sum_personalize + first_os +
#       date_account_created_dayofyear + total_secs_elapsed_median +
#       actions_secs_sum_active + actions_secs_sum_reviews + actions_secs_sum_search +
#       os_secs_elapsed_Windows + actions_secs_sum_user_profile +
#       os_secs_elapsed_Others + action_count_active, family = binomial,
#     data = train)

step_lr <- glm(formula = booking_account_diff ~ action_search_results + action_ajax_refresh_subtotal + 
                 signup_app + gender + action_ask_question + action_similar_listings + 
                 age_bkt + affiliate_channel + date_account_created_dayofyear + 
                 first_browser + first_os + first_device + signup_method + 
                 action_wishlist_content_update, family = binomial, data = train)

end <- Sys.time()
time_df$LR_Step <- end - start
# prob_train <- predict(step_lr, type = 'prob')
# prob_test <- predict(step_lr, newdata = test, type = 'prob')
prob_train <- predict(step_lr, type = 'response')
threshold <- sum(train$booking_account_diff == 'Yes')/length(train$booking_account_diff)
prob_train <- ifelse(prob_train > threshold, "Yes", 'No')
table(train$booking_account_diff, prob_train)
1 - round(mean(train$booking_account_diff != prob_train),3)

prob_test <- predict(step_lr, type = 'response', newdata = test)
prob_test <- ifelse(prob_test > threshold, "Yes", 'No')
table(test$booking_account_diff, prob_test)
1 - round(mean(test$booking_account_diff != prob_test),3)

write.csv(prob_train, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/Duration_Prob_Train_LR_Step.csv", row.names = FALSE)
write.csv(prob_test, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/Duration_Prob_Test_LR_Step.csv", row.names = FALSE)

LR_Step_OR <- exp(coef(step_lr))
write.csv(LR_Step_OR, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/duration_lr_step_or.csv")

# LR_Step_OR.CI <- exp(confint(step_lr))
# write.csv(LR_Step_OR.CI, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/duration_lr_step_orci.csv")


# # multinom function does not include p-value calculation for the regression coefficients, so we calculate p-values using Wald Z tests
# z <- summary_step_lr$coefficients/summary_step_lr$standard.errors
# pvalue <- (1 - pnorm(abs(z), 0, 1))*2  # 2-tailed test p-values
# pvalue


library(rpart)
library(rpart.plot)			# For Enhanced tree plots via PRP()
set.seed(2019)
options(digits = 3)

# default cp = 0.01. Set cp = 0 to guarantee no pruning in order to complete phrase 1: Grow tree to max.
start <- Sys.time()
cart_full <- rpart(booking_account_diff ~ ., data = train, method = 'class', control = rpart.control(minsplit = 2, cp = 0))
end <- Sys.time()
time_df$Cart_Full <- end - start
prob_train <- predict(cart_full, type = 'class')
prob_test <- predict(cart_full, newdata = test, type = 'class')
write.csv(prob_train, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/Duration_Prob_Train_Cart_Full.csv", row.names = FALSE)
write.csv(prob_test, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/Duration_Prob_Test_Cart_Full.csv", row.names = FALSE)

# print(cart_full)
# printcp(cart_full, digits = 3)
# plotcp(cart_full)

cp.opt <- cart_full$cptable[which.min(cart_full$cptable[,"xerror"]),"CP"]
start <- Sys.time()
cart_prune <- prune(cart_full, cp = cp.opt)
# cart_prune <- prune(cart_full, cp = 1.12e-04)
# cart_prune <- prune(cart_full, cp = 1.10e-04)
end <- Sys.time()
time_df$Cart_Prune <- end - start + time_df$Cart_Full
prob_train <- predict(cart_prune, type = 'class')
prob_test <- predict(cart_prune, newdata = test, type = 'class')
write.csv(prob_train, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/Duration_Prob_Train_Cart_Prune.csv", row.names = FALSE)
write.csv(prob_test, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/Duration_Prob_Test_Cart_Prune.csv", row.names = FALSE)

table(train$booking_account_diff, prob_train)
print(1 - round(mean(train$booking_account_diff != prob_train),3))

table(test$booking_account_diff, prob_test)
print(1 - round(mean(test$booking_account_diff != prob_test),3))

print(cart_prune)
printcp(cart_prune, digits = 3)
prp(cart_prune, type=2, cex = 0.5, extra=104, nn=T, fallen.leaves=T, branch.lty=3, nn.box.col = 'light blue', min.auto.cex = 0.2, nn.cex = 0.7, split.cex = 2, shadow.col="grey")
cart_prune$variable.importance
write.csv(cart_prune$variable.importance, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/duration_cart_prune_vi.csv")

library(randomForest)
start <- Sys.time()
randomForest <- randomForest(booking_account_diff ~ ., data = train, method = 'class', randomForest = TRUE, ntree = 100)
end <- Sys.time()
time_df$Random_Forest <- end - start
prob_train <- predict(randomForest)
prob_test <- predict(randomForest, newdata = test)
write.csv(prob_train, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/Duration_Prob_Train_Random_Forest.csv", row.names = FALSE)
write.csv(prob_test, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/Duration_Prob_Test_Random_Forest.csv", row.names = FALSE)

table(train$booking_account_diff, prob_train)
print(1 - round(mean(train$booking_account_diff != prob_train),3))

table(test$booking_account_diff, prob_test)
print(1 - round(mean(test$booking_account_diff != prob_test),3))

varImpPlot(randomForest)
randomForest$importance
write.csv(randomForest$importance, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/duration_forest_vi.csv")

library(earth)
start <- Sys.time()
mars <- earth(booking_account_diff~., nfold=10,  data=train, glm=list(family=binomial), degree=2)
end <- Sys.time()
time_df$Mars <- end - start
prob_train <- predict(mars, type = 'class')
prob_test <- predict(mars, newdata = test, type = 'class')
write.csv(prob_train, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/Duration_Prob_Train_Mars.csv", row.names = FALSE)
write.csv(prob_test, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/Duration_Prob_Test_Mars.csv", row.names = FALSE)

table(train$booking_account_diff, prob_train)
print(1 - round(mean(train$booking_account_diff != prob_train),3))

table(test$booking_account_diff, prob_test)
print(1 - round(mean(test$booking_account_diff != prob_test),3))

summary_mars <- summary(mars)
summary_mars$coefficients
# summary_mars

evimp(mars)
write.csv(evimp(mars), file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/duration_mars_vi.csv")
write.csv(summary_mars$coefficients, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/duration_mars_coef.csv")


time_df$Mars <- time_df$Mars * 60
# time_df$Random_Forest <- time_df$Random_Forest * 60
write.csv(time_df, file = "/Users/anqitu/Workspaces/NTU/Airbnb-new-user-bookings/training_result/Run_Time_Duration.csv", row.names = FALSE)

