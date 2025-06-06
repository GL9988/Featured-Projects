library(tidyverse)
library(caret)
library(glmnet)
library(randomForest)
library(corrplot)
library(keras)
library(tensorflow)
library(reticulate)
library(class)
library(ggplot2)
library(dplyr)

#Set data
df <- read.csv("/Users/gayathriprabhala/Documents/Duke Term 2/Data Science/Project/insurance.csv")

# Check for missing values and duplicates
sum(duplicated(df))
colSums(is.na(df))

df <- as.data.frame(data)

# Plotting scatter plots for numerical features vs 'charges'
numerical_features <- c('age', 'bmi', 'children', 'charges')
for (n in numerical_features) {
  p <- ggplot(df, aes_string(x = n, y = "charges")) +
    geom_point() +
    ggtitle(paste(n, "VS charges")) +
    xlab(n) +
    ylab("charges") +
    theme_minimal()
  
  print(p)
}  


# Plotting boxplots for categorical features vs 'charges'
categorical_features <- c("sex", "children", "smoker", "region")
for (n in categorical_features) {
  p <- ggplot(df, aes_string(x = n, y = "charges")) +
    geom_boxplot() +
    ggtitle(paste(n, "VS charges")) +
    xlab(n) +
    ylab("charges") +
    theme_minimal()
  
  print(p)  
}


# List of numerical features
numerical_features <- c('age', 'bmi', 'charges')

# Loop through the numerical features and create distplots
for (nf in numerical_features) {
  p <- ggplot(df, aes_string(x = nf)) +
    geom_histogram(aes(y = ..density..), bins = 30, color = "black", fill = "skyblue") +
    geom_density(alpha = 0.2, fill = "#FF6666") +
    ggtitle(paste(nf, "Distribution Plot")) +
    xlab(nf) +
    theme_minimal()
  
  print(p)
}


# Define the categorical features
categorical_features <- c('sex', 'children', 'smoker', 'region')

# Pie chart for each categorical feature
for (c in categorical_features) {
  s <- table(df[[c]])  # Calculate value counts
  
  # Create a pie chart
  pie(s, labels = names(s), main = paste("Percentage of", c),
      col = rainbow(length(s)), 
      radius = 1, 
      init.angle = 90)
}


# Scatter plot for categorical features vs 'charges'
for (c in categorical_features) {
  p <- ggplot(df, aes_string(x = c, y = "charges")) +
    geom_jitter(width = 0.3, height = 0, color = "blue", alpha = 0.6) +  # Scatter plot with jitter for better visibility
    ggtitle(paste(c, "VS charges")) +
    xlab(c) +
    ylab("charges") +
    theme_minimal()
  
  print(p)
}

# Handle outliers in numerical variables (IQR method)
numeric_features <- c('age', 'bmi', 'children', 'charges')
iqr_outliers <- lapply(numeric_features, function(f) {
  Q1 <- quantile(df[[f]], 0.25, na.rm = TRUE)
  Q3 <- quantile(df[[f]], 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  which(df[[f]] < Q1 - 1.5 * IQR | df[[f]] > Q3 + 1.5 * IQR)
})
iqr_outliers <- unique(unlist(iqr_outliers))
print(iqr_outliers)

# Encode categorical variables using dummy variables
df <- df %>%
  mutate(across(c(sex, smoker, region), as.factor)) %>%
  model.matrix(~.-1, .) %>%
  as.data.frame()

# Data standardization
scaler <- preProcess(df, method = c("center", "scale"))
scaled_df <- predict(scaler, df)

# Split the data into training and testing sets
set.seed(123)
train_index <- createDataPartition(df$charges, p = 0.7, list = FALSE)
train_data <- scaled_df[train_index, ]
test_data <- scaled_df[-train_index, ]

# 2. Feature Selection
# Use Lasso regression for feature selection to ensure the model only uses important features for predicting charges.
# Lasso regression for feature selection
x_train <- as.matrix(train_data %>% select(-charges))
y_train <- train_data$charges
lasso_model <- cv.glmnet(x_train, y_train, alpha = 1, family = "gaussian")

# Get the best lambda value
best_lambda <- lasso_model$lambda.min
coef(lasso_model, s = best_lambda)

# Get the coefficient matrix
coef_matrix <- as.matrix(coef(lasso_model, s = best_lambda))

# Making sure we only select features only with non-zero coefficients
important_features <- rownames(coef_matrix)[coef_matrix != 0]

# See the important features
print(important_features)

# 3. Model Selection
# Based on the data characteristics and insurance industry needs, we use Linear Regression, Lasso Regression, Random Forest, and Neural Network models to predict insurance charges.
# Linear Regression model
linear_model <- lm(charges ~ ., data = train_data)
linear_pred <- predict(linear_model, test_data)

# Lasso Regression model
lasso_pred <- predict(lasso_model, as.matrix(test_data %>% select(-charges)), s = best_lambda)

# Random Forest model
rf_model <- randomForest(charges ~ ., data = train_data, ntree = 200, mtry = 4)
rf_pred <- predict(rf_model, test_data)

# 4. Model Evaluation
# Use RMSE as the main evaluation metric to compare the performance of the models.
# Calculate RMSE
rmse_linear <- sqrt(mean((test_data$charges - linear_pred)^2))
rmse_lasso <- sqrt(mean((test_data$charges - lasso_pred)^2))
rmse_rf <- sqrt(mean((test_data$charges - rf_pred)^2))


# Print RMSE
#print(paste("Linear Regression RMSE:", rmse_linear))
#print(paste("Lasso Regression RMSE:", rmse_lasso))
#print(paste("Random Forest RMSE:", rmse_rf))


# 5. Data Visualization
# Use scatter plots, heatmaps, and feature importance plots to interpret the data and model predictions.
# 1. Due to the size, Create a numeric subset of the dataset
df_numeric <- df %>% select_if(is.numeric)

# 1.1 Plot Scatterplot Matrix
pairs(df_numeric)

# 2. Plot Correlation Heatmap
corr_matrix <- cor(df_numeric)  
corrplot(corr_matrix, method = "circle")

# 3. Random Forest Feature Importance Plot
varImpPlot(rf_model)


#5. Separate features and target for KNN
train_features <- train_data %>% select(-charges)
train_labels <- train_data$charges
test_features <- test_data %>% select(-charges)
test_labels <- test_data$charges

#KNN Model - Predict with k=5
knn_pred <- knn(train_features, test_features, train_labels, k = 5)
knn_pred <- as.numeric(as.character(knn_pred))

# RMSE for KNN 
rmse_knn <- sqrt(mean((test_labels - knn_pred)^2))

# Print KNN RMSE 
print(paste("K-Nearest Neighbors RMSE:", round(rmse_knn, 3)))

# Visualization: Actual vs Predicted for KNN
plot(test_labels, knn_pred, 
     main = "Actual vs Predicted: KNN", 
     xlab = "Actual Charges", ylab = "Predicted Charges", 
     col = "purple", pch = 20)
abline(0, 1, col = "red")

#Creating dataframe for all the RMSE Values
rmse_values <- data.frame(Model = c("Linear", "Lasso", "Random Forest", "KNN"),
                          RMSE = c(rmse_linear, rmse_lasso, rmse_rf, rmse_knn))



#Calculate and visualize the RMSE for different models
rmse_values <- data.frame(
  Model = c("Linear Regression", "Lasso Regression", "Random Forest", "KNN"),
  RMSE = c(rmse_linear, rmse_lasso, rmse_rf, rmse_knn)
)

#RMSE bar chart
ggplot(rmse_values, aes(x = Model, y = RMSE, fill = Model)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "RMSE Comparison Across Models", y = "RMSE Value") +
  scale_fill_brewer(palette = "Set1")

# Actual vs Predicted Charges for Linear Regression
plot(test_data$charges, linear_pred, 
     main = "Actual vs Predicted: Linear Regression", 
     xlab = "Actual Charges", ylab = "Predicted Charges", 
     col = "blue", pch = 20)
abline(0, 1, col = "red") 

# Actual vs Predicted Charges for Random Forest
plot(test_data$charges, rf_pred, 
     main = "Actual vs Predicted: Random Forest", 
     xlab = "Actual Charges", ylab = "Predicted Charges", 
     col = "blue", pch = 20)
abline(0, 1, col = "red") 

# Density Plot for Actual vs Predicted (Random Forest)
plot(density(test_data$charges), 
     main = "Density Plot: Actual vs Predicted (Random Forest)", 
     col = "blue", lwd = 2)
lines(density(rf_pred), col = "green", lwd = 2)
legend("topright", legend = c("Actual", "Predicted"), col = c("blue", "green"), lwd = 2)

#Function to calculate R square for all the models
calculate_r_squared <- function(actual, predicted) {
  ss_res <- sum((actual - predicted)^2)
  ss_tot <- sum((actual - mean(actual))^2)
  r_squared <- 1 - (ss_res / ss_tot)
  return(r_squared)
}

# Calculate R square for each model
r_squared_values <- c(
  Linear_Regression = calculate_r_squared(test_data$charges, linear_pred),
  Lasso_Regression = calculate_r_squared(test_data$charges, lasso_pred),
  Random_Forest = calculate_r_squared(test_data$charges, rf_pred),
  KNN = calculate_r_squared(test_labels, knn_pred)
)

#print(paste("Linear Regression R square:", round(r_squared_linear, 3)))
#print(paste("Lasso Regression R square:", round(r_squared_lasso, 3)))
#print(paste("Random Forest R square:", round(r_squared_rf, 3)))
#print(paste("K-Nearest Neighbors R square:", round(r_squared_knn, 3)))


results_df <- data.frame(
  #Model = c("Linear", "Lasso", "Random Forest", "KNN"),
  R_squared = r_squared_values,
  RMSE = c(rmse_linear, rmse_lasso, rmse_rf, rmse_knn)
)

# Print the results data frame
print(results_df)

