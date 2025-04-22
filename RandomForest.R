# Load libraries
if (!require("randomForest")) install.packages("randomForest")
if (!require("caret")) install.packages("caret")
if (!require("dplyr")) install.packages("dplyr")

library(randomForest)
library(caret)
library(dplyr)

# Load dataset
income <- read.csv("C:/Users/yeu3178/Downloads/income.csv", strip.white = TRUE)

# Clean column names (remove extra spaces)
names(income) <- trimws(names(income))

# Drop 'fnlwgt' column (not useful for prediction)
income <- income %>% select(-fnlwgt)

# Convert target to factor
income$income <- as.factor(income$income)

# Convert all character variables to factors
income <- income %>%
  mutate(across(where(is.character), as.factor))

# Remove missing or unknown values (optional)
income <- income %>%
  filter(complete.cases(.))

# Set seed for reproducibility
set.seed(123)

# Split data: 70% training, 30% testing
train_index <- createDataPartition(income$income, p = 0.7, list = FALSE)
train_data <- income[train_index, ]
test_data  <- income[-train_index, ]

# Train Random Forest model
rf_model <- randomForest(income ~ ., data = train_data, ntree = 100, importance = TRUE)

# Predict on test set
predictions <- predict(rf_model, test_data)

# Evaluate with confusion matrix
conf_matrix <- confusionMatrix(predictions, test_data$income)
print(conf_matrix)

# Show feature importance
print(importance(rf_model))
varImpPlot(rf_model)
