# Multivariable Logistic Binary Classifier - Delinquency Prediction

The panel data-set contains commercial customers’ financial information
and days past due indicator from 2000 to 2020. The goal is to build a
model to predict when customers will be 90+ days past due **(90+DPD)**
on payments.

## Prepare Training Data

``` r
setwd("/Users/jacobrichards/Desktop/DS_DA_Projects/Delinquency_Prediction")
train <- read.csv(file="/Users/jacobrichards/Desktop/DS_DA_Projects/Delinquency_Prediction/Data_Files/FITB_train.csv",header=TRUE)
test <- read.csv(file="/Users/jacobrichards/Desktop/DS_DA_Projects/Delinquency_Prediction/Data_Files/FITB_test.csv",header=TRUE)
```

From checking the distribution of the data, if you look carefully you
can see that the distribution of feature 3 (displayed by the blue curve)
has a lot of values in the extreme right tail.

``` r
library(ggplot2)
ggplot() + geom_density(data=train, aes(x=feature_3), color="blue") +
           geom_density(data=train, aes(x=feature_2), color="red") +
           geom_density(data=train, aes(x=feature_1), color="green") +
           geom_density(data=train, aes(x=feature_4), color="purple") +
           theme_minimal()
```

<div align="center">

<img src="ReadMe_files/figure-gfm/unnamed-chunk-2-1.png" width="70%">

</div>

This many outliers will weaken our model, so we remove the top and
bottom percentiles from feature 3. This is known as **Winsorization**.

``` r
library(dplyr)
train$key <- row.names(train)
feature_3_winsor <- data.frame(feature_3 = train$feature_3, key = row.names(train))
feature_3_winsor_clean <- na.omit(feature_3_winsor)

feature_3_winsor_clean <- feature_3_winsor_clean %>%
  mutate(z_score = (feature_3 - mean(feature_3)) / sd(feature_3),percentile = ecdf(feature_3)(feature_3) * 100)

feature_3_winsor_df <- feature_3_winsor_clean[!(feature_3_winsor_clean[, 4] < 1 | feature_3_winsor_clean[, 4] > 99), ]

non_matching_keys <- anti_join(train, feature_3_winsor_df, by = "key")$key

train <- train %>% mutate(feature_3 = ifelse(key %in% non_matching_keys, NA, feature_3))

colnames(train)[3] <- "feature_3_winsor"
```

We need to fill in the values we just removed, so we replace them with
the median of the non-outliers.

``` r
train[is.na(train[,3]),3] <- median(feature_3_winsor_clean$feature_3)

colnames(train)[3] <- "feature_3_impute"

test[is.na(test[,3]),3] <- median(feature_3_winsor_clean$feature_3)
colnames(test)[3] <- "feature_3_impute"
```

Feature 2 has missing values, so we will Impute them with the feature 2
value from the next year or the previous (if next year is also missing)
corresponding to the same ID.

``` r
train$date <- format(as.Date(train$date, format = "%Y-%m-%d"), "%Y")

train <- train %>%
  arrange(id, date) %>% # Sort by id and date
  group_by(id) %>%
  mutate(feature_2 = ifelse(is.na(feature_2),
                            lead(feature_2, order_by = date), # Try next year
                            feature_2)) %>%
  mutate(feature_2 = ifelse(is.na(feature_2),
                            lag(feature_2, order_by = date), # Try previous year
                            feature_2))

colnames(train)[2] <- "feature_2_impute"


test <- test %>%
  arrange(id, date) %>% 
  group_by(id) %>%
  mutate(feature_2 = ifelse(is.na(feature_2),
                            lead(feature_2, order_by = date), # Try next year
                            feature_2)) %>%
  mutate(feature_2 = ifelse(is.na(feature_2),
                            lag(feature_2, order_by = date), # Try previous year
                            feature_2))

colnames(test)[2] <- "feature_2_impute"

train <- na.omit(train)
test <- na.omit(test)
print(head(train,5))
```

    ## # A tibble: 5 × 8
    ## # Groups:   id [1]
    ##   feature_1 feature_2_impute feature_3_impute feature_4    id date  y      key  
    ##       <dbl>            <dbl>            <dbl>     <dbl> <int> <chr> <chr>  <chr>
    ## 1   39.2                60.3             138.    -35.5  50501 2000  active 1    
    ## 2  -12.6                58.0             126.     44.4  50501 2001  90+DPD 2    
    ## 3    0.0438            -39.3             139.     64.9  50501 2002  active 3    
    ## 4    2.30               50.0             124.     -3.59 50501 2003  active 4    
    ## 5    7.19              -83.5             150.     95.4  50501 2004  active 5

Our features (variables) all represent different financial measurements
quantified by vastly different units. In order for these variables to
have uniformity, we can reassign each value of each variable with it’s
corresponding z-score within it’s respective distribution.

``` r
library(dplyr)
train <- train %>%
  mutate(across(c(feature_1, feature_2_impute, feature_3_impute, feature_4), 
                ~ (.x - mean(.x, na.rm = TRUE)) / sd(.x, na.rm = TRUE)))

colnames(train) <- c("feature_1_standard","feature_2_standard","feature_3_standard","feature_4_standard","id","date","y","key")

test <- test %>%
  mutate(across(c(feature_1, feature_2_impute, feature_3_impute, feature_4), 
                ~ (.x - mean(.x, na.rm = TRUE)) / sd(.x, na.rm = TRUE)))

colnames(test) <- c("feature_1_standard","feature_2_standard","feature_3_standard","feature_4_standard","id","date","y")

ggplot() + 
  geom_density(data = train, aes(x = feature_3_standard), color = "blue") +
  geom_density(data = train, aes(x = feature_2_standard), color = "red") +
  geom_density(data = train, aes(x = feature_1_standard), color = "green") +
  geom_density(data = train, aes(x = feature_4_standard), color = "purple") +
  theme_minimal() +
  labs(x = "Normalized Features")
```

<div align="center">

<img src="ReadMe_files/figure-gfm/unnamed-chunk-6-1.png" width="70%">

</div>

``` r
print(head(train,5))
```

    ## # A tibble: 5 × 8
    ## # Groups:   id [1]
    ##   feature_1_standard feature_2_standard feature_3_standard feature_4_standard
    ##                <dbl>              <dbl>              <dbl>              <dbl>
    ## 1             -0.346             -0.131            -0.516              -0.708
    ## 2             -0.515             -0.159            -1.09                0.616
    ## 3             -0.473             -1.36             -0.481               0.956
    ## 4             -0.466             -0.257            -1.21               -0.180
    ## 5             -0.450             -1.91              0.0759              1.46 
    ## # ℹ 4 more variables: id <int>, date <chr>, y <chr>, key <chr>

The preparation of the training data is complete.

## Building The Model

Building a logistic regression model from features 1 to 4 as continuous
independent variables and column y as the binary dependent variable
(true/false).

Given historical data of these variable values and the payment status
they correspond to, we can produce a model which will generates a
probability that a customer will be **90+ days past due** on payments to
be used on current or future customers.

For explanation of logistic regression binary classifiers see the
following invaluable resource:
<https://seantrott.github.io/binary_classification_R/>

``` r
library(nnet)
train$y <- as.numeric(as.character(factor(train$y, levels = c("90+DPD", "active"), labels = c(1, 0))))
delinquency_model <- multinom(y ~ feature_1_standard + feature_2_standard + feature_3_standard + feature_4_standard, data=train,family=binomial())
```

    ## # weights:  6 (5 variable)
    ## initial  value 2730.999891 
    ## iter  10 value 1604.602929
    ## final  value 1604.602903 
    ## converged

When the model produces a probability of an individual being **90+DPD**,
we will have to decide at what probability we conclude that that
individual will be **90+DPD**. The value chosen has great impact on the
accuracy of the model.

## Fitting The Model

**decision threshold:** The minimum probability value that a customer
will be **90+ DPD** produced by the model at which it is concluded that
the customer will be **90+ DPD**.

Since the outcome of whether or not there will be late payments is known
in our testing data, we can directly compare the model’s predicted
outcome with the actual outcome to evaluate the accuracy of our model.
As well, we can evaluate the impact of our decision threshold on the
models accuracy.

As we evaluate the model on our testing data which was not included in
the data used to produce the model, the accuracy of these results will
have implications on the accuracy of the model being evaluated on future
data. Likewise the decision threshold accuracy found from this
evaluation will have similar implications.

``` r
(head(test,5))
```

    ## # A tibble: 5 × 7
    ## # Groups:   id [1]
    ##   feature_1_standard feature_2_standard feature_3_standard feature_4_standard
    ##                <dbl>              <dbl>              <dbl>              <dbl>
    ## 1             0.161              -0.991             0.0121              0.993
    ## 2             0.121              -0.327            -0.938              -0.328
    ## 3            -0.0959             -0.899            -2.18               -0.658
    ## 4             0.584               0.940             1.33               -0.279
    ## 5             0.0399             -1.45             -0.415               1.27 
    ## # ℹ 3 more variables: id <int>, date <chr>, y <chr>

To evaluate the effectiveness of the model find the optimal **decision
threshold**, we produce an ROC curve.

``` r
    library(pROC)
    test$predicted_y <- predict(delinquency_model, newdata = test, type = "class")
    test$y_numeric <- as.numeric(as.character(factor(test$y, levels = c("90+DPD", "active"), labels = c(1, 0))))
    test$Probability <- predict(delinquency_model, newdata = test, type = "probs")
    options(digits = 4)
    
roc_curve <- roc(response = test$y_numeric, predictor = test$Probability)

roc_metrics <- coords(roc_curve, x = "all", ret = c("threshold", "sensitivity", "specificity"))
auc_value <- auc(roc_curve)

roc_data <- data.frame(
  TPR = roc_metrics$sensitivity,
  FPR = roc_metrics$specificity,
  Threshold = roc_metrics$threshold
)

ggplot(roc_data, aes(x = FPR, y = TPR, color = Threshold)) +
  geom_line(size = 1) +
  geom_abline(slope = 1, intercept = 1, linetype = "dashed", color = "gray") +  
  geom_line(data = data.frame(FPR = c(1, 1, 0), TPR = c(0, 1, 1)), 
            aes(x = FPR, y = TPR), 
            color = "blue", size = 1, linetype = "dotted") +
  labs(
    title = "ROC Curve for Multinomial Logistic Regression",
    x = "Specificity",
    y = "True Positive Rate (Sensitivity)",
    caption = paste("AUC:", round(auc_value, 4)),
    color = "Decision Threshold"
  ) +
  scale_color_gradientn(colors = rev(rainbow(100))) +
  coord_fixed() +
  scale_x_reverse() +  # Reverse the x-axis
  ylim(0, 1) +
  theme_minimal() +
  theme(plot.caption = element_text(hjust = 0.5, size = 12))
```

<div align="center">

<img src="ReadMe_files/figure-gfm/unnamed-chunk-9-1.png" width="70%">

</div>

### The ROC curve is a plot of the models prediction accuracy ratios Sensitivity (y-axis) by Specificity (x-axis) as a result of the decision threshold chosen (displayed by color gradient).

**Sensitivity:** *what proportion of individuals who were **90+ DPD**
did the model correctly predict as being **90+ DPD.***

**Specificity:** *what proportion of individuals who were **not 90+
DDP** did the model correctly predict as being **not 90+ DPD.***

**AUC:** The area under the ROC curve, AUC is used as a metric for
overall model performance as the ROC curve is the result of accuracy
metrics from the entire range of decision thresholds. The blue dotted
line is a perfect model containing 100% of the area under it’s curve,
the grey line is if you were to predict the outcome by tossing a coin
and thus naturally the area under it’s curve is 50%.

The **AUC** of our model is 0.8211.

To illustrate the meaning of this curve take for example the accuracy
results if you were to select a decision threshold of 0.50 represented
as cyan blue: Your specificity would be about 94% which is good since
you did not falsely predict too many late payments. However, your
sensitivity would only be 25%, such that you only successfully predicted
25% of the late payments. Conversely if you had selected a decision
threshold of .10 represented as pink-red you would successfully predict
94% of late payments but only 25% of individuals who were not late on
payments were correctly identified as such by the model.

Therefore, the **decision threshold** we choose is a trade off between
successfully predicting late payments and successfully predicting non
late payments.

The **decision threshold** which balances these goals is visually
evident in the following plot.

``` r
ggplot(roc_metrics, aes(x = threshold)) +
    geom_smooth(aes(y = sensitivity, color = "Sensitivity")) +
    geom_smooth(aes(y = specificity, color = "Specificity")) +
    labs(title = "Sensitivity and Specificity vs. Threshold",
    x = "Threshold", y = "Metric Value") +
    scale_color_manual(name = "Metrics", values = c("Sensitivity" = "red", "Specificity" = "blue")) +
    theme_minimal()
```

<div align="center">

<img src="ReadMe_files/figure-gfm/unnamed-chunk-10-1.png" width="70%">

</div>

The balanced **decision threshold** is visually apparent by the
intersection of **Sensitivity** and **Specificity** and by the following
simple calculation.

``` r
optimal_threshold <- roc_metrics$threshold[which.min(abs(roc_metrics$sensitivity - roc_metrics$specificity))]
```

| **threshold** | **sensitivity** | **specificity** |
|:-------------:|:---------------:|:---------------:|
|    0.2244     |      0.771      |     0.7716      |

Confusion matrix displaying the results of balanced decision threshold
evaluated on the testing data.

``` r
test$predicted_class <- ifelse(test$Probability >= roc_metrics$threshold[which.min(abs(roc_metrics$sensitivity - roc_metrics$specificity))], 1, 0)

library(caret)
conf_matrix <- confusionMatrix(
  factor(test$predicted_class, levels = c(0, 1)),
  factor(test$y_numeric, levels = c(0, 1)))

confusion_table <- as.data.frame.matrix(conf_matrix$table)
rownames(confusion_table) <- c("Actual: Non-delinquent", "Actual: Delinquent")
colnames(confusion_table) <- c("Predicted: Non-delinquent", "Predicted: Delinquent")
```

|  | **Predicted: Non-delinquent** | **Predicted: Delinquent** |
|:--:|:--:|:--:|
| Actual: Non-delinquent | 652 | 49 |
| Actual: Delinquent | 193 | 165 |
