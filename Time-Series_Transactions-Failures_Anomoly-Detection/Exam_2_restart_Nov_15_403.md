


``` r
library('ggplot2')
setwd("/Users/jacobrichards/Desktop/DS assesment/DS_exam_2")
transactions <- read.csv(file="trans.csv", na.strings = c("", "NA"))
transactions[is.na(transactions)] <- "notprovided"
data <- transactions
colnames(data) <- c("t","s","mid","pmt","pg","subtype","hr","bank")
weighted_failure_rate <- numeric(nrow(data))
weighted_failure_rate <- data[,1] - data[,2] / data[,1] * log(1+data[,1])
data$weighted <- weighted_failure_rate
data_original <- data 
```



``` r
unique_hours <- unique(data$hr)
t <- aggregate(data$t,by=list(data$hr),sum)
s <- aggregate(data$s,by=list(data$hr),sum)

f <- t[,2] - s[,2]
failure_rate <- f/t[,2]
failure_count <- f

unique_hours <- unique(data$hr)
unique_hours <- sort(unique_hours)
```



``` r
failed_transactions_rate <- data.frame(hours = unique_hours, failedTransactions = failure_rate, x_index = seq(1, 72, by = 1))
ggplot(data = failed_transactions_rate, aes(x = x_index, y = failedTransactions)) + 
geom_area(fill = "blue", alpha = 0.25) + 
geom_line(color = "black") +  
scale_x_continuous(breaks = seq(1, 72, by = 6), minor_breaks = 1:72, labels = unique_hours[seq(1, length(unique_hours), by = 6)]) + 
coord_cartesian(ylim = range(failed_transactions_rate$failedTransactions, na.rm = TRUE)) +  
labs(title = "Failed Transactions Percentage by Hour", x = "Hour (72)", y = "Failed Transactions Per Hour") +
theme(axis.text.x = element_text(angle = 60, hjust = 1, size = 8), 
axis.title.x = element_text(size = 10), axis.title.y = element_text(size = 10),
plot.background = element_rect(fill = "white", color = NA),panel.background = element_rect(fill = "white", color = NA),
panel.grid.major.x = element_blank(), panel.grid.minor.x = element_blank(), panel.grid.major.y = element_blank(), 
legend.position = "none")
```

![](Exam_2_restart_Nov_15_403_files/figure-html/unnamed-chunk-3-1.png)<!-- -->

``` r
ggsave("percent_failed_before.png", plot = last_plot(), width = 10, height = 6, dpi = 300)
```


``` r
failed_transactions_rate <- data.frame(hours = unique_hours, failedTransactions = failure_rate, x_index = seq(1, 72, by = 1))
ggplot(data = failed_transactions_rate, aes(x = x_index, y = failedTransactions)) + 
geom_area(fill = "blue", alpha = 0.25) + 
geom_line(color = "black") +  
scale_x_continuous(breaks = seq(1, 72, by = 6), minor_breaks = 1:72, labels = unique_hours[seq(1, length(unique_hours), by = 6)]) + 
coord_cartesian(ylim = range(failed_transactions_rate$failedTransactions, na.rm = TRUE)) +  
labs(title = "Failed Transactions Percentage by Hour", x = "Hour (72)", y = "Failed Transactions Per Hour") +
theme(axis.text.x = element_text(angle = 60, hjust = 1, size = 8), 
axis.title.x = element_text(size = 10), axis.title.y = element_text(size = 10),
plot.background = element_rect(fill = "white", color = NA),panel.background = element_rect(fill = "white", color = NA),
panel.grid.major.x = element_blank(), panel.grid.minor.x = element_blank(), panel.grid.major.y = element_blank(), 
legend.position = "none")
```

![](Exam_2_restart_Nov_15_403_files/figure-html/unnamed-chunk-4-1.png)<!-- -->

``` r
ggsave("percent_failed_before.png", plot = last_plot(), width = 10, height = 6, dpi = 300)
```





``` r
failed_transactions <- data.frame(hours = unique_hours, failedTransactions = failure_count, x_index = seq(1, 72, by = 1))
ggplot(data = failed_transactions, aes(x = x_index, y = failedTransactions)) + 
geom_area(fill = "blue", alpha = 0.25) + 
geom_line(color = "black") +  
scale_x_continuous(breaks = seq(1, 72, by = 6), minor_breaks = 1:72, labels = unique_hours[seq(1, length(unique_hours), by = 6)]) + 
coord_cartesian(ylim = range(failed_transactions$failedTransactions, na.rm = TRUE)) +  
labs(title = "Failed Transactions Counts by Hour", x = "Hour (72)", y = "Failed Transactions Per Hour") +
theme(axis.text.x = element_text(angle = 60, hjust = 1, size = 8), 
axis.title.x = element_text(size = 10), axis.title.y = element_text(size = 10),
plot.background = element_rect(fill = "white", color = NA),panel.background = element_rect(fill = "white", color = NA),
panel.grid.major.x = element_blank(), panel.grid.minor.x = element_blank(), panel.grid.major.y = element_blank(), 
legend.position = "none")
```

![](Exam_2_restart_Nov_15_403_files/figure-html/unnamed-chunk-5-1.png)<!-- -->

``` r
ggsave("806.png", plot = last_plot(), width = 10, height = 6, dpi = 300)
```




``` r
data$pmt <- as.numeric(as.factor(data$pmt))
data$pg <- as.numeric(as.factor(data$pg))
data$bank <- as.numeric(as.factor(data$bank))
data$subtype <- as.numeric(as.factor(data$subtype))


features <- data[, c("weighted", "pmt", "pg", "bank", "subtype")]
center <- colMeans(features)
cov_matrix <- cov(features)
mahalanobis_distances <- mahalanobis(features, center, cov_matrix)

data$mahalanobis_score <- mahalanobis_distances

data <- data[order(data$mahalanobis_score,decreasing = TRUE),]

top_quartile <- quantile(data$weighted, 0.999)

filtered_data <- data[data$weighted >= top_quartile, ]

filtered_data <- filtered_data[order(filtered_data$mahalanobis_score, decreasing = TRUE), ]

original_observations_found_anamolous <- data_original[rownames(filtered_data),]

original_observations_found_anamolous
```

```
##         t    s      mid    pmt        pg             subtype            hr
## 3872 3365 2524 fanfight    UPI PAYTM_UPI             UPI_PAY 2020-02-14 15
## 4467 3340 2462 fanfight    UPI PAYTM_UPI             UPI_PAY 2020-02-12 15
## 4170 1391 1086 fanfight    UPI PAYTM_UPI             UPI_PAY 2020-02-12 02
## 4134 1357  918 fanfight    UPI PAYTM_UPI             UPI_PAY 2020-02-14 14
## 4889 1230 1001 fanfight WALLET     PAYTM DIRECT_WALLET_DEBIT 2020-02-14 15
## 4970 1197  952 fanfight WALLET     PAYTM DIRECT_WALLET_DEBIT 2020-02-12 15
## 3944 1258  767 fanfight    UPI PAYTM_UPI             UPI_PAY 2020-02-12 14
## 3838 1156  884 fanfight    UPI PAYTM_UPI             UPI_PAY 2020-02-14 11
## 3866 1078  754 fanfight    UPI PAYTM_UPI             UPI_PAY 2020-02-14 13
## 3910 1025  639 fanfight    UPI PAYTM_UPI             UPI_PAY 2020-02-12 13
## 4544  794  417 fanfight    UPI PAYTM_UPI             UPI_PAY 2020-02-13 14
## 3914  722  470 fanfight    UPI PAYTM_UPI             UPI_PAY 2020-02-12 12
## 4369  658  484 fanfight    UPI PAYTM_UPI             UPI_PAY 2020-02-14 07
## 4247  597  373 fanfight    UPI PAYTM_UPI             UPI_PAY 2020-02-13 13
## 4944  501  395 fanfight WALLET     PAYTM DIRECT_WALLET_DEBIT 2020-02-12 02
## 4862  478  396 fanfight WALLET     PAYTM DIRECT_WALLET_DEBIT 2020-02-14 11
## 4169  550  365 fanfight    UPI PAYTM_UPI             UPI_PAY 2020-02-12 11
## 3908  538  359 fanfight    UPI PAYTM_UPI             UPI_PAY 2020-02-12 07
## 4168  496  379 fanfight    UPI PAYTM_UPI             UPI_PAY 2020-02-12 03
## 3912  480  332 fanfight    UPI PAYTM_UPI             UPI_PAY 2020-02-12 06
##       bank  weighted
## 3872   UPI 3358.9083
## 4467   UPI 3334.0189
## 4170   UPI 1385.3487
## 4134   UPI 1352.1199
## 4889 PAYTM 1224.2092
## 4970 PAYTM 1191.3624
## 3944   UPI 1253.6479
## 3838   UPI 1150.6061
## 3866   UPI 1073.1152
## 3910   UPI 1020.6776
## 4544   UPI  790.4926
## 3914   UPI  717.7144
## 4369   UPI  653.2257
## 4247   UPI  593.0053
## 4944 PAYTM  496.0971
## 4862 PAYTM  472.8870
## 4169   UPI  545.8113
## 3908   UPI  533.8030
## 4168   UPI  491.2559
## 3912   UPI  475.7284
```

``` r
data <- data_original
```





``` r
paytm_subset <- data[(data[,5] %in% c("PAYTM", "PAYTM_V2", "PAYTM_UPI", "notprovided")) & (data[,6] %in% c("UPI_COLLECT")) & (data[,4] == "UPI"),]
unique_hours <- unique(data$hr)
unique_hours <- sort(unique_hours)

t <- aggregate(paytm_subset$t,by=list(paytm_subset$hr),sum)
s <- aggregate(paytm_subset$s,by=list(paytm_subset$hr),sum)
f <- t[,2] - s[,2]

proportion <- f/t[,2] * 100
failed_transactions <- data.frame(hours = unique_hours, failedTransactions = proportion, x_index = seq(1, 72, by = 1))

ggplot(data = failed_transactions, aes(x = x_index, y = failedTransactions)) + geom_area(fill = "blue", alpha = 0.25) + 
  geom_line(color = "black") + scale_x_continuous(breaks = seq(1, 72, by = 6), minor_breaks = 1:72, 
  labels = unique_hours[seq(1, length(unique_hours), by = 6)]) + 
  coord_cartesian(ylim = range(failed_transactions$failedTransactions, na.rm = TRUE)) +  
  labs(title = "Failed Transactions Percentage by Hour", x = "Hour (72)", y = "Failed Transactions Per Hour") +
  theme(axis.text.x = element_text(angle = 60, hjust = 1, size = 8), axis.title.x = element_text(size = 10),
  axis.title.y = element_text(size = 10), plot.background = element_rect(fill = "white", color = NA),
  panel.background = element_rect(fill = "white", color = NA), panel.grid.major.x = element_blank(),  
  panel.grid.minor.x = element_blank(), panel.grid.major.y = element_blank(), legend.position = "none")
```

![](Exam_2_restart_Nov_15_403_files/figure-html/unnamed-chunk-7-1.png)<!-- -->

``` r
  ggsave("failure_percent_by_category_combination.png", plot = last_plot(), width = 10, height = 6, dpi = 300)
```







``` r
paytm_subset <- data[(data[,5] %in% c("PAYTM", "PAYTM_V2", "PAYTM_UPI", "notprovided")) & (data[,6] %in% c("UPI_COLLECT")) & (data[,4] == "UPI"),]


(head(paytm_subset,5))
```

```
##       t  s            mid pmt          pg     subtype            hr bank
## 323   1  0  pharmeasytech UPI notprovided UPI_COLLECT 2020-02-12 09 Zeta
## 3837 34 25   medlife_prod UPI    PAYTM_V2 UPI_COLLECT 2020-02-14 11  UPI
## 3840 35  7 countrydelight UPI    PAYTM_V2 UPI_COLLECT 2020-02-14 07  UPI
## 3843 49 26        drivezy UPI    PAYTM_V2 UPI_COLLECT 2020-02-14 12  UPI
## 3844 28 14       fanfight UPI    PAYTM_V2 UPI_COLLECT 2020-02-14 11  UPI
##      weighted
## 323   1.00000
## 3837 31.38577
## 3840 34.28330
## 3843 46.92423
## 3844 26.31635
```

``` r
(unique(paytm_subset$mid))
```

```
## [1] "pharmeasytech"  "medlife_prod"   "countrydelight" "drivezy"       
## [5] "fanfight"       "zivame"         "urbanclap"      "purplle.com"
```

``` r
paytm_subset$f <- paytm_subset[,1] - paytm_subset[,2]

#sum transactions and failures stratified by merchants to calculate the failure rate for each merchant 

failure_sum_by_merchant <- aggregate(paytm_subset[,10],by=list(paytm_subset$mid),sum)

transaction_sum_by_merchant <- aggregate(paytm_subset[,1],by=list(paytm_subset$mid),sum)

failure_sum_by_merchant$transction_sum <- transaction_sum_by_merchant[,2]

failure_sum_by_merchant$failue_rate_merchant <- failure_sum_by_merchant[,2]/failure_sum_by_merchant[,3]

subset_failures_by_merchant <- failure_sum_by_merchant
```



``` r
data$failures <- data[,1] - data[,2]
rest_of_data_set_failure_count_by_merchant <- aggregate(data[,10],by=list(data$mid),sum)
rest_of_data_transaction_count_by_merchant <- aggregate(data[,1],by=list(data$mid),sum)
rest_of_data_transaction_count_by_merchant$failure_rate <- rest_of_data_set_failure_count_by_merchant[,2]/rest_of_data_transaction_count_by_merchant[,2]
entire_set_failures_by_merchant <- rest_of_data_transaction_count_by_merchant
entire_set_failures_by_merchant$subset_rate <- subset_failures_by_merchant[,4]
```


``` r
entire_set_failures_by_merchant$diff <- entire_set_failures_by_merchant[,4] - entire_set_failures_by_merchant[,3]

colnames(entire_set_failures_by_merchant) <- c("Merchant","Failures","Failure_rate_Before_Anamoly","Failure_Rate_During_Anamoly", "Difference_between_Failure_Rate")

entire_set_failures_by_merchant <- entire_set_failures_by_merchant[c("Merchant","Failure_rate_Before_Anamoly","Failure_Rate_During_Anamoly","Difference_between_Failure_Rate")]

entire_set_failures_by_merchant
```

```
##         Merchant Failure_rate_Before_Anamoly Failure_Rate_During_Anamoly
## 1 countrydelight                   0.2526981                   0.3792633
## 2        drivezy                   0.4115018                   0.4911779
## 3       fanfight                   0.3421881                   0.5831158
## 4   medlife_prod                   0.3588827                   0.3968254
## 5  pharmeasytech                   0.3290114                   0.4622222
## 6    purplle.com                   0.4134916                   0.5192308
## 7      urbanclap                   0.3156129                   0.2446352
## 8         zivame                   0.3603816                   0.4149131
##   Difference_between_Failure_Rate
## 1                      0.12656520
## 2                      0.07967612
## 3                      0.24092765
## 4                      0.03794269
## 5                      0.13321084
## 6                      0.10573920
## 7                     -0.07097771
## 8                      0.05453145
```

``` r
library('reshape2')
long_set_failures_by_merchant <-melt(data = entire_set_failures_by_merchant, id.vars =c("Merchant"),measured.vars =c("Failure_rate_Before_Anamoly",
        "Failure_Rate_During_Anamoly", "Difference_between_Failure_Rate"),variable.name = "Before_after_difference", value.name = "Rate")

ggplot(data = long_set_failures_by_merchant, 
       aes(x = Before_after_difference, y = Rate, fill = Merchant)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_brewer(palette = "Set2") +  # Choose a cohesive color palette
  labs(
    title = "Failure Rates Before and During Anomaly by Merchant",
    x = "Period (Before or During Anomaly)",
    y = "Failure Rate"
  ) +
  theme_minimal()
```

![](Exam_2_restart_Nov_15_403_files/figure-html/unnamed-chunk-10-1.png)<!-- -->


all were effected except ubran clap 



``` r
paytm_subset <- data[(data[,5] %in% c("PAYTM", "PAYTM_V2", "PAYTM_UPI", "notprovided")) & (data[,6] %in% c("UPI_COLLECT")) & (data[,4] == "UPI"),]
unique_hours <- unique(data$hr)
unique_hours <- sort(unique_hours)

t <- aggregate(paytm_subset$t,by=list(paytm_subset$hr),sum)
s <- aggregate(paytm_subset$s,by=list(paytm_subset$hr),sum)
f <- t[,2] - s[,2]

proportion_subset <- f/t[,2] * 100

paytm_compliment <- data[!(rownames(data) %in% rownames(paytm_subset)), ]

t <- aggregate(paytm_compliment$t,by=list(paytm_compliment$hr),sum)
s <- aggregate(paytm_compliment$s,by=list(paytm_compliment$hr),sum)
f <- t[,2] - s[,2]
proportion_c <- numeric(72)
proportion_c <- f/t[,2] * 100

hours <- seq(1,72,1)

plot <- cbind(hours,proportion_c,proportion_subset)

try <- as.data.frame(plot)

long_data <- melt(data = try, id.vars = c("hours"),
measured.vars = c("proportion_c", "proportion_subset"),
variable.name = "percentage_failure")


ggplot(data=long_data,aes(x=hours,y=value,group=percentage_failure,color=percentage_failure)) + geom_smooth() 
```

```
## `geom_smooth()` using method = 'loess' and formula = 'y ~ x'
```

![](Exam_2_restart_Nov_15_403_files/figure-html/unnamed-chunk-11-1.png)<!-- -->

``` r
ggplot(data=long_data,aes(x=hours,y=value,group=percentage_failure,color=percentage_failure)) + geom_line()
```

![](Exam_2_restart_Nov_15_403_files/figure-html/unnamed-chunk-11-2.png)<!-- -->



``` r
ggplot(data=try,aes(proportion_c)) + geom_density()
```

![](Exam_2_restart_Nov_15_403_files/figure-html/unnamed-chunk-12-1.png)<!-- -->

``` r
ggplot(data=try,aes(proportion_subset)) + geom_density()
```

![](Exam_2_restart_Nov_15_403_files/figure-html/unnamed-chunk-12-2.png)<!-- -->

``` r
compliment_before_shift <- proportion_c[1:40]
subset_before_shift <- proportion_subset[1:40]

shapiro_test <- shapiro.test(compliment_before_shift)
shapiro_test <- shapiro.test(subset_before_shift)

print(shapiro_test)
```

```
## 
## 	Shapiro-Wilk normality test
## 
## data:  subset_before_shift
## W = 0.97467, p-value = 0.4989
```

``` r
# Interpretation
if (shapiro_test$p.value > 0.05) {
  print("Fail to reject H₀: Sample appears to be from a normal distribution.")
} else {
  print("Reject H₀: Sample does not appear to be from a normal distribution.")
}
```

```
## [1] "Fail to reject H₀: Sample appears to be from a normal distribution."
```

``` r
t_test_result <- t.test(compliment_before_shift, subset_before_shift, paired = TRUE, alternative = "two.sided")

print(t_test_result)
```

```
## 
## 	Paired t-test
## 
## data:  compliment_before_shift and subset_before_shift
## t = -1.5144, df = 39, p-value = 0.138
## alternative hypothesis: true mean difference is not equal to 0
## 95 percent confidence interval:
##  -3.5387957  0.5085895
## sample estimates:
## mean difference 
##       -1.515103
```


 

