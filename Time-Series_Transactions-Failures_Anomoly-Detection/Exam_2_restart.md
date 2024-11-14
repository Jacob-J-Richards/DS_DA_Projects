

``` r
library('ggplot2')
setwd("/Users/jacobrichards/Desktop/DS_exam_2")
transactions <- read.csv(file="trans.csv", na.strings = c("", "NA"))
transactions[is.na(transactions)] <- "notprovided"
data <- transactions
colnames(data) <- c("t","s","mid","pmt","pg","subtype","hr","bank")
weighted_failure_rate <- numeric(nrow(data))
weighted_failure_rate <- data[,1] - data[,2] / data[,1] * log(1+data[,1])
data$weighted <- weighted_failure_rate
data_original <- data 
(head(data,5))
```

```
##   t s           mid    pmt          pg               subtype            hr
## 1 2 1        zivame     NB        PAYU           notprovided 2020-02-14 06
## 2 1 0     urbanclap     NB       PAYTM           notprovided 2020-02-14 06
## 3 5 1 pharmeasytech WALLET AIRTELMONEY REDIRECT_WALLET_DEBIT 2020-02-14 11
## 4 1 1 pharmeasytech   CARD        PAYU           notprovided 2020-02-14 12
## 5 1 1      fanfight   CARD    RAZORPAY           notprovided 2020-02-14 06
##                                  bank  weighted
## 1                             NB_CITI 1.4506939
## 2                             NB_SYNB 1.0000000
## 3                         AIRTELMONEY 4.6416481
## 4 THE SATARA SAHAKARI BANK LTD MUMBAI 0.3068528
## 5                                 DCB 0.3068528
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

![](Exam_2_restart_files/figure-html/unnamed-chunk-3-1.png)<!-- -->

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
labs(title = "Failed Transactions Percentage by Hour", x = "Hour (72)", y = "Failed Transactions Per Hour") +
theme(axis.text.x = element_text(angle = 60, hjust = 1, size = 8), 
axis.title.x = element_text(size = 10), axis.title.y = element_text(size = 10),
plot.background = element_rect(fill = "white", color = NA),panel.background = element_rect(fill = "white", color = NA),
panel.grid.major.x = element_blank(), panel.grid.minor.x = element_blank(), panel.grid.major.y = element_blank(), 
legend.position = "none")
```

![](Exam_2_restart_files/figure-html/unnamed-chunk-4-1.png)<!-- -->

``` r
ggsave("failed_count_before.png", plot = last_plot(), width = 10, height = 6, dpi = 300)
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

![](Exam_2_restart_files/figure-html/unnamed-chunk-6-1.png)<!-- -->

``` r
  ggsave("failure_percent_by_category_combination.png", plot = last_plot(), width = 10, height = 6, dpi = 300)
```
