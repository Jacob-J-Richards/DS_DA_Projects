``` r
library('ggplot2')
setwd("/Users/jacobrichards/Desktop/DS_DA_Projects/Anamoly_Detection")
transactions <- read.csv(file="transactions.csv", na.strings = c("", "NA"))
transactions[is.na(transactions)] <- "notprovided"
data <- transactions
colnames(data) <- c("t","s","mid","pmt","pg","subtype","hr","bank")
weighted_failure_rate <- numeric(nrow(data))
weighted_failure_rate <- data[,1] - data[,2] / data[,1] * log(1+data[,1])
data$weighted <- weighted_failure_rate; data_original <- data 
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
geom_area(fill = "blue", alpha = 0.25) + geom_line(color = "black") +  
scale_x_continuous(breaks = seq(1, 72, by = 6), minor_breaks = 1:72, labels = unique_hours[seq(1, length(unique_hours), by = 6)]) + 
coord_cartesian(ylim = range(failed_transactions_rate$failedTransactions, na.rm = TRUE)) +  
labs(title = "Failed Transactions Percentage by Hour", x = "Hour (72)", y = "Failed Transactions Per Hour") +
theme(axis.text.x = element_text(angle = 60, hjust = 1, size = 8), 
axis.title.x = element_text(size = 10), axis.title.y = element_text(size = 10),
plot.background = element_rect(fill = "white", color = NA),panel.background = element_rect(fill = "white", color = NA),
panel.grid.major.x = element_blank(), panel.grid.minor.x = element_blank(), panel.grid.major.y = element_blank(), 
legend.position = "none");
```

<div align="center">

<img src="ReadMe_files/figure-gfm/unnamed-chunk-3-1.png" width="70%">

</div>

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

<div align="center">

<img src="ReadMe_files/figure-gfm/unnamed-chunk-4-1.png" width="70%">

</div>

``` r
data$pmt <- as.numeric(as.factor(data$pmt)); data$pg <- as.numeric(as.factor(data$pg))
data$bank <- as.numeric(as.factor(data$bank)); data$subtype <- as.numeric(as.factor(data$subtype))

features <- data[, c("weighted", "pmt", "pg", "bank", "subtype")]
center <- colMeans(features); cov_matrix <- cov(features)
mahalanobis_distances <- mahalanobis(features, center, cov_matrix)
data$mahalanobis_score <- mahalanobis_distances
data <- data[order(data$mahalanobis_score,decreasing = TRUE),]
top_quartile <- quantile(data$weighted, 0.999)
filtered_data <- data[data$weighted >= top_quartile, ]

filtered_data <- filtered_data[order(filtered_data$mahalanobis_score, decreasing = TRUE), ]
original_observations_found_anamolous <- data_original[rownames(filtered_data),]
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

<div align="center">

<img src="ReadMe_files/figure-gfm/unnamed-chunk-6-1.png" width="70%">

</div>

``` r
paytm_subset <- data[(data[,5] %in% c("PAYTM", "PAYTM_V2", "PAYTM_UPI", "notprovided")) & (data[,6] %in% c("UPI_COLLECT")) & (data[,4] == "UPI"),]
paytm_subset$f <- paytm_subset[,1] - paytm_subset[,2]
failure_sum_by_merchant <- aggregate(paytm_subset[,10],by=list(paytm_subset$mid),sum)
transaction_sum_by_merchant <- aggregate(paytm_subset[,1],by=list(paytm_subset$mid),sum)
failure_sum_by_merchant$transction_sum <- transaction_sum_by_merchant[,2]
failure_sum_by_merchant$failue_rate_merchant <- failure_sum_by_merchant[,2]/failure_sum_by_merchant[,3]
subset_failures_by_merchant <- failure_sum_by_merchant; data$failures <- data[,1] - data[,2]

rest_of_data_set_failure_count_by_merchant <- aggregate(data[,10],by=list(data$mid),sum)
rest_of_data_transaction_count_by_merchant <- aggregate(data[,1],by=list(data$mid),sum)
rest_of_data_transaction_count_by_merchant$failure_rate <- rest_of_data_set_failure_count_by_merchant[,2]/rest_of_data_transaction_count_by_merchant[,2]
entire_set_failures_by_merchant <- rest_of_data_transaction_count_by_merchant
entire_set_failures_by_merchant$subset_rate <- subset_failures_by_merchant[,4]

entire_set_failures_by_merchant$diff <- entire_set_failures_by_merchant[,4] - entire_set_failures_by_merchant[,3]
colnames(entire_set_failures_by_merchant) <- c("Merchant","Failures","Failure_rate_Pre","Failure_Rate_Anamoly", "Failure_Rate_Difference")
entire_set_failures_by_merchant <- entire_set_failures_by_merchant[c("Merchant","Failure_rate_Pre","Failure_Rate_Anamoly","Failure_Rate_Difference")]

library('reshape2')
long_set_failures_by_merchant <- melt(data = entire_set_failures_by_merchant, id.vars =c("Merchant"),
   measured.vars =c("Failure_rate_Before_Anamoly", "Failure_Rate_During_Anamoly","Difference_between_Failure_Rate"),
   variable.name = "Before_after_difference", value.name = "Rate")

ggplot(data = long_set_failures_by_merchant, 
  aes(x = Before_after_difference, y = Rate, fill = Merchant)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_brewer(palette = "Set2") +  
  labs(title = "Failure Rates Before and During Anomaly by Merchant",
  x = "Period", y = "Failure Rate") +theme_minimal()
```

<div align="center">

<img src="ReadMe_files/figure-gfm/unnamed-chunk-7-1.png" width="70%">

</div>

``` r
library(webshot2)
library(gt) 
gt_table <- entire_set_failures_by_merchant %>% gt() %>%
tab_header(title = "Merchant Failure Rates", subtitle = "Comparison of Failure Rates Before and During Anomaly") %>%
cols_label(Merchant = "Merchant",Failure_rate_Pre = "Failure Rate Before Anomaly",
Failure_Rate_Anamoly = "Failure Rate During Anomaly",Failure_Rate_Difference = "Difference in Failure Rate") %>%
fmt_number(columns = c(Failure_rate_Pre, Failure_Rate_Anamoly, Failure_Rate_Difference), decimals = 4) %>%
tab_style(style = cell_text(weight = "bold"), locations = cells_column_labels(everything())) %>%
tab_options(table.font.size = "small", table.width = pct(80), heading.align = "center") %>%
data_color(columns = Failure_Rate_Difference, colors = scales::col_numeric(palette = c("lightblue", "red"), domain = NULL))

gtsave(gt_table, "~/Desktop/DS_DA_Projects/Anamoly_Detection/ReadMe_files/figure-gfm/gt_table_image.png")
knitr::include_graphics("~/Desktop/DS_DA_Projects/Anamoly_Detection/ReadMe_files/figure-gfm/gt_table_image.png")
```

<div align="center">

<img src="ReadMe_files/figure-gfm/gt_table_image.png" width="70%">

</div>

All of the merchants were effected except UrbanClap.

``` r
paytm_subset <- data[(data[,5] %in% c("PAYTM", "PAYTM_V2", "PAYTM_UPI", "notprovided")) & (data[,6] %in% c("UPI_COLLECT")) & (data[,4] == "UPI"),]
unique_hours <- unique(data$hr); unique_hours <- sort(unique_hours)

t <- aggregate(paytm_subset$t,by=list(paytm_subset$hr),sum)
s <- aggregate(paytm_subset$s,by=list(paytm_subset$hr),sum)
f <- t[,2] - s[,2]

proportion_subset <- f/t[,2] * 100
```

``` r
paytm_compliment <- data[!(rownames(data) %in% rownames(paytm_subset)), ]

t <- aggregate(paytm_compliment$t,by=list(paytm_compliment$hr),sum)
s <- aggregate(paytm_compliment$s,by=list(paytm_compliment$hr),sum)
f <- t[,2] - s[,2]

proportion_c <- f/t[,2] * 100
```

``` r
hours <- seq(1,72,1); wide <- as.data.frame(cbind(hours,proportion_c,proportion_subset))

long <- melt(data = wide, id.vars = c("hours"), measured.vars = c("proportion_c", "proportion_subset"), variable.name = "percentage_failure")

ggplot(data=long,aes(x=hours,y=value,group=percentage_failure,color=percentage_failure)) + geom_smooth() + labs(title="Smoothed Failure Percentage Curve",ylab="Percentage")  + 
  scale_color_discrete(labels = c("Non-Anamous Data", "Anamous Data")) 
```

<div align="center">

<img src="ReadMe_files/figure-gfm/unnamed-chunk-11-1.png" width="70%">

</div>

``` r
ggplot(data=long,aes(x=hours,y=value,group=percentage_failure,color=percentage_failure)) + geom_line() + labs(title="Failure Percentage Line",ylab="Percentage") + 
  scale_color_discrete(labels = c("Non-Anamous Data", "Anamous Data")) 
```

<div align="center">

<img src="ReadMe_files/figure-gfm/unnamed-chunk-11-2.png" width="70%">

</div>

``` r
ggplot(data=wide,aes(proportion_subset)) + geom_density(fill="blue",alpha=0.20) + theme_minimal() + 
  labs(title = "Anomalous Data Failure Rate Density",xlab="Failure Rate",ylab="Density")
```

<div align="center">

<img src="ReadMe_files/figure-gfm/unnamed-chunk-12-1.png" width="70%">

</div>

There are tons of plots that can be produced here but only one has a
meaningful implication.

Question: Could have this anomaly been predicted before it occurred?

1.) The node on the left is the distribution of failure rates of the
anomalous subset of data before and after the anomaly

2.) The node on the right is the distribution of failure rates of the
anomalous subset of data during the anomaly

3.) these are two district distributions representing two distinct
processes.

Therefore, real life process of which this data set is a collection of
measurements on changed around February 13th 6PM and reverted back to
it’s original state around February 14th 9AM.

``` r
Note for the future: no one helped you with this, you figured this out on your own. The only clue you had was you saw Figure 1.0 produced by someone else on discord with no indication of where it came from other than it was the result of the combination of categorical variables. 
```

blah blah blah git hub test
