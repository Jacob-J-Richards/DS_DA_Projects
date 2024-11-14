# Transaction Failures Anomaly Detection

Given: the following data sheet (abbreviated) containing transaction observations grouped by categorical variables of which those transactions are described by. Each observation (grouping) includes the number of those transactions that occured at each hour, and how many of those transactions were sucessful.

Goal: Something went wrong in the last 72 hours, find out what happened. 

<img width="1259" alt="Screenshot 2024-11-14 at 8 09 57 AM" src="https://github.com/user-attachments/assets/2bd0532e-25c7-492a-b3cf-53e1e8297240">


preliminary data cleaning 
``` r
library('ggplot2')
setwd("/Users/jacobrichards/Desktop/DS_exam_2")
transactions <- read.csv(file="trans.csv", na.strings = c("", "NA"))
transactions[is.na(transactions)] <- "notprovided"
data <- transactions
colnames(data) <- c("t","s","mid","pmt","pg","subtype","hr","bank")
```

Calculate failure percentages and totalls for each of the 72 hours within the data set.
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


percent of failed transactions per hour 
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
the failure pecentage reaches a maximum at 45% then falls before it returning to baselines.

![percent_failed_before](https://github.com/user-attachments/assets/845cd636-6765-4766-a2f9-e7053b7cc4dc)




totall failed transactions per hour
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

![806](https://github.com/user-attachments/assets/aaea92b0-aeda-4073-97c1-8601085deb52)


failure counts are at their lowest at the same time that failure % is at it's highest. This is due to transaction failure % being highly dependent on totall transaction volume. 

Since the previous were inconclusive, we will not utilise the mahalanobis method. But before we can do that we have a problem with how we quantify the signifigance of failure within an observation. This is why we should not use the % of failed transactions to quantify failure occurances.

let observation 1 have 1 transaction of which none were successful and let observation 2 have 1000 transactions of which 100 were sucessful ones.

observation 1 would have a 100% failure rate and observation 2 would have a failure rate of 90%, so we need a better metric.

Thus we utilize:

weighted_failure_rate = failure_rate * natural_logarithim(1+transactions) 

by using this weighted failure rate, we will have a quantitative value for the signifigance of failure of each observation which values the failure percentage according to how many transactions there were than comprise it.

we append this value for each observation and then run the mahalanobis method, this will prioritze the observations that have a high failure percentage and a high failure count that that hopefully will have something insightful that we can see in them.

```{r}
weighted_failure_rate <- numeric(nrow(data))
weighted_failure_rate <- data[,1] - data[,2] / data[,1] * log(1+data[,1])
data$weighted <- weighted_failure_rate
data_original <- data 
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

<img width="1308" alt="Screenshot 2024-11-14 at 8 12 03 AM" src="https://github.com/user-attachments/assets/77fc52c6-a3c3-4cdc-9b5f-6e9600e78f19">

There is definetly something happening with UPI and PAYTM services.  
``` r
data <- data_original
```
By trial and error, I found this combination of UPI and PAYTM services that produces a 80% failure rate from Febuary 13th 6PM to Febuary 14th 9AM.
payment method: UPI 
payment gateway: PAYTM or PAYTM_V2 or PAYTM_UPI 
sub-type: UPI_COLLECT

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

![failure_percent_by_category_combination](https://github.com/user-attachments/assets/b605f4bc-e1e4-400b-91bc-4bba14bd0173)




