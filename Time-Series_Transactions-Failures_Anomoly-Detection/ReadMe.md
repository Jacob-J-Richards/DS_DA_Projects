
## Transaction Failures Anomaly Detection
Given: the following data sheet (abbreviated) containing transaction observations grouped by combinations of categorical variables, number of transactions, how many were successful, and the hour in which the observation occurred.

Goal: Something went wrong in the last 72 hours, find out what happened.
![data](https://github.com/user-attachments/assets/c9ddb426-0a15-421c-b22d-265910db38d7)

Calculate and append failure totals and percentages by hour.

```r
failure_percent <- numeric(nrow(transactions))
failure_percent <- (1 - transactions[,2]/transactions[,1])*100
transactions$failure_percent <- failure_percent

transactions <- transactions[order(as.POSIXct(transactions[,7], format = "%Y-%m-%d %H")),]
unique_hours <- unique(transactions[,7])

percentage_of_failed_transactions_per_hour <- numeric(length(unique_hours))
for (i in 1:72) {
  percentage_of_failed_transactions_per_hour[i] <- ( 1 - sum(transactions[transactions[,7] == unique_hours[i],2])/sum(transactions[transactions[,7] == unique_hours[i],1]))
}

failed_transactions_per_hour <- numeric(length(unique_hours))
for (i in 1:72) {
  failed_transactions_per_hour[i] <- sum(transactions[transactions[,7] == unique_hours[i],1]) - sum(transactions[transactions[,7] == unique_hours[i],2])
}
```

```r
unique_hours_df <- data.frame(unique_hour = unique_hours, index = 1:length(unique_hours))
transactions <- transactions %>%
  mutate(unique_hour = transactions[[7]]) %>%
  left_join(unique_hours_df, by = "unique_hour") %>%
  select(-unique_hour)  
transactions_original <- transactions 
```

In examination of which combination of categorical variables caused problems: Evaluate Mahalanobis anomaly detection method.

```r
transactions$weighted_failure_score <- transactions[,9] * transactions[,1] *100 
transactions$pmt <- as.numeric(as.factor(transactions$pmt))
transactions$pg <- as.numeric(as.factor(transactions$pg))
transactions$bank <- as.numeric(as.factor(transactions$bank))
transactions$sub_type <- as.numeric(as.factor(transactions$sub_type))

features <- transactions[, c("weighted_failure_score", "pmt", "pg", "bank", "sub_type")]
center <- colMeans(features)
cov_matrix <- cov(features)
mahalanobis_distances <- mahalanobis(features, center, cov_matrix)

transactions$mahalanobis_score <- mahalanobis_distances
threshold <- quantile(mahalanobis_distances, 0.9997)
high_mahalanobis_transactions <- transactions[transactions$mahalanobis_score > threshold, ]

anomalous <- transactions_original[row.names(high_mahalanobis_transactions),]
```

```r
anomalous
```

We find the top 6 anomalous observations, of which all were with the same merchant, used the same payment method, payment gateway, sub-type, and bank.

![anomalies](https://github.com/user-attachments/assets/8ad3c78f-a637-4daa-b4bc-d621fc799457)


Let's see how these anomalies correspond to a plot of number of transaction failures per hour and percentage of transaction failures per hour (red dots).

As well, the hour in which the % of failed transactions was greatest is the big green dot.

![failed_transactions_plot856](https://github.com/user-attachments/assets/83da3c80-a417-4df8-9d96-9285fad263a3)

These red dots and spikes on top of a typical transaction failure curve are from a sports gambling service. The red dots represent the top 6 transactions in terms of anomaly rating (among 20,000 observations) by the Mahalanobis method. All users in these observations used the exact same Payment Gateway, Payment Method, sub-type, and bank. The transaction failures not along the spike are actually completely normal. Notice how the spike is a deviation from the pattern of the rest of the failure count curve.

We recommend that this merchant does not allow this combination of payment services in the future as it is abnormally failure-prone even in this context where payment failures are very normal and expected.

Further explanation: the dips in the failed transaction curve are simply a lack of consumer activity overnight. As well, the failure curve on which the spikes are plotted is expected for transactions.

![percentage_failed_transactions_plot856](https://github.com/user-attachments/assets/eb5dc813-d885-4a29-8c68-de3772ff1fd1)

It may be surprising that the plots of failure percentages and failure counts per hour are so dissimilar, but there is an explanation. There is a consistent baseline of transaction failures that occur every hour, i.e., the steady stream of staggered automatic payments combined with random frequent service interruptions and networking failures, which cause failures of such payments, etc. During the day, this baseline of transaction failures is proportionally diluted by the numeracy of successful transactions by actual people. Hence the top occurrences of payment failures and highest failure percentages were nowhere near each other on the plot. In this context, the percentage of transaction failures is not a meaningful measurement of something being wrong, as this measurement is so highly dependent upon total transactions.
