
Given: the following Data sheet (abbreviated)




``` r
print(head(transactions[,1:6],10))
```

```
##    t success           mid    pmt          pg              sub_type
## 1  2       1        zivame     NB        PAYU                      
## 2  1       0     urbanclap     NB       PAYTM                      
## 3  5       1 pharmeasytech WALLET AIRTELMONEY REDIRECT_WALLET_DEBIT
## 4  1       1 pharmeasytech   CARD        PAYU                      
## 5  1       1      fanfight   CARD    RAZORPAY                      
## 6  3       3      fanfight   CARD  GOCASHFREE                      
## 7  2       1      fanfight   CARD  GOCASHFREE                      
## 8  1       1        zivame   CARD        PAYU                      
## 9  2       2   purplle.com   CARD       PAYTM                      
## 10 1       1       drivezy   CARD  GOCASHFREE
```

Calculate and append failure totals and percentages by hour.


``` r
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




``` r
unique_hours_df <- data.frame(unique_hour = unique_hours, index = 1:length(unique_hours))
transactions <- transactions %>%
  mutate(unique_hour = transactions[[7]]) %>%
  left_join(unique_hours_df, by = "unique_hour") %>%
  select(-unique_hour)  
transactions_original <- transactions 
```

Evaluate mahalanobis anamoly detection method.


``` r
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

anamolous <- transactions_original[row.names(high_mahalanobis_transactions),]
```


``` r
anamolous
```

```
##          t success      mid pmt        pg sub_type            hr bank
## 4145  1025     639 fanfight UPI PAYTM_UPI  UPI_PAY 2020-02-12 13  UPI
## 4523  1258     767 fanfight UPI PAYTM_UPI  UPI_PAY 2020-02-12 14  UPI
## 4911  3340    2462 fanfight UPI PAYTM_UPI  UPI_PAY 2020-02-12 15  UPI
## 10806  794     417 fanfight UPI PAYTM_UPI  UPI_PAY 2020-02-13 14  UPI
## 17292 1357     918 fanfight UPI PAYTM_UPI  UPI_PAY 2020-02-14 14  UPI
## 17688 3365    2524 fanfight UPI PAYTM_UPI  UPI_PAY 2020-02-14 15  UPI
##       failure_percent index
## 4145         37.65854    14
## 4523         39.03021    15
## 4911         26.28743    16
## 10806        47.48111    39
## 17292        32.35077    63
## 17688        24.99257    64
```

Plot failed transactions per hour by line with red dots highlighting the top 6 anomalous observations.




``` r
failed_transactions_per_hour_anamoly <- failed_transactions_per_hour[anamolous$index]

failed_transactions <- data.frame(hours = unique_hours, failedTransactions = failed_transactions_per_hour, x_index = seq(1:72))

anomalous_data <- data.frame(hour_anamoly = anamolous$index,FailedTransactions_anamoly = failed_transactions_per_hour[anamolous$index])

ggplot(data = failed_transactions, aes(x = x_index, y = failedTransactions)) + 
  geom_area(fill = "blue", alpha = 0.25) + 
  geom_line(color = "black") +  
  scale_x_continuous(breaks = 1:72, labels = unique_hours) + 
  geom_point(data = anomalous_data, aes(x = hour_anamoly, y = FailedTransactions_anamoly),
  color = "red",size=2) + 
  labs(title = "Failed Transactions by Hour with Anomalous Transactions in Red", 
       x = "Hour (72)", 
       y = "Failed Transactions Per Hour") +
       theme(axis.text.x = element_text(angle = 60, hjust = 1, size = 5), 
           plot.background = element_rect(fill = "white", color = NA),
           panel.background = element_rect(fill = "white", color = NA),
           panel.grid.major = element_line(color = "black", linewidth = 0.05), 
           panel.grid.major.y = element_blank(), 
           panel.grid.major.x = element_blank())
```

![](R_files/figure-html/unnamed-chunk-9-1.png)<!-- -->

``` r
ggsave("failed_transactions_plot.png", plot = last_plot(), width = 10, height = 6, dpi = 300)
```

The table is the top 6 anomalies, the red dots in the plot of transactions failures per hour are these top 6 anomalies. 

This is a sports event gambling service, the top 6 anomalies all occurred from users failure to repay gambling debts.
All delinquent users in these observations used the exact same Payment Gateway, Payment Method, sub-type, and bank which 
cannot be a coincidence. Delinquent users most likely used this combination of services as an exploit, perhaps to avoid 
repayment. 

In futher explanation: the dips in the failed transaction curve is simply lack of consumer activity overnight. 



