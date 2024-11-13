

## Transaction Failures Anamoly Detection
Given: the following Data sheet (abbreviated) containing transaction observations grouped by combinations of categorical variables, number of transactions, how many were sucessfull, and the hour in which the observation occured.

Goal: Something went wrong in the last 72 hours, find out what happened. 

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

In examination of which combination of categorical variables caused problems: Evaluate mahalanobis anamoly detection method.



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

We find the top 6 anamolous observations, of which all were with the same merchant, used the same payment method, payment gateway, sub-type, and bank.

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

Lets see how these anamolies corespond to a plot of number of transaction failures per hour and percentage of transaction failures per hour. (red dots)

Aswell the hour in which the % of failed transactions was greatest is the big green dot.

![failed_transactions_plot856](https://github.com/user-attachments/assets/83da3c80-a417-4df8-9d96-9285fad263a3)

These red dots and spikes on top of a typical transaction failure curve are from a sports gambling service. The red dots represent the top 6 transactions in terms of anamoly rating (among 20,000) by the mahalanobis method. All users in these observations used the exact same Payment Gateway, Payment Method, sub-type, and bank which is suspicious but inconclusive. The transaction failures not along the spike are actually completely normal. Notice how the spike is a deviation from the the pattern of the rest of the failure count curve.

We recomend that this merchant does not allow this combination of payment services in the future. 

Further explanation: the dips in the failed transaction curve is simply lack of consumer activity overnight. Aswell, the failure curve which the spikes are plotted on are expected for transactions. 

![percentage_failed_transactions_plot856](https://github.com/user-attachments/assets/eb5dc813-d885-4a29-8c68-de3772ff1fd1)

It may be suprising that the plots of failure percentages and failure counts per hour are so dissimilar but there is an explanation. There is a consistent baseline of transaction failures that occur every hour, i.e. the steady stream of staggered automatic payments combined with random service interuptions and networking failures which cause failures of such payments etc.. Duiring the day, this baseline of transaction failures is proportioanlly diluted by the numeracy of sucessful transactions by actual people. Hence the top occurances of payment failures and highest failure percentages were no where near each other on the plot. In this context, percentage of transaction failures is not a meaningful measurement of there being something wrong as this measurement is so highly dependent upon totall transactions. 




