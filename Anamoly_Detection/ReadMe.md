# Transaction Anomaly Detection

## Introduction

You work as a Data Scientist at a Payments processor company. On
February 14th 2020, you get calls from many merchants reporting that
they are seeing a spike in customer complaints and want you to check if
there are any issues.

## Data set

The data available to you is the transaction count and how many of which
were successful for each combination of categorical variables (Bank,
Payment Gateway, Merchant, Payment Method, etc..) for each hour within
the 72 hours from February 12th to 14th.

``` r
setwd("/Users/jacobrichards/Desktop/DS_DA_Projects/Anamoly_Detection")
transactions <- read.csv(file = "transactions.csv", na.strings = c("", "NA"))

library(knitr)
library(kableExtra)

your_tibble <- head(transactions, 5)

kable(your_tibble, format = "html") %>%
  kable_styling(position = "center") %>% 
  save_kable(
    file = "~/Desktop/DS_DA_Projects/Anamoly_Detection/ReadMe_files/figure-gfm/AD_1.png", 
    zoom = 2
  )

knitr::include_graphics(
  "~/Desktop/DS_DA_Projects/Anamoly_Detection/ReadMe_files/figure-gfm/AD_1.png"
)
```

<div align="center">

<img src="ReadMe_files/figure-gfm/AD_1.png" width="70%">

</div>

## Initial Approach

We’re not sure what we’re looking for yet, so let’s plot the percentage
of failed transactions for each of the 72 hours within the entire data
set.

###### clean the data

``` r
library('ggplot2')

setwd("/Users/jacobrichards/Desktop/DS_DA_Projects/Anamoly_Detection")
transactions <- read.csv(file = "transactions.csv", na.strings = c("", "NA"))
transactions[is.na(transactions)] <- "notprovided"

data <- transactions
colnames(data) <- c("t", "s", "mid", "pmt", "pg", "subtype", "hr", "bank")
```

###### compute failure rate for each hour

``` r
unique_hours <- unique(data$hr)

t <- aggregate(data$t, by = list(data$hr), sum)
s <- aggregate(data$s, by = list(data$hr), sum)

f <- t[, 2] - s[, 2]
failure_rate <- f / t[, 2]
failure_count <- f

unique_hours <- sort(unique_hours)
```

###### plot failure rate for each hour

``` r
failed_transactions_rate <- data.frame(
  hours = unique_hours, 
  failedTransactions = failure_rate, 
  x_index = seq(1, 72, by = 1)
)

ggplot(data = failed_transactions_rate, aes(x = x_index, y = failedTransactions)) + 
  geom_area(fill = "blue", alpha = 0.25) + 
  geom_line(color = "black") +  
  scale_x_continuous(
    breaks = seq(1, 72, by = 6), 
    minor_breaks = 1:72, 
    labels = unique_hours[seq(1, length(unique_hours), by = 6)]
  ) + 
  coord_cartesian(ylim = range(failed_transactions_rate$failedTransactions, na.rm = TRUE)) +  
  labs(
    title = "Failed Transactions Percentage by Hour", 
    x = "Hour (72)", 
    y = "Failed Transactions Per Hour"
  ) +
  theme(
    axis.text.x = element_text(angle = 60, hjust = 1, size = 8), 
    axis.title.x = element_text(size = 10), 
    axis.title.y = element_text(size = 10),
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    panel.grid.major.x = element_blank(), 
    panel.grid.minor.x = element_blank(), 
    panel.grid.major.y = element_blank(), 
    legend.position = "none"
  )
```

<div align="center">

<img src="ReadMe_files/figure-gfm/unnamed-chunk-4-1.png" width="70%">

</div>

So we do see a spike in transaction failure rates up to 45%
afternoon/overnight the day before we started receiving complaints.

Clearly there is a problem here, but we need to find precisely what
caused this so the problem can be addressed.

###### plotting total transaction failures for each hour

``` r
failed_transactions <- data.frame(
  hours = unique_hours, 
  failedTransactions = failure_count, 
  x_index = seq(1, 72, by = 1)
)

ggplot(data = failed_transactions, aes(x = x_index, y = failedTransactions)) + 
  geom_area(fill = "blue", alpha = 0.25) + 
  geom_line(color = "black") +  
  scale_x_continuous(
    breaks = seq(1, 72, by = 6), 
    minor_breaks = 1:72, 
    labels = unique_hours[seq(1, length(unique_hours), by = 6)]
  ) + 
  coord_cartesian(ylim = range(failed_transactions$failedTransactions, na.rm = TRUE)) +  
  labs(
    title = "Failed Transactions Counts by Hour", 
    x = "Hour (72)", 
    y = "Failed Transactions Per Hour"
  ) +
  theme(
    axis.text.x = element_text(angle = 60, hjust = 1, size = 8), 
    axis.title.x = element_text(size = 10), 
    axis.title.y = element_text(size = 10),
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    panel.grid.major.x = element_blank(), 
    panel.grid.minor.x = element_blank(), 
    panel.grid.major.y = element_blank(), 
    legend.position = "none"
  )
```

<div align="center">

<img src="ReadMe_files/figure-gfm/unnamed-chunk-5-1.png" width="70%">

</div>

This isn’t noteworthy, the failure counts are higher during the day
because that’s when customers are active.

To narrow down what caused our failure rate spike, we employ the
Mahalanobis distance method which is useful for the detection of
anomalies. Just as you can detect outliers in a variable of one
dimension by an unusually high or low z-score, you can detect outliers
in higher dimensional variables by it’s z-score within it’s higher
dimensional distribution.

The variables we will produce a distribution of will be failure
percentage and transaction count.

``` r
before_anamoly_detection_data <- data

data$failures <- data[, 1] - data[, 2]
data$failure_rate <- (data[, 1] - data[, 2]) / data[, 1]

kable(head(data, 3), format = "html") %>%
  kable_styling(position = "center") %>% 
  save_kable(
    file = "~/Desktop/DS_DA_Projects/Anamoly_Detection/ReadMe_files/figure-gfm/appended_1.png", 
    zoom = 2
  )

knitr::include_graphics(
  "~/Desktop/DS_DA_Projects/Anamoly_Detection/ReadMe_files/figure-gfm/appended_1.png"
)
```

<div align="center">

<img src="ReadMe_files/figure-gfm/appended_1.png" width="70%">

</div>

Now that we have all of our variables prepared, we can form a higher
dimensional distribution from them to find which observations are the
greatest outliers.

Here is a plot of that distribution, as you can see the vast majority is
concentrated in the back corner of the volume.

``` r
distribution <- data.frame(failures = data$failures,rate = data$failure_rate )

library(plotly)
library(MASS)

kde <- kde2d(data$failures, data$failure_rate, n = 50)

  plot_ly(
    x = kde$x,
    y = kde$y,
    z = kde$z,
    type = "surface"
  )
```

``` r
knitr::include_graphics(
  "/Users/jacobrichards/Desktop/DS_DA_Projects/Anamoly_Detection/ReadMe_files/figure-gfm/3Ddistribution.png"
)
```

<div align="center">

<img src="ReadMe_files/figure-gfm/3Ddistribution.png" width="70%">

</div>

Evaluating the Mahalanobis method to find those outliers

``` r
features <- data[, c("t", "failure_rate")]

center <- colMeans(features)
cov_matrix <- cov(features)

mahalanobis_distances <- mahalanobis(features, center, cov_matrix)

data$mahalanobis_score <- mahalanobis_distances

data <- data[order(data$mahalanobis_score, decreasing = TRUE), ]
top_quartile <- quantile(data$mahalanobis_score, 0.999)
filtered_data <- data[data$mahalanobis_score >= top_quartile, ]
```

Table of the 10 observations found to have the greatest outlier score.

``` r
your_tibble <- head(filtered_data, 10)

kable(your_tibble, format = "html") %>%
  kable_styling(position = "center") %>% 
  save_kable(
    file = "~/Desktop/DS_DA_Projects/Anamoly_Detection/ReadMe_files/figure-gfm/mscoreog.png", 
    zoom = 2
  )

knitr::include_graphics(
  "~/Desktop/DS_DA_Projects/Anamoly_Detection/ReadMe_files/figure-gfm/mscoreog.png"
)
```

<div align="center">

<img src="ReadMe_files/figure-gfm/mscoreog.png" width="70%">

</div>

Notice how the top 10 outliers all have a PAYTM service as the payment
gateway with only difference in the variable name being the addition of
the suffix \_UPI to PAYTM. From that I deduced that the anomaly would be
present in the PAYTM payment gateways being PAYTM, PAYTM_V2, and
PAYTM_UPI. The most common combination of the remaining variables of
these top 10 observations is UPI for pmt, UPI_PAY for subtype, and UPI
for bank.

The combination of PAYTM services for the payment gateway, combined with
UPI for pmt, UPI_PAY for subtype, and UPI for bank may be responsible
for payment failures as deduced from the outliers.

Plotting the failure rates over the entire 72 hours of observations for
which this combination of variables is present.

``` r
data <- before_anamoly_detection_data

first_subset <- data[
  (data[, 4] %in% c("UPI")) & 
  (data[, 5] %in% c("PAYTM", "PAYTM_V2", "PAYTM_UPI")) & 
  (data[, 6] == "UPI_PAY") & 
  (data[, 8] == "UPI"), 
]

unique_hours <- unique(data$hr)
unique_hours <- sort(unique_hours)

t <- aggregate(first_subset$t, by = list(first_subset$hr), sum)
s <- aggregate(first_subset$s, by = list(first_subset$hr), sum)
f <- t[, 2] - s[, 2]

proportion <- f / t[, 2] * 100

failed_transactions <- data.frame(
  hours = unique_hours, 
  failedTransactions = proportion, 
  x_index = seq(1, 72, by = 1)
)

ggplot(data = failed_transactions, aes(x = x_index, y = failedTransactions)) + 
  geom_area(fill = "blue", alpha = 0.25) + 
  geom_line(color = "black") + 
  scale_x_continuous(
    breaks = seq(1, 72, by = 6), 
    minor_breaks = 1:72, 
    labels = unique_hours[seq(1, length(unique_hours), by = 6)]
  ) + 
  coord_cartesian(ylim = range(failed_transactions$failedTransactions, na.rm = TRUE)) +  
  labs(
    title = "Failed Transactions Percentage by Hour", 
    x = "Hour (72)", 
    y = "Failed Transactions Per Hour"
  ) +
  theme(
    axis.text.x = element_text(angle = 60, hjust = 1, size = 8), 
    axis.title.x = element_text(size = 10),
    axis.title.y = element_text(size = 10), 
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA), 
    panel.grid.major.x = element_blank(),  
    panel.grid.minor.x = element_blank(), 
    panel.grid.major.y = element_blank(), 
    legend.position = "none"
  )
```

<div align="center">

<img src="ReadMe_files/figure-gfm/unnamed-chunk-11-1.png" width="70%">

</div>

Unfortunately, the result is white noise. This is not the exact
problematic combination.

Assuming that the PAYTM payment gateways (PAYTM, PAYTM_V2, and
PAYTM_UPI) are part of the problem what we can do is plot the failure
rates of the observations which contain all possible combinations of the
remaining variables (payment method, subtype) with each of the PAYTM
payment gateways (PAYTM or PAYTM_V2 or PAYTM_UPI).

We will omit any combinations including the variable bank as it has over
300 different values within the data set.

``` r
#payment_methods <- unique(data[, 4])
#subtypes <- unique(data[, 6])
#filter_values <- c("PAYTM", "PAYTM_V2", "PAYTM_UPI")
#subset_list <- list()

write.csv(data,file="shiny_app_data.csv")


payment_methods <- unique(data[, 4])
subtypes <- unique(data[, 6])
filter_values <- c("PAYTM", "PAYTM_V2", "PAYTM_UPI")
subset_list <- list()


for (pm in payment_methods) {
  for (st in subtypes) {
    subset_name <- paste(pm, st, sep = "_")
    subset_list[[subset_name]] <- data[(data[, 4] == pm) & 
                                       (data[, 6] == st) & 
                                       (data[, 5] %in% filter_values), ]
  }
}

combinations <- data.frame(
  Payment_Method = character(),
  Subtype = character(),
  PMT_Values = character(),
  stringsAsFactors = FALSE
)

par(mfrow = c(3,3))

for (subset_name in names(subset_list)) {
  subset_data <- subset_list[[subset_name]]
  
  if (nrow(subset_data) > 0) { 
    
    t <- aggregate(subset_data$t, by = list(subset_data$hr), sum)
    s <- aggregate(subset_data$s, by = list(subset_data$hr), sum)
    f <- t[, 2] - s[, 2] 
    
    proportion <- f / t[, 2] * 100
    
    plot(
      x = seq(1, nrow(t), by = 1), 
      y = proportion, 
      main = paste("Plot for", subset_name), 
      xlab = "Time (hr)", 
      ylab = "Proportion (%)",
      type = "l"
    )
    
    unique_pmt <- unique(subset_data[, 5])
    combinations <- rbind(combinations, data.frame(
      Payment_Method = unique(subset_data[, 4]),
      Subtype = unique(subset_data[, 6]),
      PMT_Values = paste(unique_pmt, collapse = ", ")
    ))
  }
}

par(mfrow = c(1, 1)) 
```

<div align="center">

<img src="ReadMe_files/figure-gfm/unnamed-chunk-12-1.png" width="70%">

</div>

``` r
print(combinations)
```

    ##   Payment_Method               Subtype          PMT_Values
    ## 1             NB           notprovided               PAYTM
    ## 2         WALLET           notprovided               PAYTM
    ## 3         WALLET REDIRECT_WALLET_DEBIT               PAYTM
    ## 4         WALLET   DIRECT_WALLET_DEBIT               PAYTM
    ## 5           CARD           notprovided               PAYTM
    ## 6            UPI           UPI_COLLECT            PAYTM_V2
    ## 7            UPI               UPI_PAY PAYTM_V2, PAYTM_UPI

The plot of UPI for payment method and UPI_COLLECT for subtype reveals
that this is the combination of variables within the PAYTM payment
gateways that is casing customer complaints.

Better plot of the problematic subset of the data.

``` r
paytm_subset <- data[
  (data[, 5] %in% c("PAYTM", "PAYTM_V2", "PAYTM_UPI")) & 
  (data[, 6] %in% c("UPI_COLLECT")) & 
  (data[, 4] == "UPI"), 
]

unique_hours <- unique(data$hr)
unique_hours <- sort(unique_hours)

t <- aggregate(paytm_subset$t, by = list(paytm_subset$hr), sum)
s <- aggregate(paytm_subset$s, by = list(paytm_subset$hr), sum)
f <- t[, 2] - s[, 2]

proportion <- f / t[, 2] * 100

failed_transactions <- data.frame(
  hours = unique_hours, 
  failedTransactions = proportion, 
  x_index = seq(1, 72, by = 1)
)

ggplot(data = failed_transactions, aes(x = x_index, y = failedTransactions)) + 
  geom_area(fill = "blue", alpha = 0.25) + 
  geom_line(color = "black") + 
  scale_x_continuous(
    breaks = seq(1, 72, by = 6), 
    minor_breaks = 1:72, 
    labels = unique_hours[seq(1, length(unique_hours), by = 6)]
  ) + 
  coord_cartesian(ylim = range(failed_transactions$failedTransactions, na.rm = TRUE)) +  
  labs(
    title = "Failed Transactions Percentage by Hour", 
    x = "Hour (72)", 
    y = "Failed Transactions Per Hour"
  ) +
  theme(
    axis.text.x = element_text(angle = 60, hjust = 1, size = 8), 
    axis.title.x = element_text(size = 10),
    axis.title.y = element_text(size = 10), 
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA), 
    panel.grid.major.x = element_blank(),  
    panel.grid.minor.x = element_blank(), 
    panel.grid.major.y = element_blank(), 
    legend.position = "none"
  )
```

<div align="center">

<img src="ReadMe_files/figure-gfm/unnamed-chunk-13-1.png" width="70%">

</div>

The failure rate spike occurred from 5pm on the 13th to 6am on the 14th
the same day that merchants reported customer complaints.

Which merchants were impacted by this anomaly?

``` r
paytm_subset <- data[(data[, 5] %in% c("PAYTM", "PAYTM_V2", "PAYTM_UPI", "notprovided")) & 
                     (data[, 6] %in% c("UPI_COLLECT")) & 
                     (data[, 4] == "UPI"), ]

paytm_subset$failure_sum <- paytm_subset[, 1] - paytm_subset[, 2]
head(paytm_subset,5)
```

    ##       t  s            mid pmt          pg     subtype            hr bank
    ## 323   1  0  pharmeasytech UPI notprovided UPI_COLLECT 2020-02-12 09 Zeta
    ## 3837 34 25   medlife_prod UPI    PAYTM_V2 UPI_COLLECT 2020-02-14 11  UPI
    ## 3840 35  7 countrydelight UPI    PAYTM_V2 UPI_COLLECT 2020-02-14 07  UPI
    ## 3843 49 26        drivezy UPI    PAYTM_V2 UPI_COLLECT 2020-02-14 12  UPI
    ## 3844 28 14       fanfight UPI    PAYTM_V2 UPI_COLLECT 2020-02-14 11  UPI
    ##      failure_sum
    ## 323            1
    ## 3837           9
    ## 3840          28
    ## 3843          23
    ## 3844          14

``` r
failure_sum_by_merchant <- aggregate(paytm_subset[, 9], by = list(paytm_subset$mid), sum)
transaction_sum_by_merchant <- aggregate(paytm_subset[, 1], by = list(paytm_subset$mid), sum)

failure_sum_by_merchant$transction_sum <- transaction_sum_by_merchant[, 2]
failure_sum_by_merchant$failue_rate_merchant <- failure_sum_by_merchant[, 2] / failure_sum_by_merchant[, 3]
subset_failures_by_merchant <- failure_sum_by_merchant

data$failures <- data[, 1] - data[, 2]
rest_of_data_set_failure_count_by_merchant <- aggregate(data[, 9], by = list(data$mid), sum)
rest_of_data_transaction_count_by_merchant <- aggregate(data[, 1], by = list(data$mid), sum)

rest_of_data_transaction_count_by_merchant$failure_rate <- rest_of_data_set_failure_count_by_merchant[, 2] / rest_of_data_transaction_count_by_merchant[, 2]
entire_set_failures_by_merchant <- rest_of_data_transaction_count_by_merchant
entire_set_failures_by_merchant$subset_rate <- subset_failures_by_merchant[, 4]
entire_set_failures_by_merchant$diff <- entire_set_failures_by_merchant[, 4] - entire_set_failures_by_merchant[, 3]

colnames(entire_set_failures_by_merchant) <- c(
  "Merchant", 
  "Failures", 
  "Failure_rate_Pre", 
  "Failure_Rate_Anamoly", 
  "Failure_Rate_Difference"
)

entire_set_failures_by_merchant <- entire_set_failures_by_merchant[c(
  "Merchant", 
  "Failure_rate_Pre", 
  "Failure_Rate_Anamoly", 
  "Failure_Rate_Difference"
)]

library('reshape2')
long_set_failures_by_merchant <- melt(
  data = entire_set_failures_by_merchant, 
  id.vars = c("Merchant"),
  measured.vars = c(
    "Failure_rate_Before_Anamoly", 
    "Failure_Rate_During_Anamoly", 
    "Difference_between_Failure_Rate"
  ),
  variable.name = "Before_after_difference", 
  value.name = "Rate"
)

ggplot(data = long_set_failures_by_merchant, 
  aes(x = Before_after_difference, y = Rate, fill = Merchant)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_brewer(palette = "Set2") +  
  labs(
    title = "Failure Rates Before and During Anomaly by Merchant",
    x = "Period", 
    y = "Failure Rate"
  ) +
  theme_minimal()
```

<div align="center">

<img src="ReadMe_files/figure-gfm/unnamed-chunk-14-1.png" width="70%">

</div>

``` r
library(webshot2)
library(gt)

gt_table <- entire_set_failures_by_merchant %>%
  gt() %>%
  tab_header(
    title = "Merchant Failure Rates",
    subtitle = "Comparison of Failure Rates Before and During Anomaly"
  ) %>%
  cols_label(
    Merchant = "Merchant",
    Failure_rate_Pre = "Failure Rate Before Anomaly",
    Failure_Rate_Anamoly = "Failure Rate During Anomaly",
    Failure_Rate_Difference = "Difference in Failure Rate"
  ) %>%
  fmt_number(
    columns = c(Failure_rate_Pre, Failure_Rate_Anamoly, Failure_Rate_Difference),
    decimals = 4
  ) %>%
  tab_style(
    style = cell_text(weight = "bold"),
    locations = cells_column_labels(everything())
  ) %>%
  tab_options(
    table.font.size = "small",
    table.width = pct(80),
    heading.align = "center"
  ) %>%
  data_color(
    columns = Failure_Rate_Difference,
    colors = scales::col_numeric(
      palette = c("lightblue", "red"),
      domain = NULL
    )
  )

gtsave(
  gt_table, 
  "~/Desktop/DS_DA_Projects/Anamoly_Detection/ReadMe_files/figure-gfm/gt_table_image.png"
)

knitr::include_graphics(
  "~/Desktop/DS_DA_Projects/Anamoly_Detection/ReadMe_files/figure-gfm/gt_table_image.png"
)
```

<div align="center">

<img src="ReadMe_files/figure-gfm/gt_table_image.png" width="70%">

</div>

All of the merchants were effected except UrbanClap.

Could such a massive spike in payment failure rates have been predicted?
To investigate this, we separate the data with the combination of
variables we found to be problematic from the normal data and produce
the following plots to compare the two.

Then to produce a fair comparison, we make the number of observations
within each set equal by selected 552 of the normal data rows at random.

###### anomaly data

``` r
paytm_subset <- data[
  (data[, 5] %in% c("PAYTM", "PAYTM_V2", "PAYTM_UPI")) & 
  (data[, 6] %in% c("UPI_COLLECT")) & 
  (data[, 4] == "UPI"), 
]

unique_hours <- unique(data$hr)
unique_hours <- sort(unique_hours)

t <- aggregate(paytm_subset$t, by = list(paytm_subset$hr), sum)
s <- aggregate(paytm_subset$s, by = list(paytm_subset$hr), sum)
f <- t[, 2] - s[, 2]

proportion_subset <- f / t[, 2] * 100

transactions_subset <- t

cat("totall transactions within all anamolous observations",sum(transactions_subset[,2]))
```

    ## totall transactions within all anamolous observations 12348

###### normal data

``` r
paytm_compliment <- data[!(rownames(data) %in% rownames(paytm_subset)), ]
(nrow(paytm_compliment))
```

    ## [1] 18755

``` r
t <- aggregate(paytm_compliment$t, by = list(paytm_compliment$hr), sum)
s <- aggregate(paytm_compliment$s, by = list(paytm_compliment$hr), sum)
f <- t[, 2] - s[, 2]

proportion_compliment <- f / t[, 2] * 100

transactions_compliment <- t

(mean(proportion_compliment))
```

    ## [1] 34.63494

###### normal data but equal number of observations selected at random as the anomalous subset

``` r
paytm_compliment_sample <- paytm_compliment[sample(nrow(paytm_compliment), 1200), ]

t <- aggregate(paytm_compliment_sample$t, by = list(paytm_compliment_sample$hr), sum)

s <- aggregate(paytm_compliment_sample$s, by = list(paytm_compliment_sample$hr), sum)

f <- t[, 2] - s[, 2]

proportion_compliment_sample <- f / t[, 2] * 100

compliment_sample_sizes <- t
(sum(compliment_sample_sizes[,2]))
```

    ## [1] 12929

``` r
cat("totall transactions in sample of observations from normal data of equal size to number of anamoly observations.",sum(compliment_sample_sizes[,2]))
```

    ## totall transactions in sample of observations from normal data of equal size to number of anamoly observations. 12929

``` r
hours <- seq(1, 72, 1)
wide <- as.data.frame(cbind(hours, proportion_compliment, proportion_subset,proportion_compliment_sample))

long <- melt(
  data = wide, 
  id.vars = c("hours"), 
  measured.vars = c("proportion_compliment", "proportion_subset","proportion_compliment_sample"), 
  variable.name = "percentage_failure"
)
```

``` r
ggplot(data = long, aes(x = hours, y = value, group = percentage_failure, color = percentage_failure)) + 
  geom_smooth() + 
  labs(
    title = "Smoothed Failure Percentage Curve", 
    ylab = "Percentage"
  ) + 
  scale_color_discrete(labels = c("Non-Anomalous Data Entire Set", "Anomalous Data","Non-Anomalous Data Sample")) +
  geom_hline(yintercept = 34.63494, linetype = "dashed", color = "red")
```

<div align="center">

<img src="ReadMe_files/figure-gfm/unnamed-chunk-20-1.png" width="70%">

</div>

``` r
cat("totall transactions in sample of observations from normal data of equal size to number of anamoly observations.",sum(compliment_sample_sizes[,2]))
```

    ## totall transactions in sample of observations from normal data of equal size to number of anamoly observations. 12929

To make a fair comparison of the anomalous data and normal data before
the anomaly event, the blue line is the failure rate of the normal data
from 552 randomly selected observations within it, such that it of equal
sample size to the anomalous data comprised of an equal number of
observations.

Despite the transaction count within the normal data sample size
controlled being greater than the totall transactions of the anomalous
data, it’s variance and mean is consistently

``` r
ggplot(data = long, aes(x = hours, y = value, group = percentage_failure, color = percentage_failure)) + 
  geom_line() + 
  labs(
    title = "Failure Percentage Line", 
    ylab = "Percentage"
  ) + 
  scale_color_discrete(labels = c("Non-Anomalous Data", "Anomalous Data"))
```

<div align="center">

<img src="ReadMe_files/figure-gfm/unnamed-chunk-21-1.png" width="70%">

</div>

``` r
write.csv(wide,file="density_data.csv")
```

You can see that the anomalous data has much higher variance than the
normal data, but this is expected as the anomalous data only has ~500
observations and the normal data has 1800.

The following graphics displays the distribution of the anomalous data
over a shifting 18 hour time window of first 18 hours of the dataset
until the last 18 hours of the data set out of the 72.

![Density Plot Animation](ReadMe_files/figure-gfm/density_animation_high_quality.gif)

The distribution of the anomalous data and a sample of the normal data
of equal size to anomalous.

Notice that the anomalous data is synthetic by it’s perfect
distribution.

``` r
library(plotly)

ggplot(data = wide, aes(proportion_subset)) + 
  geom_density(data=wide,aes(proportion_compliment_sample,fill="red",alpha=0.20)) + 
  geom_density(fill = "blue", alpha = 0.20) + 
  theme_minimal() + 
  labs(
    title = "Anomalous Data Failure Rate Density", 
    x = "Failure Rate", 
    y = "Density"
  )
```

<div align="center">

<img src="ReadMe_files/figure-gfm/unnamed-chunk-22-1.png" width="70%">

</div>

2)  how could the issue have been detected earlier to prevent merchants
    and customers from discovering it. A dashboard visualizing metrics
    like success rate and volume across dimensions is then requested.

ok what if for each payment method we made a plot of a curve for each
payment gate way that somehow visualized the combination of transaction
volume and failure rate

``` r
paytm_subset <- data[
  (data[, 5] %in% c("PAYTM", "PAYTM_V2", "PAYTM_UPI")) & 
  (data[, 6] %in% c("UPI_COLLECT")) & 
  (data[, 4] == "UPI"), 
]

unique_hours <- unique(data$hr)
unique_hours <- sort(unique_hours)

t <- aggregate(paytm_subset$t, by = list(paytm_subset$hr), sum)
s <- aggregate(paytm_subset$s, by = list(paytm_subset$hr), sum)
f <- t[, 2] - s[, 2]

proportion <- f / t[, 2] * 100

failed_transactions <- data.frame(
  hours = unique_hours, 
  failedTransactions = proportion, 
  x_index = seq(1, 72, by = 1)
)

ggplot(data = failed_transactions, aes(x = x_index, y = failedTransactions)) + 
  geom_area(fill = "blue", alpha = 0.25) + 
  geom_line(color = "black") + 
  scale_x_continuous(
    breaks = seq(1, 72, by = 6), 
    minor_breaks = 1:72, 
    labels = unique_hours[seq(1, length(unique_hours), by = 6)]
  ) + 
  coord_cartesian(ylim = range(failed_transactions$failedTransactions, na.rm = TRUE)) +  
  labs(
    title = "Failed Transactions Percentage by Hour", 
    x = "Hour (72)", 
    y = "Failed Transactions Per Hour"
  ) +
  theme(
    axis.text.x = element_text(angle = 60, hjust = 1, size = 8), 
    axis.title.x = element_text(size = 10),
    axis.title.y = element_text(size = 10), 
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA), 
    panel.grid.major.x = element_blank(),  
    panel.grid.minor.x = element_blank(), 
    panel.grid.major.y = element_blank(), 
    legend.position = "none"
  )
```

<div align="center">

<img src="ReadMe_files/figure-gfm/unnamed-chunk-24-1.png" width="70%">

</div>

i have the shiny working but i don’t know where to go from here on
seeing the anamoly sooner
