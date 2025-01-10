setwd("~/Desktop/DS_DA_Projects/3-90+_DaysPastDue")
train <- read.csv(file="data/train.csv", header=TRUE)
test <- read.csv(file="data/test.csv", header=TRUE)

data <- rbind(train,test)

############################################################

data[,2][is.na(data[,2])] <- median(data[,2], na.rm=TRUE)

############################################################

library(tidyverse)

data <- data %>%
  arrange(id, date) %>%
  group_by(id) %>%
  mutate(feature_3 = case_when(
    is.na(feature_3) & !is.na(lag(feature_3)) ~ lag(feature_3),
    is.na(feature_3) & !is.na(lead(feature_3)) ~ lead(feature_3),
    TRUE ~ feature_3
  )) %>%
  filter(!is.na(feature_3)) %>%
  ungroup()


print(head(data,5))

############################################################


data$y <- as.numeric(ifelse(data$y == "active", 0, ifelse(data$y == "90+DPD", 1, data$y)))

############################################################

data$id <- factor(data$id, levels = unique(data$id))

############################################################

data$date <- substr(data$date, 1, 4)

############################################################

data$feature_1 <- scale(data$feature_1)
data$feature_2 <- scale(data$feature_2)
data$feature_3 <- scale(data$feature_3)
data$feature_4 <- scale(data$feature_4)

############################################################

data$f1_f2 <- data$feature_1 * data$feature_2
data$f1_f3 <- data$feature_1 * data$feature_3
data$f1_f4 <- data$feature_1 * data$feature_4
data$f2_f3 <- data$feature_2 * data$feature_3
data$f2_f4 <- data$feature_2 * data$feature_4
data$f3_f4 <- data$feature_3 * data$feature_4

############################################################

data_1 <- data

library(lme4)



library(skimr)
print(skim(data_1))