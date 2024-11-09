
#1.1.1 Which dimension combination caused the issue?
#  Explore the data and visualization to understand when the issue (possibly a significant number of failures in transactions ) 
#happened and which combination of dimension (pmt, pg, bank and sub_type) has the impact.
#Tip: Identify the method to detect an anomaly in a metric across 4+ dimensions and apply that method to find the above.


setwd("C:/Users/jake pc/Desktop/DS_Exam_2")
transactions <- read.csv(file="transactions(1).csv",header=TRUE)

#add column for failure % of each observation 
failure_percent <- numeric(nrow(transactions))
failure_percent <- (1 - transactions[,2]/transactions[,1])*100
transactions$failure_percent <- failure_percent




