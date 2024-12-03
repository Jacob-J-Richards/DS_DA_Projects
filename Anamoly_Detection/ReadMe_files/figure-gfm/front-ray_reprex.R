#' ---
#' output:
#'   reprex::reprex_document:
#'     advertise: FALSE
#' ---

library(tibble)
print(as_tibble(head(test[c(1,2,3,4,6),], 5)))
