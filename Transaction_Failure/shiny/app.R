library(shiny)
library(shinyjs)

# Load the data
data <- read.csv("shiny_app_data.csv")

# Precompute valid combinations
valid_combinations <- unique(data[, c("pmt", "pg", "subtype")])

ui <- fluidPage(
  useShinyjs(),
  # Add custom CSS for disabled buttons
  tags$style(HTML("
    .btn:disabled {
      background-color: grey !important;
      border-color: grey !important;
      color: #ffffff !important;
    }
  ")),
  titlePanel("Failure Rate Tracker"),
  sidebarLayout(
    sidebarPanel(
      h4("Select Payment Method (PMT):"),
      div(
        id = "pmt_buttons",
        lapply(unique(valid_combinations$pmt), function(pmt) {
          actionButton(inputId = paste0("pmt_", pmt), label = pmt, class = "btn btn-primary")
        })
      ),
      br(),
      h4("Select Payment Gateway (PG):"),
      uiOutput("pg_buttons"),
      br(),
      h4("Selected Variables:"),
      textOutput("selected_pmt"),
      textOutput("selected_pg")
    ),
    mainPanel(
      uiOutput("subtypePlots")  # Dynamically generate plots
    )
  )
)

server <- function(input, output, session) {
  # Reactive values to store current selections
  selections <- reactiveValues(pmt = NULL, pg = NULL, y_range = c(0, 100))
  
  # Observe PMT button clicks
  observe({
    lapply(unique(valid_combinations$pmt), function(pmt) {
      observeEvent(input[[paste0("pmt_", pmt)]], {
        selections$pmt <- pmt
        selections$pg <- NULL  # Reset PG selection
        
        # Enable/disable PMT buttons
        lapply(unique(valid_combinations$pmt), function(other_pmt) {
          if (other_pmt == pmt ) {
            disable(paste0("pmt_", other_pmt))
          } else {
            enable(paste0("pmt_", other_pmt))
          }
        })
        
        # Update PG buttons based on selected PMT
        updatePgButtons(pmt)
      })
    })
  })
  
  # Function to update PG buttons sorted by transaction volume
  updatePgButtons <- function(selected_pmt) {
    pg_data <- data[data$pmt == selected_pmt, ]
    if (nrow(pg_data) > 200) {
      # Aggregate total transactions for each PG
      pg_totals <- aggregate(pg_data$t, by = list(pg = pg_data$pg), FUN = sum)
      pg_totals <- pg_totals[pg_totals$x >= 200, ]  # Filter out PGs with less than 200 transactions
      pg_totals <- pg_totals[order(-pg_totals$x), ]  # Sort by total transactions descending
      
      # Create buttons for each PG
      pg_buttons <- lapply(pg_totals$pg, function(pg) {
        actionButton(inputId = paste0("pg_", gsub(" ", "_", pg)), label = paste(pg, "-", pg_totals$x[pg_totals$pg == pg]), class = "btn btn-success")
      })
      
      if (nrow(pg_totals) > 0) {
        output$pg_buttons <- renderUI({ do.call(tagList, pg_buttons) })
      } else {
        output$pg_buttons <- renderUI({ h4("No Payment Gateways with 200+ Transactions Available") })
      }
    } else {
      output$pg_buttons <- renderUI({ h4("No Payment Gateways Available") })
    }
  }
  
  
  # Observe PG button clicks
  observe({
    unique_pg <- unique(data$pg)
    lapply(unique_pg, function(pg) {
      observeEvent(input[[paste0("pg_", gsub(" ", "_", pg))]], {
        selections$pg <- pg
        
        # Update global y-axis range
        calculateGlobalYRange()
      })
    })
  })
  
  # Function to calculate global y-axis range
  calculateGlobalYRange <- function() {
    if (!is.null(selections$pmt) && !is.null(selections$pg)) {
      relevant_data <- data[data$pmt == selections$pmt & data$pg == selections$pg, ]
      if (nrow(relevant_data) > 0) {
        proportions <- (relevant_data$t - relevant_data$s) / relevant_data$t * 100
        proportions <- proportions[!is.na(proportions) & is.finite(proportions)]
        selections$y_range <- range(proportions, na.rm = TRUE)
      } else {
        selections$y_range <- c(0, 100)
      }
    }
  }
  
  # Text outputs for selected variables
  output$selected_pmt <- renderText({
    paste("Selected Payment Method (PMT):", selections$pmt %||% "None")
  })
  
  output$selected_pg <- renderText({
    paste("Selected Payment Gateway (PG):", selections$pg %||% "None")
  })
  
  # Dynamically generate plots for all available subtypes
  output$subtypePlots <- renderUI({
    if (!is.null(selections$pmt) && !is.null(selections$pg)) {
      # Filter subtypes with 10+ transactions
      subtype_data <- data[data$pmt == selections$pmt & data$pg == selections$pg, ]
      subtype_totals <- aggregate(subtype_data$t, by = list(subtype = subtype_data$subtype), FUN = sum)
      valid_subtypes <- subtype_totals[subtype_totals$x >= 200, ]
      
      plot_outputs <- lapply(seq_len(nrow(valid_subtypes)), function(i) {
        subtype <- valid_subtypes$subtype[i]
        transaction_count <- valid_subtypes$x[i]
        plotname <- paste0("plot_", subtype)
        
        tagList(
          h4(paste("Subtype:", subtype, "- Transactions:", transaction_count)),
          plotOutput(outputId = plotname)
        )
      })
      
      if (nrow(valid_subtypes) > 0) {
        do.call(tagList, plot_outputs)
      } else {
        h4("No Subtypes with 10+ Transactions Available")
      }
    } else {
      h4("Please select a Payment Method and Payment Gateway to view plots.")
    }
  })
  
  # Render plots for each subtype
  observe({
    if (!is.null(selections$pmt) && !is.null(selections$pg)) {
      # Filter subtypes with 10+ transactions
      subtype_data <- data[data$pmt == selections$pmt & data$pg == selections$pg, ]
      subtype_totals <- aggregate(subtype_data$t, by = list(subtype = subtype_data$subtype), FUN = sum)
      valid_subtypes <- subtype_totals[subtype_totals$x >= 10, ]
      
      lapply(valid_subtypes$subtype, function(subtype) {
        output[[paste0("plot_", subtype)]] <- renderPlot({
          subset_data <- data[
            data$pmt == selections$pmt &
              data$pg == selections$pg &
              data$subtype == subtype, 
          ]
          output[[paste0("plot_", subtype)]] <- renderPlot({
            subset_data <- data[
              data$pmt == selections$pmt &
                data$pg == selections$pg &
                data$subtype == subtype, 
            ]
            
            if (sum(subset_data$t) < 4000) {
              showNotification(
                "Warning: The subset contains less than 4000 transactions. The failure curve may not be representative of the subset.",
                type = "warning", duration = 5
              )
            }
            
            if (nrow(subset_data) > 0) {
              # Aggregate data by hour
              t <- aggregate(subset_data$t, by = list(hr = subset_data$hr), sum)
              s <- aggregate(subset_data$s, by = list(hr = subset_data$hr), sum)
              
              # Calculate failures and proportions
              f <- t[, 2] - s[, 2]
              proportion <- f / t[, 2] * 100
              
              # Create a data frame for ggplot
              plot_data <- data.frame(
                hour = seq(1,length(proportion),1),
                total_transactions = t[, 2],
                failure_proportion = proportion
              )
              
  
                
              plot_data_smooth <- data.frame(hour = plot_data$hour, failure_proportion = plot_data$failure_proportion  * 600 / 100 )
              
              
              library(ggplot2)
              
              plot_data_smooth <- data.frame(
                hour = plot_data$hour,
                failure_proportion = plot_data$failure_proportion * 600 / 100
              )
              
              max_transactions <- max(plot_data$total_transactions, na.rm = TRUE)
              y_axis_limit <- ifelse(max_transactions > 600, max_transactions, 600)
              
              ggplot() +
                geom_bar(
                  data = plot_data,
                  aes(x = hour, y = total_transactions),
                  stat = "identity",
                  fill = "steelblue",
                  alpha = 0.7
                ) +
                geom_smooth(
                  data = plot_data_smooth,
                  aes(x = hour, y = failure_proportion, group = 1),
                  color = "red",
                  size = 1.2
                ) +
                scale_y_continuous(
                  name = "Total Transactions",
                  limits = c(0,y_axis_limit),
                  sec.axis = sec_axis(~ . * 100 / 600, name = "Failure Proportion (%)")
                ) +
                labs(
                  title = paste("Failure Proportion and Total Transactions by Hour for Subtype", subtype),
                  x = "Hour"
                ) +
                theme_minimal() +
                theme(
                  axis.title.y.left = element_text(color = "steelblue"),
                  axis.title.y.right = element_text(color = "red"),
                  axis.text.y.right = element_text(color = "red"),
                  axis.text.y.left = element_text(color = "steelblue")
                )
              
              
              
            } else {
              plot(1, type = "n", xlab = "", ylab = "", main = "No Data Available")
            }
          })
        })
      })
    }
  })
}

shinyApp(ui = ui, server = server)