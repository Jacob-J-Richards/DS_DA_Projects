library(shiny)
library(shinyjs)

# Load the data
data <- read.csv("/Users/jacobrichards/Desktop/DS_DA_Projects/Anamoly_Detection/shiny/shiny_app_data.csv")

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
          if (other_pmt == pmt) {
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
    if (nrow(pg_data) > 0) {
      # Aggregate total transactions for each PG
      pg_totals <- aggregate(pg_data$t, by = list(pg = pg_data$pg), FUN = sum)
      pg_totals <- pg_totals[pg_totals$x >= 20, ]  # Filter out PGs with less than 20 transactions
      pg_totals <- pg_totals[order(-pg_totals$x), ]  # Sort by total transactions descending
      
      # Create buttons for each PG
      pg_buttons <- lapply(pg_totals$pg, function(pg) {
        actionButton(inputId = paste0("pg_", gsub(" ", "_", pg)), label = paste(pg, "-", pg_totals$x[pg_totals$pg == pg]), class = "btn btn-success")
      })
      
      if (nrow(pg_totals) > 0) {
        output$pg_buttons <- renderUI({ do.call(tagList, pg_buttons) })
      } else {
        output$pg_buttons <- renderUI({ h4("No Payment Gateways with 20+ Transactions Available") })
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
      valid_subtypes <- subtype_totals[subtype_totals$x >= 10, ]
      
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
          
          if (nrow(subset_data) > 0) {
            t <- aggregate(subset_data$t, by = list(hr = subset_data$hr), sum)
            s <- aggregate(subset_data$s, by = list(hr = subset_data$hr), sum)
            f <- t[, 2] - s[, 2]
            
            proportion <- f / t[, 2] * 100
            
            plot(
              x = seq(1, nrow(t), by = 1),
              y = proportion,
              main = paste("Failure Rate for Subtype:", subtype),
              xlab = "Time (hr)",
              ylab = "Proportion (%)",
              type = "l",
              ylim = selections$y_range  # Dynamically set y-axis range
            )
          } else {
            plot(1, type = "n", xlab = "", ylab = "", main = "No Data Available")
          }
        })
      })
    }
  })
}

shinyApp(ui = ui, server = server)
