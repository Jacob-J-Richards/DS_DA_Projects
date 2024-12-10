library(shiny)
library(shinyjs)

# Load the data
data <- read.csv("/Users/jacobrichards/Desktop/DS_DA_Projects/Anamoly_Detection/shiny/shiny_app_data.csv")

# Ensure hr is numeric
if(!is.numeric(data$hr)) {
  data$hr <- as.numeric(data$hr)
}

# Precompute valid combinations
valid_combinations <- unique(data[, c("pmt", "pg", "subtype")])

ui <- fluidPage(
  useShinyjs(),
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
  selections <- reactiveValues(pmt = NULL, pg = NULL, y_range = c(0, 100))
  
  # Observe PMT button clicks
  observe({
    lapply(unique(valid_combinations$pmt), function(pmt) {
      observeEvent(input[[paste0("pmt_", pmt)]], {
        selections$pmt <- pmt
        selections$pg <- NULL
        
        lapply(unique(valid_combinations$pmt), function(other_pmt) {
          if (other_pmt == pmt) {
            disable(paste0("pmt_", other_pmt))
          } else {
            enable(paste0("pmt_", other_pmt))
          }
        })
        
        updatePgButtons(pmt)
      })
    })
  })
  
  updatePgButtons <- function(selected_pmt) {
    pg_data <- data[data$pmt == selected_pmt, ]
    if (nrow(pg_data) > 0) {
      pg_totals <- aggregate(pg_data$t, by = list(pg = pg_data$pg), FUN = sum)
      pg_totals <- pg_totals[pg_totals$x >= 20, ]  
      pg_totals <- pg_totals[order(-pg_totals$x), ]
      
      pg_buttons <- lapply(pg_totals$pg, function(pg) {
        actionButton(inputId = paste0("pg_", gsub(" ", "_", pg)), 
                     label = paste(pg, "-", pg_totals$x[pg_totals$pg == pg]), 
                     class = "btn btn-success")
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
  
  observe({
    unique_pg <- unique(data$pg)
    lapply(unique_pg, function(pg) {
      observeEvent(input[[paste0("pg_", gsub(" ", "_", pg))]], {
        selections$pg <- pg
        calculateGlobalYRange()
      })
    })
  })
  
  calculateGlobalYRange <- function() {
    if (!is.null(selections$pmt) && !is.null(selections$pg)) {
      relevant_data <- data[data$pmt == selections$pmt & data$pg == selections$pg, ]
      if (nrow(relevant_data) > 0) {
        proportions <- (relevant_data$t - relevant_data$s) / relevant_data$t * 100
        proportions <- proportions[is.finite(proportions)]
        if (length(proportions) > 0) {
          selections$y_range <- range(proportions)
        } else {
          selections$y_range <- c(0, 100)
        }
      } else {
        selections$y_range <- c(0, 100)
      }
    }
  }
  
  output$selected_pmt <- renderText({
    paste("Selected Payment Method (PMT):", selections$pmt %||% "None")
  })
  
  output$selected_pg <- renderText({
    paste("Selected Payment Gateway (PG):", selections$pg %||% "None")
  })
  
  output$subtypePlots <- renderUI({
    if (!is.null(selections$pmt) && !is.null(selections$pg)) {
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
  
  observe({
    if (!is.null(selections$pmt) && !is.null(selections$pg)) {
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
          
          if (nrow(subset_data) == 0) {
            plot(1, type = "n", main = "No Data Available", xlab = "", ylab = "")
            return()
          }
          
          # Aggregate both t and s together
          hr_data <- aggregate(cbind(t, s) ~ hr, data = subset_data, FUN = sum)
          hr_data <- hr_data[order(hr_data$hr), ]
          
          if (nrow(hr_data) == 0) {
            plot(1, type = "n", main = "No Data Available", xlab = "", ylab = "")
            return()
          }
          
          f <- hr_data$t - hr_data$s
          proportion <- (f / hr_data$t) * 100
          
          # Valid data checks
          valid_idx <- is.finite(proportion) & !is.na(proportion) & is.finite(hr_data$t) & hr_data$t > 0
          if (!any(valid_idx)) {
            plot(1, type = "n", main = "No Data Available", xlab = "", ylab = "")
            return()
          }
          
          hours <- hr_data$hr[valid_idx]
          vol <- hr_data$t[valid_idx]
          prop <- proportion[valid_idx]
          
          if (length(hours) == 0) {
            plot(1, type = "n", main = "No Data Available", xlab = "", ylab = "")
            return()
          }
          
          # Determine volume y-limits
          max_vol <- max(vol, na.rm = TRUE)
          if (!is.finite(max_vol)) {
            plot(1, type = "n", main = "No Data Available", xlab = "", ylab = "")
            return()
          }
          
          vol_ylim <- c(0, max_vol * 1.2)
          
          # Ensure y_range is finite
          if (any(!is.finite(selections$y_range))) {
            selections$y_range <- c(0, 100)
          }
          
          # Plot volume (left axis)
          plot(
            x = hours,
            y = vol,
            type = "h",
            lwd = 5,
            lend = "butt",
            col = "lightblue",
            main = paste("Failure Rate and Volume for Subtype:", subtype),
            xlab = "Time (hr)",
            ylab = "Volume (Transactions)",
            ylim = vol_ylim
          )
          
          # Overlay failure rate (right axis)
          par(new = TRUE)
          plot(
            x = hours,
            y = prop,
            type = "l",
            lwd = 2,
            col = "red",
            axes = FALSE,
            xlab = "",
            ylab = "",
            ylim = selections$y_range
          )
          
          axis(side = 4, col.axis = "red", col = "red")
          mtext("Failure Rate (%)", side = 4, line = 3, col = "red")
          
          legend("top", legend = c("Volume", "Failure Rate"), 
                 col = c("lightblue", "red"), 
                 lty = c(1, 1),
                 lwd = c(5, 2),
                 bty = "n", cex = 0.8, text.col = c("black","red"))
        })
      })
    }
  })
}

shinyApp(ui = ui, server = server)
