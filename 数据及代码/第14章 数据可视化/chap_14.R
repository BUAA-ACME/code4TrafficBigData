library(RODBC)
library(leaflet)
library(leaflet.extras)
library(shiny)
library(shinyjs)

# 建立连接  
conn <-odbcConnect("Database", "用户名", "密码") 
data = sqlQuery(conn, "SELECT * FROM traj")  


# UI
fieldsMandatory <- c("name", "favourite_pkg")

labelMandatory <- function(label) {
  tagList(
    label,
    span("*", class = "mandatory_star")
  )
}



# Define UI for app that draws a histogram ----
ui <- fluidPage(
  
  # App title ----
  titlePanel("轨迹数据可视化"),
  
  
  fluidRow(
    
    column(12,wellPanel(
      leafletOutput("mymap")
    ))
    
  ),
  
  fluidRow(
    column(12,wellPanel(
           sliderInput("slider","数据范围", min=1, max=nrow(data), value=c(25,500), ticks=FALSE)
          ))
    
  )
)


# Server
server <- function(input, output) {
  
  # output associated with the leafletOutput in the UI script
  output$mymap <- renderLeaflet({
    
    # get the data
    df <- data[c(input$slider[1]:input$slider[2]),]
    
    # definition of the leaflet map 
    m <- leaflet(data = df) %>%
      addTiles() %>%
      addPolylines(lng = ~lon, lat = ~lat,color = "#F00", weight = 8, opacity = 0.5,)
    # return the map
    m
  })
    
}


# Run the app ---- 
shinyApp(ui = ui, server = server) 
