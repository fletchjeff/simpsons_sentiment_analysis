## Part 4: Shiny Application
### Using the Application
# After the Application deploys, click on the blue-arrow next to the name to launch the 
# applicatio. This application is self explanitory, type in a sentence and choose which model
# to send it to to get a sentiment prediction back.

library(jsonlite)
library(dplyr)
library(sparklyr)
library(sparklyr.nested)
library(sparknlp)
library(yardstick)
library(shiny)

# Create the Spark connection
config <- spark_config()
config$spark.driver.memory <- "4g"
config$spark.master <- "local" #This is here because the master='local' in the spark_connect function is having issues. If you're running on your local machine, you shouldn't need it.
config$`sparklyr.cores.local` <- 2
config$`sparklyr.shell.driver-memory` <- "4G"
config$spark.memory.fraction <- 0.9

sc <- spark_connect(master="local", config = config)

pipeline_w2v <- ml_load(sc,"models/pipeline_w2v/")
lr_model_w2v <- ml_load(sc,"models/lr_model_w2v/")

pipeline_bert <- ml_load(sc,"models/pipeline_bert/")
lr_model_bert <- ml_load(sc,"models/lr_model_bert/")

#sentence = "I was born an oaf and I'll die an oaf"

result <- ""


get_result <- function (sentence, model) {

  test_text_df <- as.data.frame(list(sentence,-1,"positive","positive"))
  colnames(test_text_df) <- c("spoken_words","weighted_sum","sent_binary","sent_multi")
  
  sdf_copy_to(sc, test_text_df, name="test_text", overwrite = TRUE)
  test_text <- tbl(sc, "test_text")
  
  if (model == "w2v") {
    test_text_w2v <- ml_transform(pipeline_w2v,test_text)
    result <- ml_predict(lr_model_w2v,test_text_w2v)
    result <- as.data.frame(result)
  } else {
    test_text_bert <- ml_transform(pipeline_bert,test_text)
    test_text_bert <- test_text_bert %>% sdf_separate_column(column = "finished_sentence_embeddings")

    result <- ml_predict(lr_model_bert,test_text_bert)
    result <- as.data.frame(result)
  }
  
  return(paste("That sentence is",result$predicted_label,"with confidence of",paste(as.character(round(result$probability[[1]],5)), collapse = ' ')))
         
}


app <- shinyApp(ui <- fluidPage(
  titlePanel("Sentiment Analysis Model Application"),
  
  sidebarLayout(
    sidebarPanel(
      textAreaInput( 
        "caption", "Test Sentence", "I was born an oaf and I'll die an oaf"
      ),
      radioButtons(
        "model", "Choose model:", c("Word2Vec" = "w2v", "Bert" = "bert")
      ),
      submitButton("Get Sentiment", icon("arrow-right"))
    ),
    
    mainPanel(
      markdown(
        "
        #### Model Result Output
        The _Test Sentence_ will be sent to the selected model and the response will be displayed below
        "
      ),
     
      verbatimTextOutput("value")
    )
  )
),

server <- function(input, output) {
  output$value <- renderText({
    get_result(input$caption, input$model)
  })
})

runApp(app, port = as.numeric(Sys.getenv("CDSW_READONLY_PORT")), host = "127.0.0.1", launch.browser = "FALSE")
