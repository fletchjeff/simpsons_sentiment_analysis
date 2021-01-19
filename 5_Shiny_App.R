## 5 Shiny Application
# 
# This script will run a Shiny application (https://shiny.rstudio.com/) that can
# take a new sentence and make a sentiment prediction using either the Word2Vec or BERT
# model. This application is self explanatory, type in a sentence and choose which model
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

# Load all the transformation pipelines and the logistic regression models.
# Note: These are a bit too big for github, so the models need to be built
# using the `3_Word2vec_model.R` and `4_Bert_Model.R` scripts.

pipeline_w2v <- ml_load(sc,"models/pipeline_w2v/")
lr_model_w2v <- ml_load(sc,"models/lr_model_w2v/")

pipeline_bert <- ml_load(sc,"models/pipeline_bert/")
lr_model_bert <- ml_load(sc,"models/lr_model_bert/")


# This function will make a new prediction based on the input sentence.
# It takes the sentence, creates a single-row spark data frame, and runs
# the chosen models' `ml_transform` and `ml_predict` function on 
# the sentence and returns the predict class and the the confidence level.

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

# The 2 functions below (`app` and `server`) are basic implementations needed to create
# a Shiny application. See the Shiny docs for 

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


# The port number and other settings in the `runApp` below are specific to make it work
# on CML. To run it yourself, just uncomment the line below and comment the one
# below that.

#runApp(app, port = 8080)
runApp(app, port = as.numeric(Sys.getenv("CDSW_READONLY_PORT")), host = "127.0.0.1", launch.browser = "FALSE")
