#install.packages("remotes")
#remotes::install_github("r-spark/sparknlp")

library(dplyr)
library(sparklyr)
library(sparklyr.nested)

config <- spark_config()
config$spark.driver.memory <- "8g"

#storage <- Sys.getenv("STORAGE")
#config$spark.executor.memory <- "8g"
#config$spark.executor.cores <- "2"
#config$spark.yarn.access.hadoopFileSystems <- storage

# yarn-client",
sc <- spark_connect(master="local", config = config)

#paste("http://spark-",Sys.getenv("CDSW_ENGINE_ID"),".",Sys.getenv("CDSW_DOMAIN"),sep="")


cols = list(
  raw_character_text = "character",
  spoken_words = "character"
)

spark_read_csv(
  sc,
  name = "simpsons_spark_table",
  path = "data/simpsons_dataset.csv",
  infer_schema = FALSE,
  columns = cols,
  header = TRUE
)

spark_read_csv(
  sc,
  name = "afinn_table",
  path = "data/AFINN-en-165.txt",
  infer_schema = TRUE,
  delimiter = ",",
  header = FALSE
)

simpsons_spark_table <- tbl(sc, "simpsons_spark_table")
afinn_table <- tbl(sc, "afinn_table")
afinn_table <- afinn_table %>% rename(word = V1, value = V2)

simpsons_spark_table <- 
   simpsons_spark_table %>% 
   rename(raw_char = raw_character_text)

# Drop NA's
simpsons_spark_table <- 
 simpsons_spark_table %>% 
 na.omit()

# 
simpsons_spark_table <- 
 simpsons_spark_table %>% 
 mutate(spoken_words = regexp_replace(spoken_words, "\'", "")) %>%
 mutate(spoken_words = regexp_replace(spoken_words, "[_\"():;,.!?\\-]", " "))

simpsons_spark_table <- 
 simpsons_spark_table %>% 
 ft_tokenizer(input_col="spoken_words",output_col= "word_list")

simpsons_spark_table <- 
 simpsons_spark_table %>% 
 ft_stop_words_remover(input_col = "word_list", output_col = "wo_stop_words")

## Create Sentiment values per Sentence
# This is explained in the previous step and its how to create a sentiment value for each 
# line of dialogue. 

sentences <- simpsons_spark_table %>%  
  mutate(word = explode(wo_stop_words)) %>% 
  select(spoken_words, word) %>%  
  filter(nchar(word) > 2) %>% 
  compute("simpsons_spark_table")

sentence_values <- sentences %>% 
  inner_join(afinn_table) %>% 
  group_by(spoken_words) %>% 
  summarise(weighted_sum = sum(value, na.rm=TRUE))

#spark_write_table(simpsons_spark_table,"simpsons_spark_table",mode = "overwrite")
#spark_write_table(afinn_table,"afinn_table",mode = "overwrite")

### Convert value to Binary Label
# As there is a range of values, the next step take the mean value as the center point for Positive vs 
# Negative sentiment and adds the `sent_score` column which will be used as the dependent variable
# to train the model.

weighted_sum_summary <- sentence_values %>% sdf_describe(cols="weighted_sum")

weighted_sum_mean <- as.data.frame(weighted_sum_summary)$weighted_sum[2]

sentence_scores <- sentence_values %>% 
  mutate(sent_score = ifelse(weighted_sum > weighted_sum_mean,1,0))

sentence_values_tokenized <- 
  sentence_scores %>% 
  ft_tokenizer(input_col="spoken_words",output_col= "word_list") %>%
  ft_stop_words_remover(input_col = "word_list", output_col = "wo_stop_words")


### Creating a word2vec model
# To train the model we need a numeric representation of the sentence that can be passed to the 
# Logistic Regression classifier model. This as know as word embedding and the process we're
# using here is the built in Spark [Word2Vec](https://spark.rstudio.com/reference/ft_word2vec/)
# function. 

# _Note:_ If your model has already been saved, you can bypass this process by commenting out the following code:
# _Comment from here:_
# ============
w2v_model <- ft_word2vec(sc,
                        input_col = "wo_stop_words",
                        output_col = "result",
                        min_count = 5,
                        max_iter = 25,
                        vector_size = 400,
                        step_size = 0.0125
                       )

w2v_model_fitted <- ml_fit(w2v_model,sentence_values_tokenized)



ml_save(
  w2v_model_fitted,
  "w2v_model_fitted",
  overwrite = TRUE
)
# =============
# _to here.
# 
# _And uncomment the lines below from:_
# ==============
# w2v_model_fitted <- ml_load(
#   sc, 
#   paste(Sys.getenv("STORAGE"),"/datalake/data/sentiment/w2v_model_fitted",sep="")
# )
# ==============
# _to here._


# `word2vec` is a transformer and will create a new column with a numeric representation of each 
# sentence. The data set is split into a test and training set for later validation.

w2v_transformed <- ml_transform(w2v_model_fitted, sentence_values_tokenized)

w2v_transformed_split <- w2v_transformed %>% sdf_random_split(training=0.7, test = 0.3)

### Creating a Logistic Regression model
# The next step is to train a logistic regression model using the `sent_score` binary label calculated earlier
# and the word2vec numeric representation calcuated in the previous step.
# 
# _Note:_ If your model has already been saved, you can bypass this process by commenting out the following code:
# _Comment from here:_
# ============

# lr_model <- result %>% 
#   ml_logistic_regression(
#     sent_score ~ clean_embeddings,
#     max_iter=500, 
#     elastic_net_param=0.0,
#     reg_param = 0.01
#   )

lr_model <- w2v_transformed_split$training %>% select(result,sent_score) %>% 
  ml_logistic_regression(
    sent_score ~ result,
    max_iter=500, 
    elastic_net_param=0.0,
    reg_param = 0.01
  )

pred_lr_training <- ml_predict(lr_model, w2v_transformed_split$training)

pred_lr_test<- ml_predict(lr_model, w2v_transformed_split$test)

ml_binary_classification_evaluator(pred_lr_training,label_col = "sent_score",
                                   prediction_col = "prediction", metric_name = "areaUnderROC")

ml_binary_classification_evaluator(pred_lr_test,label_col = "sent_score",
                                   prediction_col = "prediction", metric_name = "areaUnderROC")

# ml_save(
#    lr_model,
#    "lr_model",
#    overwrite = TRUE
# )
# =============
# _to here.
# 
# _And load the model by uncommenting the lines below from here:_
# ==============
# lr_model <- ml_load(
#   sc, 
#   paste(Sys.getenv("STORAGE"),"/datalake/data/sentiment/lr_model",sep="")
# )
# ==============
# _to here._

### Showing the Model Performance
# The model performance can be shown using the `ml_binary_classification_evaluator`
# function from sparklyr. 


# 89% seems reasonable.