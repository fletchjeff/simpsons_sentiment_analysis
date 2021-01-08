#install.packages("remotes")
#remotes::install_github("r-spark/sparknlp")

library(dplyr)
library(sparklyr)
library(sparklyr.nested)
library(sparknlp)
library(yardstick)

config <- spark_config()
config$spark.driver.memory <- "16g"
config$spark.master <- "local"

config$`sparklyr.cores.local` <- 4
config$`sparklyr.shell.driver-memory` <- "16G"
config$spark.memory.fraction <- 0.9

#storage <- Sys.getenv("STORAGE")
#config$spark.executor.memory <- "8g"
#config$spark.executor.cores <- "2"
#config$spark.yarn.access.hadoopFileSystems <- storage

# yarn-client",
sc <- spark_connect(master="local", config = config) #, spark_home = "/etc/spark")

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

simpsons_spark_table <- 
 simpsons_spark_table %>% 
 na.omit()

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

spark_write_table(simpsons_spark_table,"simpsons_spark_table",mode = "overwrite")
spark_write_table(afinn_table,"afinn_table",mode = "overwrite")

### Convert value to Binary Label
# As there is a range of values, the next step take the mean value as the center point for Positive vs 
# Negative sentiment and adds the `sent_score` column which will be used as the dependent variable
# to train the model.

weighted_sum_summary <- sentence_values %>% sdf_describe(cols="weighted_sum")

weighted_sum_mean <- as.data.frame(weighted_sum_summary)$weighted_sum[2]

sentence_scores <- sentence_values %>% 
  mutate(sent_binary = ifelse(weighted_sum > weighted_sum_mean,"positive","negative")) %>%
  mutate(sent_multi = ifelse(weighted_sum > 2,"positive",ifelse(weighted_sum < -2,"negative","neutral")))




## Spark NLP Stuff




document_assembler <- nlp_document_assembler(sc, input_col = "spoken_words", output_col = "document")
tokenizer <- nlp_tokenizer(sc, input_cols = c("document"), output_col = "token")
normalizer <- nlp_normalizer(sc, input_cols = c("token"), output_col = "normalized")
stopwords_cleaner <- nlp_stop_words_cleaner(sc, input_cols = c("normalized"), output_col = "cleanTokens", case_sensitive = FALSE)


bert_sentence_embeddings <- 
  nlp_bert_sentence_embeddings_pretrained(
    sc, 
    input_cols = "document", 
    output_col = "sentence_embeddings",
    case_sensitive = FALSE,
  )

embeddings_finisher <- 
  nlp_embeddings_finisher(
    sc, input_cols = c("sentence_embeddings"), output_cols = c("finished_sentence_embeddings"),
                                               output_as_vector = TRUE, clean_annotations = FALSE)

label_stringIdx <- ft_string_indexer(sc, input_col = "sent_multi", output_col = "label")





nlp_pipeline_bert <- ml_pipeline(
  document_assembler,
  tokenizer,
  normalizer,
  stopwords_cleaner,
  bert_sentence_embeddings,
  embeddings_finisher,
  label_stringIdx
)

nlp_model_bert <- ml_fit(nlp_pipeline_bert, sentence_scores)
processed_bert <- ml_transform(nlp_model_bert, sentence_scores)



processed_bert <- 
  processed_bert %>% 
  sdf_separate_column(column = "finished_sentence_embeddings") %>%
  sdf_persist(storage.level = "MEMORY_AND_DISK")




processed_bert <- processed_bert %>% 
  mutate(features = explode(finished_sentence_embeddings))

processed_bert <- processed_bert %>% sdf_persist(storage.level = "MEMORY_AND_DISK")

processed_bert %>% 
  mutate(description = substr(original_words, 1, 30)) %>% 
  select(description, sent_multi, label,features) %>% 
  head(20)


splits <- sdf_random_split(processed_bert, training = 0.7, test = 0.3, seed = 100)
trainingData <- splits$training
testData <- splits$test


lr_model <- trainingData %>% 
  ml_logistic_regression(
    label ~ finished_sentence_embeddings_1,
    max_iter=500, 
    elastic_net_param=0.0,
    reg_param = 0.01
  )


pred <- ml_predict(lr_model,testData)

ml_multiclass_classification_evaluator(pred, metric_name = "accuracy")

cm <- conf_mat(pred_df %>% mutate(label = factor(sent_multi), prediction = factor(predicted_label)),label, prediction)
summary(cm)

ggplot2::autoplot(cm, type = "heatmap")

