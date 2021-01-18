## 4 - BERT Model

#install.packages("remotes")
#remotes::install_github("r-spark/sparknlp")

library(dplyr)
library(sparklyr)
library(sparklyr.nested)
library(sparknlp)
library(yardstick)

# Create the Spark connection
config <- spark_config()
config$spark.driver.memory <- "16g"
config$spark.master <- "local" #This is here because the master='local' in the spark_connect function is having issues. If you're running on your local machine, you shouldn't need it.
config$`sparklyr.cores.local` <- 2
config$`sparklyr.shell.driver-memory` <- "16G"
config$spark.memory.fraction <- 0.9

# If you want to run in distributed mode, uncomment these lines and change master = 'local' to 
# master = 'yarn-client'. Both above for the config parameter and below in the in 
# spark_connect function.

#config$spark.executor.memory <- "8g"
#config$spark.executor.cores <- "2"

# If you're running this on CML in distributed mode, uncomment the following and make 
# sure your project has environment variable named STORAGE that points to the right
# hive warehouse storage location. 
# See: https://github.com/fastforwardlabs/cml_churn_demo_mlops/blob/master/0_bootstrap.py

#storage <- Sys.getenv("STORAGE")
#config$spark.yarn.access.hadoopFileSystems <- storage

sc <- spark_connect(master="local", config = config)

sentence_scores <- spark_read_parquet(sc,"data/sentence_scores/")


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

ml_save(
  nlp_model_bert,
  "models/pipeline_bert",
  overwrite = TRUE
)

processed_bert <- ml_transform(nlp_model_bert, sentence_scores)


processed_bert <- 
  processed_bert %>% 
  sdf_separate_column(column = "finished_sentence_embeddings") %>%
  sdf_persist(storage.level = "MEMORY_AND_DISK")


processed_bert <- processed_bert %>% sdf_persist(storage.level = "MEMORY_AND_DISK")

processed_bert %>% 
  mutate(description = substr(spoken_words, 1, 30)) %>% 
  select(description, sent_multi, label,finished_sentence_embeddings_1) %>% 
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

cm <- conf_mat(pred %>% as.data.frame() %>% mutate(label = factor(sent_multi), prediction = factor(predicted_label)),label, prediction)
summary(cm)

ggplot2::autoplot(cm, type = "heatmap")

ml_save(
  lr_model,
  "models/lr_model_bert",
  overwrite = TRUE
)


