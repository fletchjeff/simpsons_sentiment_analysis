## 4 - BERT Model
#
# Similar to the word2vec model, this script will load the sentiment score dataset created 
# by the `2_Sentiment_Score_Creator.R` script. The main difference is that the word embedding
# process is based on a pre-trained BERT model use the SparkNLP library.
# This script also implements a multi-class classifier, rather than a binary classifier.

# To install sparknlp, use the code below.
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


## Spark NLP
#
# The pipeline below follows a similar process to the word2vec process, the primary difference
# is that this is using the SparkNLP library to do the same jobs. There is similar pipeline
# that does the remove punctuations, tokenizing and stop word removal. There after the vector
# representation is done using a pre-trained BERT model from Google. 


document_assembler <- nlp_document_assembler(sc, input_col = "spoken_words", output_col = "document")
tokenizer <- nlp_tokenizer(sc, input_cols = c("document"), output_col = "token")
normalizer <- nlp_normalizer(sc, input_cols = c("token"), output_col = "normalized")
stopwords_cleaner <- nlp_stop_words_cleaner(sc, input_cols = c("normalized"), output_col = "cleanTokens", case_sensitive = FALSE)

# This can take a while as is a 900MB model download. Spark on k8s in distributed mode
# has issues with where to store the model. I got it working on CML - eventually - but staying
# in local mode makes it easier.

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

# This is a bit quicker than the Word2Vec pipeline / model as its using the pre-trained version, not
# training from the start.

nlp_model_bert <- ml_fit(nlp_pipeline_bert, sentence_scores)

# Save the pipeline for the Shiny Application

ml_save(
  nlp_model_bert,
  "models/pipeline_bert",
  overwrite = TRUE
)

## Logistic Regression Model
#
# For this logistic regression model, the BERT vector created above will be used to 
# predict the `sent_multi` values. There is some work done to optimise the Spark dataFrame
# to make it easier to work with for additional testing. 

processed_bert <- ml_transform(nlp_model_bert, sentence_scores)


# The SparkNLP library puts the output from the BERT model in a column of lists, and the
# the actual embeddings vector that will be used have to be coerced out of the data. 
# If you want to see the output from the BERT model, uncomment the `sdf_persist` and 
# the `%>%` on the line above to keep it in memory. Makes it faster to use.

processed_bert <- 
  processed_bert %>% 
  sdf_separate_column(column = "finished_sentence_embeddings") # %>% 
  # sdf_persist(storage.level = "MEMORY_AND_DISK")

# This is just to display some results in R-Studio
processed_bert %>% 
  mutate(description = substr(spoken_words, 1, 30)) %>% 
  select(description, sent_multi, label,finished_sentence_embeddings_1) %>% 
  head(20)

# Split the data into a 70/30 train test split. 

splits <- sdf_random_split(processed_bert, training = 0.7, test = 0.3, seed = 100)
trainingData <- splits$training
testData <- splits$test

# Next step is to train a Logistic Regression classifier against the multiple classes
# This part takes a long time as the BERT vector is much wider than the Word2Vec one.

lr_model <- trainingData %>% 
  ml_logistic_regression(
    label ~ finished_sentence_embeddings_1,
    max_iter=500, 
    elastic_net_param=0.0,
    reg_param = 0.01
  )

# Checking the model performance is slightly different for multi-class classifiers. 
# This uses the `ml_multiclass_classification_evaluator` function along with a confusion
# matrix. 

pred <- ml_predict(lr_model,testData)

ml_multiclass_classification_evaluator(pred, metric_name = "accuracy")

# 70% is not great, but I don't think the process of making the different class labels is
# particularly good. The aim here is to show how BERT works though.

cm <- conf_mat(pred %>% as.data.frame() %>% mutate(label = factor(sent_multi), prediction = factor(predicted_label)),label, prediction)
summary(cm)

ggplot2::autoplot(cm, type = "heatmap")

# Save the model to use in the Shiny App

ml_save(
  lr_model,
  "models/lr_model_bert",
  overwrite = TRUE
)


