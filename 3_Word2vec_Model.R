## 3 - Word2Vec Model
#
# This script will load the sentiment score dataset created by the `2_Sentiment_Score_Creator.R`
# script and then go through the process of build a classifier to make new sentiment predictions
# based on this data. This particular script uses Word2Vec to create the word embeddings 
# and a standard Spark ML logistic regression classifier to create the model.

library(dplyr)
library(sparklyr)

# Create the Spark connection
config <- spark_config()
config$spark.driver.memory <- "8g"
config$spark.master <- "local" #This is here because the master='local' in the spark_connect function is having issues. If you're running on your local machine, you shouldn't need it.
config$`sparklyr.cores.local` <- 2
config$`sparklyr.shell.driver-memory` <- "8G"
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

# Load the dataset from the previous step
sentence_scores <- spark_read_parquet(sc,"data/sentence_scores/")


## Creating a word2vec representation
#
# To train the model we need a numeric representation of the sentence that can be passed to the 
# Logistic Regression classifier model. This as know as word embedding and the process we're
# using here is the built in Spark Word2Vec - https://spark.rstudio.com/reference/ft_word2vec/)
# function. `ft_word2vec` is a transformer and will create a new column with a numeric 
# representation of each sentence. The data set is split into a test and training set for later 
# validation. There some other steps needed to get each sentence into the right format
# for the word2vec operation to run that are part of the overall pipeline.
# This includes:
# - removing punctuation
# - tokenizing (separating the sentence into individual words)
# - stop word removal (words like: "a, the, if, and" are not useful)


# To make this a single step pipeline and not have to use `regexp_replace` separately, the
# code below creates and new function `ft_remove_punctuation` and uses the `ft_dplyr_transformer`
# function to put into a format the makes it compatible with an `ml_pipeline` pipeline.

ft_remove_punctuation <- sentence_scores %>% 
  mutate(spoken_words = regexp_replace(spoken_words, "\'", "")) %>%
  mutate(spoken_words = regexp_replace(spoken_words, "[_\"():;,.!?\\-]", " "))

word2vec_pipeline <- ml_pipeline(sc) %>%
  ft_dplyr_transformer(ft_remove_punctuation) %>% 
  ft_tokenizer(input_col="spoken_words",output_col= "word_list") %>%
  ft_stop_words_remover(input_col = "word_list", output_col = "wo_stop_words") %>%
  ft_word2vec(input_col = "wo_stop_words",
              output_col = "result",
              min_count = 5,
              max_iter = 25,
              vector_size = 400,
              step_size = 0.0125
  )

# Now the pipeline will be fitted. This can take a loooong time. Go get coffee.

w2v_model_fitted <- ml_fit(word2vec_pipeline,sentence_scores)

# The pipeline is saved to be used in the shiny app. This is saved in the local
# directory as this is local mode Spark. Change the path if you are running this in 
# distributed mode.

ml_save(
  w2v_model_fitted,
  "models/pipeline_w2v",
  overwrite = TRUE
)

## Logistic Regression Model
#
# For this logistic regression model, the word2vec vector created above will be used to 
# predict the `sent_binary` values. This is a fairly basic operation. The complexity is in 
# creating the vector representation of the sentences. 

w2v_transformed <- ml_transform(w2v_model_fitted, sentence_scores)

# Split the data into a 70/30 train / test set
w2v_transformed_split <- w2v_transformed %>% sdf_random_split(training=0.7, test = 0.3)

# Train the model
lr_model_w2v <- w2v_transformed_split$training %>% 
  ml_logistic_regression(
    sent_binary ~ result,
    max_iter=500, 
    elastic_net_param=0.0,
    reg_param = 0.01
  )


# The model can be evaluated using the `ml_binary_classification_evaluator` function.

pred_lr_test<- ml_predict(lr_model_w2v, w2v_transformed_split$test)

ml_binary_classification_evaluator(pred_lr_test,label_col = "label",
                                   prediction_col = "prediction", metric_name = "areaUnderROC")

# 89% seems reasonable

# Save the model to use with the Shiny App.
ml_save(
  lr_model_w2v,
  "models/lr_model_w2v",
  overwrite = TRUE
)



