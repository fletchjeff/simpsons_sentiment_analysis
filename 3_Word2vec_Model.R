## 3 - Word2Vec Model
# 

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

# Load the data from step 2
sentence_scores <- spark_read_parquet(sc,"data/sentence_scores/")

# word2vec_pipeline <- ml_pipeline(sc) %>%
#   ft_dplyr_transformer(ft_remove_punction) %>% 
#   ft_tokenizer(input_col="spoken_words",output_col= "word_list") %>%
#   ft_stop_words_remover(input_col = "word_list", output_col = "wo_stop_words")
# 
# sentence_values_tokenized <- 
#   sentence_scores %>% 
#   ft_tokenizer(input_col="spoken_words",output_col= "word_list") %>%
#   ft_stop_words_remover(input_col = "word_list", output_col = "wo_stop_words")

### Creating a word2vec model
# To train the model we need a numeric representation of the sentence that can be passed to the 
# Logistic Regression classifier model. This as know as word embedding and the process we're
# using here is the built in Spark [Word2Vec](https://spark.rstudio.com/reference/ft_word2vec/)
# function. `ft_word2vec` is a transformer and will create a new column with a numeric representation 
# of each sentence. The data set is split into a test and training set for later validation.

ft_remove_punction <- sentence_scores %>% 
  mutate(spoken_words = regexp_replace(spoken_words, "\'", "")) %>%
  mutate(spoken_words = regexp_replace(spoken_words, "[_\"():;,.!?\\-]", " "))

word2vec_pipeline <- ml_pipeline(sc) %>%
  ft_dplyr_transformer(ft_remove_punction) %>% 
  ft_tokenizer(input_col="spoken_words",output_col= "word_list") %>%
  ft_stop_words_remover(input_col = "word_list", output_col = "wo_stop_words") %>%
  ft_word2vec(input_col = "wo_stop_words",
              output_col = "result",
              min_count = 5,
              max_iter = 25,
              vector_size = 400,
              step_size = 0.0125
  )

w2v_model_fitted <- ml_fit(word2vec_pipeline,sentence_scores)

ml_save(
  w2v_model_fitted,
  "models/pipeline_w2v",
  overwrite = TRUE
)

# Logistic Regression Model

w2v_transformed <- ml_transform(w2v_model_fitted, sentence_scores)

w2v_transformed_split <- w2v_transformed %>% sdf_random_split(training=0.7, test = 0.3)

lr_model_w2v <- w2v_transformed_split$training %>% 
  ml_logistic_regression(
    sent_binary ~ result,
    max_iter=500, 
    elastic_net_param=0.0,
    reg_param = 0.01
  )

pred_lr_test<- ml_predict(lr_model_w2v, w2v_transformed_split$test)

ml_binary_classification_evaluator(pred_lr_test,label_col = "label",
                                   prediction_col = "prediction", metric_name = "areaUnderROC")

# 89% seems reasonable

ml_save(
  lr_model_w2v,
  "models/lr_model_w2v",
  overwrite = TRUE
)



