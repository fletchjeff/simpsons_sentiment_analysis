## 2 - Sentiment Score Creator
# This file creates a new data set based on the concepts presented in 1_Data_Analysis.Rmd
# The final output is a line of dialogue with a corresponding sentiment label. 

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

# The code below is replicate from the 1_Data_Analysis.Rmd file and is used to get an
# integer "sentiment" value per line of dialogue. 

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

sentences <- simpsons_spark_table %>%  
  mutate(word = explode(wo_stop_words)) %>% 
  select(spoken_words, word) %>%  
  filter(nchar(word) > 2) %>% 
  compute("simpsons_spark_table")

sentence_values <- sentences %>% 
  inner_join(afinn_table) %>% 
  group_by(spoken_words) %>% 
  summarise(weighted_sum = sum(value, na.rm=TRUE))

sentence_values

### Convert Value to Labels
# As there is a range of values, the next step take these values and assign a label. The 2
# options presented to below are:
#
#### 1 - Create a binary label
# Find the average for each sentence and if its above, treat it as positive and 
# if its below, treat it as negative
#
#### 2 - Create multiclass labels
# The histogram in the 1_Data_Analysis.Rmd file shows a bimodal like distribution, with the peaks
# at -2 and 2. So we can create 3 labels: "negative" for values below -2, "neutral" from -2 to 2 and 
# "positive" for values above 2. Many of the values are exactly -2 or 2, so you can change the code
# below and make the '>' and '<' into '>=' and '<=' rather and see what difference it makes
# to the models.


weighted_sum_summary <- sentence_values %>% sdf_describe(cols="weighted_sum")

weighted_sum_mean <- as.data.frame(weighted_sum_summary)$weighted_sum[2]

sentence_scores <- sentence_values %>% 
  mutate(sent_binary = ifelse(weighted_sum > weighted_sum_mean,"positive","negative")) %>%
  mutate(sent_multi = ifelse(weighted_sum > 2,"positive",ifelse(weighted_sum < -2,"negative","neutral")))

# We need to use this data for the 2 models we're going to build, so lets write it out to a parquet
# file. Why parquet, because its better than CSV, always.

spark_write_parquet(sdf_coalesce(sentence_scores,1),"data/sentence_scores",mode="overwrite")
