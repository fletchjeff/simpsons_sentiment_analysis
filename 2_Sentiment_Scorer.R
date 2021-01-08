#install.packages("remotes")
#remotes::install_github("r-spark/sparknlp")

library(dplyr)
library(sparklyr)
library(sparklyr.nested)
library(sparknlp)

config <- spark_config()
config$spark.driver.memory <- "8g"
config$spark.master <- "local"
config$`sparklyr.cores.local` <- 2
config$`sparklyr.shell.driver-memory` <- "8G"
config$spark.memory.fraction <- 0.9

#config$spark.executor.memory <- "8g"
#config$spark.executor.cores <- "2"

#storage <- Sys.getenv("STORAGE")
#config$spark.yarn.access.hadoopFileSystems <- storage

# yarn-client",
sc <- spark_connect(master="local", config = config) #, spark_home = "/etc/spark")

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


### Convert value to Binary Label
# As there is a range of values, the next step take the mean value as the center point for Positive vs 
# Negative sentiment and adds the `sent_score` column which will be used as the dependent variable
# to train the model.

weighted_sum_summary <- sentence_values %>% sdf_describe(cols="weighted_sum")

weighted_sum_mean <- as.data.frame(weighted_sum_summary)$weighted_sum[2]

sentence_scores <- sentence_values %>% 
  mutate(sent_binary = ifelse(weighted_sum > weighted_sum_mean,"positive","negative")) %>%
  mutate(sent_multi = ifelse(weighted_sum > 2,"positive",ifelse(weighted_sum < -2,"negative","neutral")))

spark_write_table(simpsons_spark_table,"simpsons_spark_table",mode = "overwrite")
spark_write_table(afinn_table,"afinn_table",mode = "overwrite")
