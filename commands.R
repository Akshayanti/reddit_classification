library(tm)
library(foreign)
library(SnowballC)
library(wordcloud)
library(caret)

set.seed(125)


# Data Loading and Preprocessing ------------------------------------------

bigboob = VCorpus(DirSource("text_files/bigboobproblems/"), readerControl = list(language="en"))
bigdick = VCorpus(DirSource("text_files/bigdickproblems/"), readerControl = list(language="en"))
both = c(bigboob, bigdick)

both.trans = tm_map(both, removeNumbers)
both.trans = tm_map(both.trans, removePunctuation)
both.trans = tm_map(both.trans, content_transformer(tolower))
both.trans = tm_map(both.trans, removeWords, stopwords("english"))
both.trans = tm_map(both.trans, stripWhitespace)
both.trans = tm_map(both.trans, stemDocument)

both.dtm = DocumentTermMatrix(both.trans)
both.dtm.90 = removeSparseTerms(both.dtm, sparse = 0.90)

data = data.frame(as.matrix(both.dtm.90))
type = c(rep("boob",1000),rep("dick",1000))

working = cbind(both.dtm.90, type)
write.arff(cbind(data,type),file = "tdf-weka.arff")

# Some preliminary patterns -----------------------------------------------

wordFreqs_boobs = sort(colSums(as.matrix(both.dtm.90)[1:1000,]),decreasing = TRUE)
png("boob.png")
wordcloud(words = names(wordFreqs_boobs), freq = wordFreqs_boobs)
dev.off()

wordFreqs_dick = sort(colSums(as.matrix(both.dtm.90)[1001:2000,]),decreasing = TRUE)
png("dick.png")
wordcloud(words = names(wordFreqs_dick), freq = wordFreqs_dick)
dev.off()

distMatrix=dist(t(scale(as.matrix(both.dtm.90))))
termClustering=hclust(distMatrix,method="complete")
png("cluster.png", width=1600)
plot(termClustering)
dev.off()


# Building Models ---------------------------------------------------------

both.matrix = as.data.frame(as.matrix(working))
colnames(both.matrix)[109]="type"
validation_index <- createDataPartition(both.matrix$type, p=0.80, list=FALSE)
training <- both.matrix[validation_index,]
testing <- both.matrix[-validation_index,]


# Training Control Parameters ---------------------------------------------

lgocv <- trainControl(method = "LGOCV",
                     number = 5,
                     search = "grid",
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary,
                     verboseIter = TRUE)
repeatedcv <- trainControl(method = "repeatedcv",
                                number = 5,
                                repeats = 5,
                                classProbs = TRUE,
                                summaryFunction = twoClassSummary,
                                search = "grid",
                                verboseIter = TRUE)
lgocv_random <- trainControl(method = "LGOCV",
                      number = 5,
                      search = "random",
                      classProbs = TRUE,
                      summaryFunction = twoClassSummary,
                      verboseIter = TRUE)
repeatedcv_random <- trainControl(method = "repeatedcv",
                           number = 5,
                           repeats = 5,
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary,
                           search = "random",
                           verboseIter = TRUE)

# Model Definitions and Training -------------------------------------------------------

# Model1: Adaboost with grid search, and LGOCV
adaboost_lgocv <- train(type ~ ., data=training, 
                             method="adaboost", 
                             metric="ROC",
                             tuneLength = 5,
                             trControl=lgocv)
predictions_adaboost_lgocv <- predict(adaboost_lgocv, newdata=testing, type="raw")
confusionMatrix(predictions_adaboost_lgocv, testing$type)
saveRDS(adaboost_lgocv, "adaboost_lgocv.model")

# Model2: Adaboost with random search, and LGOCV
adaboost_lgocv_random <- train(type ~ ., data=training, 
                        method="adaboost", 
                        metric="ROC",
                        tuneLength = 5,
                        trControl=lgocv_random)
predictions_adaboost_lgocv_random <- predict(adaboost_lgocv_random, newdata=testing, type="raw")
confusionMatrix(predictions_adaboost_lgocv_random, testing$type)
saveRDS(adaboost_lgocv_random, "adaboost_lgocv_random.model")

# Model3 : RDA with random search, and RepeatedCV
rda_repeatedcv <- train(type ~ ., data = training, 
                 method = "rda",
                 metric = "ROC",
                 tuneLength = 5,
                 trControl = repeatedcv_random)
predictions_rda_repeatedcv <- predict(rda_repeatedcv, newdata=testing, type="raw")
confusionMatrix(predictions_rda_repeatedcv, testing$type)
saveRDS(rda_repeatedcv, "rda_repeatedcv.model")

# Model4 : RDA with random search, and LGOCV
rda_lgocv <- train(type ~ ., data = training, 
                        method = "rda",
                        metric = "ROC",
                        tuneLength = 5,
                        trControl = lgocv_random)
predictions_rda_lgocv <- predict(rda_lgocv, newdata=testing, type="raw")
confusionMatrix(predictions_rda_lgocv, testing$type)
saveRDS(rda_lgocv, "rda_lgocv.model")

# Performace Comparison --------------------------------------------------------------

scales <- list(x=list(relation="free"), y=list(relation="free"))

# Compare same method models, with different search strategy
LGOCV_both <- resamples(list(grid=adaboost_lgocv, random=adaboost_lgocv_random))
png("boxplot_lgocv.png")
bwplot(LGOCV_both, scales=scales)
dev.off()

# Compare all together
all_adaboost_compare <- resamples(list(
  adaboost_lr=adaboost_lgocv_random,
  adaboost_lg=adaboost_lgocv,
  rda_l=rda_lgocv,
  rda_r=rda_repeatedcv
))
png("all_models.png")
bwplot(all_compare, scales=scales)
dev.off()

aucRoc(adaboost_lgocv)
aucRoc(adaboost_lgocv_random)

# Deep Learning -----------------------------------------------------------

library(h2o)
h2o.init(ip="localhost", port=1234, nthreads=1, max_mem_size = "4g")
data.hex = h2o.uploadFile("tdf-weka.arff", destination_frame = "data.hex", parse_type = "ARFF")
hex.split = h2o.splitFrame(data=data.hex, ratios=0.80)
hex.train = hex.split[[1]]
hex.test = hex.split[[2]]

# Model Definition ----------------------------------------------------------

model <- h2o.deeplearning(x=1:nrow(hex.train), y=109, training_frame = hex.train, hidden=c(100, 100, 100, 100), shuffle_training_data = TRUE, stopping_metric = "AUC", loss = "CrossEntropy", epochs=10, train_samples_per_iteration = -2, fold_assignment = "AUTO", activation="TanhWithDropout")

# Performance Evaluation --------------------------------------------------

predictions_dl =h2o.predict(object = model, newdata = hex.test)
performance_dl = h2o.performance(model, hex.test)
performance_dl
h2o.shutdown()