#### [0] Libraries Loading ####

install.packages("tidyverse")
install.packages("cluster")
install.packages("factoextra")
library(tidyverse)  # data manipulation
library(cluster)    # clustering algorithms
library(factoextra) # clustering algorithms & visualization

data<- readRDS("/Users/mohamedkhanafer/Desktop/Currently/project R working on/solar_dataset.RData")
View(data)

#for values without NAs:
data<-data[c(1:5113),]


#### [1] Clustering the stations based on their outputs ####


#### [1.1] Loading the data ####
df<-data[,1:99]
# first remember the names
n <- df$Date
# transpose all but the first column (name)
df<- as.data.frame(t(df[,-1]))
colnames(df) <- n
View(df) # Check the column types

#### [1.2] Trying the model ####
k2 <- kmeans(df, centers = 2, nstart = 25)
str(k2)
fviz_cluster(k2, data = df)

#vizualising different numbers of clusters:
k3 <- kmeans(df, centers = 3, nstart = 25)
k4 <- kmeans(df, centers = 4, nstart = 25)
k5 <- kmeans(df, centers = 5, nstart = 25)

# plots to compare
p1 <- fviz_cluster(k2, geom = "point", data = df) + ggtitle("k = 2")
p2 <- fviz_cluster(k3, geom = "point",  data = df) + ggtitle("k = 3")
p3 <- fviz_cluster(k4, geom = "point",  data = df) + ggtitle("k = 4")
p4 <- fviz_cluster(k5, geom = "point",  data = df) + ggtitle("k = 5")

#install.packages("gridExtra")
library(gridExtra)
grid.arrange(p1, p2, p3, p4, nrow = 2)


# Conclusion from the plots: either k=2 or k=3 or k=4 looks like a good option, 
                            #thus we use the following methods to Determine the Optimal Clusters:


#### [1.3] Determining Optimal Clusters  ####

#### [1.3.1] Elbow Method ####
set.seed(123)

# function to compute total within-cluster sum of square 
wss <- function(k) {
  kmeans(df, k, nstart = 10 )$tot.withinss
}

# Compute and plot wss for k = 1 to k = 15
k.values <- 1:15

# extract wss for 2-15 clusters
wss_values <- map_dbl(k.values, wss)

plot(k.values, wss_values,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")

# doing what we did in a single function:
set.seed(123)
fviz_nbclust(df, kmeans, method = "wss")

#Conclusion: the graph suggests that 4 is the optimal number of clusters because it is the bend


#### [1.3.2] Silhouette Method ####
# function to compute average silhouette for k clusters
avg_sil <- function(k) {
  km.res <- kmeans(df, centers = k, nstart = 25)
  ss <- silhouette(km.res$cluster, dist(df))
  mean(ss[, 3])
}

# Compute and plot wss for k = 2 to k = 15
k.values <- 2:15

# extract avg silhouette for 2-15 clusters
avg_sil_values <- map_dbl(k.values, avg_sil)

plot(k.values, avg_sil_values,
     type = "b", pch = 19, frame = FALSE, 
     xlab = "Number of clusters K",
     ylab = "Average Silhouettes")

#same, here is a single function:
fviz_nbclust(df, kmeans, method = "silhouette")

#Conclusion: 4 is the optimum number of clusters according to silhouette



#### [2] Clustering the stations based on their locations and altitude ####
#### [2.1] Loading the data ####
df<-fread("/Users/mohamedkhanafer/Desktop/Currently/project R working on/station_info.csv")
View(df)
df<-as.data.frame(df)
n <- df$stid
row.names(df)<-n
df<-df[,c(2:4)]

#### [2.2] Trying the model ####
k2 <- kmeans(df, centers = 2, nstart = 25)
str(k2)
fviz_cluster(k2, data = df)

#vizualising different numbers of clusters:
k3 <- kmeans(df, centers = 3, nstart = 25)
k4 <- kmeans(df, centers = 4, nstart = 25)
k5 <- kmeans(df, centers = 5, nstart = 25)

# plots to compare
p1 <- fviz_cluster(k2, geom = "point", data = df) + ggtitle("k = 2")
p2 <- fviz_cluster(k3, geom = "point",  data = df) + ggtitle("k = 3")
p3 <- fviz_cluster(k4, geom = "point",  data = df) + ggtitle("k = 4")
p4 <- fviz_cluster(k5, geom = "point",  data = df) + ggtitle("k = 5")

#install.packages("gridExtra")
library(gridExtra)
grid.arrange(p1, p2, p3, p4, nrow = 2)


# Conclusion from the plots: the first observation is that compared to the 1st clusters where we run 
#using only the output as variable for clustering, here the clusters are less defined and separated.
#Here, either k=2 or k=4 looks like a good option, 
#thus we use the following methods to Determine the Optimal Clusters:


#### [2.3] Determining Optimal Clusters ####

#### [2.3.1] Elbow Method ####
# doing what we did in a single function:
set.seed(123)
fviz_nbclust(df, kmeans, method = "wss")

#Conclusion: the graph suggests that 2 is the optimal number of clusters because it is the bend


#### [2.3.2] Silhouette Method ####
#same, here is a single function:
fviz_nbclust(df, kmeans, method = "silhouette")

#Conclusion: 2 is the optimum number of clusters according to silhouette when we have location and elevation as variables

#### [3] Conclusion of clustering ####

#Given that the clusters seems more defined running the clustering using only as variable the 
#output produces per station, we chose to cluster stations according to outputs.

# The result is: K-means clustering with 4 clusters of sizes 27, 29, 18, 24






#### [4] Stations per Cluster ####
df<-data[,1:99]
n <- df$Date
df<- as.data.frame(t(df[,-1]))
colnames(df) <- n
k4 <- kmeans(df, centers = 4, nstart = 25)
fviz_cluster(k4, data = df)
k4
str(k4)
#getting the stations with their respective clusters:
clusters<-as.data.frame(k4$cluster)
stations<-row.names(clusters)
named_col<-cbind(stations,clusters)
cluster1<-named_col[named_col$`k4$cluster`==1,]
cluster1
cluster2<-named_col[named_col$`k4$cluster`==2,]
cluster3<-named_col[named_col$`k4$cluster`==3,]
cluster4<-named_col[named_col$`k4$cluster`==4,]

#getting the clusters' centers so we could use them in the model later on:
centers<-k4$centers
centers_col<-t(centers)
centers_cluster1<-centers_col[,1]
centers_cluster2<-centers_col[,2]
centers_cluster3<-centers_col[,3]
centers_cluster4<-centers_col[,4]

# Testing to see if the centers of cluster 1 are not too far from the actual station output
testing<-data[,"BIXB"]-centers_col[,1]


#### [5] Building model and Optimization of Hyper parameters ####

#### [5.1] Model for the centroid of cluster 1 and all PCAs ####

#### [5.1.1] Setting the data to be used ####
data<- readRDS("/Users/mohamedkhanafer/Desktop/Currently/project R working on/solar_dataset.RData")
data_centro1<- cbind(centers_cluster1,data[1:5113,100:456])
full_data<-readRDS("/Users/mohamedkhanafer/Desktop/Currently/project R working on/solar_dataset.RData")
View(data_centro1)

# Dividing  Test Datasets

test <- as.data.table(full_data[5114:nrow(full_data), ])
test<-test[,100:456]

not_test <- as.data.table(data_centro1)

# Dividing Training and Validation Datasets
set.seed(100)
train_index <- sample(1:nrow(not_test), 0.7*nrow(not_test))
train <- not_test[train_index, ]
validation <- not_test[-train_index, ]

##### [5.1.2] Model For Cluster 1 using centroids ####

## Building The Model

model1 <- svm(centers_cluster1 ~., data = train)

## Getting Predictions

predictions_train1 <- predict(model1, newdata = train)
predictions_val1 <- predict(model1, newdata = validation)

## Calculating Errors

errors_train1 <- predictions_train1 - train$centers_cluster1
errors_val1 <- predictions_val1 - validation$centers_cluster1

## Computing Metrics

mse_train1 <- round(mean((errors_train1)^2), 2);
mae_train1 <- round(mean(abs(errors_train1)), 2);

mse_val1 <- round(mean(errors_val1^2), 2);
mae_val1 <- round(mean(abs(errors_val1)), 2);

## Comparision Table

comp <- data.table(model = c("standard svm"), 
                   mse_train = mse_train1, mae_train = mae_train1,
                   mse_val = mse_val1, mae_val = mae_val1);

comp

## Hyperparameter Optimisation - Grid

c_values <- 10^seq(from = -3, to = 3, by = 1);
eps_values <- 10^seq(from = -3, to = 0, by = 1);
gamma_values <- 10^seq(from = -3, to = 3, by = 1);


## Hyperparameter Optimisation - Building And Evaluating Models

grid_results <- data.table();

for (c in c_values){
  for (eps in eps_values){
    for (gamma in gamma_values){
      
      print(sprintf("Start of c = %s - eps = %s - gamma = %s", c, eps, gamma));
      
      # train SVM model with a particular set of hyperparamets
      model <- svm(centers_cluster1 ~ ., data = train,
                   cost = c, epsilon = eps, gamma = gamma);
      
      # Get model predictions
      predictions_train <- predict(model, newdata = train);
      predictions_val <- predict(model, newdata = validation);
      
      # Get errors
      errors_train <- predictions_train - train$centers_cluster1;
      errors_val <- predictions_val - validation$centers_cluster1;
      
      # Compute Metrics
      mse_train <- round(mean(errors_train^2), 2);
      mae_train <- round(mean(abs(errors_train)), 2);
      
      mse_val <- round(mean(errors_val^2), 2);
      mae_val <- round(mean(abs(errors_val)), 2);
      
      # Get Comparision Results
      grid_results <- rbind(grid_results,
                            data.table(c = c, eps = eps, gamma = gamma, 
                                       mse_train = mse_train, mae_train = mae_train,
                                       mse_val = mse_val, mae_val = mae_val));
    }
  }
}


grid_results <- grid_results[order(mse_val, mae_val)]
View(grid_results)
best <- grid_results[1]

## Train  model using best hyperparameters on train and validation data for one station in the cluster:

## Data for station BIXB in cluster 1:

# Dividing  Test Datasets
data<-full_data[,-1]
test <- data[5114:nrow(data), ]
not_test <- data[1:5113, ]

# Dividing Training and Validation Datasets
set.seed=100
train_index <- sample(1:nrow(not_test), 0.7*nrow(not_test))
train <- not_test[train_index, ]
validation <- not_test[-train_index, ]

#### 1.Model For Station BIXB based on cluster centroids ####

## Removing Other Stations
grep("BIXB", colnames(train)) #this gives us the index of BIXB that we use in the model
train1 <- train[, c(8, 99:455)]
validation1 <- validation[, c(8, 99:455)]
test1 <- test[, 99:455]

# train SVM model with best found set of hyperparamets for cluster 1:
model <- svm(BIXB ~ ., data = rbind(train1,validation1), 
             cost = best$c, epsilon = best$eps, gamma = best$gamma);

# Get model predictions
predictions_train <- predict(model, newdata = train1);
predictions_val <- predict(model, newdata = validation1);
predictions_test <- predict(model, newdata = test1);

# Get errors
errors_train <- predictions_train - train1$BIXB;
errors_val <- predictions_val - validation1$BIXB;

# Compute Metrics
mse_train <- round(mean(errors_train^2), 2);
mae_train <- round(mean(abs(errors_train)), 2);

mse_val <- round(mean(errors_val^2), 2);
mae_val <- round(mean(abs(errors_val)), 2);

mae_train;
mae_val

# We keep these values to compare them to the next station where we run the same analysis but now using
# hyper parameters found from using the means of the stations in the cluster and not the centroids.



##### [5.2] Model For Cluster 1 using means of clusters ####

c1<-as.vector(cluster1[,1])
data_only_cluster1<-data[,c("ARNE", "BEAV", "BOIS", "BUFF", "CAMA", "CHER", "CHEY", 
                            "FAIR", "FREE", "GOOD", "HOOK", "KENT", "LAHO", "MAYR","PUTN", "SEIL", "SLAP", "WOOD")]

# row averages for the stations in cluster 1
mu <- rowMeans(data_only_cluster1)
View(mu)


#### [5.2.1] Setting the data to be used ####
data<- readRDS("/Users/mohamedkhanafer/Desktop/Currently/project R working on/solar_dataset.RData")
data_means1<- cbind(mu,data[,100:456])

# Dividing  Test Datasets

test2 <- as.data.table(full_data[5114:nrow(full_data), ]) #still to run
test2 <- test2[100:456]

not_test2 <- as.data.table(data_means1)
not_test2<-not_test2[1:5113,]

View(not_test2)

# Dividing Training and Validation Datasets
set.seed(100)
train_index2 <- sample(1:nrow(not_test2), 0.7*nrow(not_test2))
train2 <- not_test2[train_index2, ]
validation2 <- not_test2[-train_index2, ]

##### [5.2.2] Model For Cluster 1 using means of clusters ####

model2 <- svm(mu ~., data = train2)

## Getting Predictions

predictions_train2 <- predict(model2, newdata = train2)
predictions_val2 <- predict(model2, newdata = validation2)

## Calculating Errors

errors_train2 <- predictions_train2 - train2$mu
errors_val2 <- predictions_val2 - validation2$mu

## Computing Metrics

mse_train2 <- round(mean((errors_train2)^2), 2);
mae_train2 <- round(mean(abs(errors_train2)), 2);

mse_val2 <- round(mean(errors_val2^2), 2);
mae_val2 <- round(mean(abs(errors_val2)), 2);

## Comparision Table

comp2 <- data.table(model2 = c("standard svm2"), 
                   mse_train = mse_train2, mae_train = mae_train2,
                   mse_val = mse_val2, mae_val = mae_val2);

comp2

## Hyperparameter Optimisation - Grid

c_values <- 10^seq(from = -3, to = 3, by = 1);
eps_values <- 10^seq(from = -3, to = 0, by = 1);
gamma_values <- 10^seq(from = -3, to = 3, by = 1);


## Hyperparameter Optimisation - Building And Evaluating Models

grid_results <- data.table();

for (c in c_values){
  for (eps in eps_values){
    for (gamma in gamma_values){
      
      print(sprintf("Start of c = %s - eps = %s - gamma = %s", c, eps, gamma));
      
      # train SVM model with a particular set of hyperparamets
      model2 <- svm(mu ~ ., data = train2,
                   cost = c, epsilon = eps, gamma = gamma);
      
      # Get model predictions
      predictions_train2 <- predict(model2, newdata = train2);
      predictions_val2 <- predict(model2, newdata = validation2);
      
      # Get errors
      errors_train2 <- predictions_train2 - train2$mu;
      errors_val2 <- predictions_val2 - validation2$mu;
      
      # Compute Metrics
      mse_train2 <- round(mean(errors_train2^2), 2);
      mae_train2 <- round(mean(abs(errors_train2)), 2);
      
      mse_val2 <- round(mean(errors_val2^2), 2);
      mae_val2 <- round(mean(abs(errors_val2)), 2);
      
      # Get Comparision Results
      grid_results <- rbind(grid_results,
                            data.table(c = c, eps = eps, gamma = gamma, 
                                       mse_train = mse_train2, mae_train = mae_train2,
                                       mse_val = mse_val2, mae_val = mae_val2));
    }
  }
}


grid_results <- grid_results[order(mse_val, mae_val)]
View(grid_results)
best <- grid_results[1]


## Train  model using best hyperparameters on train and validation data for one station in the cluster:

## Data for station BIXB in cluster 1:

# Dividing  Test Datasets
data<-full_data[,-1]
test <- data[5114:nrow(data), ]
not_test <- data[1:5113, ]

# Dividing Training and Validation Datasets
set.seed=100
train_index <- sample(1:nrow(not_test), 0.7*nrow(not_test))
train <- not_test[train_index, ]
validation <- not_test[-train_index, ]

##### 2. Model For Station BIXB based on cluster means ####

## Removing Other Stations
grep("BIXB", colnames(train)) #this gives us the index of BIXB that we use in the model
train1 <- train[, c(8, 99:455)]
validation1 <- validation[, c(8, 99:455)]
test1 <- test[, 99:455]

# train SVM model with best found set of hyperparamets for cluster 1:
model <- svm(BIXB ~ ., data = rbind(train1,validation1), 
             cost = best$c, epsilon = best$eps, gamma = best$gamma);

# Get model predictions
predictions_train <- predict(model, newdata = train1);
predictions_val <- predict(model, newdata = validation1);
predictions_test <- predict(model, newdata = test1);

# Get errors
errors_train <- predictions_train - train1$BIXB;
errors_val <- predictions_val - validation1$BIXB;

# Compute Metrics
mse_train <- round(mean(errors_train^2), 2);
mae_train <- round(mean(abs(errors_train)), 2);

mse_val <- round(mean(errors_val^2), 2);
mae_val <- round(mean(abs(errors_val)), 2);

mae_train;
mae_val


#### [5.3] Conclusion 1 on the approach to choose for hyper parameter optimization ####

# Given that the optimization of HyperParameters using the means gave us better results than using 
# the centroids of clusters, we will choose to follow this approach to get the HyperParameters of the 
# other clusters. But before, we run a last trial on a model using normalized PCAs to see if it will lead 
# us to better results than without normalizing PCAs.




##### [5.4] Model For Cluster 1 using means of clusters and Standardized PCAs ####

c1<-as.vector(cluster1[,1])
data_only_cluster1<-data[,c("ARNE", "BEAV", "BOIS", "BUFF", "CAMA", "CHER", "CHEY", 
                            "FAIR", "FREE", "GOOD", "HOOK", "KENT", "LAHO", "MAYR","PUTN", "SEIL", "SLAP", "WOOD")]

# row averages for the stations in cluster 1
mu <- rowMeans(data_only_cluster1)
View(mu)

#### [5.4.1] Setting the data to be used ####
data<- readRDS("/Users/mohamedkhanafer/Desktop/Currently/project R working on/solar_dataset.RData")
scaled_PCAs<-scale(data[,100:456])
data_means1<- cbind(mu,scaled_PCAs)

# Dividing  Test Datasets
test3 <- as.data.table(full_data[5114:nrow(full_data), ]) 
test3 <- test3[100:456]

not_test3 <- as.data.table(data_means1)
not_test3<-not_test3[1:5113,]

View(not_test3)

# Dividing Training and Validation Datasets
set.seed(100)
train_index3 <- sample(1:nrow(not_test3), 0.7*nrow(not_test3))
train3 <- not_test3[train_index3, ]
validation3 <- not_test3[-train_index3, ]

##### [5.4.2] Model For Cluster 1 using means of clusters and standardized PCAs ####

## Hyperparameter Optimisation - Grid

c_values <- 10^seq(from = -3, to = 3, by = 1);
eps_values <- 10^seq(from = -3, to = 0, by = 1);
gamma_values <- 10^seq(from = -3, to = 3, by = 1);


## Hyperparameter Optimisation - Building And Evaluating Models

grid_results <- data.table();

for (c in c_values){
  for (eps in eps_values){
    for (gamma in gamma_values){
      
      print(sprintf("Start of c = %s - eps = %s - gamma = %s", c, eps, gamma));
      
      # train SVM model with a particular set of hyperparamets
      model3 <- svm(mu ~ ., data = train3,
                    cost = c, epsilon = eps, gamma = gamma);
      
      # Get model predictions
      predictions_train3 <- predict(model3, newdata = train3);
      predictions_val3 <- predict(model3, newdata = validation3);
      
      # Get errors
      errors_train3 <- predictions_train3 - train3$mu;
      errors_val3 <- predictions_val3 - validation3$mu;
      
      # Compute Metrics
      mse_train3 <- round(mean(errors_train3^2), 2);
      mae_train3 <- round(mean(abs(errors_train3)), 2);
      
      mse_val3 <- round(mean(errors_val3^2), 2);
      mae_val3 <- round(mean(abs(errors_val3)), 2);
      
      # Get Comparision Results
      grid_results <- rbind(grid_results,
                            data.table(c = c, eps = eps, gamma = gamma, 
                                       mse_train = mse_train3, mae_train = mae_train3,
                                       mse_val = mse_val3, mae_val = mae_val3));
    }
  }
}


grid_results <- grid_results[order(mse_val3, mae_val3)]
View(grid_results)
best <- grid_results[1]


## Train  model using best hyperparameters on train and validation data for one station in the cluster:

## Data for station BIXB in cluster 1:

# Dividing  Test Datasets
data<-full_data[,-1]
scaled_PCAs<-scale(data[,100:455])
data<- cbind(data[,1:98],scaled_PCAs)

test <- data[5114:nrow(data), ]
not_test <- data[1:5113, ]


# Dividing Training and Validation Datasets
set.seed=100
train_index <- sample(1:nrow(not_test), 0.7*nrow(not_test))
train <- not_test[train_index, ]
validation <- not_test[-train_index, ]

##### 1. Model For Station BIXB based on cluster means and Standardized PCAs ####

## Removing Other Stations
grep("BIXB", colnames(train)) #this gives us the index of BIXB that we use in the model
train1 <- train[, c(8, 99:454)]
validation1 <- validation[, c(8, 99:454)]
test1 <- test[, 99:454]

# train SVM model with best found set of hyperparamets for cluster 1:
model <- svm(BIXB ~ ., data = rbind(train1,validation1), 
             cost = best$c, epsilon = best$eps, gamma = best$gamma);

# Get model predictions
predictions_train <- predict(model, newdata = train1);
predictions_val <- predict(model, newdata = validation1);
predictions_test <- predict(model, newdata = test1);

# Get errors
errors_train <- predictions_train - train1$BIXB;
errors_val <- predictions_val - validation1$BIXB;

# Compute Metrics
mse_train <- round(mean(errors_train^2), 2);
mae_train <- round(mean(abs(errors_train)), 2);

mse_val <- round(mean(errors_val^2), 2);
mae_val <- round(mean(abs(errors_val)), 2);

mae_train;
mae_val


#### [5.5] Final Conclusion on approach chosen to run the models ####

# The results given by this last model did not seem convincing as when compared to the ones we got 
# for the previous 2 models. We thus chose to choose the means of the clusters as an approach to set
# the hyper parameters for each model. This is what we turn next to. We will run the optimized hyper 
# parameters for each cluster and then we will use for loops to get the predictions per station.






data<- readRDS("/Users/mohamedkhanafer/Desktop/Currently/project R working on/solar_dataset.RData")
data_centro1<- cbind(centers_cluster1,data[,100:456])
full_data<-readRDS("/Users/mohamedkhanafer/Desktop/Currently/project R working on/solar_dataset.RData")
View(data_centro1)

# Dividing  Test Datasets

test <- as.data.table(full_data[5114:nrow(full_data), ])
test<-test[,100:456]

not_test <- as.data.table(data_centro1)

# Dividing Training and Validation Datasets
set.seed(100)
train_index <- sample(1:nrow(not_test), 0.7*nrow(not_test))
train <- not_test[train_index, ]
validation <- not_test[-train_index, ]




##### [6] Re-running the clusters ####
data<- readRDS("/Users/mohamedkhanafer/Desktop/Currently/project R working on/solar_dataset.RData")
View(data)

#for values without NAs:
data<-data[c(1:5113),]

##Stations per Cluster
df<-data[,1:99]
n <- df$Date
df<- as.data.frame(t(df[,-1]))
colnames(df) <- n
k4 <- kmeans(df, centers = 4, nstart = 25)
fviz_cluster(k4, data = df)
k4
str(k4)
#getting the stations with their respective clusters:
clusters<-as.data.frame(k4$cluster)
stations<-row.names(clusters)
named_col<-cbind(stations,clusters)
cluster1<-named_col[named_col$`k4$cluster`==1,]
cluster2<-named_col[named_col$`k4$cluster`==2,]
cluster3<-named_col[named_col$`k4$cluster`==3,]
cluster4<-named_col[named_col$`k4$cluster`==4,]



##### [7] HYPER PARAMETER OPTIMIZATION FOR CLUSTER 1 ####

c1<-as.vector(cluster1[,1])
data_only_cluster1<-data[,c("ACME", "ALTU", "APAC", "BESS", "BUTL", "CHIC", "ELRE", "ERIC", "FTCB", "HINT", "HOBA", "HOLL", "KETC", "MANG",
                            "MEDI", "MINC", "NINN", "RETR", "RING", "TIPT", "WASH", "WATO", "WAUR", "WEAT")]

# row averages for the stations in cluster 1
mu1 <- rowMeans(data_only_cluster1)
View(mu1)


####Setting the data to be used ####
data<- readRDS("/Users/mohamedkhanafer/Desktop/Currently/project R working on/solar_dataset.RData")
full_data<-readRDS("/Users/mohamedkhanafer/Desktop/Currently/project R working on/solar_dataset.RData")
data_means1<- cbind(mu1,data[,100:456])
View(data_means1)

# Dividing  Test Datasets
test <- as.data.table(full_data[5114:nrow(full_data), ])
test <- test[,100:456]
View(test)

not_test<- as.data.table(data_means1)
not_test<-not_test[1:5113,]
View(not_test)

# Dividing Training and Validation Datasets
set.seed(100)
train_index <- sample(1:nrow(not_test), 0.7*nrow(not_test))
train <- not_test[train_index, ]
validation <- not_test[c(-train_index), ]

View(train)
View(validation)

## Building The Model

model <- svm(mu1 ~., data = train)

## Getting Predictions

predictions_train <- predict(model, newdata = train)
predictions_val <- predict(model, newdata = validation)

## Calculating Errors

errors_train <- predictions_train - train$mu1
errors_val <- predictions_val - validation$mu1

## Computing Metrics

mse_train <- round(mean((errors_train)^2), 2);
mae_train <- round(mean(abs(errors_train)), 2);

mse_val <- round(mean(errors_val^2), 2);
mae_val <- round(mean(abs(errors_val)), 2);

## Comparision Table

comp <- data.table(model = c("standard svm"), 
                   mse_train = mse_train, mae_train = mae_train,
                   mse_val = mse_val, mae_val = mae_val);

comp

## Hyperparameter Optimisation - Grid

c_values <- 10^seq(from = -3, to = 3, by = 1);
eps_values <- 10^seq(from = -3, to = 0, by = 1);
gamma_values <- 10^seq(from = -3, to = 3, by = 1);


## Hyperparameter Optimisation - Building And Evaluating Models

grid_results <- data.table();

for (c in c_values){
  for (eps in eps_values){
    for (gamma in gamma_values){
      
      print(sprintf("Start of c = %s - eps = %s - gamma = %s", c, eps, gamma));
      
      # train SVM model with a particular set of hyperparamets
      model <- svm(mu1 ~ ., data = train,
                   cost = c, epsilon = eps, gamma = gamma);
      
      # Get model predictions
      predictions_train <- predict(model, newdata = train);
      predictions_val <- predict(model, newdata = validation);
      
      # Get errors
      errors_train <- predictions_train - train$mu1;
      errors_val <- predictions_val - validation$mu1;
      
      # Compute Metrics
      mse_train <- round(mean(errors_train^2), 2);
      mae_train <- round(mean(abs(errors_train)), 2);
      
      mse_val <- round(mean(errors_val^2), 2);
      mae_val <- round(mean(abs(errors_val)), 2);
      
      # Get Comparision Results
      grid_results <- rbind(grid_results,
                            data.table(c = c, eps = eps, gamma = gamma, 
                                       mse_train = mse_train, mae_train = mae_train,
                                       mse_val = mse_val, mae_val = mae_val));
    }
  }
}


grid_results <- grid_results[order(mse_val, mae_val)]
View(grid_results)
best <- grid_results[1]





##### [8] HYPER PARAMETER OPTIMIZATION FOR CLUSTER 2 ####

c2<-as.vector(cluster2[,1])
data_only_cluster2<-data[,c("ADAX", "BOWL", "BURN", "BYAR", "CENT", "CLAY", "CLOU", "COOK", "DURA", "EUFA", "HUGO", "IDAB", "LANE", "MADI",
                            "MCAL", "MTHE", "OKEM", "OKMU", "PAUL", "SALL", "STIG", "STUA", "SULP", "TAHL", "TALI", "TISH", "WEST", "WILB","WIST")]

# row averages for the stations in cluster 1
mu2 <- rowMeans(data_only_cluster2)
View(mu2)


####Setting the data to be used ####
data<- readRDS("/Users/mohamedkhanafer/Desktop/Currently/project R working on/solar_dataset.RData")
full_data<-readRDS("/Users/mohamedkhanafer/Desktop/Currently/project R working on/solar_dataset.RData")
data_means2<- cbind(mu2,data[,100:456])
View(data_means2)

# Dividing  Test Datasets
test <- as.data.table(full_data[5114:nrow(full_data), ])
test <- test[,100:456]
View(test)

not_test<- as.data.table(data_means2)
not_test<-not_test[1:5113,]
View(not_test)

# Dividing Training and Validation Datasets
set.seed(100)
train_index <- sample(1:nrow(not_test), 0.7*nrow(not_test))
train <- not_test[train_index, ]
validation <- not_test[c(-train_index), ]

View(train)
View(validation)

## Building The Model

model <- svm(mu2 ~., data = train)

## Getting Predictions

predictions_train <- predict(model, newdata = train)
predictions_val <- predict(model, newdata = validation)

## Calculating Errors

errors_train <- predictions_train - train$mu2
errors_val <- predictions_val - validation$mu2

## Computing Metrics

mse_train <- round(mean((errors_train)^2), 2);
mae_train <- round(mean(abs(errors_train)), 2);

mse_val <- round(mean(errors_val^2), 2);
mae_val <- round(mean(abs(errors_val)), 2);

## Comparision Table

comp <- data.table(model = c("standard svm"), 
                   mse_train = mse_train, mae_train = mae_train,
                   mse_val = mse_val, mae_val = mae_val);

comp

## Hyperparameter Optimisation - Grid

c_values <- 10^seq(from = -3, to = 3, by = 1);
eps_values <- 10^seq(from = -3, to = 0, by = 1);
gamma_values <- 10^seq(from = -3, to = 3, by = 1);


## Hyperparameter Optimisation - Building And Evaluating Models

grid_results <- data.table();

for (c in c_values){
  for (eps in eps_values){
    for (gamma in gamma_values){
      
      print(sprintf("Start of c = %s - eps = %s - gamma = %s", c, eps, gamma));
      
      # train SVM model with a particular set of hyperparamets
      model <- svm(mu2 ~ ., data = train,
                   cost = c, epsilon = eps, gamma = gamma);
      
      # Get model predictions
      predictions_train <- predict(model, newdata = train);
      predictions_val <- predict(model, newdata = validation);
      
      # Get errors
      errors_train <- predictions_train - train$mu2;
      errors_val <- predictions_val - validation$mu2;
      
      # Compute Metrics
      mse_train <- round(mean(errors_train^2), 2);
      mae_train <- round(mean(abs(errors_train)), 2);
      
      mse_val <- round(mean(errors_val^2), 2);
      mae_val <- round(mean(abs(errors_val)), 2);
      
      # Get Comparision Results
      grid_results <- rbind(grid_results,
                            data.table(c = c, eps = eps, gamma = gamma, 
                                       mse_train = mse_train, mae_train = mae_train,
                                       mse_val = mse_val, mae_val = mae_val));
    }
  }
}


grid_results <- grid_results[order(mse_val, mae_val)]
View(grid_results)
best <- grid_results[1]





##### [9] HYPER PARAMETER OPTIMIZATION FOR CLUSTER 3 ####

c3<-as.vector(cluster3[,1])
data_only_cluster3<-data[,c("ARNE", "BEAV", "BOIS", "BUFF", "CAMA", "CHER", "CHEY", "FAIR", "FREE", "GOOD", "HOOK", "KENT", "LAHO", "MAYR",
                            "PUTN", "SEIL", "SLAP", "WOOD")]

# row averages for the stations in cluster 1
mu3 <- rowMeans(data_only_cluster3)
View(mu3)


####Setting the data to be used ####
data<- readRDS("/Users/mohamedkhanafer/Desktop/Currently/project R working on/solar_dataset.RData")
full_data<-readRDS("/Users/mohamedkhanafer/Desktop/Currently/project R working on/solar_dataset.RData")
data_means3<- cbind(mu3,data[,100:456])
View(data_means3)

# Dividing  Test Datasets
test <- as.data.table(full_data[5114:nrow(full_data), ])
test <- test[,100:456]
View(test)

not_test<- as.data.table(data_means3)
not_test<-not_test[1:5113,]
View(not_test)

# Dividing Training and Validation Datasets
set.seed(100)
train_index <- sample(1:nrow(not_test), 0.7*nrow(not_test))
train <- not_test[train_index, ]
validation <- not_test[c(-train_index), ]

View(train)
View(validation)

## Building The Model

model <- svm(mu3 ~., data = train)

## Getting Predictions

predictions_train <- predict(model, newdata = train)
predictions_val <- predict(model, newdata = validation)

## Calculating Errors

errors_train <- predictions_train - train$mu3
errors_val <- predictions_val - validation$mu3

## Computing Metrics

mse_train <- round(mean((errors_train)^2), 2);
mae_train <- round(mean(abs(errors_train)), 2);

mse_val <- round(mean(errors_val^2), 2);
mae_val <- round(mean(abs(errors_val)), 2);

## Comparision Table

comp <- data.table(model = c("standard svm"), 
                   mse_train = mse_train, mae_train = mae_train,
                   mse_val = mse_val, mae_val = mae_val);

comp

## Hyperparameter Optimisation - Grid

c_values <- 10^seq(from = -3, to = 3, by = 1);
eps_values <- 10^seq(from = -3, to = 0, by = 1);
gamma_values <- 10^seq(from = -3, to = 3, by = 1);


## Hyperparameter Optimisation - Building And Evaluating Models

grid_results <- data.table();

for (c in c_values){
  for (eps in eps_values){
    for (gamma in gamma_values){
      
      print(sprintf("Start of c = %s - eps = %s - gamma = %s", c, eps, gamma));
      
      # train SVM model with a particular set of hyperparamets
      model <- svm(mu3 ~ ., data = train,
                   cost = c, epsilon = eps, gamma = gamma);
      
      # Get model predictions
      predictions_train <- predict(model, newdata = train);
      predictions_val <- predict(model, newdata = validation);
      
      # Get errors
      errors_train <- predictions_train - train$mu3;
      errors_val <- predictions_val - validation$mu3;
      
      # Compute Metrics
      mse_train <- round(mean(errors_train^2), 2);
      mae_train <- round(mean(abs(errors_train)), 2);
      
      mse_val <- round(mean(errors_val^2), 2);
      mae_val <- round(mean(abs(errors_val)), 2);
      
      # Get Comparision Results
      grid_results <- rbind(grid_results,
                            data.table(c = c, eps = eps, gamma = gamma, 
                                       mse_train = mse_train, mae_train = mae_train,
                                       mse_val = mse_val, mae_val = mae_val));
    }
  }
}


grid_results <- grid_results[order(mse_val, mae_val)]
View(grid_results)
best <- grid_results[1]



##### [10] HYPER PARAMETER OPTIMIZATION FOR CLUSTER 4 ####

c4<-as.vector(cluster4[,1])
data_only_cluster4<-data[,c("BIXB", "BLAC", "BREC", "BRIS", "BURB", "CHAN", "COPA", "FORA", "GUTH", "HASK", "JAYX", "MARE", "MEDF", "MIAM",
                            "NEWK", "NOWA", "OILT", "PAWN", "PERK", "PRYO", "REDR", "SHAW", "SKIA", "SPEN", "STIL", "VINI", "WYNO")]

# row averages for the stations in cluster 1
mu4 <- rowMeans(data_only_cluster4)
View(mu4)


####Setting the data to be used ####
data<- readRDS("/Users/mohamedkhanafer/Desktop/Currently/project R working on/solar_dataset.RData")
full_data<-readRDS("/Users/mohamedkhanafer/Desktop/Currently/project R working on/solar_dataset.RData")
data_means4<- cbind(mu4,data[,100:456])
View(data_means4)

# Dividing  Test Datasets
test <- as.data.table(full_data[5114:nrow(full_data), ])
test <- test[,100:456]
View(test)

not_test<- as.data.table(data_means4)
not_test<-not_test[1:5113,]
View(not_test)

# Dividing Training and Validation Datasets
set.seed(100)
train_index <- sample(1:nrow(not_test), 0.7*nrow(not_test))
train <- not_test[train_index, ]
validation <- not_test[c(-train_index), ]

View(train)
View(validation)

## Building The Model

model <- svm(mu4 ~., data = train)

## Getting Predictions

predictions_train <- predict(model, newdata = train)
predictions_val <- predict(model, newdata = validation)

## Calculating Errors

errors_train <- predictions_train - train$mu4
errors_val <- predictions_val - validation$mu4

## Computing Metrics

mse_train <- round(mean((errors_train)^2), 2);
mae_train <- round(mean(abs(errors_train)), 2);

mse_val <- round(mean(errors_val^2), 2);
mae_val <- round(mean(abs(errors_val)), 2);

## Comparision Table

comp <- data.table(model = c("standard svm"), 
                   mse_train = mse_train, mae_train = mae_train,
                   mse_val = mse_val, mae_val = mae_val);

comp

## Hyperparameter Optimisation - Grid

c_values <- 10^seq(from = -3, to = 3, by = 1);
eps_values <- 10^seq(from = -3, to = 0, by = 1);
gamma_values <- 10^seq(from = -3, to = 3, by = 1);


## Hyperparameter Optimisation - Building And Evaluating Models

grid_results <- data.table();

for (c in c_values){
  for (eps in eps_values){
    for (gamma in gamma_values){
      
      print(sprintf("Start of c = %s - eps = %s - gamma = %s", c, eps, gamma));
      
      # train SVM model with a particular set of hyperparamets
      model <- svm(mu4 ~ ., data = train,
                   cost = c, epsilon = eps, gamma = gamma);
      
      # Get model predictions
      predictions_train <- predict(model, newdata = train);
      predictions_val <- predict(model, newdata = validation);
      
      # Get errors
      errors_train <- predictions_train - train$mu4;
      errors_val <- predictions_val - validation$mu4;
      
      # Compute Metrics
      mse_train <- round(mean(errors_train^2), 2);
      mae_train <- round(mean(abs(errors_train)), 2);
      
      mse_val <- round(mean(errors_val^2), 2);
      mae_val <- round(mean(abs(errors_val)), 2);
      
      # Get Comparision Results
      grid_results <- rbind(grid_results,
                            data.table(c = c, eps = eps, gamma = gamma, 
                                       mse_train = mse_train, mae_train = mae_train,
                                       mse_val = mse_val, mae_val = mae_val));
    }
  }
}


grid_results <- grid_results[order(mse_val, mae_val)]
View(grid_results)
best <- grid_results[1]





data_only_cluster1<-data[,c("ACME", "ALTU", "APAC", "BESS", "BUTL", "CHIC", "ELRE", "ERIC", "FTCB", "HINT", "HOBA", "HOLL", "KETC", "MANG",
                            "MEDI", "MINC", "NINN", "RETR", "RING", "TIPT", "WASH", "WATO", "WAUR", "WEAT")]



#### [11] Conclusion on running Hyper parameter optimization per cluster #### 

# For the 4 clusters, we get the same optimized hyper parameters, this seemed a bit weird to us.
# Before deciding to use those hyper parameters, we decided to do a last attempt to see if we could better cluster.
# We thus clustered based on the variances per station. Here is the code we used to do this:

#### [11.1] Last attempt at getting better clusters ####

data <- readRDS("/Users/mohamedkhanafer/Desktop/Currently/project R working on/solar_dataset.RData")
clustering_data <- data[1:5113, 2:99]
setDF(clustering_data)

means <- sapply(clustering_data, mean)

diff <- data.frame()

for (c in 1:ncol(clustering_data)){
  for(r in 1:nrow(clustering_data)){
    diff[r, c] <- (clustering_data[r, c] - means[c])^2
  }
}


View(diff)
diff<-as.data.frame(t(diff))

#vizualising different numbers of clusters:
k2 <- kmeans(diff, centers = 2, nstart = 25)
k3 <- kmeans(diff, centers = 3, nstart = 25)
k4 <- kmeans(diff, centers = 4, nstart = 25)
k5 <- kmeans(diff, centers = 5, nstart = 25)

# plots to compare
p1 <- fviz_cluster(k2, geom = "point", data = diff) + ggtitle("k = 2")
p2 <- fviz_cluster(k3, geom = "point",  data = diff) + ggtitle("k = 3")
p3 <- fviz_cluster(k4, geom = "point",  data = diff) + ggtitle("k = 4")
p4 <- fviz_cluster(k5, geom = "point",  data = diff) + ggtitle("k = 5")

#install.packages("gridExtra")
library(gridExtra)

grid.arrange(p1, p2, p3, p4, nrow = 2)

View(diff)
k3 <- kmeans(diff, centers = 3, nstart = 25)
fviz_cluster(k3, data = diff)
k3
str(k3)
#getting the stations with their respective clusters:
clusters<-as.data.frame(k3$cluster)
stations<-row.names(clusters)
named_col<-cbind(stations,clusters)
cluster1<-named_col[named_col$`k3$cluster`==1,]
cluster1
cluster2<-named_col[named_col$`k4$cluster`==2,]
cluster3<-named_col[named_col$`k4$cluster`==3,]

#getting the clusters' centers so we could use them in the model later on:
centers<-k3$centers
centers_col<-t(centers)
centers_cluster1<-centers_col[,1]
centers_cluster2<-centers_col[,2]
centers_cluster3<-centers_col[,3]

data<- readRDS("/Users/mohamedkhanafer/Desktop/Currently/project R working on/solar_dataset.RData")
data_centro1<- cbind(centers_cluster1,data[,100:456])
full_data<-readRDS("/Users/mohamedkhanafer/Desktop/Currently/project R working on/solar_dataset.RData")
View(data_centro1)

# Dividing  Test Datasets

test <- as.data.table(full_data[5114:nrow(full_data), ])
test<-test[,100:456]

not_test <- as.data.table(data_centro1)

# Dividing Training and Validation Datasets
set.seed(100)
train_index <- sample(1:nrow(not_test), 0.7*nrow(not_test))
train <- not_test[train_index, ]
validation <- not_test[-train_index, ]

## Building The Model

model1 <- svm(centers_cluster1 ~., data = train)

## Getting Predictions

predictions_train1 <- predict(model1, newdata = train)
predictions_val1 <- predict(model1, newdata = validation)

## Calculating Errors

errors_train1 <- predictions_train1 - train$centers_cluster1
errors_val1 <- predictions_val1 - validation$centers_cluster1

## Computing Metrics

mse_train1 <- round(mean((errors_train1)^2), 2);
mae_train1 <- round(mean(abs(errors_train1)), 2);

mse_val1 <- round(mean(errors_val1^2), 2);
mae_val1 <- round(mean(abs(errors_val1)), 2);

## Comparision Table

comp <- data.table(model = c("standard svm"), 
                   mse_train = mse_train1, mae_train = mae_train1,
                   mse_val = mse_val1, mae_val = mae_val1);

comp

## Hyperparameter Optimisation - Grid

c_values <- 10^seq(from = -3, to = 3, by = 2);
eps_values <- 10^seq(from = -3, to = 0, by = 2);
gamma_values <- 10^seq(from = -3, to = 3, by = 2);


## Hyperparameter Optimisation - Building And Evaluating Models

grid_results <- data.table();

for (c in c_values){
  for (eps in eps_values){
    for (gamma in gamma_values){
      
      print(sprintf("Start of c = %s - eps = %s - gamma = %s", c, eps, gamma));
      
      # train SVM model with a particular set of hyperparamets
      model <- svm(centers_cluster1 ~ ., data = train,
                   cost = c, epsilon = eps, gamma = gamma);
      
      # Get model predictions
      predictions_train <- predict(model, newdata = train);
      predictions_val <- predict(model, newdata = validation);
      
      # Get errors
      errors_train <- predictions_train - train$centers_cluster1;
      errors_val <- predictions_val - validation$centers_cluster1;
      
      # Compute Metrics
      mse_train <- round(mean(errors_train^2), 2);
      mae_train <- round(mean(abs(errors_train)), 2);
      
      mse_val <- round(mean(errors_val^2), 2);
      mae_val <- round(mean(abs(errors_val)), 2);
      
      # Get Comparision Results
      grid_results <- rbind(grid_results,
                            data.table(c = c, eps = eps, gamma = gamma, 
                                       mse_train = mse_train, mae_train = mae_train,
                                       mse_val = mse_val, mae_val = mae_val));
    }
  }
}


grid_results <- grid_results[order(mse_val, mae_val)]
View(grid_results)
best <- grid_results[1]


#### [11.2] Final Conclusion on Hyper Parameters ####

#As this last trial also gave unfavorable outcomes, we chose to use the optimized hyperparameters found while 
#clustering based on the means of the clusters and using k=4. 
# These hyper parameters were: cost = 10, eps = 0.001, gamma = 0.001


#### [12] Predictions per station ####


#For loops for predictions:
final_predictions <- data.table()

data <- readRDS("/Users/mohamedkhanafer/Desktop/Currently/project R working on/solar_dataset.RData")
data <- data[, 2:ncol(data)]
setDF(data)
class(data)

# Dividing  Test Datasets

test <- data[5114:nrow(data), ]
not_test <- data[1:5113, ]



final_predictions <- data.table()

for(i in 1:98) {
  print(sprintf("model%s", i))
  model <- svm(x = not_test[, 99:455], y = not_test[, i], cost = 10, epsilon = 0.001, gamma = 0.001)
  predictions_test <- predict(model, newdata = test[, 99:455])
  final_predictions <- data.table(final_predictions, predictions_test)
}


