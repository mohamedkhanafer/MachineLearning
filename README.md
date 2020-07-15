# MachineLearning
In this repository, I share a bunch of projects I have completed individually and as part of a team in developping predictive models. These include personnal projects, online competitions, as well as projects done during my studies at IE University.

## 1. Pump it up Competition (DrivenData)
The goal here is to try helping the Tanzanian Ministry of Water to predict the operating condition of a waterpoint accross the country. We want to predict which pumps are going to continue working, which ones are going to need repairs and which ones are going to fail. We are thus dealing here with a multiclass classification problem for which we develop and Ensemble model. 

The link to the competition is the following: https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/. 

This work was done with the help of my colleagues Begoña Frigolet and Dwight Alexander. The model we developped ended up in the top 15% out of more than 8900 people listed on the Leaderboard with a score of 0.8129.

The script associated with this model is found in the file 1_Pump_it_up_competition. It contains the following: a detailed Exploratory Data Analysis, Feature Engineering and Modeling, Feature Creation, a lightGBM, an ensemble VotingClassifier and a trial with H2O's Random Forrest.

## 2. Clustering Footballers' playing styles (Using Dataiku)
The scope of our analysis is to aid modern football teams in the process of individuating, selecting and acquiring the best possible fit as an addition to their current line-up.
The rationale behind our idea comes from the fact that, in the last few years, European teams have seen a lot of new players acquisitions that turned out to be a failure because a given footballer’s abilities were not in line with what was expected of him.
As no such thing as an optimal cluster exists, our approach consisted of categorizing players based on a number of different attributes derived from their playing style. As we are moving towards data-backed decisions in most industries, we believe that the process of individuating the right members of a team will be ever more reliant on solutions such as ours. This is because once teams understand what is missing from their line-up through playing-data analytics, our cluster analysis will show them which players on the market have the right attributes.

The dataset used: we derived all the attributes from the players’ in-match performance came from the football game Football Manager and can be found here: https://www.kaggle.com/ajinkyablaze/football-manager-data.

This work was done with the help of my colleague Aayush Kejriwal. We used Dataiku to perform our modeling and elaborate on all the results in the document 2_Clustering_footballers_playing_styles. 

## 3. The HR Dataset: Determining the probability of employees' attrition from the company
The goal here is to model the probability of attrition (employees leaving, either on their own or because they got fired) of each individual, as well as to understand which variables are the most important ones and need to be addressed right away.
The goal of this notebook is to highlight the importance of the feature engineering process.

To be able to build a predictive model on this data, I first start a thorough exploration of the data, I then create a baseline model that I improve on by applying Feature Engineering.

This is an individual work and the code can be found in the folder 3_HR_Dataset_Feature_Engineering and the dataset is from Kaggle and can be found here: https://www.kaggle.com/giripujar/hr-analytics.

## 4. Housing Regression Model on Scrapped Data from Idealista (Using Dataiku)
This model was built for two different business scenarios:
- To assist investors looking for underpriced houses;
- To use as a reference in the construcion of new buildings/houses given certain features.

We built the model based on data scrapped over 3 days from Idealista, the largest estate website in Spain. And our model is built on data from 12.000 listings accross the Madrid region. 

This work was done with the help of my colleague Aayush Kejriwal, and the technical report of our model can be found in the folder 4_Housing_Model_Regression. This model was built using Daitaiku

## 5. Model for predicting short-term Solar Energy Production
The goal of this project is to discover which machine learning methods would provide the best short term predictions of solar energy production. 
This project is based on a Kaggle Competition: https://www.kaggle.com/c/ams-2014-solar-energy-prediction-contest/overview.

We were given pre-processed data with PCA that we had to model. We tried various approaches before and the last model chosen is a SVM on the clustered stations' data with hyperparameter tunning. 

Our model's score would be ranked in the top 20% of the competition. This work was done with the help of my colleagues Aayush Kejriwal and Arda Pekkucukyan. It can be found in the file 5_Predicting_solar_energy_production.

