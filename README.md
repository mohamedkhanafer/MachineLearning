# MachineLearning
In this repository, I share a bunch of projects I have completed individually and as part of a team in developping predictive models. These include personnal projects, online competitions, as well as projects done during my studies at IE University.

# 1. Pump it up Competition (DrivenData)
The goal here is to try helping the Tanzanian Ministry of Water to predict the operating condition of a waterpoint accross the country. We want to predict which pumps are going to continue working, which ones are going to need repairs and which ones are going to fail. We are thus dealing here with a multiclass classification problem for which we develop and Ensemble model. 

The link to the competition is the following: https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/. 

This work was done with the help of my colleagues Begoña Frigolet and Dwight Alexander. The model we developped ended up in the top 15% out of more than 8900 people listed on the Leaderboard with a score of 0.8129.

The script associated with this model is found in the file 1_Pump_it_up_competition. It contains the following:
#### 1. The datasets
#### 2. Exploratory Data Analysis
#### 3. First drop of features
#### 4. Feature Engineering and Modeling
#### 4.2 Running the First Model (Hyper parameter tunning)
#### 4.3 Feature Creation 
#### 4.4 Trying a lightGBM model (Hyper parameter tunning)
#### 5. Trying an ensemble: the VotingClassifier
#### 6. Conclusion about the main model
#### 7. Extra: A trial with H20's Random Forrest 

# 2. Clustering Footballers' playing styles (Using Dataiku)
The scope of our analysis is to aid modern football teams in the process of individuating, selecting and acquiring the best possible fit as an addition to their current line-up.
The rationale behind our idea comes from the fact that, in the last few years, European teams have seen a lot of new players acquisitions that turned out to be a failure because a given footballer’s abilities were not in line with what was expected of him.
As no such thing as an optimal cluster exists, our approach consisted of categorizing players based on a number of different attributes derived from their playing style. As we are moving towards data-backed decisions in most industries, we believe that the process of individuating the right members of a team will be ever more reliant on solutions such as ours. This is because once teams understand what is missing from their line-up through playing-data analytics, our cluster analysis will show them which players on the market have the right attributes.

The dataset used: we derived all the attributes from the players’ in-match performance came from the football game Football Manager and can be found here: https://www.kaggle.com/ajinkyablaze/football-manager-data.

This work was done with the help of my colleague Aayush Kejriwal. We used Dataiku to perform our modeling and elaborate on all the results in the document 2_Clustering_footballers_playing_styles. 
