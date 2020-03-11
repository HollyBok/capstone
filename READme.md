# Exploration and Modeling of Active NBA Player Salaries
Author: Holly Bok


## Executive Summary

The goal of this project is to develop a machine learning model that can predict the salaries of active NBA players. The NBA has a massive range of salaries (from ~79.5k to ~40 million) and just as massive a range of player skills and records. In this project I will use a combination of unsupervised machine learning models (KMeans, DBSCAN, and PCA) and supervised machine learning models (LassoCV, RidgeCV, Random Forest Regression, and SVR) to model the salaries of NBA players as accurately as possible. I believe this will make an interesting target for modeling as the entire population of interest is very small (with only around 450-500 players listed as 'active' in the NBA at any given time) and the range of salaries and skills is so large. This tests the limits of machine learning modeling and will be a lot of fun to model and analyze.

Data is gathered using the NBA API and a web scrape of hoopshype.com. Statistics are collected for the career stats, common information, and salaries of each active NBA player. The data is explored and analyzed and several models that predict 19/20 season salaries will be fit and evaluated. The performance of the models will be evaluated using the models' r2 scores, which show the percent of variability in the data that is accounted for by each model. The best model will be chosen and predictions will be made. Finally, a discussion will be had about the strengths and weaknesses of the model, inferences that can be made from the model's predictions, and further research that should be done.

## Data Gathering

Data is gathered from two sources, the 'nba_api' package for the NBA.com API and a web scrape of www.hoopshype.com/salaries.

**nba_api**

Documentation and further information on the nbi_api package can be found at https://github.com/swar/nba_api. The package is maintained by the user Swar and functions as an easy way to communicate with the NBA.com API. A list of all players is gathered (including the Player Name, Player ID, and Active Status of each player) from the nba_api.stats.static module 'players', active players are selected, and a list of Active Player IDs is created.

Statistical and common information for each active player is gathered using two nba_api endpoints, 'playercareerstats' and 'commonplayerinfo'. Playercareerstats provides game statistics for each player, such as assists or minutes played. All features gathered represent *career* stats, meaning that the value of any feature such as points or rebounds will be the sum of points / rebounds that the player has made in all games he played. Commonplayerinfo provides non-game features for each player, such as height, weight, and country of origin.

A few errors occurred while collecting information from the NBA API. Five player IDs did not return any data for career stats. These players are either extremely new to the NBA or recently transitioned out of the NBA after only a few games. As it would be unrealistic to find all of the career statistics for unknown players  from reliable sources, these players are not included in further analysis. Additionally, the team abbreviations, heights, and weights of 25 players were not available although all of the other statistics for these players were returned. These features were manually gathered for each player using Wikipedia.

**Scrape of hoopyshype.com**

Salary information for all active players is gathered from 'https://hoopshype.com/salaries/players/' using a web scrape (using BeautifulSoup). The information taken from hoopshype includes current salary (19/20 season) and future salary (20/21 season) where applicable.

**Final Features**

After all of the raw data is gathered it is cleaned and reformatted. Details of specific reformatting can be found in the '01 - Data Gathering' file of this repository under the 'DataFrame Merging and Cleaning' subheader.

Additionally, new features are created using some of the original features. The 20/21 salary feature is replaced with a binary feature called 'future_salary' where the player receives a 0 if he does not have a 20/21 salary listed and a 1 if he does. Players without future salaries do not have settled contracts or will no longer be playing basketball. The 'draft round' feature is converted to dummies and 'draft number' is grouped into 10s and ranked from 1-8 where earlier draft picks (1-10) are ranked 1, draft picks 2-20 are ranked 2, and so on. New features are created for average points per game (total points / total games) and percent of games started (games started / total games).

The final dataset includes 467 active NBA players and is in this datasets folder of this repository as 'NBAPlayers.csv'

The final features are as follows:

|Feature|Type|Description|
|---|---|---|
|player_id|int| The official NBA Player ID Number for each baseketball player| 
|player_name|object| Full name of player as he appears on the roster| 
|team|object| The 3 letter abbreviation code for the player's current team|
|height|object| The height of the player in standard feet-inches format|
|weight|float| Most recently recoreded weight of player in lbs| 
|seasons|int| Total seasons that the player has been in the NBA|
|points|int| Total points scored in all games throughout career| 
|games|int| The number of total games the player has played in throughout career|
|games_started|int| The total number of games a player has played where they were on the starting lineup|
|minutes_played|float| The total number of minutes a player has played in all games throughout career|
|field_goals_made|int| Total field goals made throughout career. Field goals include to all baskets made (2 or 3 pointers) that are not free throws| 
|3_pntrs_made|int| Total 3-point field goals made throughout career|
|field_goals_pct_made|float| Percentage of successful field goal attempts (2 or 3 pointer)| 
|3_pntrs_pct_made|float| Percentage of successful 3 point field goal attempts|
|ft_pct|float| Percentage of free throws made|
|field_goals_assisted|int| Total field goals made throughought career where the player acted as an assist (where an assist is a maneuver that directly results in another player successfully scoring points)| 
|3_pntrs_assists|int| Total 3-point field goals made throughought career where the player acted as an assist|
|rebounds|int| Total rebounds made throughout career. Rebounds apply to maneuvers in which a player is able to gain control of the ball after the opposing team has made a basket |
|assists_to_turnovers|int| Total assists in which the assisted player makes a turnover. Turnover applies to situations in which a player is able to gain control of the ball from the opposing team before the opposing team has attempted to score points|
|offensive_rebounds|int| Total rebounds on the opposing teams court-side throughought career|
|defensive_rebounds|int| Total rebounds on the player's teams' court-side throughout career|
|steals|int| Total maneuvers in which the player is able to legally steal the ball from the opposing team throughout career|
|blocks|int| Total blocks throughout career where blocks applies to situations in which the player is able to directly stop a member of the opposing team from scoring points| 
|turnovers|int| Total maneuvers throughout career in which the player is able to recover the ball from the opposing team without stealing before the opposing team has attempted to score points|
|personal_fouls|int| Total fouls that have been assigned to the player thoughout career|
|start_year|int| The year the player joined the NBA| 
|19_20_salary|int| Current salary for the 2019/2020 season in USD|
|future salary|int| Whether or not the player has a settled salary for the next season (2020/2021). Players that do not have a settled salary have a value of 0 and players that have a settled salary have a value of 1|
|avg_pnts_per_game|float| Average points made per all games played. Calculated by dividing 'points' by 'games'|
|pct_of_games_started|float| Percentage of total games played that the player was on the starting lineup for. Calculated by dividing 'games_started' by 'games'|
|heights_inches|int| The most recent recorded height expressed in inches rather than the traditional feet-inches format|
|draft_1st_pick|int| Whether or not the player was drafted during the 1st draft round. Players who were drated in the first round receive a 1 and players who were not receive a 0|
|draft_2nd_pick|int| Whether or not the player was drafted during the 2nd draft round. Players who were drated in the second round receive a 1 and players who were not receive a 0|
|draft_undrafted|int| Whether or not the player was drafted. Players who were not drafted receive a 1 and players who were drafted receive a 0. Non-drafted players can enter the NBA through direct contracts with indivudual teams|
|draft_number_group|object| The 'grouping' of draft pick numbers where players who were picked 1st through 10th are assigned draft group 1-10, players who were picked 11th through 20th are assigned draft group 2-20, and so on|
|draft_nmbr_grp_rank|int| A ranking of draft number group where the 1-10 group is ranked at 1| 


## Initial EDA

Several visuals are created to aid in initial data analysis using matplotlib and seaborn.

A heatmap is made to visualize the correlative relationship between each numeric feature and current salary. More positive correlations exist than negative correlations and the positive correlations are generally stronger than the negative. The most negatively correlated feature is start year (-0.58) while there are many positively correlated features with correlation scores above 0.58 (from least correlated to most: seasons, 3 pointers, 3 pointer assists, rebounds, games, personal fouls, defensive rebounds, percent of games started, assists to turnovers, steals, minutes played, turnovers, average points per game, games started, field goals assisted, field goals made, and total points).

The distribution of salaries (fig.1) ranges from a minimum of $79,568 (44 different players) to $40,231,758 (Steph Curry) and is heavily left leaning with the vast majority of players' salaries at 5 million or below. Several distributions are created to show the range of salaries at different cutoffs (500k, 2 million, 10 million - fig2). Large spikes appear at a few major salary values including $79,568, $898,310, $1,416,852, $2,564,753, and $4,767,000. These correlate to contract minimums for players. Often contracts specify salary ranges rather than specific, precise values and the player's yearly salary depends on performance. Players that are considered to be meeting their contract but not exceeding expectations have salaries at these values, especially at the league minimum of $79,568.

Fig. 1
![fig1](/figures/figure1.png)

Fig. 2
![fig2](/figures/figure2.png)

Histograms of each numerical feature are made. The majority of histograms are left leaning and right-skewed. Most of these histograms represent career statistics where it is unusual and exceptional for a player to have large values and more common for the values to be towards the low end of all players (such as points, fig3). Histograms become more normal for percentage values (as opposed to total career values) such as percentage of field goals made or average points per game (fig4). A few histograms are right leaning and left-skewed, such as future salary. Start year is right leaning because smaller start years are less common. If this feature were formatted differently, such as a feature for total years in the league, the histogram would be right-skewed.

Fig.3
![fig3](/figures/figure3.png)

Fig. 4
![fig4](/figures/figure4.png)  

Scatter plots are created between each numerical feature and the target variable. These scatter plots generally provide the same information as the correlation heatmap in that we can see how linearly positive and / or linearly negative features are related to salary. Further information can be gathered from these plots because it is clear which features have the most outliers, the largest spreads, and the most uneven spreads. The majority of these plots have the same shape (positive correlation with greater spread as salary increases) and show a similar amount of outliers. The most evenly spread features include the percentage of free throws made, the percentage of field goals made, and the percent of 3 pointers made. This suggests that percentage features might be better predictors of salary than total career features, although that is not necessarily echoed in the heatmap.


## Modeling

Unsupervised machine learning models are used in tandem with supervised machine learning models. Unsupervised models are utilized in two ways: clustering for transfer learning and feature selection through PCA. Supervised learning models (including LassoCV, RidgeCV, Random Forest Regressor, and SVR) are created and compared. The best model is used to make predictions for analysis.

**KMeans and DBSCAN**

The purpose of using clustering unsupervised machine learning models is to identify trends, patterns, and groupings in the data that are not otherwise obvious. Clustering models split the data into "clusters", or groups based on similar features. Clustering can be useful in modeling because players with similar features may also have similar salaries. Put another way, clusters can be good predictors of target variables. Clusters are also useful in analysis because we may observe trends and groupings within the clusters that were not obvious before. This is especially useful for datasets with many features (and thus many dimensions) such as the NBA player dataset.

In order to prepare for clustering, all non-numerical columns are dropped. Since I am ultimately using cluster assignments as a feature in the model it is important to also drop the features that refer to salary directly. '19/20 salary' and 'future salary' are dropped so that clusters will not be created based on similarities in salaries. I do not want clusters created around similarities in salaries because then the cluster assignments would not be valid predictive variables - I would be using salary to predict salary.

Clustering is done using two methods: DBSCAN and KMeans. My original cluster models were done using the DBSCAN method. I chose this method because DBSCAN tends to be less sensitive to outliers and the NBA dataset has many outliers. However, the KMeans method was more successful in creating useful clusters. The KMeans method outperformed the DBSCAN method in two ways: silhouette score and evenness of grouping counts. Silhouette score is a measure of how dense and far apart the values in each cluster are. It ranges from -1 to 1 where -1 means the clusters are very far apart (and badly clustered) and a score of 1 means the clusters are very distinct. The silhouette scores for all of the DBSCAN methods tried maxed out at 0.16. Additionally, the clusters that were created by the DBSCAN were extremely uneven where some clusters had ~250 players and some clusters only had ~10 players. The DBSCAN method was tried using several different combinations of input features but was still not able to outperform the KMeans method. Details on the various trials of the DBSCAN can be found in the '03 - Clustering' file of this repository under the 'DBSCAN' subheader.

The KMeans method produced a higher silhouette score of 0.202. The KMeans method also produced more evenness within the clusters - the number of players grouped into each cluster is much more consistent than it was with the DBSCAN. This method proved better for this modeling process because you are able to select the number of clusters used in the KMeans method. While you can adjust hyperparameters with the DBSCAN method you cannot directly specify the number of total clusters. With KMeans this specification is a requirement. This proved a better method for the NBA Player dataset, and a new feature is created in the existing dataset for the cluster assignment of each player. The new dataset with the clusters is exported as 'NBAPlayersClustered.csv' and can be found in the datasets folder of this repository.


**Regression Modeling**

Several models were created and compared. Model performance is evaluated using the R2 score, or the proportion of the variance in the data that the model explains. R2 scores are given for the training portion of the data (the portion of NBA players that the model learns from) as well as the testing portion (the proportion of the NBA players that the model makes predictions for). The best model should have the highest R2 scores and be relatively stable between the testing R2 score and the training R2 score.

The regression models that are evaluated include LassoCV, RidgeCV, LassoCV and RidgeCV with PCA, LassoCV and RidgeCV with Polynomial Features, LassoCV and RidgeCV with PCA *and* Polynomial features, a Random Forest Regressor, and an SVR model. Polynomial features and PCA are both methods of feature selection. Polynomial features creates interaction terms for all of the independent features in the model and all of the interaction terms are used in the modeling process. PCA is an unsupervised machine learning model approach in which the original features are transformed into a new set of features where each individual new feature is composed of portions of the old features that account for high amounts of variability.

The model with the best performance is used to make predictions and a new dataset (including predictions) is made for the testing data for use in analysis. The dataset with the predictions can be found in the datasets folder of this repository and is called 'NBAPlayersClustered.csv'.

## Analysis

**Overall Conclusions**

The model which preformed the best is the LassoCV model with polynomial features (training r2 = 0.77, testing r2 =0.60). While the RidgeCV model with polynomial features preformed best overall (training r2 = 0.94, testing r2 =0.63), the training and testing scores had a very large differential (0.31) which is evidence that there is severe overfitting with this model. The same is true for the Random Forest Model (training r2 = 0.90, testing r2 =0.62, differential = 0.28). The differential of the LassoCV model with polynomial features (0.17) is much smaller and the model still preforms better than the remainder of the models.

The errors are not distributed evenly for the predictions of this model (fig7). There is an upwards trend between size of residuals and seasons played (fig5, fig6). That is, the predictions that the model is making for the salary of a player are off by a greater amount for players who have played for longer. The model almost always makes large predictions for salary for players who have been playing for a long time. That is, even if a player is making a smaller amount of money the model will predict they are making a larger amount of money because they have played so many seasons and have large numbers for points, games played, games started, etc. However, this causes a problem with modeling because there are many instances of veteran basketball players with strong career records who have decreased in skill so much over the years that their salary is MUCH lower than the model would predict.

Fig. 5
![fig5](/figures/figure5.png)
Fig. 6
![fig6](/figures/figure6.png)
Fig. 7
![fig7](/figures/figure7.png)


This makes sense intuitively because of the presence of players who have become almost "grandfathered in". To illustrate this point I will look at two different players, Carmelo Anthony (Player_Id: 2159029) and Chris Paul (Player_Id: 101108). Carmelo Anthony, who currently plays for the Portland Trailblazers, has been playing professional basketball for 16 seasons. Chris Paul, who is currently with the Oklahoma City Thunder, has been playing for 14 years. Carmelo Anthony is ahead of Chris Paul in almost every statistic ; Anthony has a higher average points per game (23.6 vs. 18.5), almost 3000 more total rebounds, almost 8000 more total points, etc. Both players are extremely well known and have been considered top tier players during their career. The model predicts these players' salaries at ~40 million for Chris Paul and ~31.7 million for Carmelo Anthony. This is very close to correct for Chris Paul who makes ~38.5 million (a residual of 1.5 million), but is extremely off for Anthony who makes only ~2.2 million (a residual of ~29.5 million!).

Fans of NBA will understand exactly why Carmelo Anthony makes less money. After a long career of exceptional success his skills have deteriorated to the point that he is likely to retire in the coming season. Anthony has been shuffled between teams in the last several years without much success, and critics have gone so far as to say that his placement on high-performing teams such as the Trailblazers is due to being "grandfathered-in". That is, Carmelo Anthony has made such a huge overall impact in his career that he is still considered an asset and still picked up even though he does not play as well anymore. The teams will continue to hire him because he is Carmelo Anthony, but they will not pay him anywhere close to what they will pay an asset such as Steph Curry.

Chris Paul is a different story. While Anthony has better overall stats, Paul has won more awards (including Rookie of the Year) and participated in high honors such as being on the Olympic National Basketball Team. Recently Chris Paul had a slight decline in skill (only lasting around 1 year) but has maintained relatively stable statistics throughout his career. While it is difficult for the model to understand the difference between these two highly decorated players with similar statistics, a human NBA fan would be able to predict the difference easily. Paul is still generally considered to be one of the best, Anthony *used* to be one of the best.

To test how this model would perform without "grandfathered" players, I ran a version of the same model but only included players who had been playing for less than 12 seasons. This test can be seen at the bottom of the '05 - Analysis' file of this repository under the 'Low Season Model' subheader. This model performed better than the original model (training r2 = 0.80, testing r2 =0.69). Plotting of residuals shows that this model has fewer outliers but is not necessarily more homoskedastic than the original model (fig10). We see fewer outliers but it is not better spread. The residuals by season played follow a distinct upwards trend with this model (fig8, fig9). While it makes sense that this model would perform better than the original model, it is not as useful as it only represents a portion of the entire population and only moderately improves the R2 scores.

Fig. 8
![fig8](/figures/figure8.png)
Fig. 9
![fig9](/figures/figure9.png)
Fig. 10
![fig10](/figures/figure10.png)

There are only 19 total coefficients for features in the model that are not equal to 0 . The majority of these coefficients are positive (only 3 out of 19 are negative) and the negative coefficients are smaller than the majority of the positive coefficients (fig13). This suggests that most of the coefficients in the model are adding money to the final salary. It is easy for a player's prediction of salary to be higher than the player's actual salary because as long as a player checks a certain amount of boxes his salary will be assumed higher. This is not necessarily the case in real life and shows that the model would likely improve with the addition of features that decrease salary (such as negative press or a negative trend in skill).

Fig. 13
![fig13](/figures/figure13.png)

**Cluster Analysis**

In general the KMeans clustering model created very easy to differentiate and evenly spread clusters(fig12). While the cluster feature was not a major predictor of salary (no features that included clusters had non-0 coefficients) the clusters are still very interesting to analyze. The relationships between the players in each group are very clear. For example, cluster 1 is the cluster of superstars. This cluster includes Steph Curry, James Harden, Russel Westbrook, Kevin Durant, Chris Paul, etc. and only has the highest paid and most notable players in the league. Another cluster, cluster 19, is the "promising rookie" group. This group includes Trae Young (of the Atlanta Hawks), Lonzo Ball, Luka Doncic, and other new players that have been in the league for a short time but have already made a name for themselves. Another cluster, cluster 4, is for new players who have made very little impact. These players have started 0 games and have only just begun in the NBA. For a full analysis of several other clusters and a look at all of the players in some of the groups I have discussed see the '05-Analysis' file of this repository under the 'Cluster analysis' subheader.

Fig. 12
![fig12](/figures/figure12.png)

The clustering models had the same difficulties with differentiating long-standing players (fig11). The plot of residuals by cluster shows large residuals in cluster 13. Cluster 13 has only 3 total players: Carmelo Anthony, LeBron James, and Vince Carter. These players have been playing for 16-21 years and are in the same cluster even though their salaries are vastly different (LeBron James is still one of the highest paid players in the NBA).

Fig. 11
![fig11](/figures/figure11.png)


## Conclusions

The best model choice for predicting the salaries of active NBA players is a LassoCV model in combination with polynomial features. The Random Forest Model and RidgeCV with polynomial features performed better in both the training and testing sets than the LassoCV with polynomial features, but both had much larger variance. The best clustering method for grouping active NBA players is KMeans clustering as this method results in more clusters, more even clusters, and a higher silhouette score than the DBSCAN method. These result surprises me as I expected a model with polynomial features *and* PCA would produce the best results as it would optimize the variability explained by the model. I also expected DBSCAN to outperform KMeans as the NBA players dataset has a lot of outliers and KMeans is more sensitive to that. The model predicted the salaries of the testing set with an R2 score of 0.60, meaning the model explains 60% variability in the testing set.

These are thee coefficients that the model thought were useful.

Although the model performed decently overall, there were several weaknesses. The major weakness of the model is the fact that the residuals increase as the players total seasons increases, which I believe is explained by an inability to differentiate between long time players who's skills have recently faltered and long time players who are still at the top of their game. This problem could be addressed by creating a feature that shows the trend in skill of the player. However, this could become complicated quickly as some skills may be trending upwards and some skills trending downwards for the same player. More importantly, finding concrete information about trends in skill of each player would be difficult to find and interpret, especially in the case of a single player having skills trending in different directions. More importantly, this would still likely only improve the model somewhat as players who have declining skills may have other attributes (such as strong fan base or an ability to increase team morale) that contribute to a higher salary.

These issues relate to a larger issue with this model which is that there are countless other features that likely contribute to a NBA players salary, many of which are impossible or extremely difficult to quantify or collect data on. I have used Carmelo Anthony as an example in this project as a player who is making less money because his skills have declined, but it is likely there are other players who have similar records *and* a similar downward trend in skill that are making more money because they have less bad press or have experienced fewer recent trades.

In general the model over-predicts salary as opposed to under-predicts. This is likely due to the fact that the model has a higher ratio of positive correlations to negative correlations and positive coefficients to negative coefficients. This suggests that a player will have a lot of attributes that result in a positive salary addition and very little attributes that result in a salary decrease.

This dataset proved particularly difficult to model because there are so many outliers. This was slightly improved by decreasing the dataset to only players that have been in the NBA for less than 12 seasons, but that has very little practical application. This model is unusual in that I am able to model almost my entire population (all active basketball players), so any problems with the way the model handles variation in the data are problems with the way the model handles variation in the population of interest. This dataset is also difficult to model because there is a vast number of potential features that I do not have access to. Although the population I am observing is extremely controlled, the features that affect the target variable are not. It would be nearly impossible to even make a complete list of possible predictive features, much less quantify, collect, and model with them. Put another way, an NBA player's salary is worth more than the sum of its parts.

All in all this model performs well considering the variability in the target variable. The model does not necessarily have use as a salary predictor (the NBA does not need a salary predictor as they are very good at handling salary offers without one), but it is an excellent way of learning more about the differences and patterns in the career statistics and overall performance of NBA players. This project has been very useful for me in particular as I have learned a lot about players I enjoy and I feel I have a better understanding of the relationship between basketball skill and salary. In this way, the model has more inferential power than it does predictive power.


## Future Considerations

I have many suggestions for further research. Firstly, I would love to gather more information on individual players. It would be very interesting to have information on a player's fan base, the public perception of the player, the types of press that the player gets (negative / positive), the player's relationship with their teammates and coaches, how many trades the player has recently experienced, and which direction their skills are trending. Gathering more information, particularly for features that could have negative impact on salary, would likely help the model's residuals become more homoskedastic and help the model better predict the salaries of long-time players. Unfortunately, as I have discussed earlier, these features are not only difficult to collect but they are difficult to quantify or measure in the first place.

I also suggest that this analysis be continued to include entire teams. During my initial data collection I had planned on finding information about teams such as the sizes of the fan bases, the proportion of stadium seats that are filled on average, the amount of money brought in yearly by each team, the total spending cost of each team, etc. This analysis would likely prove interesting as there are probably complex relationships between the total salaries of all players on a team and the success of the team. Unfortunately, while a lot of this data is available from many reliable sources I was not able to find a source that did not have a pay wall.
