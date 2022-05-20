#!/usr/bin/env python
# coding: utf-8

# In[414]:


get_ipython().system(' pip install opendatasets')


# In[415]:


import opendatasets as od


# In[416]:


od.download ("https://www.kaggle.com/datasets/stefanoleone992/fifa-21-complete-player-dataset")


# In[417]:


od.download ("https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset")


# In[418]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import math


# In[419]:


df="fifa-21-complete-player-dataset.zip to .\fifa-21-complete-player-dataset"


# In[420]:


df=("fifa-22-complete-player-dataset.zip to .\fifa-22-complete-player-dataset")


# In[421]:


fifa21 = pd.read_csv('players_21.csv')


# In[422]:


fifa22 = pd.read_csv('players_22.csv')


# In[423]:


print(fifa21.shape) 


# In[424]:


print(fifa22.shape)  


# In[425]:


fifa_complete_data = pd.DataFrame.merge(fifa22, fifa21, how="outer")


# In[426]:


print(fifa_complete_data.shape)


# In[427]:


fifa_complete_data.head(10)


# In[428]:


fifa_complete_data.describe()
fifa_complete_data.info()


# In[429]:


number_of_duplicated_players = fifa_complete_data.duplicated("short_name").sum()
print("The number of duplicated players is: ", number_of_duplicated_players)


# In[430]:


fifa_complete_data = fifa_complete_data.drop_duplicates("short_name", keep="first")


# In[431]:


fifa_complete_data.shape  


# In[432]:


fifa_complete_data = fifa_complete_data[
fifa_complete_data["overall"] >= 80]  # filtering dataset for players 80+
fifa_complete_data.shape  


# In[433]:


# split the data on the player_position column as a number of players have more than one position.


players_with_comma = fifa_complete_data[fifa_complete_data["player_positions"].str.contains(",")]
players_no_comma = fifa_complete_data[~fifa_complete_data["player_positions"].str.contains(",")]


# use regex to capture the players position, the first position before the ",".


regex = r'^(.+?),'


# In[434]:


fifa_complete_data = pd.merge(players_with_comma, players_no_comma, how="outer").sort_values("overall", ascending=False)     .reset_index().drop(columns=['index'])  # remove index column as it is just a duplicate of ids.

print(fifa_complete_data.shape)  # checkpoint for data


# In[435]:


# sorting of the player positions
# create a dictionary for new positions -defense-midfield or forward.


player_position_fifa = {"RB": "Defender", "RWB": "Defender", "RCB": "Defender", "CB": "Defender", "LB": "Defender",
                        "LWB": "Defender", "LCB": "Defender", "CDM": "Midfield", "CM": "Midfield", "LCM": "Midfield",
                        "RCM": "Midfield", "LM": "Midfield", "RM": "Midfield", "CAM": "Midfield", "LF": "Forward",
                        "ST": "Forward", "CF": "Forward", "RF": "Forward", "RW": "Forward", "LW": "Forward"}


# In[436]:


# Updated player positions.
for key, value in player_position_fifa.items():
    print(key + ":" + str(value))


# In[437]:


# iterate over columns and replace values with dictionary values.
for key, value in fifa_complete_data["player_positions"].iteritems():
    fifa_complete_data["player_positions"] = fifa_complete_data["player_positions"].apply(lambda x:
                                                                                          player_position_fifa.get(x,
                                                                                                                   x))


# In[438]:


# validation for null values.
fifa_complete_data["player_positions"].isnull().sum()  # "o" = no null values


# In[439]:


# check position count.
position_count = pd.crosstab(index=fifa_complete_data.player_positions, columns="Positions")
print(position_count)


# In[440]:


# review null values in overall df
null_values = fifa_complete_data.isnull().sum()
# pace,shooting,passing,dribbling,defending,physic

df_to_use = fifa_complete_data.dropna(axis=0, how="any",
                                      subset=['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic'])


# In[441]:


hist(var="Skill_rating")
hist(var="Player_value_euros")
hist(var="wage_eur")
hist(var="release_clause_eur")
hist(var="movement_reactions")
hist(var="mentality_composure")
hist(var="skill_ball_control")
hist(var="dribbling")
hist(var="attacking_short_passing")


# In[443]:


def player_positions_plot():
    """ Make a visualisation of number of positions in df."""
    player_position_plot = ["GK", "Defender", "Midfield", "Forward"]
    plot1 = sns.catplot("player_positions", data=fifa_complete_data, kind="count",
                        legend=True, order=player_position_plot)
    plot1.fig.set_size_inches(10, 6)
    plot1.fig.subplots_adjust(top=0.81, right=0.86)
    # extract the matplotlib axes_subplot objects from the FacetGrid
    ax = plot1.facet_axis(0, 0)
    # iterate through the axes containers
    for i in ax.containers:
        labels = [f'{(v.get_height()):.0f}' for v in i]
        ax.bar_label(i, labels=labels, label_type='edge')
    plt.title("Player position totals FIFA 21/22")
    plt.xlabel("Positions")
    plt.ylabel("Frequency")
    plt.show()
player_positions_plot()


# In[444]:


def hist(var="plot"):
    """use histograms to check variable distribution"""
    plt.hist(x=var, data=df_to_use)
    plt.title("Histogram of variable distribution")
    plt.xlabel(var)
    plt.show()


# In[445]:


# view the distributions of target variable 

hist(var="Skill_rating")
hist(var="Player_value_euros")
hist(var="wage_eur")
hist(var="movement_reactions")
hist(var="mentality_composure")
hist(var="skill_ball_control")
hist(var="dribbling")
hist(var="attacking_short_passing")


# In[446]:


#checking correlations with target variable with dataset.

fifa_complete_data = fifa_complete_data.rename(columns={"overall": "Skill_rating", "value_eur": "Player_value_euros"})
# recode player foot, 0 for left and 1 for right.
df_to_use["preferred_foot"] = df_to_use["preferred_foot"].replace("Left", 0).replace("Right", 1)
df_to_use = df_to_use.rename(columns={"overall": "Skill_rating", "value_eur": "Player_value_euros"})
pearson_corr = df_to_use.corr()["Player_value_euros"]
positive_corr_features = pearson_corr[pearson_corr > 0.5].round(decimals=2)
print("The variables that correlate with our target variable are:" "\n", positive_corr_features)


# In[447]:


# now selecting the most highly correlated features or variables from the data.
features_corr_positive = pearson_corr[pearson_corr > 0.5]
# printing the most postively and most negatively correlated factors:
print('Most Positively Correlated Features:''\n',
      features_corr_positive[features_corr_positive.values != 1.0].round(decimals=2))



# In[448]:


fifa_complete_data.head(5)


# In[449]:


def linear_relationship(variable="dummy", xlab="label", title="test"):
    """Scatter-plot to show the top correlating factors to skill level in fifa to illustrate the linear relationship"""
    sns.set()
    sns.scatterplot(x=variable, y="Player_value_euros", data=df_to_use)
    sns.set_theme("talk")
    plt.xlabel(xlab)
    plt.ylabel("Players valuation (Millions in Euro)")
    plt.title(title)
    plt.show()


# plot of linear relationship for the players movement reactions metric.
linear_relationship("movement_reactions", xlab="Players movement_reactions",
                    title="Linear Relationship plot")

# plot of linear relationship for the players movement composure metric.
linear_relationship("mentality_composure", xlab="Players mentality_composure value",
                    title="Linear Relationship plot")

# plot of linear relationship for the players offensive passing metric.
linear_relationship("attacking_short_passing", xlab="Players offensive short passing value",
                    title="Linear Relationship plot")

# plot of linear relationship for the players ball skill metric.
linear_relationship("skill_ball_control", xlab="Players ball skill value",
                    title="Linear Relationship plot")

# plot of linear relationship for the players ball skill metric.
linear_relationship("wage_eur", xlab="Players wage value (thousands of euro per week)",
                    title="Linear Relationship plot")

