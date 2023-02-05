"""
@author: Cem Baran KURUKAYA
"""
import pandas as pd
import numpy as np
import time
import xgboost as xgb
version_of_xgb = xgb.__version__ #to learn which xgboost version I use

df = pd.read_csv("Netflix_movies.csv")

# rename specific column because it was unnamed. Now I have "id" column
df.rename(columns = {'Unnamed: 0':'id'}, inplace = True)

# data cleaning
df.drop("enter_in_netflix",inplace=True, axis =1)
df.drop("actors",inplace=True, axis =1)

# finding shape of our dataset
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

# to learn the data types of each column
print(df.info())

# checking for missing values
print("Missing values:", df.isnull().sum()) #there are no any empty column or missing values

# checking for duplicate data
duplicateData = df.duplicated().any()
print("Duplicate Data:", duplicateData) # ---> there are no duplicate data

# Getting statistics about the dataset
statisticsData = df.describe() 
statisticsData.drop("id",inplace=True, axis =1)
        
df_old = df.copy() # just for observing the old dataframe

# to get first element of genre, director and country to getting closer result.
for i, row in df.iterrows():
    df.loc[i, "genre"] = row.genre.split(",")[0]
    df.loc[i, "director"] = row.director.split(",")[0]
    df.loc[i,"country"] = row.country.split(",")[0]
    
# ratings of directors
directorRatings = df.groupby("director")["rating"].mean().sort_values(ascending=False)
    
#Finding unique values from genre so we can seperate them
list1 = []
for value in df["genre"]:
    list1.append(value.split(","))
    
list2 = []
for item in list1:
    for item1 in item:
        list2.append(item1)
        
list2 = [word.strip() for word in list2] #removing spaces on the beggining of genres name.

unique_list = [] # this list gives us the all genres names.
for item in list2:
    if item not in unique_list:
        unique_list.append(item)
        
"""
unique_list = Genres:
    ['Dramas',
     'Comedies',
     'Horror Movies',
     'Action & Adventure',
     'Documentaries',
     'Independent Movies',
     'Children & Family Movies',
     'Movies',
     'Stand-Up Comedy',
     'International Movies',
     'Anime Features',
     'Classic Movies',
     'Thrillers',
     'Music & Musicals',
     'Cult Movies',
     'Sci-Fi & Fantasy']
"""

## importing label encoder to convert the string values to numeric values
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

# converting "director","country" columns to numeric values
df["director"] = le.fit_transform(df["director"])
df["country"] = le.fit_transform(df["country"])

#now we are going to seperate genres to columns
"""
When I tried it by converting the genre to numerical data 
without separating it into columns, 
the mean absoulte eror of linear Regression was 0.098. 
I managed to reduce this to 0.092 by separating the columns.
"""

uni_genres = pd.get_dummies(df[["genre"]])
df = pd.concat([df, uni_genres],axis=1)


"""
MinMaxscaler is a type of scaler that scales the minimum and maximum values to be 0 and 1 respectively. 
 ML algorithm works better when features are relatively on a similar scale and close to Normal Distribution."""
 
#using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
df[["director","rating","country"]] = mms.fit_transform(df[["director","rating","country"]])

#X = df[["genre","director","country"]] # ---> it is for when I don't divide genre into columns
X = df.drop(["id","movie_name","Duration","year","genre","rating"], axis = 1)
y = df["rating"]

#Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 100)

##start time of Liner Regression
st_lr = time.time()

### Liner Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
ypred = lr.predict(X_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, ypred)

from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
mae_lr_p = mean_absolute_percentage_error(y_test,ypred)

#to learn mean absolute error
mae_lr = mean_absolute_error(y_test,ypred)
print("Mean absolute error of linear regression:", mae_lr)

##end time Liner Regression
end_lr = time.time()

#Execution time of linear regression
exc_lr = end_lr - st_lr
print("Execution time of linear regression:", exc_lr)


################ squeeze to one dimension
conc=pd.DataFrame({'act':np.squeeze(y_test),
             'predicted':np.squeeze(ypred)})


import matplotlib.pyplot as plt
#######################################The distribution of the predicted data on the real data.
plt.figure(figsize=(12,8))
plt.scatter(conc.act,conc.predicted,color='g', label = 'Linear Regression Prediction')
plt.plot(conc.act,conc.act,color='r',label='The Real data')
plt.legend()
###################################

plt.figure(figsize=(12,8))
plt.scatter(y_test,ypred,color='r', label = 'Linear Regression Prediction not conc')
plt.plot(conc.act,conc.act,color='r',label='The Real data')
plt.legend()

##start time of xgboost
st_xgb = time.time()

### Xgboost 
from xgboost import XGBRegressor

xgb = XGBRegressor()
xgb.fit(X_train,y_train)
xgbprd=xgb.predict(X_test)

r2x = r2_score(y_test,xgbprd)

#to learn mean absolute error
mae_xgb = mean_absolute_error(y_test,xgbprd)
print("Mean absolute error of XGreg:", mae_xgb)

#end time of xgboost
end_xgb = time.time()

#Execution time of XGBoost
exc_xgboost = end_xgb - st_xgb
print("Execution time of XGBoost:", exc_xgboost)


import matplotlib.pyplot as plt
####################### The distribution of the predicted data on the real data.
plt.figure(figsize=(12,8))
plt.plot(y_test,y_test)
plt.scatter(y_test,xgbprd, color = 'g', label = 'XGBooster Regression Model')
plt.xlabel('Rating')
plt.legend()
######################

####################### comparison of methods with a graph
plt.figure(figsize=(12,8))
plt.plot(y_test,y_test)
plt.scatter(y_test,xgbprd, color = 'g', label = 'XGBooster Regression Model')
plt.scatter(y_test,ypred, color = 'r', label = 'Lineaer Regression Model')
plt.xlabel('Rating')
plt.legend()
######################

# Comparison of XGBoost and Linear Regression
from sklearn import metrics
MEA = metrics.mean_absolute_error

metrics = {'XGBoost' : [MEA(y_test,xgbprd),r2x, exc_xgboost],     # Dictionary
        'Linear Regression' : [MEA(y_test,ypred),r2, exc_lr]}

comparison = pd.DataFrame(metrics,index=['Mean Absolute Error',"R2 score","Execution Time"])  # DataFrame
comparison =comparison.transpose() 
print(comparison)


