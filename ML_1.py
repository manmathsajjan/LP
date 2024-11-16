#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df  = pd.read_csv("uber.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[6]:


df.columns


# In[7]:


df.shape


# In[8]:


df = df.drop(['Unnamed: 0', 'key'], axis= 1) 


# In[9]:


df.head()


# In[10]:


df.shape


# In[11]:


df.dtypes


# In[12]:


df.info()


# In[13]:


df.describe()


# In[14]:


df.isnull().sum()


# In[15]:


df['dropoff_latitude'].fillna(value=df['dropoff_latitude'].mean(),inplace = True)
df['dropoff_longitude'].fillna(value=df['dropoff_longitude'].median(),inplace = True)


# In[16]:


df['dropoff_latitude']


# In[17]:


df.isnull().sum()


# In[18]:


df.dtypes


# In[19]:


df.pickup_datetime = pd.to_datetime(df.pickup_datetime,)


# In[20]:


df.dtypes


# In[21]:


df= df.assign(hour = df.pickup_datetime.dt.hour,
             day= df.pickup_datetime.dt.day,
             month = df.pickup_datetime.dt.month,
             year = df.pickup_datetime.dt.year,
             dayofweek = df.pickup_datetime.dt.dayofweek)


# In[22]:


df.head()


# In[23]:


# drop the column 'pickup_daetime' using drop()
# 'axis = 1' drops the specified column

df = df.drop('pickup_datetime',axis=1)


# In[24]:


df.head()


# In[25]:


df.dtypes


# In[26]:


df.plot(kind = "box",subplots = True,layout = (7,2),figsize=(15,20)) #Boxplot to check the outliers


# In[27]:


def remove_outlier(df1 , col):
    Q1 = df1[col].quantile(0.25)
    Q3 = df1[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_whisker = Q1-1.5*IQR
    upper_whisker = Q3+1.5*IQR
    df[col] = np.clip(df1[col] , lower_whisker , upper_whisker)
    return df1

def treat_outliers_all(df1 , col_list):
    for c in col_list:
        df1 = remove_outlier(df , c)
    return df1


# In[28]:


df = treat_outliers_all(df , df.iloc[: , 0::])


# In[29]:


df.plot(kind = "box",subplots = True,layout = (7,2),figsize=(15,20)) #Boxplot shows that dataset is free from outliers


# In[30]:


get_ipython().system('pip install haversine')
import haversine as hs  #Calculate the distance using Haversine to calculate the distance between to points. Can't use Eucladian as it is for flat surface.
travel_dist = []
for pos in range(len(df['pickup_longitude'])):
        long1,lati1,long2,lati2 = [df['pickup_longitude'][pos],df['pickup_latitude'][pos],df['dropoff_longitude'][pos],df['dropoff_latitude'][pos]]
        loc1=(lati1,long1)
        loc2=(lati2,long2)
        c = hs.haversine(loc1,loc2)
        travel_dist.append(c)

print(travel_dist)
df['dist_travel_km'] = travel_dist
df.head()


# In[31]:


travel_dist


# In[32]:


#Uber doesn't travel over 130 kms so minimize the distance
df= df.loc[(df.dist_travel_km >= 1) | (df.dist_travel_km <= 130)]
print("Remaining observastions in the dataset:", df.shape)


# In[33]:


#Finding inccorect latitude (Less than or greater than 90) and longitude (greater than or less than 180)
incorrect_coordinates = df.loc[(df.pickup_latitude > 90) |(df.pickup_latitude < -90) |
                                   (df.dropoff_latitude > 90) |(df.dropoff_latitude < -90) |
                                   (df.pickup_longitude > 180) |(df.pickup_longitude < -180) |
                                   (df.dropoff_longitude > 180) |(df.dropoff_longitude < -180)
                                    ]


# In[34]:


incorrect_coordinates


# In[35]:


df.drop(incorrect_coordinates, inplace = True, errors = 'ignore')


# In[36]:


df.head()


# In[37]:


df.isnull().sum()


# In[38]:


sns.heatmap(df.isnull()) #Free for null values


# In[39]:


corr = df.corr()


# In[40]:


corr


# In[43]:


fig,axis = plt.subplots(figsize = (10,6))
sns.heatmap(df.corr(),annot = True) #Correlation Heatmap (Light values means highly correlated)


# In[44]:


x = df[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count','hour','day','month','year','dayofweek','dist_travel_km']]
y = df['fare_amount']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.20)


# In[45]:


X_train


# In[46]:


X_test


# In[47]:


y_train


# In[48]:


y_test


# In[49]:


from sklearn.linear_model import LinearRegression
regression = LinearRegression()


# In[50]:


regression.fit(X_train,y_train)


# In[51]:


regression.intercept_ #To find the linear intercept


# In[52]:


regression.coef_ #To find the linear coeeficient


# In[53]:


prediction = regression.predict(X_test) #To predict the target values


# In[54]:


print(prediction)


# In[55]:


y_test


# In[56]:


from sklearn.metrics import r2_score


# In[57]:


from sklearn.metrics import mean_squared_error


# In[58]:


from sklearn.metrics import mean_squared_error


# In[59]:


MSE = mean_squared_error(y_test,prediction)


# In[60]:


MSE


# In[61]:


RMSE = np.sqrt(MSE)


# In[62]:


RMSE


# In[63]:


from sklearn.ensemble import RandomForestRegressor


# In[64]:


rf = RandomForestRegressor(n_estimators=50) #Here n_estimators means number of trees you want to build before making the prediction


# In[65]:


rf.fit(X_train,y_train)


# In[66]:


y_pred = rf.predict(X_test)


# In[67]:


y_pred


# In[68]:


R2_Random = r2_score(y_test,y_pred)


# In[69]:


R2_Random


# In[70]:


MSE_Random = mean_squared_error(y_test,y_pred)


# In[71]:


MSE_Random


# In[72]:


RMSE_Random = np.sqrt(MSE_Random)


# In[73]:


RMSE_Random


# In[ ]:




