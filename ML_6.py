#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#Importing the required libraries.


# In[2]:


from sklearn.cluster import KMeans, k_means #For clustering
from sklearn.decomposition import PCA # Linear Dimensionality reduction.


# In[3]:


df = pd.read_csv("sales_data_sample.csv",encoding='latin1') #Loading the dataset.


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


df.info()


# In[8]:


df.isnull().sum()

df.dtypes


# In[9]:


df_drop  = ['ADDRESSLINE1', 'ADDRESSLINE2', 'STATUS','POSTALCODE', 'CITY', 'TERRITORY', 'PHONE', 'STATE', 'CONTACTFIRSTNAME', 'CONTACTLASTNAME', 'CUSTOMERNAME', 'ORDERNUMBER']
df = df.drop(df_drop, axis=1) #Dropping the categorical uneccessary columns along with columns having null values. Can't fill the null values are there are alot of null values.


# In[10]:


df.isnull().sum()


# In[11]:


df.dtypes


# In[12]:


df.duplicated( keep='first').sum()


# In[13]:


df.isna().sum() #finding missing values


# In[14]:


# Checking the categorical columns.
df['COUNTRY'].unique()


# In[15]:


df['PRODUCTLINE'].unique()


# In[16]:


df['DEALSIZE'].unique()


# In[17]:


productline = pd.get_dummies(df['PRODUCTLINE']) #Converting the categorical columns.
Dealsize = pd.get_dummies(df['DEALSIZE'])


# In[18]:


df = pd.concat([df,productline,Dealsize], axis = 1)


# In[19]:


df


# In[20]:


df_drop  = ['COUNTRY','PRODUCTLINE','DEALSIZE'] #Dropping Country too as there are alot of countries.
df = df.drop(df_drop, axis=1)


# In[21]:


df['PRODUCTCODE'] = pd.Categorical(df['PRODUCTCODE']).codes #Converting the datatype.


# In[22]:


df.drop('ORDERDATE', axis=1, inplace=True) #Dropping the Orderdate as Month is already included.


# In[23]:


df.dtypes #All the datatypes are converted into numeric


# In[24]:


# before we implement the k-means and assign the centers of our data, we can also make a quick analyze to

# find the optimal number (centers) of clusters using Elbow Method.

# Elbow Method is one of the most popular methods to determine this optimal value of k.

# Distortion: It is calculated as the average of the squared distances from the cluster centers of the respective clusters. Typically, the Euclidean distance metric is used.
# Inertia: It is the sum of squared distances of samples to their closest cluster center.

# The KMeans algorithm clusters data by trying to separate samples in n groups of equal variances, minimizing a criterion known as inertia, or within-cluster sum-of-squares Inertia, or
# the within-cluster sum of squares criterion, can be recognized as a measure of how internally coherent clusters are.

# The k-means algorithm divides a set of N samples X into K disjoint clusters C, each described by the mean j of the samples in the cluster.
# The means are commonly called the cluster centroids.

# The K-means algorithm aims to choose centroids that minimize the inertia, or within-cluster sum of squared criterion.

from sklearn.cluster import KMeans
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(df)
    distortions.append(kmeanModel.inertia_)


# In[25]:


kmeanModel


# In[26]:


kmeanModel.cluster_centers_


# In[27]:


kmeanModel.inertia_

# The lower values of inertia are better and zero is optimal.

# We can see that the model has very high inertia. So, this is not a good model fit to the data.


# In[29]:


# check how many of the samples were correctly labeled

label = kmeanModel.labels_


# In[30]:


label


# In[31]:


plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()


# In[32]:


X_train = df.values #Returns a numpy array.


# In[33]:


X_train.shape


# In[34]:


model = KMeans(n_clusters=3,random_state=2) #Number of cluster = 3
model = model.fit(X_train) #Fitting the values to create a model.
predictions = model.predict(X_train) #Predicting the cluster values (0,1,or 2)


# In[35]:


predictions

#3 clusters within 0, 1, and 2 numbers. We can also merge the result of the clusters with our original data table like this:


# In[36]:


unique,counts = np.unique(predictions,return_counts=True)


# In[37]:


unique


# In[39]:


counts


# In[40]:


counts = counts.reshape(1,3)   # 1 row and 3 column


# In[41]:


counts


# In[42]:


counts_df = pd.DataFrame(counts,columns=['Cluster1','Cluster2','Cluster3'])


# In[43]:


counts_df.head()


# In[44]:


pca = PCA(n_components=2) #Converting all the features into 2 columns to make it easy to visualize using Principal COmponent Analysis.


# In[45]:


reduced_X = pd.DataFrame(pca.fit_transform(X_train),columns=['PCA1','PCA2']) #Creating a DataFrame.


# In[46]:


reduced_X.head()


# In[48]:


#Plotting the normal Scatter Plot
plt.figure(figsize=(14,10))
plt.scatter(reduced_X['PCA1'],reduced_X['PCA2'])


# In[49]:


model.cluster_centers_ #Finding the centriods. (3 Centriods in total. Each Array contains a centroids for particular feature )


# In[50]:


reduced_centers = pca.transform(model.cluster_centers_) #Transforming the centroids into 3 in x and y coordinates


# In[51]:


reduced_centers


# In[52]:


plt.figure(figsize=(14,10))
plt.scatter(reduced_X['PCA1'],reduced_X['PCA2'])
plt.scatter(reduced_centers[:,0],reduced_centers[:,1],color='black',marker='x',s=300) #Plotting the centriods


# In[53]:


reduced_X['Clusters'] = predictions #Adding the Clusters to the reduced dataframe.


# In[54]:


reduced_X.head()


# In[55]:


#Plotting the clusters
plt.figure(figsize=(14,10))
#                     taking the cluster number and first column           taking the same cluster number and second column      Assigning the color
plt.scatter(reduced_X[reduced_X['Clusters'] == 0].loc[:,'PCA1'],reduced_X[reduced_X['Clusters'] == 0].loc[:,'PCA2'],color='slateblue')
plt.scatter(reduced_X[reduced_X['Clusters'] == 1].loc[:,'PCA1'],reduced_X[reduced_X['Clusters'] == 1].loc[:,'PCA2'],color='springgreen')
plt.scatter(reduced_X[reduced_X['Clusters'] == 2].loc[:,'PCA1'],reduced_X[reduced_X['Clusters'] == 2].loc[:,'PCA2'],color='indigo')


plt.scatter(reduced_centers[:,0],reduced_centers[:,1],color='black',marker='x',s=300)


# In[ ]:




