#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics


# In[4]:


df=pd.read_csv("emails.csv")


# In[5]:


df

df.head()

df.info

df.shape

df.columns

df.isnull().sum()


# In[6]:


df.dropna(inplace = True)
df.drop(['Email No.'],axis=1,inplace=True)
X = df.drop(['Prediction'],axis = 1)
y = df['Prediction']


# In[7]:


df


# In[8]:


from sklearn.preprocessing import scale
X = scale(X)
# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# In[9]:


X


# In[10]:


X_train


# In[11]:


X_test


# In[12]:


y_train


# In[14]:


y_test


# In[15]:


X_train.shape


# In[16]:


X_test.shape


# In[17]:


y_train.shape


# In[18]:


y_test.shape


# In[19]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)


# In[20]:


print("Prediction",y_pred)   # 1 for spam 0 for not spam


# In[21]:


print("KNN accuracy = ",metrics.accuracy_score(y_test,y_pred))


# In[22]:


print("Confusion matrix",metrics.confusion_matrix(y_test,y_pred))


# In[23]:


# cost C = 1
model = SVC()   # C is an offset value to account for some mis-classification of data that can happen.

# fit
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)


# In[24]:


print("SVM accuracy = ",metrics.accuracy_score(y_test,y_pred))


# In[25]:


metrics.confusion_matrix(y_true=y_test, y_pred=y_pred)


# In[ ]:




