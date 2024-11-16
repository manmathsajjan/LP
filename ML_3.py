#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt #Importing the libraries


# In[2]:


df = pd.read_csv("Churn_Modelling.csv")


# In[3]:


df.head()

df.shape

df.describe()

df.isnull()

df.isnull().sum()

df.info()

df.dtypes

df.columns


# In[4]:


df.duplicated().sum()


# In[5]:


df['Exited'].value_counts()    # 0 indicate people stay with bank and 1 left the bank shown in balance


# In[6]:


df['Geography'].value_counts()


# In[7]:


df['Gender'].value_counts()


# In[8]:


df.drop(columns=['RowNumber','CustomerId','Surname'],inplace=True)  #dropping unnecssary column inplace means permanantly removed


# In[9]:


df.head()


# In[10]:


# Now convert categorical column into one hot encoder Geography and gender

df=pd.get_dummies(df,columns=['Geography','Gender'],drop_first=True)   #drop first it helps in reducing the extra column created during dummy variable creation. see gender


# In[11]:


df


# In[12]:


# Now see column X and Y
X = df.drop(columns=['Exited'])
y = df['Exited']



# In[13]:


X


# In[14]:


y


# In[15]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=1)


# In[16]:


X_train.shape


# In[17]:


# Now scale the values

# Normalizing the values with mean as 0 and Standard Deviation as 1

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# In[18]:


X_train_scaled   # shown 2 D array with small values


# In[19]:


X_train_scaled.shape


# In[20]:


X_test_scaled


# In[21]:


def visualization(x, y, xlabel):
    plt.figure(figsize=(10,5))
    plt.hist([x, y], color=['red', 'green'], label = ['exit', 'not_exit'])
    plt.xlabel(xlabel,fontsize=20)
    plt.ylabel("No. of customers", fontsize=20)
    plt.legend()


# In[22]:


df_churn_exited = df[df['Exited']==1]['Tenure']     #customer left the bank
df_churn_not_exited = df[df['Exited']==0]['Tenure']  #customer not left the bank


# In[23]:


visualization(df_churn_exited, df_churn_not_exited, "Tenure")


# In[24]:


df_churn_exited2 = df[df['Exited']==1]['Age']
df_churn_not_exited2 = df[df['Exited']==0]['Age']


# In[25]:


visualization(df_churn_exited2, df_churn_not_exited2, "Age")


# In[26]:


import keras #Keras is an Open Source Neural Network library written in Python that runs on top of Theano or Tensorflow.
# we Can use Tenserflow as well but won't be able to understand the errors initially.


# In[27]:


from keras.models import Sequential #To create sequential neural network layers in a sequential order
from keras.layers import Dense #To create hidden layers


# In[28]:


classifier = Sequential()  # sequential is class name ie a predictive modeling problem where you have some sequence of inputs over space or time, and the task is to predict a category for the sequence
#Units: it denotes the output size of the layer, normally average of no of node in input layer (no of independent variable) which is 11 and no of node in output layer which is 1, we took 6 as average.
#Kernel_initializer : The initializer parameters tell Keras how to initialize the values of our layer, weight matrix and our bias vector
#Activation: Element-wise activation function to be used in the dense layer. read more about Rectified Linear Unit (ReLU)
#Input_dim: for first layer only, number of input independent variable. only for first hidden layer
#Bias : if we are going with advance implementation

classifier.add(Dense(units =3 , activation='sigmoid', kernel_initializer='uniform', input_dim = 11))   #input layer 11 hidden layer=3 #uniform is type of distribution

classifier.add(Dense(units =1 , activation='sigmoid', kernel_initializer='uniform',))  # output layer is 1


# In[29]:


classifier.summary()

# 11*3 + 3 bias=36
# 3*1 + 1 bias = 4
# total= 36+4= 40


# In[30]:


# Now compile model using loss function it is binary classification problem


# Optimizer: update the weight parameters to minimize the loss function..
# Loss function: acts as guides to the terrain telling optimizer if it is moving in the right direction to reach the bottom of the valley, the global minimum.
# Metrics: A metric function is similar to a loss except that the results from evaluating a metric are not used when training the model.
# Batch size: hyper-parameter related to sample
# Epochs: hyper-parameter related to iteration

classifier.compile(optimizer="adam",loss = 'binary_crossentropy',metrics = ['accuracy']) #To compile the Artificial Neural Network. Ussed Binary crossentropy as we just have only two output
classifier.fit(X_train_scaled,y_train,batch_size = 10,epochs=10, validation_split=0.2 )


# In[31]:


# now check the value of weight and bias value

classifier.layers[0].get_weights()

# output shown 33 layers connection 3 bias


# In[32]:


classifier.layers[1].get_weights()

# output shown 3 layers connection 1 bias



# In[33]:


# Now predict the model

classifier.predict(X_test_scaled)

# output not shown 1 or 0 because you use sigmoid due to this we need convert the probability into 0 and 1


# In[34]:


# assume threshold 0.5
# if threshold less than 0.5 customer left the bank
# if threshold greater than 0.5 customer not left the bank

y_log= classifier.predict(X_test_scaled)  # y_log is just name of varriable

y_pred= np.where(y_log>0.5,1,0)


# In[35]:


# Now check accuracy of model

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[36]:


from keras.metrics import Accuracy
# now check accuracy
classifier.compile(loss='binary_crossentropy',optimizer = 'Adam', metrics=['Accuracy'])  # Adam perform good for our gradient decent algorithm

history = classifier.fit(X_train_scaled,y_train,batch_size = 10,epochs=50, validation_split=0.2 )   # validation_split=0.2 mean seperate 20% customer out of avalible 10,000 customer

# output shown loss on training data with accuracy and validation loss and accuracy for 20% testing data ie 0.2 we taken earlier


# In[37]:


classifier.layers[0].get_weights()


# In[38]:


classifier.layers[1].get_weights()


# In[40]:


y_log= classifier.predict(X_test_scaled)  # y_log is just name of varriable

y_pred= np.where(y_log>0.5,1,0)


# In[41]:


# Now check accuracy of model

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

# accuracy shown reduce due to overfitting of model but we need more accuracy


# In[42]:


import matplotlib.pyplot as plt


# In[43]:


acc=history.history['loss']
val_acc=history.history['Accuracy']
loss=history.history['val_loss']
val_loss=history.history['val_Accuracy']
# so history dictionary created
# out shown training loss , training Accuracy, validation loss, validation accuracy


# In[44]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])


# In[45]:


plt.plot(history.history['Accuracy'])

plt.plot(history.history['val_Accuracy'])


# In[46]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


# In[47]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


# In[48]:


cm = confusion_matrix(y_test,y_pred)


# In[49]:


cm


# In[50]:


accuracy = accuracy_score(y_test,y_pred)


# In[51]:


accuracy


# In[52]:


plt.figure(figsize = (10,7))
sns.heatmap(cm,annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[53]:


print(classification_report(y_test,y_pred))

#Precision of the model is 83 %. It looks good on paper but we should easily be able to get 100% with a more complex model.


# In[ ]:




