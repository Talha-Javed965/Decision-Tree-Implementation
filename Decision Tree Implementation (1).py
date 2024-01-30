#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# In[5]:


# Reading data from github 
url = 'https://raw.githubusercontent.com/mGalarnyk/Tutorial_Data/master/King_County/kingCountyHouseData.csv'
df = pd.read_csv(url)
# Selecting columns I am interested in
columns = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','price']
df = df.loc[:, columns]
df.head(10)


# In[6]:


# Creating Feature and Target Variables for further processing 
features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors']
X = df.loc[:, features]
y = df.loc[:, ['price']]


# In[7]:


# Splitting the dataset into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size = .75)


# In[8]:


# Creating the instance of Decision Tree Model
reg = DecisionTreeRegressor(max_depth = 2, random_state = 0)


# In[9]:


# Training the dataset on this model
reg.fit(X_train, y_train)


# In[10]:


# Testing the model
reg.predict(X_test[0:10])


# In[11]:


# Assessing the performance of the model
score = reg.score(X_test, y_test)
print(score)


# In[12]:


from sklearn.ensemble import RandomForestClassifier


# In[40]:



forest = RandomForestClassifier(n_estimators=50,max_depth= 10,random_state=0).fit(X_train,y_train )


# In[41]:


forest.predict(X_test[0:10])


# In[43]:


score = forest.score(X_test, y_test)
print(score)


# ## https://builtin.com/data-science/train-test-split

# ## the above link can be used to understand the code in detail
# 

# In[ ]:




