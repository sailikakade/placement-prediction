#!/usr/bin/env python
# coding: utf-8

# # PLACEMENT PACKAGE PREDICTION USING PAST DATA(DIPLOMA) UPDATED

# In[1]:


import numpy as np
import pickle
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import pylab as pl


# In[2]:


dataframe0= pd.read_csv("C:\\Users\\abc\\Desktop\\SAILI\\PROJECT\\SEM 8\\updated\\Deployment-flask-master\\Batch 2020 (Diploma).csv")


# In[3]:


dataframe0.head()


# # DATA PREPROCESSING

# In[4]:


data0=dataframe0.drop(['First Name','Last Name','College Roll No.','Percentage (XII)','B.E. Sem 1 CGPA','B.E. Sem 2 CGPA','Department','Contact No.','Email address','1st Offer'],axis=1)


# In[5]:


data0.head()


# In[6]:


#Converting words to integer values
def convert_to_int(word):
    word_dict = {'Yes':1, 'No':0}
    return word_dict[word]

data0['Worked as Resource Person for Workshop'] = data0['Worked as Resource Person for Workshop'].apply(lambda data0 : convert_to_int(data0))


# In[7]:


data0


# In[8]:


data0.shape


# In[9]:


#Finding Median of attribute Package of Regular Dataset
data0["Package"].median()


# In[10]:


data0.fillna(data0.median())


# In[11]:


data0['Package'] = data0['Package'].fillna(0)


# In[12]:


data0['Package'] = data0['Package'].astype(float)


# In[13]:


data0


# In[14]:


data0 = data0.astype(str)


# #  Decision Tree Regression

# In[15]:


X = data0.iloc[:, :-1]
y = data0['Package']


# In[16]:


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0).fit(X, y)


# In[17]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[18]:


regressor.score(X_test,y_test)


# In[19]:


lr=DecisionTreeRegressor()
lr.fit(X_train,y_train)


# In[20]:


pickle.dump(regressor, open('model2.pkl','wb'))


# In[21]:


model2 = pickle.load(open('model2.pkl','rb'))

