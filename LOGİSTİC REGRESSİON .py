#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import pandas as pd 
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.preprocessing import LabelEncoder 


# In[21]:


import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# In[35]:


df = pd.read_csv('gender.csv')


# In[36]:


le=LabelEncoder()
df["FavoriteColor"]= le.fit_transform(df["FavoriteColor"])


# In[37]:


df["FavoriteMusicGenre"]= le.fit_transform(df["FavoriteMusicGenre"])


# In[38]:


df["FavoriteBeverage"]= le.fit_transform(df["FavoriteBeverage"])


# In[39]:


df["FavoriteSoftDrink"]= le.fit_transform(df["FavoriteSoftDrink"])


# In[41]:


df["Gender"]= le.fit_transform(df["Gender"])


# In[42]:


y= df["Gender"] #bağımlıdeğisken


# In[47]:


X = df.drop(["Gender"], axis=1)#bağımsızdeğiskenler


# In[44]:


df["Gender"].value_counts()


# In[45]:


df.info()


# In[46]:


X.head()


# In[48]:


y.head()


# In[53]:


loj_model = LogisticRegression(solver="liblinear").fit(X,y)


# In[54]:


loj_model.intercept_


# In[56]:


loj_model.coef_


# In[59]:


loj_model.predict(X)[0:10]


# In[60]:


y[0:10]


# In[61]:


y_pred = loj_model.predict(X)


# In[62]:


confusion_matrix(y,y_pred)


# In[64]:


accuracy_score(y,y_pred) #doğruluk oranı


# # model doğrulama

# In[124]:


X_train , X_test , y_train, y_test =train_test_split( X,y,test_size=0.9, random_state=10)


# In[125]:


loj_model = LogisticRegression(solver="liblinear").fit(X_train,y_train)


# In[126]:


y_pred= loj_model.predict(X_test)


# In[129]:


accuracy_score(y_test , y_pred)

