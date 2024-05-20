#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.diagnostic import linear_rainbow

import scipy.stats as stats
import statsmodels.formula.api as sfa
import statsmodels.api as sma
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
# VIF 
from statsmodels.stats.outliers_influence import variance_inflation_factor 

import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sma
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import roc_auc_score,roc_curve


# In[3]:


train=pd.read_csv('Train_NLP.csv')
test = pd.read_csv('Test_NLP.csv')
subm = pd.read_csv('sample submission_NLP.csv')


# In[4]:


train.head()


# In[5]:


combined = pd.concat([train,test],ignore_index=True)


# In[6]:


combined.head()


# In[7]:


combined.shape


# In[9]:


combined.info()


# In[11]:


# Convert Date into Datetime format
combined['Travel Date']= pd.to_datetime(combined['Travel Date'])


# In[13]:


combined['Date']=combined['Travel Date'].dt.day
combined['Month']=combined['Travel Date'].dt.month
combined['Year']=combined['Travel Date'].dt.year


# In[15]:


combined.head()


# In[16]:


# Lets analyse the target variable
sns.displot(x='Per Person Price',data=combined)


# In[ ]:


# Lets analyse the target variable with respect to Month


# In[20]:


max_month =combined.groupby('Month')['Per Person Price'].max()
max_month


# In[25]:


plt.plot(max_month.index,max_month,color='blue',marker='*')
plt.xlabel('Month')
plt.ylabel('Per Person Price')
plt.grid()
plt.show()


# In[26]:


# The peak months are 4,5,7,9,10


# In[29]:


combined['Peak_Month']=combined.Month.apply(lambda x:'peak_month' if x in[4,5,7,9,10]else 'normal_month')


# In[31]:


sns.boxplot(x='Peak_Month',y='Per Person Price',data=combined)


# In[32]:


plt.plot(max_month.index,max_month,color='blue',marker='*')
plt.xlabel('Day')
plt.ylabel('Per Person Price')
plt.grid()
plt.show()


# In[ ]:





# In[35]:


max_date =combined.groupby('Date')['Per Person Price'].max()
max_date


# In[36]:


plt.plot(max_date.index,max_date,color='blue',marker='*')
plt.xlabel('Date')
plt.ylabel('Per Person Price')
plt.grid()
plt.show()


# In[37]:


# Peak Dates = 8,10,11,13,15,19,22,24,27,29


# In[38]:


combined['Peak_Date']=combined.Month.apply(lambda x:'peak_date' if x in[8,10,11,13,15,19,22,24,27,29]else 'normal_date')


# In[39]:


sns.boxplot(x='Peak_Date',y='Per Person Price',data=combined)


# In[ ]:





# In[ ]:





# In[ ]:




