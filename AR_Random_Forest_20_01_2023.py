#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Importing of Processed Data

# In[2]:


X_test = pd.read_csv(r'C:\Users\ALLRUSSELL\OneDrive - Deloitte (O365D)\Kaggle sprint\Processed data\X_test.csv')

print(f'Dimention of raw data is {X_test.shape}')


# In[3]:


X_test.head() # to display the first 5 lines of loaded data


# In[4]:


X_train = pd.read_csv (r'C:\Users\ALLRUSSELL\OneDrive - Deloitte (O365D)\Kaggle sprint\Processed data\X_train.csv')
print(f'Dimention of raw data is {X_train.shape}')


# In[5]:


y_test = pd.read_csv(r"C:\Users\ALLRUSSELL\OneDrive - Deloitte (O365D)\Kaggle sprint\Processed data\y_test.csv")
print(f'Dimention of raw data is {y_test.shape}')


# In[6]:


y_train = pd.read_csv(r'C:\Users\ALLRUSSELL\OneDrive - Deloitte (O365D)\Kaggle sprint\Processed data\y_train.csv')
print(f'Dimention of raw data is {y_train.shape}')


# Importing of Random forest

# In[7]:


from sklearn.ensemble import RandomForestClassifier
rfc= RandomForestClassifier(n_estimators=100, bootstrap = True, max_features = 'sqrt')


# In[8]:


rfc.fit(X_train, y_train.values.ravel())


# In[9]:


y_pred = rfc.predict(X_test)


# In[10]:


print(y_pred)


# In[11]:


check = sum(y_pred)
print(check)


# In[12]:


y_probs = rfc.predict_proba(X_test)[:,1]


# In[13]:


from sklearn.metrics import roc_auc_score

# Calculate roc auc
roc_value = roc_auc_score(y_test, y_probs)


# In[14]:


print(roc_value)


# In[15]:


from imblearn.ensemble import BalancedRandomForestClassifier


# In[16]:


brf = BalancedRandomForestClassifier(n_estimators=100, bootstrap = True, max_features = 'sqrt')
brf.fit(X_train, y_train.values.ravel())
y_pred2 = brf.predict(X_test)


# In[17]:


print(y_pred2)


# In[18]:


Check2 = sum(y_pred2)
print(Check2)


# In[19]:


y_probs2 = brf.predict_proba(X_test)[:,1]
roc_value2 = roc_auc_score(y_test, y_probs2)
print(roc_value2)


# In[ ]:




