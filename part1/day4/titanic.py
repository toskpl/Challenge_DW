#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import bibliotek
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier


# In[2]:


#dane Titanic
df = pd.read_csv('http://raw.githubusercontent.com/dataworkshop/titanic/master/vladimir/input/train.csv')
df.head()


# In[3]:


#zbior cech
feats = ['Pclass', 'Fare']
X = df[feats].values
y = df['Survived'].values

#zbior testowy , treningowy
X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.3)
X_train.shape,X_test.shape


# In[4]:


#model Dummy
model = DummyClassifier()
model.fit (X_train, y_train)
y_pred = model.predict(X_test)
accuracy_score(y_test,y_pred)


# In[5]:


#model DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=20)
model.fit (X_train, y_train)
y_pred = model.predict(X_test)
accuracy_score(y_test,y_pred)


# In[6]:


#model RandomForestClassifier
model = RandomForestClassifier(max_depth=10, n_estimators=50)
model.fit (X_train, y_train)
y_pred = model.predict(X_test)
accuracy_score(y_test,y_pred)


# In[7]:


model = RandomForestClassifier(max_depth=10, n_estimators=50)
model.fit (X_train, y_train)
y_pred = model.predict(X_test)
accuracy_score(y_test,y_pred)


# In[8]:


#factorize zamiana ciagu znakow na liczby np  male = 0, female = 1
df['Sex_cat'] = df['Sex'].factorize()[0]


# In[9]:


#zbior cech o nową ceche Sex_cat
feats = ['Pclass', 'Fare','Sex_cat']
X = df[feats].values
y = df['Survived'].values

#zbior testowy , treningowy
X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.3)
X_train.shape,X_test.shape


# In[10]:


#model DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=20)
model.fit (X_train, y_train)
y_pred = model.predict(X_test)
accuracy_score(y_test,y_pred)


# In[ ]:




