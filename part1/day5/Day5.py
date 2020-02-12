#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import bibliotek
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#wczytanie danych
df = pd.read_csv('https://raw.githubusercontent.com/aczepielik/KRKtram/master/reports/report_07-23.csv')
df.head()


# In[4]:


#przykaldowy trpid
df[df.tripId == 6351558574044883205]


# In[7]:


#rozklad wartosci delay
df.delay.value_counts()


# In[8]:


#normalizacja wartosci delay
df.delay.value_counts(normalize=True)


# In[9]:


#histogram
df.delay.hist(bins=15);


# In[10]:


#podstawowe informacje statystyczne
df.delay.describe()


# In[11]:


###########################################################################################################
#modele
X = df [['number']].values
y = df ['delay'].values

model = DecisionTreeRegressor(max_depth=10)
cross_val_score(model,X,y,cv=3,scoring='neg_mean_absolute_error')

scores = cross_val_score(model,X,y,cv=3,scoring='neg_mean_absolute_error')

np.mean(scores), np.std(scores)


# In[21]:


###########################################################################################################
#modele
X = df [['number','stop']].values
y = df ['delay'].values

model = DecisionTreeRegressor(max_depth=10)
cross_val_score(model,X,y,cv=3,scoring='neg_mean_absolute_error')

scores = cross_val_score(model,X,y,cv=3,scoring='neg_mean_absolute_error')

np.mean(scores), np.std(scores)


# In[34]:


###########################################################################################################
#opytmalizcja o:
# - delay na sek
# - direction

#df ['delay_secs'] = df ['delay'].map(lambda x: x*60)
df ['direction_cat'] = df ['direction'].factorize()[0]

X = df [['number','stop','direction_cat']].values
y = df ['delay'].values

model = DecisionTreeRegressor(max_depth=10)
cross_val_score(model,X,y,cv=3,scoring='neg_mean_absolute_error')

scores = cross_val_score(model,X,y,cv=3,scoring='neg_mean_absolute_error')

np.mean(scores), np.std(scores) 


# In[36]:


###########################################################################################################
#opytmalizcja o:
# - delay na sek
# - direction
# - vehicleId

df ['delay_secs'] = df ['delay'].map(lambda x: x*60)
df ['direction_cat'] = df ['direction'].factorize()[0]
df ['vehicleId'].fillna(-1, inplace = True)

X = df [['number','stop','direction_cat','vehicleId']].values
y = df ['delay_secs'].values

model = DecisionTreeRegressor(max_depth=10)
cross_val_score(model,X,y,cv=3,scoring='neg_mean_absolute_error')

scores = cross_val_score(model,X,y,cv=3,scoring='neg_mean_absolute_error')

np.mean(scores), np.std(scores)


# In[38]:


###########################################################################################################
#opytmalizcja o:
# - delay na sek
# - direction
# - vehicleId
# - seq_num

df ['delay_secs'] = df ['delay'].map(lambda x: x*60)
df ['direction_cat'] = df ['direction'].factorize()[0]
df ['vehicleId'].fillna(-1, inplace = True)
df ['seq_num'].fillna(-1, inplace = True)


X = df [['number','stop','direction_cat','vehicleId','seq_num']].values
y = df ['delay_secs'].values

model = DecisionTreeRegressor(max_depth=10)
cross_val_score(model,X,y,cv=3,scoring='neg_mean_absolute_error')

scores = cross_val_score(model,X,y,cv=3,scoring='neg_mean_absolute_error')

np.mean(scores), np.std(scores)


# In[51]:


###########################################################################################################
#opytmalizcja o:
# - delay na sek
# - direction
# - vehicleId
# - seq_num
# - number_direction_id
# -stop_direction_id

df ['delay_secs'] = df ['delay'].map(lambda x: x*60)
df ['direction_cat'] = df ['direction'].factorize()[0]
df ['vehicleId'].fillna(-1, inplace = True)
df ['seq_num'].fillna(-1, inplace = True)

def gen_id_num_direction(x):
	return '{} {}'.format( x['number'], x['direction'])

#df.apply(lambda x: '{} {}'.format( x[number], x[direction] ),axis =1).factorize[](0)
df ['number_direction_id'] = df.apply(gen_id_num_direction,axis =1).factorize()[0]


def gen_id_stop_direction(x):
	return '{} {}'.format( x['stop'], x['direction'])
#df.apply(lambda x: '{} {}'.format( x[stop], x[direction] ),axis =1).factorize[](0)
df ['stop_direction_id'] = df.apply(gen_id_stop_direction,axis =1).factorize()[0]


feats = [
'number',
'stop',
'direction_cat',
'vehicleId',
'seq_num',
'stop_direction_id',
'number_direction_id'
]
X = df [feats].values
y = df ['delay_secs'].values
#model = RandomForestRegressor(max_depth=10,n_estimators=50)
model = DecisionTreeRegressor(max_depth=10,random_state=0)

cross_val_score(model,X,y,cv=3,scoring='neg_mean_absolute_error')

cores = cross_val_score(model,X,y,cv=3,scoring='neg_mean_absolute_error')

np.mean(scores), np.std(scores)

