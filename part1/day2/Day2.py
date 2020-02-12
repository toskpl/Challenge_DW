
# coding: utf-8

# In[19]:

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# tekst

# In[9]:

np.random.seed(2018);
np.random.rand(4,5)


# In[33]:

x = np.linspace(-10,10,50)
y = np.sin(x)
z = np.cos(x)
k = np.tan(x)

plt.plot(x,y);
plt.plot(x,z);
plt.plot(x,k);


# In[34]:

1+1

