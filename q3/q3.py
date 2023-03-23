#!/usr/bin/env python
# coding: utf-8

# # Q3

# In[1]:


import pandas as pd


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


pbook = pd.read_csv("prog_book.csv")


# In[4]:


pbook.head()


# In[5]:


pbook.shape


# ## Task 1: Univariate Outlier detection (4 points)
# Box plots are efficient in identifying the outliers in numerical values. For this dataset, use the IQR method to identify any univariate outliers in this dataset.  Plot the outliers in the context of the feature's distribution for all the four numerical features (Rating, reviews, number of pages, price).
# Your output should be the box plots of each feature.
# 

# In[6]:


pbook.Reviews.unique() # they're strings


# In[7]:


import locale


# In[8]:


locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


# In[9]:


locale.atof('3,208')


# In[10]:


pbook['Reviews'] = [locale.atof(pbook.Reviews[num]) if "," in pbook.Reviews[num] else float(pbook.Reviews[num]) for num in range(len(pbook.Reviews))]


# In[11]:


# Plot the outliers in the context of the feature's distribution
boxplot = pbook.boxplot(column=['Rating', 'Reviews', 'Number_Of_Pages', 'Price'])  


# ### Rating

# In[12]:


plt.hist(pbook.Rating);


# In[13]:


pbook.Rating.describe()


# In[14]:


# use the IQR method to identify any univariate outliers in this dataset.
iqr = 4.25 - 3.915
low_thresh = 3.915 - 1.5*iqr
high_thresh = 4.25 + 1.5*iqr


# In[15]:


pbook.loc[pbook.Rating < low_thresh] # low outliers


# In[16]:


pbook.loc[pbook.Rating > high_thresh] # high outliers


# In[17]:


rating = pbook.boxplot(['Rating']) # visualized


# ### Reviews

# In[18]:


plt.hist(pbook.Reviews);


# In[19]:


pbook['Reviews'].describe()


# In[20]:


iqr = 116.5 - 5.5
low_thresh = 5.5 - 1.5*iqr
high_thresh = 116.5 + 1.5*iqr


# In[21]:


pbook.loc[pbook['Reviews'] < low_thresh] # none


# In[22]:


pbook.loc[pbook['Reviews'] > high_thresh] # lots


# In[23]:


reviews = pbook.boxplot(['Reviews'])


# ### Number of Pages

# In[24]:


plt.hist(pbook.Number_Of_Pages);


# In[25]:


pbook['Number_Of_Pages'].describe()


# In[26]:


iqr = 572.5 - 289
low_thresh = 289 - 1.5*iqr
high_thresh = 572.5 + 1.5*iqr


# In[27]:


pbook.loc[pbook['Number_Of_Pages'] < low_thresh] # none again


# In[28]:


pbook.loc[pbook['Number_Of_Pages'] > high_thresh] # lots again


# In[29]:


pages = pbook.boxplot(['Number_Of_Pages'])


# ### Price

# In[30]:


plt.hist(pbook.Price);


# In[31]:


pbook['Price'].describe()


# In[32]:


iqr = 67.854412 - 30.751471
low_thresh = 30.751471 - 1.5*iqr
high_thresh = 67.854412 + 1.5*iqr


# In[33]:


pbook.loc[pbook['Price'] < low_thresh] # none again


# In[34]:


pbook.loc[pbook['Price'] > high_thresh] # lots again


# In[35]:


price = pbook.boxplot(['Price'])


# ## Task 2: Multivariate Outlier detection (6 points)
# Lets use the DBSCAN Method for multivariate outlier detection. For this exercise, use the numerical and categorical columns to fit DBSCAN.
# Your first task is to perform a bivariate analysis on all possible pairs of the above features and identify any outliers. The output should be all the DBSCAN plots and the outlier data row(index&value) for each combination.

# In[36]:


pbook.columns


# In[37]:


pbook.shape


# In[38]:


from sklearn.cluster import DBSCAN


# In[39]:


from sklearn.neighbors import NearestNeighbors as NN


# In[40]:


from sklearn.preprocessing import StandardScaler


# In[41]:


# if don't use this, scale of features might affect how distance is measured in outlier detection 
scaler = StandardScaler()


# ### Rating x Reviews

# In[42]:


nei = NN(n_neighbors = 2)
nn = nei.fit(pbook[['Rating', 'Reviews']])
dist, ind = nn.kneighbors(pbook[['Rating', 'Reviews']])


# In[43]:


import numpy as np


# In[44]:


dist = np.sort(dist, axis=0)
dist = dist[:,1]
plt.plot(dist);


# In[45]:


dist[250:]


# In[46]:


# min_samples should be >= dimension of dataset + 1, generally 2x dim
clustering = DBSCAN(eps=350, min_samples = 14).fit(pbook[['Rating', 'Reviews']])


# In[47]:


plt.scatter(pbook["Rating"], pbook["Reviews"], c = clustering.labels_); # outliers are in purple


# In[48]:


clustering.labels_ # Noisy samples are given the label -1. 0's are core points


# In[49]:


# outlier data row(index&value)
print(pbook.loc[clustering.labels_ == -1, ['Rating','Reviews']])


# ### Rating x Number_Of_Pages

# In[50]:


nei = NN(n_neighbors = 2)
nn = nei.fit(pbook[['Rating', 'Number_Of_Pages']])
dist, ind = nn.kneighbors(pbook[['Rating', 'Number_Of_Pages']])


# In[51]:


dist = np.sort(dist, axis=0)
dist = dist[:,1]
plt.plot(dist);


# In[52]:


dist[260:]


# In[53]:


# min_samples should be >= dimension of dataset + 1, generally 2x dim
clustering = DBSCAN(eps=56, min_samples = 14).fit(pbook[['Rating', 'Number_Of_Pages']])


# In[54]:


plt.scatter(pbook["Rating"], pbook["Number_Of_Pages"], c = clustering.labels_);


# In[55]:


# outlier data row(index&value)
print(pbook.loc[clustering.labels_ == -1, ['Rating','Number_Of_Pages']])


# ### Rating x Type

# In[56]:


# need to preprocess categorical variable
pbook.Type.unique()


# In[57]:


pbook['Type_Dummies'] = pbook["Type"].replace({"Paperback": 0, "Hardcover": 1, 'Boxed Set - Hardcover':2, 'Kindle Edition':3, "ebook":4, 'Unknown Binding':5})


# In[58]:


pbook.head()


# In[152]:


nei = NN(n_neighbors = 2)
nn = nei.fit(scaler.fit_transform(pbook[['Rating', 'Type_Dummies']]))
dist, ind = nn.kneighbors(scaler.fit_transform(pbook[['Rating', 'Type_Dummies']]))


# In[153]:


dist = np.sort(dist, axis=0)
dist = dist[:,1]
plt.plot(dist);


# In[166]:


dist[250:]


# In[163]:


clustering = DBSCAN(eps=0.6, min_samples = 2).fit(scaler.fit_transform(pbook[['Rating', 'Type_Dummies']]))


# In[164]:


plt.scatter(pbook["Rating"], pbook["Type_Dummies"], c = clustering.labels_);


# In[165]:


# outlier data row(index&value)
print(pbook.loc[clustering.labels_ == -1, ['Rating','Type_Dummies','Type']])


# ### Rating x Price

# In[65]:


nei = NN(n_neighbors = 2)
nn = nei.fit(scaler.fit_transform(pbook[['Rating', 'Price']]))
dist, ind = nn.kneighbors(scaler.fit_transform(pbook[['Rating', 'Price']]))


# In[66]:


dist = np.sort(dist, axis=0)
dist = dist[:,1]
plt.plot(dist);


# In[67]:


dist[245:]


# In[68]:


clustering = DBSCAN(eps=0.5, min_samples = 14).fit(scaler.fit_transform(pbook[['Rating', 'Price']]))


# In[69]:


plt.scatter(pbook["Rating"], pbook["Price"], c = clustering.labels_);


# In[70]:


# outlier data row(index&value)
print(pbook.loc[clustering.labels_ == -1, ['Rating','Price']])


# ### Reviews x Number_Of_Pages

# In[71]:


nei = NN(n_neighbors = 2)
nn = nei.fit(scaler.fit_transform(pbook[['Reviews', 'Number_Of_Pages']]))
dist, ind = nn.kneighbors(scaler.fit_transform(pbook[['Reviews', 'Number_Of_Pages']]))


# In[72]:


dist = np.sort(dist, axis=0)
dist = dist[:,1]
plt.plot(dist);


# In[73]:


dist[250:]


# In[74]:


clustering = DBSCAN(eps=1, min_samples = 14).fit(scaler.fit_transform(pbook[['Reviews', 'Number_Of_Pages']]))


# In[75]:


plt.scatter(pbook["Reviews"], pbook["Number_Of_Pages"], c = clustering.labels_);


# In[76]:


# outlier data row(index&value)
print(pbook.loc[clustering.labels_ == -1, ['Reviews','Number_Of_Pages']])


# ### Reviews x Type

# In[77]:


nei = NN(n_neighbors = 2)
nn = nei.fit(pbook[['Reviews', 'Type_Dummies']])
dist, ind = nn.kneighbors(pbook[['Reviews', 'Type_Dummies']])


# In[78]:


dist = np.sort(dist, axis=0)
dist = dist[:,1]
plt.plot(dist);


# In[79]:


dist[250:]


# In[80]:


clustering = DBSCAN(eps=1300, min_samples = 14).fit(pbook[['Reviews', 'Type_Dummies']])


# In[81]:


plt.scatter(pbook["Reviews"], pbook["Type_Dummies"], c = clustering.labels_);


# In[82]:


# outlier data row(index&value)
print(pbook.loc[clustering.labels_ == -1, ['Reviews','Type_Dummies','Type']])


# ### Reviews x Price

# In[83]:


nei = NN(n_neighbors = 2)
nn = nei.fit(pbook[['Reviews', 'Price']])
dist, ind = nn.kneighbors(pbook[['Reviews', 'Price']])


# In[84]:


dist = np.sort(dist, axis=0)
dist = dist[:,1]
plt.plot(dist);


# In[85]:


dist[250:]


# In[86]:


clustering = DBSCAN(eps=1300, min_samples = 14).fit(pbook[['Reviews', 'Price']])


# In[87]:


plt.scatter(pbook["Reviews"], pbook["Price"], c = clustering.labels_);


# In[88]:


# outlier data row(index&value)
print(pbook.loc[clustering.labels_ == -1, ['Reviews','Price']])


# ### Number_Of_Pages x Type

# In[89]:


nei = NN(n_neighbors = 2)
nn = nei.fit(pbook[['Number_Of_Pages', 'Type_Dummies']])
dist, ind = nn.kneighbors(pbook[['Number_Of_Pages', 'Type_Dummies']])


# In[90]:


dist = np.sort(dist, axis=0)
dist = dist[:,1]
plt.plot(dist);


# In[91]:


dist[250:]


# In[92]:


clustering = DBSCAN(eps=272, min_samples = 14).fit(pbook[['Number_Of_Pages', 'Type_Dummies']])


# In[93]:


plt.scatter(pbook["Number_Of_Pages"], pbook["Type_Dummies"], c = clustering.labels_);


# In[94]:


# outlier data row(index&value)
print(pbook.loc[clustering.labels_ == -1, ['Number_Of_Pages','Type_Dummies','Type']])


# ### Number_Of_Pages x Price

# In[95]:


nei = NN(n_neighbors = 2)
nn = nei.fit(pbook[['Number_Of_Pages', 'Price']])
dist, ind = nn.kneighbors(pbook[['Number_Of_Pages', 'Price']])


# In[96]:


dist = np.sort(dist, axis=0)
dist = dist[:,1]
plt.plot(dist);


# In[97]:


dist[250:]


# In[98]:


clustering = DBSCAN(eps=300, min_samples = 14).fit(pbook[['Number_Of_Pages', 'Price']])


# In[99]:


plt.scatter(pbook["Number_Of_Pages"], pbook["Price"], c = clustering.labels_);


# In[100]:


# outlier data row(index&value)
print(pbook.loc[clustering.labels_ == -1, ['Number_Of_Pages','Price']])


# Your second task is to look for all combinations of three variables in the above dataset to identify multivariate outliers. The output should again be all the dbscan plots(3D) and the outlier data row for all combinations.

# ### Rating x Reviews x Number_Of_Pages

# In[101]:


nei = NN(n_neighbors = 3)
nn = nei.fit(pbook[['Rating', 'Reviews', 'Number_Of_Pages']])
dist, ind = nn.kneighbors(pbook[['Rating', 'Reviews', 'Number_Of_Pages']])


# In[102]:


dist = np.sort(dist, axis=0)
dist = dist[:,1]
plt.plot(dist);


# In[103]:


dist[250:]


# In[104]:


clustering = DBSCAN(eps=350, min_samples = 14).fit(pbook[['Rating', 'Reviews', 'Number_Of_Pages']])


# In[105]:


fig = plt.figure(figsize = (10, 7))
ax = fig.add_subplot(projection='3d')
ax.scatter(pbook["Rating"], pbook["Reviews"], pbook["Number_Of_Pages"], c = clustering.labels_)
# ax.scatter3D

ax.set_xlabel('Rating')
ax.set_ylabel('Reviews')
ax.set_zlabel('Number_Of_Pages')

plt.show()


# In[106]:


print(pbook.loc[clustering.labels_ == -1, ['Rating','Reviews','Number_Of_Pages']])


# ### Rating x Reviews x Type

# In[107]:


nei = NN(n_neighbors = 3)
nn = nei.fit(pbook[['Rating', 'Reviews', 'Type_Dummies']])
dist, ind = nn.kneighbors(pbook[['Rating', 'Reviews', 'Type_Dummies']])


# In[108]:


dist = np.sort(dist, axis=0)
dist = dist[:,1]
plt.plot(dist);


# In[109]:


dist[250:]


# In[110]:


clustering = DBSCAN(eps=1201, min_samples = 14).fit(pbook[['Rating', 'Reviews', 'Type_Dummies']])


# In[111]:


fig = plt.figure(figsize = (10, 7))
ax = fig.add_subplot(projection='3d')
ax.scatter(pbook["Rating"], pbook["Reviews"], pbook["Type_Dummies"], c = clustering.labels_)
# ax.scatter3D

ax.set_xlabel('Rating')
ax.set_ylabel('Reviews')
ax.set_zlabel('Type_Dummies')

plt.show()


# In[112]:


print(pbook.loc[clustering.labels_ == -1, ['Rating','Reviews','Type']])


# ### Rating x Reviews x Price

# In[113]:


nei = NN(n_neighbors = 3)
nn = nei.fit(pbook[['Rating', 'Reviews', 'Price']])
dist, ind = nn.kneighbors(pbook[['Rating', 'Reviews', 'Price']])


# In[114]:


dist = np.sort(dist, axis=0)
dist = dist[:,1]
plt.plot(dist);


# In[115]:


dist[250:]


# In[116]:


clustering = DBSCAN(eps=1201, min_samples = 14).fit(pbook[['Rating', 'Reviews', 'Price']])


# In[117]:


fig = plt.figure(figsize = (10, 7))
ax = fig.add_subplot(projection='3d')
ax.scatter(pbook["Rating"], pbook["Reviews"], pbook["Price"], c = clustering.labels_)
# ax.scatter3D

ax.set_xlabel('Rating')
ax.set_ylabel('Reviews')
ax.set_zlabel('Price')

plt.show()


# In[118]:


print(pbook.loc[clustering.labels_ == -1, ['Rating','Reviews','Price']])


# ###  Reviews x Number_Of_Pages x Type

# In[119]:


nei = NN(n_neighbors = 3)
nn = nei.fit(pbook[['Number_Of_Pages', 'Reviews', 'Type_Dummies']])
dist, ind = nn.kneighbors(pbook[['Number_Of_Pages', 'Reviews', 'Type_Dummies']])


# In[120]:


dist = np.sort(dist, axis=0)
dist = dist[:,1]
plt.plot(dist);


# In[121]:


dist[250:]


# In[122]:


clustering = DBSCAN(eps=342, min_samples = 14).fit(pbook[['Number_Of_Pages', 'Reviews', 'Type_Dummies']])


# In[123]:


fig = plt.figure(figsize = (10, 7))
ax = fig.add_subplot(projection='3d')
ax.scatter(pbook["Number_Of_Pages"], pbook["Reviews"], pbook["Type_Dummies"], c = clustering.labels_)
# ax.scatter3D

ax.set_xlabel('Number_Of_Pages')
ax.set_ylabel('Reviews')
ax.set_zlabel('Type_Dummies')

plt.show()


# In[124]:


print(pbook.loc[clustering.labels_ == -1, ['Number_Of_Pages', 'Reviews', 'Type']])


# ###  Reviews x Number_Of_Pages x Price

# In[125]:


nei = NN(n_neighbors = 3)
nn = nei.fit(pbook[['Number_Of_Pages', 'Reviews', 'Price']])
dist, ind = nn.kneighbors(pbook[['Number_Of_Pages', 'Reviews', 'Price']])


# In[126]:


dist = np.sort(dist, axis=0)
dist = dist[:,1]
plt.plot(dist);


# In[127]:


dist[250:]


# In[128]:


clustering = DBSCAN(eps=342, min_samples = 14).fit(pbook[['Number_Of_Pages', 'Reviews', 'Price']])


# In[129]:


fig = plt.figure(figsize = (10, 7))
ax = fig.add_subplot(projection='3d')
ax.scatter(pbook["Number_Of_Pages"], pbook["Reviews"], pbook["Price"], c = clustering.labels_)
# ax.scatter3D

ax.set_xlabel('Number_Of_Pages')
ax.set_ylabel('Reviews')
ax.set_zlabel('Price')

plt.show()


# In[130]:


print(pbook.loc[clustering.labels_ == -1, ['Price','Reviews','Number_Of_Pages']])


# ###  Number_Of_Pages x Type x Price

# In[131]:


nei = NN(n_neighbors = 3)
nn = nei.fit(pbook[['Number_Of_Pages', 'Type_Dummies', 'Price']])
dist, ind = nn.kneighbors(pbook[['Number_Of_Pages', 'Type_Dummies', 'Price']])


# In[132]:


dist = np.sort(dist, axis=0)
dist = dist[:,1]
plt.plot(dist);


# In[133]:


dist[250:]


# In[134]:


clustering = DBSCAN(eps=300, min_samples = 14).fit(pbook[['Number_Of_Pages', 'Type_Dummies', 'Price']])


# In[135]:


fig = plt.figure(figsize = (10, 7))
ax = fig.add_subplot(projection='3d')
ax.scatter(pbook["Number_Of_Pages"], pbook["Type_Dummies"], pbook["Price"], c = clustering.labels_)
# ax.scatter3D

ax.set_xlabel('Number_Of_Pages')
ax.set_ylabel('Type_Dummies')
ax.set_zlabel('Price')

plt.show()


# In[136]:


print(pbook.loc[clustering.labels_ == -1, ['Type','Price','Number_Of_Pages']])


# # Turning it in
# 1. A Jupyter Notebook(q3.jpynb) with all code, output and graphs.
# 2. Please export your q3.jpynb to a q3.py and upload this q3.py as well. We will need to run a plagiarism check on the code. (do not zip these two files)
# 

# In[ ]:




