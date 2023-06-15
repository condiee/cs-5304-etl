#!/usr/bin/env python
# coding: utf-8

# ## CS/INFO 5304 Assignment 1: Data Preparation <br />
# 

# In[1]:


import pandas as pd # all the functions in the panda library are allowed


# In[2]:


# Numpy, Scipy.stats, .etc are also allowed.
import matplotlib.pyplot as plt


# In[3]:


ind = pd.read_pickle("Extrasensory_individual_data.p")


# In[4]:


sen = pd.read_pickle("Extrasensory_sensor_data.p")


# ### Case 1: Actual screen time
# 

# In[5]:


## Case 1 Problem A code (and optional graph)
ind.head()


# In[6]:


ind.columns # DataFrame with the participant ID and columns about demographics/phone usage


# In[7]:


ind.shape


# In[8]:


len(sen) # dict of the modified dataset with participant ID as the key
# There are 60 participants in this dataset. 


# In[9]:


ind.isna().any()


# In[10]:


ind.actual_average_screen_time.describe()


# #### Writeup Answer to Problem A: 
# How are missing values represented for this feature?
# 

# *ANSWER:* Missing values are represented for actual_average_screen_time with a value of -1, which is possible here since the real value would have to be non-negative.

# In[11]:


## Case 1 Problem B code and graph
plt.hist(ind.actual_average_screen_time, bins=60);


# In[12]:


no_nulls = ind[ind.actual_average_screen_time>=0]
no_nulls.actual_average_screen_time.describe()


# In[13]:


plt.hist(no_nulls.actual_average_screen_time, bins=len(no_nulls));


# In[106]:


# An outlier is defined as being any point of data that lies over 1.5 IQRs
# below the first quartile (Q1) or above the third quartile (Q3) in a data set

# IQR = 75th percentile - 25th percentile
iqr = 4.5 - 3.175
iqr


# In[105]:


# no low outliers (magic percentile numbers are from .describe() above)
3.175000 - 1.5*iqr


# In[108]:


no_nulls.loc[no_nulls.actual_average_screen_time < 1.1874999999999996] # 2


# In[107]:


# high outliers
4.56 + 1.5*iqr


# In[109]:


no_nulls.loc[no_nulls.actual_average_screen_time > 6.547499999999999] # 2


# In[18]:


# skewness
from scipy.stats import skew


# In[19]:


# For normally distributed data, the skewness should be about zero. 
skew(no_nulls.actual_average_screen_time)
# a skewness value greater than zero means that there is more weight
# in the right tail of the distribution.


# #### Writeup Answer to Problem B: 
#  Does it have outliers?If so, how many?
#  Is it skewed? If so, is it left skewed or right skewed? What’s the skewness? 

# *ANSWER:* The distribution is skewed slightly right (positively); the tail is on the right and there are also two outliers on the high end.

# In[20]:


## Case 1 Problem C code and graph
# method 1: mean
imp_mean = ind.copy()


# In[21]:


imp_mean.loc[imp_mean.actual_average_screen_time < 0, "actual_average_screen_time"] = imp_mean.actual_average_screen_time.mean()


# In[22]:


imp_mean.actual_average_screen_time.min()


# In[23]:


# method 2: median
imp_med = ind.copy()
imp_med.loc[imp_med.actual_average_screen_time < 0, "actual_average_screen_time"] = imp_med.actual_average_screen_time.median()
imp_med.actual_average_screen_time.min()


# In[24]:


# method 3: random value within range
no_nulls.actual_average_screen_time.describe()


# In[25]:


import random


# In[26]:


imp_rand = ind.copy()
imp_rand.loc[imp_rand.actual_average_screen_time < 0, "actual_average_screen_time"] = random.randint(3,5)
imp_rand.actual_average_screen_time.min()


# In[27]:


plt.figure(figsize=(15,8))
plt.hist([ind.actual_average_screen_time, imp_mean.actual_average_screen_time, imp_med.actual_average_screen_time, imp_rand.actual_average_screen_time],bins=60,label=['original','mean', 'median', 'random']);
plt.legend(loc='upper right');


# #### Writeup Answer to Problem C: 
# How did you choose the random value from method 3)? 
# How do the distributions look like after you implement the three filling methods? (Compare them)
# 

# In[28]:


import numpy as np


# In[29]:


## Case 1 Problem D code and graph
dist = np.random.normal(3.75, 1.25, len(ind))


# In[30]:


plt.hist(dist, bins=60);


# In[31]:


from scipy.stats import ttest_ind


# In[32]:


#mean
ttest_ind(dist, imp_mean.actual_average_screen_time)


# In[33]:


#median
ttest_ind(dist, imp_med.actual_average_screen_time)


# In[34]:


#random
ttest_ind(dist, imp_rand.actual_average_screen_time)


# #### Answer to Problem D: 
# Report the three p-values. Which one of the filling methods reconstruct this feature to be closest to the research distribution? Why do you think this is the case?
# 

# *ANSWER:* The p-value for the t-test between the randomly generated distribution and the mean-imputed data is 0.06 > 0.05. The one for the median-imputed data is also 0.06. But the one for the randomly imputed data is 0.03<0.05 and therefore statistically significant, indicating that it is closest in resembling the research distribution (they could be drawn from the same distribution instead of by chance). This makes sense since both are randomly generated and therefore more representative of real data than when we simply fill all missing values with the same number hoping the missing values were around the average or median value.

# ### Case 2: Perceived average screen time

# In[87]:


## Case 2 Problem A code and histogram
no_missing = ind[ind.perceived_average_screen_time>=0]
plt.hist(no_missing.perceived_average_screen_time, bins=60);


# In[88]:


no_missing.perceived_average_screen_time.describe()


# In[110]:


# IQR = 75th percentile - 25th percentile
iqr = 4.2 - 3.1
iqr


# In[111]:


# low outliers
3.1 - 1.5 * iqr


# In[112]:


no_missing[no_missing.perceived_average_screen_time<=1.45]


# In[113]:


# high outliers
4.2 + 1.5 * iqr


# In[114]:


no_missing[no_missing.perceived_average_screen_time>=5.8500000000000005]


# In[93]:


skew(no_missing.perceived_average_screen_time)
# a skewness value greater than zero means that there is more weight
# in the right tail of the distribution.


# #### Writeup Answer to Problem B: 
#  Does it have outliers?If so, how many? 
#  Is it skewed? If so, is it left skewed or right skewed? What’s the skewness? 
# 

# *ANSWER:* There are 2 outliers, one low and one high. The distribution is skewed very slightly to the left, with a skewness of -0.2.

# In[144]:


## Case 2 Problem B code
no_nulls[no_nulls.actual_average_screen_time > (no_nulls.actual_average_screen_time.mean() + no_nulls.actual_average_screen_time.std())]
# use no_nulls again for rows where actual_average_screen_time is not null


# How many of them are intense phone users?

# *ANSWER:* 4 individuals are "intense phone users" according to their actual average screen time when looking at individuals for whom we have actual screen time data.

# In[165]:


## Case 2 Problem C code and graph
no_nulls['missing'] = [True if time < 0 else False for time in no_nulls.perceived_average_screen_time]


# In[167]:


# If the user’s actual screen time is missing, you should not count that user as either intensive OR NON-INTENSIVE
no_nulls['intense'] = [True if screen_time > (no_nulls.actual_average_screen_time.mean() + no_nulls.actual_average_screen_time.std()) else False for screen_time in no_nulls.actual_average_screen_time]


# In[152]:


# chi-square test: compare observed results with expected results
from scipy.stats import chi2_contingency


# What is the p-value? Do you think they are correlated? What does this mean? Do you think this feature is MAR or MNAR? 

# In[173]:


# defining the table
cont_tbl = pd.crosstab(no_nulls['missing'], no_nulls['intense'])
stat, p, dof, expected = chi2_contingency(cont_tbl)
  
print("p value is ", p)
if p <= 0.05:
    print('Variables are dependent (reject null hypothesis)')
else:
    print('Variables are independent (null hypothesis holds true)')


# *ANSWER:* 
# 
# It seems like missing values for perceived screen time are not correlated with intense phone users. Therefore, I think that perceived average screen time is Missing At Random, meaning that the cause of the missing data is unrelated to the missing values of this variable (ex. people are *not* necessarily embarrassed by their screen time) but may be related to the observed values of other variables.

# ### Case 3: Location

# In[38]:


## Case 3 Problem A code (graph)

# location data (location:raw_latitude) are consistently lost when their
# battery level (lf_measurements:battery_level) is below 15%


# In[233]:


# example entry: participant at index 0
sen[ind['uuid'][0]]


# In[234]:


sen[ind['uuid'][0]]['location:raw_latitude'].isna().any()


# In[235]:


sen[ind['uuid'][0]].loc[sen[ind['uuid'][0]]['location:raw_latitude'].isna()]


# In[236]:


# check how battery level is recorded: 0-1
sen[ind['uuid'][0]]['lf_measurements:battery_level'].describe()


# In[245]:


sen[ind['uuid'][0]].loc[(sen[ind['uuid'][0]]['location:raw_latitude'].isna()) & (sen[ind['uuid'][0]]['lf_measurements:battery_level'] < 0.15)]


# In[250]:


len(sen[ind['uuid'][0]].loc[(sen[ind['uuid'][0]]['location:raw_latitude'].isna()) & (sen[ind['uuid'][0]]['lf_measurements:battery_level']< 0.15)])


# In[257]:


# identify all people who turn off location services to save battery when the battery level is low
keys = []
for person in sen.keys(): # for each of the 60 people
    low = len(sen[person].loc[sen[person]['lf_measurements:battery_level'] < 0.15])
    low_off = len(sen[person].loc[(sen[person]['location:raw_latitude'].isna()) & (sen[person]['lf_measurements:battery_level']< 0.15)])
    if low==low_off & low > 0: # record them if they show this behavior CONSISTENTLY
        keys += [person]


# In[258]:


len(keys)


# In[259]:


print(keys)


# In[261]:


# For each of these individuals, how many minutes of location data have we lost due to turning off location?
# The number of records corresponds to the number of minutes that the participant underwent the study
loss = {key: len(sen[key].loc[(sen[key]['location:raw_latitude'].isna()) & (sen[key]['lf_measurements:battery_level']< 0.15)]) for key in keys}


# In[262]:


loss


# **explanation of implementation:** Explain how you find these people in the Notebook. For each of these individuals, how many minutes of location data have we lost due to turning-off of location service?  

# *ANSWER:* I used a dictionary comprehension to filter the dictionary by my conditions and keep the entries that matched. I found 6 individuals who matched the conditions of always having no location data when their phone battery was below 15%. We lost 277 minutes of location data from the first participant out of the 6, 7 for the second, 104 for the third, 108 for the fourth, 182 for the fifth, and 4 for the sixth.

# In[263]:


## Case 3 Problem B code and graph
subject = 'F50235E0-DD67-4F2A-B00B-1F31ADA998B9'
sen[subject]


# In[264]:


sen[subject].columns


# In[267]:


sen[subject].loc[sen[subject]['location:raw_latitude'].isna()]


# In[309]:


# Forward filling: Assume the missing value occurs at tn, fill in the value using tn-1. 
forward = sen[subject].copy()

for entry in range(1, len(forward['location:raw_latitude'])):
    if pd.isna(forward['location:raw_latitude'][entry]):
        forward['location:raw_latitude'][entry] = forward['location:raw_latitude'][entry-1]


# In[310]:


forward['location:raw_latitude'].isna().any()


# In[321]:


# Backward filling: Assume the missing value occurs at tn, fill in the value using tn+1.
backward = sen[subject].copy()

for entry in range(len(backward['location:raw_latitude'])-1, 0, -1): # step back
    if pd.isna(backward['location:raw_latitude'][entry]):
        backward['location:raw_latitude'][entry] = backward['location:raw_latitude'][entry+1]
    
backward['location:raw_latitude'].isna().any()


# In[429]:


# Linear interpolation: For each CHUNK of missing values, perform linear interpolation with
# the value of one sample before the missing chunk, and one value after the missing chunk.

interp = sen[subject].copy()
interp['location:raw_latitude'].isna().sum()


# In[362]:


sen[subject]['location:raw_latitude'][0] # doesn't start null


# In[430]:


start = False # True when in middle of chunk of nulls
for current in range(1, len(interp['location:raw_latitude'])):
    if (start==False) & (pd.isna(interp['location:raw_latitude'][current])): # find first null value in a chunk
        start = current # save location
    if (start!=False) & (pd.isna(interp['location:raw_latitude'][current]) != True): # find next non-null value index
        steps = current - start # number of indices to step through
        if interp['location:raw_latitude'][current] > interp['location:raw_latitude'][start-1]: # step up
            rng = interp['location:raw_latitude'][current] - interp['location:raw_latitude'][start-1]
            inc = rng/steps
            new_val = interp['location:raw_latitude'][start-1] + inc
            for i in range(start,current): # loop through again to fill
                interp['location:raw_latitude'][i] = new_val
                new_val += inc # step up in equal increments
        if interp['location:raw_latitude'][current] < interp['location:raw_latitude'][start-1]: # step down
            rng = interp['location:raw_latitude'][start-1] - interp['location:raw_latitude'][current]
            inc = rng/steps
            new_val = interp['location:raw_latitude'][start-1] - inc
            for i in range(start,current):
                interp['location:raw_latitude'][i] = new_val
                new_val -= inc # step down in equal increments
        else: # they are the same
            for i in range(start,current):
                interp['location:raw_latitude'][i] = interp['location:raw_latitude'][current]
        
        start=False # reset for next chunk
        
interp['location:raw_latitude'].isna().sum() # yay


# In[435]:


# Produce a figure with 4 traces overlaid on each other
# (4 traces including the original, unfilled trace and trace produced by your three filling methods)

plt.figure(figsize=(15,8))
plt.plot(sen[subject]['location:raw_latitude'],label = "original", linestyle="-", alpha=0.7)
plt.plot(forward['location:raw_latitude'], label = "forward-filling", linestyle="--", alpha=0.7)
plt.plot(backward['location:raw_latitude'],label = "backward-filling", linestyle="-.", alpha=0.7)
plt.plot(interp['location:raw_latitude'], label = "linear interpolation", linestyle=":", alpha=0.7)
plt.legend(); # hard to see...


# In[440]:


# orig & forward-fill
plt.figure(figsize=(10,8))
plt.plot(sen[subject]['location:raw_latitude'],label = "original", linestyle="-", alpha=0.7, color = "yellow")
plt.plot(forward['location:raw_latitude'], label = "forward-filling", linestyle=":", alpha=0.7);


# In[441]:


# orig & backward-fill
plt.figure(figsize=(10,8))
plt.plot(sen[subject]['location:raw_latitude'],label = "original", linestyle="-", alpha=0.7, color = "yellow")
plt.plot(backward['location:raw_latitude'], label = "backward-filling", linestyle=":", alpha=0.7);


# In[442]:


# orig & linear interpolation
plt.figure(figsize=(10,8))
plt.plot(sen[subject]['location:raw_latitude'],label = "original", linestyle="-", alpha=0.7, color = "yellow")
plt.plot(interp['location:raw_latitude'], label = "linear interpolation", linestyle=":", alpha=0.7);


# In[443]:


# back & forward-fill: pretty much the same
plt.figure(figsize=(10,8))
plt.plot(backward['location:raw_latitude'],label = "backward-filling", linestyle="-", alpha=0.7, color = "yellow")
plt.plot(forward['location:raw_latitude'], label = "forward-filling", linestyle=":", alpha=0.7);


# In[444]:


# back & linear interp
plt.figure(figsize=(10,8))
plt.plot(backward['location:raw_latitude'],label = "backward-filling", linestyle="-", alpha=0.7, color = "yellow")
plt.plot(interp['location:raw_latitude'], label = "interpolation", linestyle=":", alpha=0.7);


# Compare the 4 traces. What do you see? If you were to use this dataset for further analysis, which filling method will you choose? 
# 

# *ANSWER:* They all look pretty good! If I were to use this dataset for further analysis, I would probably choose  linear interpolation just because it was fun to implement and seems like it would be the most accurate based on common sense.

# In[ ]:




