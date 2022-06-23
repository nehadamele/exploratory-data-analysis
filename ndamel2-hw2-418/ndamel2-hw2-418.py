#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
pd.set_option('display.max_columns', None)


# ### Going over the dataset to gain some insights

# In[2]:


crash_census_df = pd.read_csv('Traffic_Crashes_-_Crashes.csv')
crash_census_df.head(5)


# ##### Number of rows and columns in the dataset
# 

# In[3]:


crash_census_df.shape


# ##### Listening all the columns in the dataset
# 

# In[4]:


crash_census_df.columns


# ##### Summary statistics of the dataset
# 

# In[5]:


crash_census_df.describe()


# ### 1. The data set need cleaning. Decide what to do with missing values and extra attributes.

# ##### Listening all the columns containing missing values in the dataset
# 

# In[6]:


crash_census_df.isna().sum()


# ##### Finding missing data percent and listening all the columns where percent is greater than zero
# 

# In[7]:


missing_value = crash_census_df.isna().sum().sort_values(ascending=False)
missing_percent = (missing_value/len(crash_census_df))*100
missing_percent[missing_percent!=0]


# ###### Plotting missing percent to get the visual view
# 

# In[8]:


missing_percent[missing_percent!=0].plot(kind='bar')
plt.title("Missing data percent");


# ####  Looking at the visuals,  NOT_RIGHT_OF_WAY column have 95% of the values missing, thus analysis using NOT_RIGHT_OF_WAY column will distort finding, hence dropping this column
# 

# In[9]:


crash_census = crash_census_df.copy()
crash_census.drop(['NOT_RIGHT_OF_WAY'], axis = 1, inplace = True)
crash_census.shape


# ##### Filling the NaN values in INTERSECTION_RELATED_I with unknown
# 

# In[10]:


crash_census["INTERSECTION_RELATED_I"].fillna("unknown", inplace = True)
crash_census['INTERSECTION_RELATED_I'].value_counts()


# ##### Filling the NaN values in HIT_AND_RUN_I with unknown
# 

# In[11]:


crash_census["HIT_AND_RUN_I"].fillna("unknown", inplace = True)
crash_census['HIT_AND_RUN_I'].value_counts()


# ##### Checking unique values of INJURIES_REPORTED_NOT_EVIDENT
# 

# In[12]:


crash_census['INJURIES_REPORTED_NOT_EVIDENT'].value_counts()


# ##### Checking unique values of INJURIES_NON_INCAPACITATING
# 

# In[13]:


crash_census['INJURIES_NON_INCAPACITATING'].value_counts()


# ##### Checking unique values of INJURIES_INCAPACITATING
# 

# In[14]:


crash_census['INJURIES_INCAPACITATING'].value_counts()


# ##### Checking unique values of INJURIES_FATAL
# 

# In[15]:


crash_census['INJURIES_FATAL'].value_counts()


# ##### Checking unique values of INJURIES_TOTAL
# 

# In[16]:


crash_census['INJURIES_TOTAL'].value_counts()


# ##### Filling the NaN values in INJURIES_NON_INCAPACITATING,  INJURIES_INCAPACITATING,  INJURIES_FATAL, INJURIES_TOTAL with '0'
# 

# In[17]:


crash_census[['INJURIES_NON_INCAPACITATING', 'INJURIES_INCAPACITATING', 'INJURIES_FATAL', 'INJURIES_TOTAL']].fillna(value=0)


# ##### Dropping CRASH_RECORD_ID as it is not playing any major role in doing the analysis, its an extra attribute 
# 

# In[18]:


crash_census.drop(['CRASH_RECORD_ID'], axis = 1, inplace = True)
crash_census.shape


# #### Cleaing the data:
# The following columns had missing values "NOT_RIGHT_OF_WAY, INTERSECTION_RELATED_I, HIT_AND_RUN_I, INJURIES_REPORTED_NOT_EVIDENT, INJURIES_NON_INCAPACITATING, INJURIES_INCAPACITATING, INJURIES_FATAL, INJURIES_TOTAL"
# 
# ###### Dealing with missing values:
# 
#  -Filled the NaN values in INTERSECTION_RELATED_I, HIT_AND_RUN_I with __'unknown'__
#  
#  -Filled the NaN values in INJURIES_NON_INCAPACITATING,  INJURIES_INCAPACITATING,  INJURIES_FATAL, INJURIES_TOTAL with __'0'__
#  
#  -Dropped NOT_RIGHT_OF_WAY column since __95%__ of the values were missing
#  
#  -Dropped CRASH_RECORD_ID as it was an __extra attribute__

# ### 2. Some attributes are more useful if you break them into several attributes.

# ##### Breaking down CRASH_DATE attribute into smaller attributes to gain more information
# 

# In[19]:


crash_census['CRASH_DATE']


# ##### 'CRASH_DATE'  data type is objet, converting it to datetime data type
# 

# In[20]:


crash_census['CRASH_DATE'] = pd.to_datetime(crash_census['CRASH_DATE'])
crash_census['CRASH_DATE']


# ##### Extracting year from the 'CRASH_DATE' column
# 

# In[21]:


crash_census['CRASH_YEAR'] = pd.DatetimeIndex(crash_census['CRASH_DATE']).year
crash_census['CRASH_YEAR']


# ##### Breaking down CRASH_DATE attribute into smaller attributes to gain more information
# 

# In[22]:


crash_census.head()


# ==> Broken down __CRASH_DATE__ attribute.
# 
#    Droping __CRASH_DATE__ attribute as this attribute has now been broken down into several attributes in the data set where the time, day, and month, year of the crash are given as separate attributes.
# 

# In[23]:


crash_census.drop(['CRASH_DATE'], axis = 1, inplace = True)
crash_census.shape


# ### 3. What are some insights about the crashes and date/time? You can look into season, day of the week, day/night, lightning, weather, etc.

# ##### Observing CRASH_HOUR
# 

# In[24]:


sns.distplot(crash_census['CRASH_HOUR'], bins=24, norm_hist=True ,kde=False)


# ==> From the above graph we can conclude that most of the accidents occur during __14:00 to 16:00 hour__ timeframe
# 

# ##### Observing CRASH_DAY_OF_WEEK
# 

# In[25]:


sns.distplot(crash_census['CRASH_DAY_OF_WEEK'], bins=7, norm_hist=True, kde=False)


# ==> From the above graph we can conclude that almost everyday of the week has the same number of accidents but __Saturday__ has the most accident.
# 

# ##### Observing CRASH_MONTH
# 

# In[26]:


sns.distplot(crash_census['CRASH_MONTH'], bins=12, norm_hist=True, kde=False)


# ==> From the above graph we can conclude that most of the accidents occur during __October__ month
# 

# ##### Observing Weekend (Saturday and Sunday) data
# 

# In[27]:


weekend_data = crash_census[(crash_census['CRASH_DAY_OF_WEEK']==6) | (crash_census['CRASH_DAY_OF_WEEK']==7)]
weekend_data['CRASH_HOUR'].value_counts().sort_values(ascending=False)
sns.distplot(weekend_data['CRASH_HOUR'], bins=24, norm_hist=True, kde=False)


# ==> From the above graph we can conclude that most accidents occur during during __14:00 to 15:00 hour__ timeframe on __weekends__.
# 

# ##### Observing Weekday (Monday - Friday) data
# 

# In[28]:


weekday_data = crash_census[(crash_census['CRASH_DAY_OF_WEEK']!=6) & (crash_census['CRASH_DAY_OF_WEEK']!=7)]
weekday_data['CRASH_HOUR'].value_counts().sort_values(ascending=False)
sns.distplot(weekday_data['CRASH_HOUR'], bins=24, norm_hist=True, kde=False)


# ==> From the above graph we can conclude that most accidents occur during __14:00 to 16:00 hour__ timeframe on __weekdays__ (Monday to Friday)
# 
# 
# 

# ### 4. Has number of deadly crashes increased recently? Look at the data over the years. Can you identify any significant increase/decrease?

# In[29]:


#crash data over the years.
pd.DataFrame(crash_census['CRASH_YEAR'].value_counts().sort_index()).plot(kind = 'bar', figsize = (20,10))


# ==> From the above graph we can conclude that the injuries increased till the year __2018 to 2019__. (There is not enough data for the year 2020 and 2021, possibly because of pandemic)
# 

# ### 5. Investigate number and type of injuries based on the speed limit.

# In[30]:



injury_types = ['INJURIES_FATAL', 'INJURIES_INCAPACITATING', 'INJURIES_NON_INCAPACITATING', 'INJURIES_REPORTED_NOT_EVIDENT']

crash_census_no_zero = crash_census[crash_census['INJURIES_TOTAL'] > 0]

crash_by_types = crash_census_no_zero.groupby(['POSTED_SPEED_LIMIT'])[
    injury_types
].sum()
plt = crash_by_types.plot(figsize=(15,9))
plt.set_xlabel("Speed limit")
plt.set_ylabel("Total")


# ==> From the above graph we can infer that there is a spike in number of injuries when __speed limit is 30__.
# 

# ### 6. Is there a relationship between hit and run crashes and number of fatal injuries?

# In[31]:


df_hit_and_run = crash_census['HIT_AND_RUN_I']
df_hit_and_run.head(5)


# In[32]:


df1_witout_unknown = crash_census[crash_census['HIT_AND_RUN_I'] != 'unknown']

df2__witout_unknown = df1_witout_unknown.replace({'HIT_AND_RUN_I' : { 'Y' : 1, 'N' : 0}})
df2__witout_unknown['HIT_AND_RUN_I'].value_counts()


# In[33]:


df2_Corr = df2__witout_unknown[['INJURIES_FATAL', 'HIT_AND_RUN_I']].corr()
df2_Corr


# In[34]:


sns.heatmap(df2_Corr.corr(), annot = True, fmt='.2g',cmap= 'coolwarm')


# ==> After observing the values from the heatmap, we can conclude that there is __no co-realtion__ between INJURIES_FATAL and HIT_AND_RUN_I as we are getting a negative co-relation

# ### 7. Do intersection-related crashes result in more fatal injuries?

# In[35]:



crash_census = crash_census[crash_census['INTERSECTION_RELATED_I'] != 'unknown']

crash_census.replace({'INTERSECTION_RELATED_I' : { 'Y' : 1, 'N' : 0}}, inplace=True)
crash_census['INTERSECTION_RELATED_I'].corr(crash_census['INJURIES_FATAL'])

intersection_injuries_df = crash_census[['INTERSECTION_RELATED_I', 'INJURIES_FATAL']].corr()

sns.heatmap(intersection_injuries_df.corr(), annot = True, fmt='.2g',cmap= 'coolwarm')


# ==> After observing the values from the heatmap, we can conclude that there is __no co-relation__ between intersection-related crashes and fatal injuries

# ### 8. Come up with at least two more interesting insights and visualize them.

# ##### 8.1 Observing WEATHER_CONDITION
# 

# In[36]:


crash_census['WEATHER_CONDITION'].value_counts()


# In[37]:


pd.DataFrame(crash_census['WEATHER_CONDITION'].value_counts()).plot(kind = 'bar', figsize = (20,10))
plt.set_title('Crashes in different weather conditions');


# ==> From the above graph we can infer that most of the accidents happen on a __clear day__. Maybe people are careless while driving on a clear day and more careful while driving on a rainy/snow day.

# ##### 8.2 Finding what is the LIGHTING_CONDITION  at time of crash usually
# 

# In[38]:


pd.DataFrame(crash_census['LIGHTING_CONDITION'].value_counts()).plot(kind = 'bar', figsize = (20,10))


# ==> From the above graph we can infer that most of the crashes happen during __day time__ . This trend in the above graph is maybe because people tend to be more careful while driving when there is more darkness.
# 
