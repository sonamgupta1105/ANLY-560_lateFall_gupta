#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Ques 1: Import pandas under the name pd
# Importing useful libraries
import pandas as pd
import numpy as np


# In[2]:


# Ques 2 : Print the version of pandas that has been imported.
# Checking pandas version
pd.__version__


# In[3]:


# Ques 4 : Create a DataFrame df from this dictionary data which has the index labels.
# Created dictionary data 
data = {'animal': ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],
        'age': [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],
        'visits': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']
       }

# List of labels
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

# Creating dataframe from the above dictionary
data_df = pd.DataFrame(data, index = labels)

# Print dataframe
print(data_df)


# In[4]:


# Ques 5 : Display a summary of the basic information about this DataFrame and its data.

# Using describe func for pandas, printing the summary of data_df built from dictionary and list created above
data_df.describe()


# In[5]:


# Ques 6 :  Return the first 3 rows of the DataFrame df.

# printing the first 3 rows using head() and passing 3 as parameter as we want first 3 rows
print(data_df.head(3))


# In[6]:


# Ques 7 : Select just the 'animal' and 'age' columns from the DataFrame df.

# Selecting the specific columns by passing the column name from dataframe
print(data_df['animal'])
print(data_df['age'])


# In[7]:


# Ques 8 : Select the data in rows [3, 4, 8] and in columns ['animal', 'age'].

# Print rows data from particular columns using loc()
print(data_df.loc[['c', 'd', 'h'],['animal', 'age']])


# In[8]:


# Ques 9 : Select only the rows where the number of visits is greater than 3.

# Using loc() print all the rows from all columns based on the condition for visits > 3
visits3 = data_df.loc[data_df['visits'] > 3]
print(visits3) # Result is empty dataframe because there are no values for visits column which are > 3


# In[9]:


# Ques 10 : Select the rows where the age is missing, i.e. is NaN.

# using isnull() we can see which are those rows that column 'age' is  NaN for 
ages_none = data_df.loc[data_df['age'].isnull()]
print(ages_none)


# In[10]:


# Ques 11: Select the rows where the animal is a cat and the age is less than 3.

# Selecting data based on animal being cat and age < 3 using loc()
cat_3 = data_df.loc[(data_df['animal'] == 'cat') & (data_df['age'] < 3)]
print(cat_3)


# In[11]:


# Ques 12 : Select the rows the age is between 2 and 4 (inclusive).

# selecting data based on inclusive conditions
age_2_4 = data_df[data_df['age'].between(2,4)]
print(age_2_4)


# In[12]:


# Ques 13 : Change the age in row 'f' to 1.5.

# changing a value in row for a particular row
#field_age = data_df.loc['f']['age']

data_df.loc['f', 'age'] = 1.5
print(data_df['age'])


# In[13]:


# Ques 14 : Calculate the sum of all visits (the total number of visits).

# calculate the sum of elements in a column using sum()
print(data_df['visits'].sum())


# In[14]:


# Ques 15 : Calculate the mean age for each different animal in df.
print(data_df.groupby('animal')['age'].mean())


# In[15]:


# Ques 16 : Append a new row 'k' to df with your choice of values for each column. Then delete that row to return the original DataFrame.

data_df.loc['k'] = [4.5, 'dog', 'no', 2]
data_df = data_df.drop('k')
print(data_df)


# In[16]:


# Ques 17 :  Count the number of each type of animal in df
data_df['animal'].count() # Total number of animals in the data
print(data_df['animal'].value_counts()) # Returns the count of each animal in the data


# In[17]:


# Ques 18 : Sort df first by the values in the 'age' in decending order, then by the value in the 'visit' column in ascending order.

# Using sort_values() to sort by different columns in different orders
print(data_df.sort_values(by = ['age', 'visits'], ascending=[False, True]))


# In[18]:


# Ques 19 : The 'priority' column contains the values 'yes' and 'no'. Replace this column with a column of boolean values: 'yes' should be True and 'no' should be False.
data_df['priority'] = data_df['priority'].map({'yes' : True, 'no' : False})
data_df['priority']


# In[19]:


# Ques 20 :  In the 'animal' column, change the 'snake' entries to 'python'.
data_df['animal'].replace('snake', 'python')


# In[20]:


# Ques 21 : For each animal type and each number of visits, find the mean age. In other words, each row is an animal, 
# each column is a number of visits and the values are the mean ages (hint: use a pivot table).

# Pandas have pivot_table() which works exactly like pivot function in Excel. WIth the help of this function, the following command
# answers the question to calculate mean age based on visits and animal columns
data_df.pivot_table(index='animal', columns='visits', values= 'age', aggfunc= 'mean')


# In[21]:


# Ques 22 : You have a DataFrame df with a column 'A' of integers. How do you filter out rows which contain the same integer as the row immediately above?
df = pd.DataFrame({'A': [1, 2, 2, 3, 4, 5, 5, 5, 6, 7, 7]})
print(df)
# Filtering the rows that contain the same integer as the row above it
#for row in df:
#df['A'] = np.roll(df['A'], 1)
#print(df['A'])

# using shift() lets us to shift and then compare with the value in row above
# Ref: https://stackoverflow.com/questions/10982089/how-to-shift-a-column-in-pandas-dataframe
df_filtered = df.loc[df['A'].shift() != df['A']]
print(df_filtered)


# In[22]:


# Ques 23 :  Given a DataFrame of numeric values,how do you subtract the row mean from each element in the row?

df = pd.DataFrame(np.random.random(size=(5, 3))) # a 5x3 frame of float values
print(df)
# Ref: https://stackoverflow.com/questions/22149584/what-does-axis-in-pandas-mean
sub_row = df.sub(df.mean(axis=1), axis=0)
print(sub_row)


# In[23]:


# Ques 24 : Suppose you have DataFrame with 10 columns of real numbers
# Which column of numbers has the smallest sum? (Find that column's label.)

df = pd.DataFrame(np.random.random(size=(5, 10)), columns=list('abcdefghij'))
print(df)
df_sum = df.sum().min()
#print(df_sum)
print(df.sum().idxmin()) # J is the column with minium sum out of others


# In[24]:


# Ques 25 : How do you count how many unique rows a DataFrame has (i.e. ignore all rows that are duplicates)?
df.drop_duplicates(keep=False).count() #This counts the number of rows per column
len(df.drop_duplicates(keep=False)) # this is the count of unique rows in the df


# In[25]:


# Ques 26 : You have a DataFrame that consists of 10 columns of floating--point numbers.
#Suppose that exactly 5 entries in each row are NaN values. For each row of the DataFrame, 
#find the column which contains the third NaN value.You should return a Series of column labels.

# Ref: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.idxmax.html
print((df.isna().cumsum(axis=1)==3).idxmax(axis=1))


# In[26]:


# Ques 27 :  A DataFrame has a column of groups 'grps' and and column of numbers 'vals'.
#For each group, find the sum of the three greatest values.
df = pd.DataFrame({'grps': list('aaabbcaabcccbbc'), 
                   'vals': [12,345,3,1,45,14,4,52,54,23,235,21,57,3,87]})
df_27 = pd.DataFrame({'grps': list('aaabbcaabcccbbc'), 
                   'vals': [12,345,3,-1,45,14,4,52,54,23,235,21,57,3,87]})

# Sort the dataframe by vals since we want 3 greatest values
df = df.sort_values('vals', ascending=False)

# Find the sum of 3 greatest values, grouped by grps
sum_vals = df.groupby('grps')['vals'].nlargest(3).sum(level=0) # WIthout level=0, it returns the sum of values for each group rather than separating
print(sum_vals)


# In[27]:


# Ques 28 : A DataFrame has two integer columns 'A' and 'B'. The values in 'A' are between 1 and 100 (inclusive). 
# For each group of 10 consecutive integers in 'A' (i.e. (0, 10], (10, 20], ...),
# calculate the sum of the corresponding values in column 'B'.

# Build two lists with values ranging from 0 to 100 (inclusive)
a = list(range(101))
b = list(range(101))

# Combine the two lists created above and make a dataframe
df = pd.DataFrame({
    'A' : a,
    'B' : b
})

# Using pandas cut() , create buckets for numbers with interval of 10, group them together 
# Calculate the sum of column B
# Ref: https://stackoverflow.com/questions/21441259/pandas-groupby-range-of-values
print(df.groupby(pd.cut(df['A'], np.arange(0, 101, 10)))['B'].sum())


# In[28]:


# Ques 29 : For each value, count the difference back to the previous zero (or the start of the Series, whichever is closer). 
# These values should therefore be [1, 2, 0, 1, 2, 3, 4, 0, 1, 2]. Make this a new column 'Y'.
# all are solvable using just the usual pandas/NumPy methods (and so avoid using explicit for loops)

df = pd.DataFrame({'X': [7, 2, 0, 3, 4, 2, 5, 0, 3, 4]})

# Ref: https://stackoverflow.com/questions/30730981/how-to-count-distance-to-the-previous-zero-in-pandas-series

# Calculate length of the X column
len_df = np.arange(len(df))
#len_df

# We need to adjust the 0th index for this question cause we have to count backward so concatenate on first axis of the dataframe
zero_index_adjust = np.r_[-1, (df['X'] == 0).nonzero()[0]]

#print(zero_index_adjust) -- [-1  2  7]

# Now we count the difference back to the previous zero or the start of X
y = len_df - zero_index_adjust[np.searchsorted(zero_index_adjust - 1, len_df) - 1]

# Add y as another column in df
df.loc[:,'Y'] = y 
print(df)


# In[29]:


# Ques 30 : Consider a DataFrame containing rows and columns of purely numerical data. Create a list of the row-column index locations of the 3 largest values.

print(df)

# Sort the values of both the columns in dataframe
# Get the index of 3 largest values and convert it to a list, therefore .index.tolist()
df.sort_values(by='Y')[-3:].index.tolist() # THis gives the 3 largest values if we sort by the column Y

# We want row-column, thus pivoting the dataframe row-column using unstack()

print(df.unstack().sort_values()[-3:].index.tolist())


# In[30]:


# Ques 31 :  Given a DataFrame with a column of group IDs, 'grps', and a column of corresponding integer values, 'vals', 
#replace any negative values in 'vals' with the group mean.

# Calculate mean of the vals column
mean_vals = df_27['vals'].mean()


# Get numerical value from the dataframe
# num = df_27._get_numeric_data()

# num[num<0] = mean_vals

# for val in df_27.iterrows():
#     df_27['vals'][df_27['vals'] < 0 ] = mean_vals
    
# df_27

# Ref: https://stackoverflow.com/questions/14760757/replacing-values-with-groupby-means

def replace(group):
    mask = group<0
    # Select those values where it is < 0, and replace
    # them with the mean of the values which are not < 0.
    group[mask] = group[~mask].mean()
    return group

# Groupy the group ID column, use the replace function on vals column of the dataframe
print(df_27.groupby(['grps'])['vals'].transform(replace))


# In[41]:


# Ques 32: Implement a rolling mean over groups with window size 3, which ignores NaN value. 
df = pd.DataFrame({'group': list('aabbabbbabab'),
                       'value': [1, 2, 3, np.nan, 2, 3, 
                                 np.nan, 1, 7, 3, np.nan, 8]})

# Ref: https://stackoverflow.com/questions/13996302/python-rolling-functions-for-groupby-object
# groupby group and value, to calculate rolling mean for window size 3
# Calculate the mean and replace NAs with 0
df.groupby('group')['value'].rolling(3).mean()
'''The above approach does not ignore NAs'''

# Fill NAs with 0 and group them by groups
fill_na = df.fillna(0).groupby('group')['value']
group_vals = df.groupby(['group'])['value']

# Ref: https://stackoverflow.com/questions/36988123/pandas-groupby-and-rolling-apply-ignoring-nans

# Now calculate the rolling mean with window size 3
result = fill_na.rolling(3, min_periods = 1).sum() / group_vals.rolling(3, min_periods = 1).count()

# Adjust the index to remove the other level as the dataframe has multi-index
result.reset_index(level=0, drop=True).sort_index()
#s.reset_index(level=0, drop=True).sort_index()  # drop/sort index


# In[10]:


# Ques 33 : Create a DatetimeIndex that contains each business day of 2015 and use it to index a Series of random numbers. 
# Let's call this Series s.

# Create data time index using date_range() from pandas.
# Ref: https://stackoverflow.com/questions/29370057/select-dataframe-rows-between-two-dates
# Ref: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html

date_time_2015 = pd.date_range(start = '2015-01-01', end= '2015-12-31', freq = 'B') # Lists all the business days
#date_time_2015

# Create a series of random numbers with date_time_2015 as index
# Using numpy random number generator
series_date = pd.Series(np.random.rand(len(date_time_2015)), index = date_time_2015)
print(series_date)


# In[13]:


# Ques 34 :  Find the sum of the values in s for every Wednesday.

# Ref: https://stackoverflow.com/questions/28009370/get-weekday-day-of-week-for-datetime-column-of-dataframe
sum_wednesday = series_date[series_date.index.weekday == 2].sum()
print(sum_wednesday)


# In[18]:


# Ques 35: For each calendar month in s, find the mean of values.

mean_calendar_month = series_date[series_date.index.month].mean()
#mean_calendar_month
''' The above approach gave the mean of all the months together'''

# To get mean of all the months separately, .resample() is used
# Ref: https://stackoverflow.com/questions/42191697/resample-daily-data-to-monthly-with-pandas-date-formatting

mean_calendar_month = series_date.resample('M').mean()
print(mean_calendar_month)


# In[24]:


# Ques 36: For each group of four consecutive calendar months in s, find the date on which the highest value occurred

# We have to group the calendar months in series_date
# 'M' denotes months for frequency, providing 4M means 4 continuous/consecutive months
# pandas idxmax() returns the maximum value first
print(series_date.groupby(pd.Grouper(freq = '4M')).idxmax())


# In[25]:


# Ques 37 : Create a DateTimeIndex consisting of the third Thursday in each month for the years 2015 and 2016.

# Using similar idea from Ques 33 of date_range
# Frequency  will be third thursday = 3THU, WOM = week of month
# Ref: https://github.com/pandas-dev/pandas/issues/2289#issuecomment-269616457

date_2015_2016 = pd.date_range(start = '2015-01-01', end = '2016-12-31', freq= 'WOM-3THU')
print(date_2015_2016)


# In[65]:


# Ques 38: Some values in the the FlightNumber column are missing. These numbers are meant to increase by 10 with
#each row so 10055 and 10075 need to be put in place. 
#Fill in these missing numbers and make the column an integer column (instead of a float column).

df = pd.DataFrame({'From_To': ['LoNDon_paris', 'MAdrid_miLAN', 'londON_StockhOlm', 
                               'Budapest_PaRis', 'Brussels_londOn'],
              'FlightNumber': [10045, np.nan, 10065, np.nan, 10085],
              'RecentDelays': [[23, 47], [], [24, 43, 87], [13], [67, 32]],
                   'Airline': ['KLM(!)', '<Air France> (12)', '(British Airways. )', 
                               '12. Air France', '"Swiss Air"']})

# .astype(int) can convert the float column to int column
# Since we are dealing only with a particular column, we will just access that column
# interpolate is a very useful function to fill in the gaps between a sequence, in this case
# the increment in flight number happens by 10. method is linear by default which works perfect for this question
df['FlightNumber'] = df['FlightNumber'].interpolate().astype(int)
print(df['FlightNumber'])


# In[63]:


# Ques 39: The From_To column would be better as two separate columns! Split each string on the underscore 
# delimiter _ to give a new temporary DataFrame with the correct values. 
# Assign the correct column names to this temporary DataFrame.

# Using str.split('_') on From to column as accessing one column of a dataframe is a series

temp_df = df.From_To.str.split('_', expand = True)

# assigning the header to temp_df columns
temp_df.columns = ['From', 'To']
print(temp_df)


# In[64]:


# Ques 40: Notice how the capitalisation of the city names is all mixed up in this temporary DataFrame. 
# Standardise the strings so that only the first letter is uppercase (e.g. "londON" should become "London".)

# Ref: https://www.geeksforgeeks.org/string-capitalize-python/

temp_df['From'] = temp_df['From'].str.capitalize()
temp_df['To'] = temp_df['To'].str.capitalize()
print(temp_df)


# In[67]:


# Ques 41: Delete the From_To column from df and attach the temporary DataFrame from the previous questions.
df = pd.DataFrame({'From_To': ['LoNDon_paris', 'MAdrid_miLAN', 'londON_StockhOlm', 
                               'Budapest_PaRis', 'Brussels_londOn'],
              'FlightNumber': [10045, np.nan, 10065, np.nan, 10085],
              'RecentDelays': [[23, 47], [], [24, 43, 87], [13], [67, 32]],
                   'Airline': ['KLM(!)', '<Air France> (12)', '(British Airways. )', 
                               '12. Air France', '"Swiss Air"']})
# Deleting a column from dataframe
df = df.drop(['From_To'], axis=1)

# Attach the temp_df to df

df = df.join(temp_df)
print(df)


# In[54]:


# Ques 42: In the Airline column, you can see some extra puctuation and symbols have appeared around the airline names. 
# Pull out just the airline name. E.g. '(British Airways. )' should become 'British Airways'.

'''Using simple regex for capturing just the letters, both uppercase and lowercase, and then strip() any leading characters or spaces'''
df['Airline'] = df['Airline'].str.extract('([a-zA-Z\s]+)', expand = False).str.strip()
print(df['Airline'])


# In[70]:


# Ques 43: In the RecentDelays column, the values have been entered into the DataFrame as a list.
#We would like each first value in its own column, each second value in its own column, and so on. 
#If there isn't an Nth value, the value should be NaN.Expand the Series of lists into a DataFrame named delays,
#rename the columns delay_1, delay_2, etc. and replace the unwanted RecentDelays column in df with delays.

df = pd.DataFrame({'From_To': ['LoNDon_paris', 'MAdrid_miLAN', 'londON_StockhOlm', 
                               'Budapest_PaRis', 'Brussels_londOn'],
              'FlightNumber': [10045, np.nan, 10065, np.nan, 10085],
              'RecentDelays': [[23, 47], [], [24, 43, 87], [13], [67, 32]],
                   'Airline': ['KLM(!)', '<Air France> (12)', '(British Airways. )', 
                               '12. Air France', '"Swiss Air"']})
# Ref: https://stackoverflow.com/questions/27263805/pandas-when-cell-contents-are-lists-create-a-row-for-each-element-in-the-list

'''In the Ref used for this ques, the list is split into different rows using apply(). SOmething similar is used for this question as follows'''

# Implement apply() on pd.Series as mentioned in the question
delays = df['RecentDelays'].apply(pd.Series)
#print(delays)

# format the columns ranging from 1 to number of columns in delays
delays.columns = ['delay_{}'.format(n) for n in range(1, len(delays.columns)+1)]

# Drop unwanted RecentDelays column and replace with new delays values
df = df.drop('RecentDelays', axis=1).join(delays)
print(df) 


# In[74]:


# Ques 44: Given the lists letters = ['A', 'B', 'C'] and numbers = list(range(10)), 
# construct a MultiIndex object from the product of the two lists. 
# Use it to index a Series of random numbers. Call this Series s.

letters = ['A', 'B', 'C']
numbers = list(range(10))

# Ref: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.MultiIndex.from_product.html

mi_prod = pd.MultiIndex.from_product([letters, numbers])
#mi_prod

# Create a series of random numbers with mi_prod as index. 
# We need 30 random numbers because i_prod has 30 values, otherwise we get ValueERror
s = pd.Series(np.random.rand(30), index=mi_prod)
print(s)


# In[75]:


# Ques 45: Check the index of s is lexicographically sorted 
#(this is a necessary proprty for indexing to work correctly with a MultiIndex).

# Ref: https://stackoverflow.com/questions/31427466/ensuring-lexicographical-sort-in-pandas-multiindex

print(s.index.is_lexsorted())


# In[78]:


# Ques 46: Select the labels 1, 3 and 6 from the second level of the MultiIndexed Series.

'''This can be solved simmply by using .loc function and provide 1,3,6 as a list for second level'''
print(s.loc[:,[1,3,6]])


# In[80]:


# Ques 47: Slice the Series s; slice up to label 'B' for the first level and from label 5 onwards for the second level.

'''At first we use .loc() to get to first and second level, then use slice() to get labels'''

print(s.loc[slice(None, 'B'), slice(5, None)])


# In[81]:


# Ques 48:  Sum the values in s for each label in the first level 
# (you should have Series giving you a total for labels A, B and C).

# Ref: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sum.html
'''passing level = 0 gets the first level of multiindex series'''
print(s.sum(level = 0))


# In[82]:


# Ques 49: Suppose that sum() (and other methods) did not accept a level keyword argument. 
# How else could you perform the equivalent of s.sum(level=1)?

#Ref: http://www.datasciencemadesimple.com/reshape-using-stack-unstack-function-pandas-python/

'''We can reshape the series using unstack() and then calculate sum on axis 0'''
print(s.unstack().sum(axis=0))


# In[86]:


# Ques 50: Exchange the levels of the MultiIndex so we have an index of the form (letters, numbers). 
# Is this new Series properly lexsorted? If not, sort it.

# Ref: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.swaplevel.html

'''Using swaplevel() to exchange the levels of multiindex seriex, is_lex_sorted() to check the order of sort and if not, 
then use sort_index to correctly sort the series'''

new_series = s.swaplevel(0,1)
print(new_series)

# Checking the sort order
print(new_series.index.is_lexsorted()) # Not sorted lexicographically

# Sorting the new_series
new_series = new_series.sort_index()
print(new_series)



