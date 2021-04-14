# Importing the necessary library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Apply the default seaborn theme, scaling, and color pallet
sns.set()

# Read the data file
births = pd.read_csv("births.csv")

# Fill the empty cells and change the datatype to int
births['day'].fillna(0, inplace=True)
births['day'] = births['day'].astype(int)
# print(births.head())

births['decade'] = 10 * (births['year'] // 10)
births.pivot_table('births', index='decade', columns='gender', aggfunc='sum')
print(births.head())

# Visualize the total number of births per year
birth_decade = births.pivot_table('births', index='decade', columns='gender', aggfunc='sum')
birth_decade.plot()
plt.ylabel('Total births per year')
plt.show()

# Further data exploration
# Sigma-clipping operation: To cut outliers

quartiles = np.percentile(births['births'], [25, 50, 75])
mu = quartiles[1]
sig = 0.74 * (quartiles[2] - quartiles[0])

# This final line is a robust estimate of the sample mean, where the 0.74 comes from the inter-quartile range of a
# Gaussian distribution, With this we can use the query() method to filter out rows with births outside these values:

births = births.query('(births > @mu - 5 * @sig) & (births < @mu + 5 * @sig)')
births['day'] = births['day'].astype(int)
births.index = pd.to_datetime(10000 * births.year + 100 * births.month + births.day, format='%Y%m%d')
births['dayofweek'] = births.index.dayofweek
births.pivot_table('births', index='dayofweek', columns='decade', aggfunc='mean').plot()
plt.gca().set_xticklabels(['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'])
plt.ylabel('Average births by day');
plt.show()

# Plot the mean number of births by the day of the year
births_month = births.pivot_table('births', [births.index.month, births.index.day])
print(births_month.head())
births_month.index = [pd.datetime(2012, month, day) for (month, day) in births_month.index]
print(births_month.head())

# Plot the data
fig, ax = plt.subplots(figsize=(12, 4))
births_month.plot(ax=ax)
plt.show()