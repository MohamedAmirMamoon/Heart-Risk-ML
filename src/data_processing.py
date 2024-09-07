import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
cleveland = pd.read_csv('../dataset/heart.csv')

# Remove missing data (indicated with '?')
data = cleveland[~cleveland.isin(['?'])].dropna()

# Convert data to numeric
data = data.apply(pd.to_numeric)

# Describe the data
print(data.describe())

# Plot histograms
data.hist(figsize=(12, 12))
plt.show()

# Plot heart disease frequency for ages
pd.crosstab(data.age, data.target).plot(kind="bar", figsize=(20, 6))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Plot correlation heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(data.corr(), annot=True, fmt='.1f')
plt.show()

# Save the processed data
data.to_csv('../dataset/processed_heart.csv', index=False)
