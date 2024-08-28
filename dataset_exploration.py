import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets

# Load the dataset
iris = datasets.load_iris()

# Create a DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['target_name'] = df['target'].apply(lambda x: iris.target_names[x])

# Display the first few rows
print(df.head())

# Get basic statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Histograms of each feature
df.hist(bins=20, figsize=(10, 8))
plt.suptitle('Histograms of Each Feature')
plt.show()

# Boxplots for each feature
plt.figure(figsize=(10, 6))
sns.boxplot(data=df.drop(columns=['target', 'target_name']))
plt.title('Boxplots of Each Feature')
plt.show()

# Pair plot
sns.pairplot(df, hue='target_name', markers=["o", "s", "D"])
plt.suptitle('Pairwise Relationships', y=1.02)
plt.show()

# Correlation matrix
corr_matrix = df.corr()

# Heatmap of the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Scatter plot of specific features
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', hue='target_name', style='target', palette='deep', s=100)
plt.title('Sepal Length vs Sepal Width')
plt.show()

# Violin plots for each feature
plt.figure(figsize=(10, 8))
for i, feature in enumerate(df.columns[:-2]):
    plt.subplot(2, 2, i+1)
    sns.violinplot(x='target_name', y=feature, data=df)
    plt.title(f'Violin plot of {feature}')
plt.tight_layout()
plt.show()

# Class distribution
sns.countplot(x='target_name', data=df, palette='Set2')
plt.title('Class Distribution')
plt.show()
