# imports:
import os
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

#########################################################################################
# FUNCTIONS:

def numerical(df, feature) :
    print(f'='*30)
    print(f'{feature.upper()} ANALISYS')
    print(f'='*30)

    nuniques = df[feature].nunique()
    counts = df[feature].value_counts()
    print(f'There are {nuniques} different values in {feature}')
    print(f'{counts}')

def graphs(df, feature, bins) :
    # hist:
    if bins == 0 :
        df.hist(feature, grid=True, figsize=(6,6), bins=df[feature].nunique())
    else :
        df.hist(feature, grid=True, figsize=(6,6), bins=bins)
    plt.show()
    
    # boxplot:
    sns.catplot(y=feature, kind='box', data=df)
    plt.show()
#########################################################################################

# importing the CSV:
df_raw= pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv')
df_raw.to_csv('../data/raw/housing_raw.csv')


# data size:
print(f'The Dataset has {df_raw.shape[0]} "observations" with {df_raw.shape[1]} columns')

# duplicates?
print(f'Dataframe has {df_raw.duplicated().sum()} duplicates')

# Will keep onle the desired features:
df_interim = df_raw[['Latitude', 'Longitude', 'MedInc']]

# Remove high outliers for MedInc
df_interim = df_interim.drop(df_interim[df_interim['MedInc'] > 8.01].index)

# scale the data:
scaler = MinMaxScaler()
scaler.fit_transform(df_interim)

# save df_interim:
df_interim.to_csv('../data/interim/housing_interim.csv')
df = df_interim.copy()


# K-means model and fit:
model_kmeans = KMeans(n_clusters=6, random_state=13)
model_kmeans.fit(df)

# K-means extra info:
print(f'K-Means Cluster Centers: \n {model_kmeans.cluster_centers_}')

# K-means clustering for df:
prediction = model_kmeans.predict(df)

# Assign 'Cluster' to a new column on the dataset
df['Cluster'] = pd.Series(prediction, index=df.index)

# Cluster analysis:
numerical(df, 'Cluster')

# define the cluster as categorical:
df['Cluster'] = df['Cluster'].astype('category')

# plot Lat, Long and Cluster
# sns.relplot(x='Latitude', y='Longitude', hue='Cluster', data=df)

# export the model:
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, '../models/K-means_housing.pkl')
joblib.dump(model_kmeans, filename)