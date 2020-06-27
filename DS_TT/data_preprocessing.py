import pandas as pd
import numpy as np
import sklearn.preprocessing as pp
from sklearn.compose import ColumnTransformer

data = pd.read_csv("diamonds.csv", index_col=0)

# Drop columns
data.drop(['depth', 'table'], axis='columns', inplace=True)

# Nan value handling
# data.dropna(axis=0, inplace=True)
# data.reset_index(drop=True, inplace=True)
data['cut'].fillna(method='bfill', inplace=True)
data['color'].fillna(method='bfill', inplace=True)
data['clarity'].fillna(method='bfill', inplace=True)
data.interpolate(method='linear', inplace=True)


# Run either MR or KNN.


# VER. Multiple Regression
""""""
num_attr = ['carat', 'price', 'x', 'y', 'z']
cat_attr = ['cut', 'color', 'clarity']
transformer = ColumnTransformer([
    ('num', pp.StandardScaler(), num_attr),  # numeric standard scaling
    ('cat', pp.OneHotEncoder(), cat_attr)  # categorical one-hot encoding
])
data_transformed = transformer.fit_transform(data)

data_transformed = pd.DataFrame(data_transformed)
data_transformed.rename(columns={0: 'carat', 1: 'price', 2: 'x', 3: 'y', 4: 'z'}, inplace=True)

print(data_transformed.head())

# Save transformed data
data_transformed.to_csv("diamonds_MR.csv")
""""""

# VER. K-NN
"""
num_attr = ['carat', 'x', 'y', 'z']
transformer = ColumnTransformer([
    ('num', pp.StandardScaler(), num_attr)  # numeric standard scaling
])
data_transformed = transformer.fit_transform(data)

data_transformed = pd.DataFrame(data_transformed)
data_transformed.rename(columns={0: 'carat', 1: 'x', 2: 'y', 3: 'z'}, inplace=True)

# Label encoding
data_labeled = pd.DataFrame()
data_labeled['cut'] = (pp.LabelEncoder().fit_transform(data['cut']))
data_labeled['color'] = (pp.LabelEncoder().fit_transform(data['color']))
data_labeled['clarity'] = (pp.LabelEncoder().fit_transform(data['clarity']))

# concat dataframe
data_transformed = pd.concat([data_transformed, data_labeled], axis=1)

# Data range
label_range = [x for x in range(0, 20001, 2000)]  # divide

price_label = pd.DataFrame()
price_label['price'] = data['price']

for i in range(len(label_range) - 1):
    price_label['price'] = price_label['price'].apply(lambda x: i if label_range[i] < x <= label_range[i + 1] else x)

# concat dataframe
data_transformed = pd.concat([data_transformed, price_label], axis=1)

# categorical value scaling
num_attr = ['cut', 'color', 'clarity']
transformer = ColumnTransformer([
    ('num', pp.StandardScaler(), num_attr)  # numeric standard scaling
])
new_data = transformer.fit_transform(data_transformed)

new_data = pd.DataFrame(new_data)
new_data.rename(columns={0: 'cut', 1: 'color', 2: 'clarity'}, inplace=True)

data_transformed['cut'] = new_data['cut']
data_transformed['color'] = new_data['color']
data_transformed['clarity'] = new_data['clarity']

print(data_transformed.head(5))
data_transformed.to_csv("diamonds_KNN.csv")
"""
