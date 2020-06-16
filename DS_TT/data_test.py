import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("diamonds_KNN.csv", index_col=0)
data['price'] = data['price'].astype(float)

# Correlation Coefficient
"""
corr = data.drop(['cut', 'clarity', 'color'], axis='columns').corr()
plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True)
plt.show()
"""
## carat, x,y,z have strong correlation

plt.hist(data['price'].tolist(), bins=10)
plt.show()
"""
"""