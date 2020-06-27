import pandas as pd
import numpy as np
data = pd.read_csv("diamonds.csv")

# data_del = data.drop(['num', 'price'], axis=1)
#
# # print(data_del.head(10))
# # print(data_del.shape);
#
# randfloat = np.random.random([53940, 9])
#
# masks = ((randfloat < 0.2) & (randfloat >= 0.18))
#
# # print(masks)
# data_del.mask(masks, inplace=True)
#
# data['carat'] = data_del['carat']
# data['color'] = data_del['color']
# data['clarity'] = data_del['clarity']
# data['cut'] = data_del['cut']
# data['table'] = data_del['table']
# data['depth'] = data_del['depth']
# data['x'] = data_del['x']
# data['y'] = data_del['y']
# data['z'] = data_del['z']
#
# # data.drop(['num'], axis=1).to_csv("diamonds.csv", mode="w")

print(data.isnull().sum())
print(data.dropna())
