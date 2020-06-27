import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("diamonds.csv", index_col=0)
data['price'] = data['price'].astype(float)

# Correlation Coefficient
corr = data.drop(['cut', 'clarity', 'color'], axis='columns').corr()
plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True)
plt.show()
# carat, x,y,z have strong correlation


# Check outlier
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

carat = data['carat']
fig = plt.figure(figsize=(5, 8))
plotData = [carat]
green_diamond = dict(markerfacecolor='b', marker='s')
plt.boxplot(carat, flierprops=green_diamond)
plt.title('Boxplot of carat')
plt.xticks([1], ['carat'])
plt.ylabel("Values")
plt.title("Box plot of carat")
plt.show()

df = [data['depth'], data['table'], data['x'], data['y'], data['z']]
fig = plt.figure(figsize=(5, 8))
green_diamond = dict(markerfacecolor='b', marker='s')
plt.boxplot(df, flierprops=green_diamond)
plt.xticks([1, 2, 3, 4, 5], ['depth', 'table', 'x', 'y', 'z'])
plt.ylabel("Values")
plt.title("multiple sample box plot")
plt.show()

# count of outlier
floatColumns = ['carat', 'depth', 'table', 'x', 'y', 'z']
for c in floatColumns:
    print("[", c, "]")
    count = 0
    pc = np.mean(data[c]) + 3 * np.std(data[c])
    mc = np.mean(data[c]) - 3 * np.std(data[c])
    for i in range(data.shape[0]):
        if (data.loc[i, c]) > pc:
            # print(i, data[c][i])
            count += 1
        elif (data.loc[i, c]) < mc:
            # print(i, data[c][i])
            count += 1
    print('count:', count)