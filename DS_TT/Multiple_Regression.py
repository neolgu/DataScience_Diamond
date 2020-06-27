import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv("diamonds_MR.csv", index_col=0)

# init X, y
X = data.drop(['price'], axis='columns', inplace=False)
y = data['price']

# model
multiple_regression = LinearRegression()
# Kfold
kfold = KFold(5, shuffle=True)

# model validation
score = cross_val_score(multiple_regression, X=X, y=y, cv=kfold)
print("cross validation scroes: {}".format(score))
print("Mean score: {}".format(np.mean(score)))

# Test Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

multiple_regression.fit(X_train, y_train)
y_pred = multiple_regression.predict(X_test)
errors = y_test - y_pred

# Show Graph
plt.scatter(y_test, y_pred)
x_lim = plt.xlim()
y_lim = plt.ylim()
plt.plot(x_lim, y_lim, "r:")
plt.show()

plt.scatter(range(y_test.shape[0]), errors)
plt.axhline(y=0, color='r', linestyle=':')
plt.show()
