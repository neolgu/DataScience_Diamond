import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("diamonds_KNN.csv", index_col=0)

X = data.drop(['price'], axis='columns', inplace=False)
y = data['price']

kfold = KFold(5, shuffle=True)

gcv = GridSearchCV(KNeighborsClassifier(),
                   param_grid={'n_neighbors': np.arange(1, 10, 1),
                               'p': [1, 2]},
                   cv=kfold,
                   scoring='f1_micro',
                   n_jobs=-1)
gcv.fit(X, y)

print('final params: ', gcv.best_params_)
print('best score: ', gcv.best_score_)
print("{}".format(gcv.best_estimator_))

# best n = 5
