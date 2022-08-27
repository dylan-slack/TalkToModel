"""Train compas model."""
import numpy as np
import pandas as pd
import pickle as pkl

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

np.random.seed(90210)

X_values = pd.read_csv("./data/compas.csv", index_col=0)
y_values = X_values.pop("y")

X_train, X_test, y_train, y_test = train_test_split(X_values, y_values, test_size=0.15)

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1).fit(X_train.values, y_train.values)
print("Score:", clf.score(X_test.values, y_test.values))
print("Portion y==0:", np.sum(y_test.values == 1) * 1. / y_test.values.shape[0])

with open("./data/compas_model_grad_boosted_tree.pkl", "wb") as f:
    pkl.dump(clf, f)

X_train['y'] = y_train
X_test['y'] = y_test

X_train.to_csv('./data/compas_train.csv')
X_test.to_csv('./data/compas_test.csv')
