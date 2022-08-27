"""Train compas model."""
import numpy as np
import pandas as pd
import pickle as pkl

from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

np.random.seed(3)

X_values = pd.read_csv("./data/diabetes.csv")
y_values = X_values.pop("y")

scalar = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(
    X_values, y_values, test_size=0.40)
cols = X_train.columns
# Save data before transformations
X_train['y'] = y_train
X_test['y'] = y_test
X_train.to_csv('./data/diabetes_train.csv')
X_test.to_csv('./data/diabetes_test.csv')
X_train.pop("y")
X_test.pop("y")

X_train = X_train.values
X_test = X_test.values

# Setup pipeline
# lr_pipeline = Pipeline([('scaler', StandardScaler()),
#                         ('lr', LogisticRegression(C=1.0, max_iter=10_000))])
lr_pipeline = Pipeline([('scaler', StandardScaler()),
                        ('lr', GradientBoostingClassifier())])
lr_pipeline.fit(X_train, y_train)

print("Train Score:", lr_pipeline.score(X_train, y_train))
print("Score:", lr_pipeline.score(X_test, y_test))
print("Portion y==0:", np.sum(y_test.values == 0)
      * 1. / y_test.values.shape[0])

print("Column names: ", cols)
# print("Coefficients: ", lr_pipeline.named_steps["lr"].coef_)

with open("./data/diabetes_model_grad_tree.pkl", "wb") as f:
    pkl.dump(lr_pipeline, f)

print("Saved model!")
