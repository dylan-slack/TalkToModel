"""Train titanic model."""
import sys
from os.path import dirname, abspath

import numpy as np
import pickle as pkl
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

np.random.seed(0)
parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)

from data.processing_functions import get_and_preprocess_titanic

titanic_data, categorical_mapping = get_and_preprocess_titanic()

# Split df in x and y values
y_train = titanic_data['train']['Survived'].values
X_train = titanic_data['train']
y_test = titanic_data['test']['Survived'].values
X_test = titanic_data['test']

# Save to CSV
X_train.to_csv('titanic_train.csv')
X_test.to_csv('titanic_test.csv')

X_train.pop("Survived")
X_test.pop("Survived")
X_train = X_train.values
X_test = X_test.values


data_columns = titanic_data['column_names']
preprocessor = ColumnTransformer(
    [
        ('onehot_sex', OneHotEncoder(categories=[[0, 1]]), [data_columns.index('Sex')]),
        ('onehot_embarked', OneHotEncoder(categories=[[0, 1, 2]]), [data_columns.index('Embarked')]),
    ]
)
lr_pipeline = Pipeline([('preprocessing', preprocessor),
                        ('lr', SVC(probability=True))])
lr_pipeline.fit(X_train, y_train)

print("Train Score:", lr_pipeline.score(X_train, y_train))
print("Test Score:", lr_pipeline.score(X_test, y_test))
print("Portion y==1:", np.sum(y_test == 1)
      * 1. / y_test.shape[0])

x_test_pred = lr_pipeline.predict(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, x_test_pred, pos_label=1)
print("AUC_test", metrics.auc(fpr, tpr))

with open("titanic_model_short_grad_tree.pkl", "wb") as f:
    pkl.dump(lr_pipeline, f)

print("Saved model!")
