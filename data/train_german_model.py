"""Train german model."""
import sys
from os.path import dirname, abspath

import numpy as np
import pickle as pkl
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data.utils import TypeSelector

np.random.seed(0)
parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)

from data.processing_functions import get_and_preprocess_german_short

german_data, categorical_mapping = get_and_preprocess_german_short()

# savings_categories = [['NA', 'little', 'moderate', 'quite rich', 'rich']]
savings_categories = [[0, 1, 2, 3, 4]]
# checking_categories = [['NA', 'little', 'moderate', 'rich']]
checking_categories = [[0, 1, 2, 3]]
one_hot_col_names = ['Sex', 'Housing', 'Purpose']
sex_categories = [list(german_data['x_values']['Sex'].unique())]
housing_categories = [list(german_data['x_values']['Housing'].unique())]
purpose_categories = [list(german_data['x_values']['Purpose'].unique())]
standard_scaler_col_list = ['Age', 'Credit amount', 'Duration']
job_categories = [[0, 1, 2, 3]]

X_values = german_data["x_values"]
y_values = german_data["y_values"]

# Transform categorical names to according int values
scalar = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(
    X_values, y_values, test_size=0.20)
# Save data before transformations
X_train['y'] = y_train
X_test['y'] = y_test
X_train.to_csv('german_train.csv')
X_test.to_csv('german_test.csv')
X_train.pop("y")
X_test.pop("y")

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

# Setup pipeline
# lr_pipeline = Pipeline([('scaler', StandardScaler()),
#                         ('lr', LogisticRegression(C=1.0, max_iter=10_000))])
# save_column_id_to_possible_values_mapping(X_train, categorical_col_ids)
# save_column_id_to_value_index_mapping(X_train, categorical_col_ids)


data_columns = german_data['column_names']
preprocessor = ColumnTransformer(
    [
        ('onehot_sex', OneHotEncoder(categories=sex_categories), [data_columns.index('Sex')]),
        ('onehot_housing', OneHotEncoder(categories=housing_categories), [data_columns.index('Housing')]),
        ('onehot_purpose', OneHotEncoder(categories=purpose_categories), [data_columns.index('Purpose')]),
        ('ordinal_job', OrdinalEncoder(categories=job_categories), [data_columns.index('Job')]),
        ('ordinal_saving', OrdinalEncoder(categories=savings_categories), [data_columns.index('Saving accounts')]),
        ('ordinal_checking', OrdinalEncoder(categories=checking_categories), [data_columns.index('Checking account')]),
        ('scaler', StandardScaler(), [data_columns.index(col) for col in standard_scaler_col_list])
    ],
    remainder='drop'
)
lr_pipeline = Pipeline([('dtype', TypeSelector("object")),
                        ('preprocessing', preprocessor),
                        ('lr', GradientBoostingClassifier())])
lr_pipeline.fit(X_train, y_train)

print("Train Score:", lr_pipeline.score(X_train, y_train))
print("Test Score:", lr_pipeline.score(X_test, y_test))
print("Portion y==1:", np.sum(y_test == 1)
      * 1. / y_test.shape[0])

x_test_pred = lr_pipeline.predict(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, x_test_pred, pos_label=1)
print("AUC_test", metrics.auc(fpr, tpr))

with open("german_model_short_grad_tree.pkl", "wb") as f:
    pkl.dump(lr_pipeline, f)

print("Saved model!")
