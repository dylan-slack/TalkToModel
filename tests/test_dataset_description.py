"""Tests dataset description functionality."""
from os.path import dirname, abspath, join  # noqa: E402, F401
import sys

import gin
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression

parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)

from explain.dataset_description import DatasetDescription  # noqa: E402, F401


def write_random_dataset_model(filepath):
    """Write a dataset with a random number of features and columns."""
    dataset_file_path = join(filepath, "dataset.csv")

    x, y = 100, 100
    data = np.random.rand(x, y)

    features = [f"f_named_{i}" for i in range(y)]
    df = pd.DataFrame(data, columns=features)

    y = data[:, 0] > .8
    for i in range(y.shape[0]):
        if np.random.choice([0, 1]) == 0:
            y[i] = np.random.choice([0, 1])
    y = y.astype(int)

    lr = LogisticRegression().fit(df.values, y)

    df["y"] = y

    df.to_csv(dataset_file_path)
    return dataset_file_path, lr, df


def test_dataset_description(tmp_path):
    """Tests the dataset description routines."""
    np.random.seed(90210)

    # Loop through a couple random datasets
    for _ in range(3):
        eval_filepath, lr, df = write_random_dataset_model(tmp_path)

        objective = "predict random numbers"
        dataset_description = "random numbers"
        model = "logistic regression"

        gin_config = f"""

        DatasetDescription.dataset_objective = "{objective}"
        DatasetDescription.dataset_description = "{dataset_description}"
        DatasetDescription.eval_file_path = "{eval_filepath}"
        DatasetDescription.model_description = "{model}"
        """

        gin.parse_config(gin_config)
        data_obj = DatasetDescription(index_col=0, target_var_name='y')

        assert data_obj.get_dataset_objective() == objective
        assert data_obj.get_dataset_description() == dataset_description

        y = df.pop("y")
        X = df.values
        y_pred = lr.predict(X)
        score = str(round(accuracy_score(y, y_pred)*100, 3))

        score_string = f"The model scores <em>{score}% accuracy</em> on the data."
        assert data_obj.get_eval_performance(lr) == score_string

        score_string = f"The model scores <em>{round(f1_score(y, y_pred), 3)} f1</em> on the data."
        assert data_obj.get_eval_performance(lr, metric_name="f1") == score_string

        score_string = f"The model scores <em>{round(roc_auc_score(y, y_pred), 3)} roc</em> on the data."
        assert data_obj.get_eval_performance(lr, metric_name="roc") == score_string

        score_string = f"The model scores <em>{round(precision_score(y, y_pred), 3)} precision</em> on the data."
        assert data_obj.get_eval_performance(lr, metric_name="precision") == score_string

        score_string = f"The model scores <em>{round(recall_score(y, y_pred), 3)} recall</em> on the data."
        assert data_obj.get_eval_performance(lr, metric_name="recall") == score_string
