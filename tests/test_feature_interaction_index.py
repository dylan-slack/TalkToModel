"""Tests for feature interaction index."""
from os.path import dirname, abspath
import sys

import gin
import pandas as pd
import pytest
import numpy as np

np.random.seed(0)
parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)

from explain.logic import load_sklearn_model  # noqa: E402
from explain.utils import read_and_format_data  # noqa: E402
from explain.feature_interaction import FeatureInteraction  # noqa: E402

gin.parse_config_file('./tests/tests-config.gin')


@pytest.fixture
def model_and_data():

    def mock_f(data: np.ndarray):
        results = []
        for val in data:
            x, y, z = val[0], val[1], val[2]
            results.append(x + y ** 2 + 10 * x * z - z)
        return np.array(results)

    random_num = np.random.rand(1_000, 2)
    random_cat = np.random.choice([0, 1], size=(1_000, 1))
    random_data = np.concatenate((random_num, random_cat), axis=1)

    dataset = pd.DataFrame(random_data, columns=['x', 'y', 'z'])
    model = mock_f
    num_features = ['x', 'y']
    cat_features = ['z']

    return model, dataset, num_features, cat_features


@pytest.fixture
def sklearn_model_and_data():
    data = read_and_format_data()
    dataset, cat_features, num_features = data[0], data[2], data[3]
    model = load_sklearn_model()
    return model, dataset, num_features, cat_features


def test_feature_interactions_mock(model_and_data):
    model, dataset, _, cat_features = model_and_data

    feature_interaction_explainer = FeatureInteraction(data=dataset, prediction_fn=model, cat_features=cat_features,
                                                       verbose=False)
    xy_interaction = feature_interaction_explainer.feature_interaction('y', 'x', sub_sample_pct=5.0)
    zx_interaction = feature_interaction_explainer.feature_interaction('z', 'x', sub_sample_pct=5.0)
    assert zx_interaction > xy_interaction, "z and x feature interaction should be greater!"


def test_feature_interaction_sklearn(sklearn_model_and_data):
    model, dataset, _, cat_features = sklearn_model_and_data

    feature_interaction_explainer = FeatureInteraction(data=dataset,
                                                       prediction_fn=model.predict_proba,
                                                       cat_features=cat_features,
                                                       class_ind=None,
                                                       verbose=False)

    j = 0

    for i, f1 in enumerate(dataset.columns):
        for j in range(i, len(dataset.columns)):
            f2 = dataset.columns[j]
            if f1 == f2:
                continue

            effect = feature_interaction_explainer.feature_interaction(f1,
                                                                       f2,
                                                                       number_sub_samples=20)
            print(effect)
            j += 1
            if j > 5:
                break
