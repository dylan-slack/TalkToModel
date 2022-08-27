"""Mega Explainer tests."""
from os.path import dirname, abspath
import sys

import gin
import pytest
import numpy as np

np.random.seed(0)
parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)

from explain.logic import load_sklearn_model  # noqa: E402
from explain.mega_explainer.explainer import Explainer  # noqa: E402
from explain.utils import read_and_format_data  # noqa: E402

gin.parse_config_file('./tests/tests-config.gin')


@pytest.fixture
def model_and_data():
    data = read_and_format_data()
    dataset, cat_features, num_features = data[0], data[2], data[3]
    model = load_sklearn_model()
    return model, dataset, num_features, cat_features


def conv_features_to_inds(data, features):
    """Converts feature names to indices using dataset"""
    result = []
    all_features = list(data.columns)
    for i, f in enumerate(all_features):
        if f in features:
            result.append(i)
    return result


def test_mega_exp(model_and_data):
    """Tests the mega explainer"""
    model, dataset, _, cat_features = model_and_data
    cat_feature_inds = conv_features_to_inds(dataset, cat_features)

    mega_mega = Explainer(explanation_dataset=dataset,
                          explanation_model=model.predict_proba,
                          feature_names=list(dataset.columns),
                          discrete_features=cat_feature_inds)

    for ind in list(dataset.index)[:2]:
        exp = mega_mega.explain_instance(dataset.loc[[ind]])
        print(exp.list_exp)
