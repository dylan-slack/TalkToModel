"""Explanation tests.

TODO(dylan): add tests for explanation regeneration e.g., when ids_to_regenerate + save_to_cache flags
             are set in different ways.
"""
import collections
from os.path import dirname, abspath, join
import sys

import gin
import pandas as pd
import pytest
import numpy as np

np.random.seed(0)
parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)

from explain.logic import load_sklearn_model  # noqa: E402
from explain.explanation import TabularDice, MegaExplainer, Explanation, load_cache  # noqa: E402
from explain.utils import read_and_format_data  # noqa: E402

gin.parse_config_file('./tests/tests-config.gin')


@pytest.fixture
def model_and_data():
    data = read_and_format_data()
    dataset, cat_features, num_features = data[0], data[2], data[3]
    model = load_sklearn_model()
    return model, dataset, num_features, cat_features


# This test results in the test suite to hang on github though it passes
# locally... a bit confusing/concerning?
def test_tabulardice(tmp_path, model_and_data):
    """Test the tabular dice generating method."""
    model, dataset, num_features, _ = model_and_data
    cache_file = join(tmp_path, "cache.pkl")
    tabular_dice = TabularDice(model=model,
                               data=dataset,
                               num_features=num_features,
                               cache_location=cache_file)
    ids = np.random.choice(list(dataset.index), size=3)
    explanations = tabular_dice.get_explanations(ids,
                                                 dataset,
                                                 save_to_cache=True)
    for id_c in ids:
        orig_pred = model.predict(dataset.loc[[id_c]])

        cf_e = explanations[id_c].cf_examples_list[0].final_cfs_df
        cf_e.pop('y')
        cfe_pred = model.predict(cf_e.values)

        # Make sure we actually have cfe's!
        for pred in cfe_pred:
            assert pred != orig_pred[0]

    # Check that ids of retrieved explanations are correct
    assert set(explanations.keys()) == set(ids.tolist())

    summary = tabular_dice.summarize_explanations(dataset.loc[[ids[0]]])

    assert isinstance(summary, tuple)
    assert isinstance(summary[0], str)
    assert isinstance(summary[1], str)


def test_megaexplainer_string_format(tmp_path, model_and_data):
    """Test the tabular lime string formatting."""
    model, dataset, _, cat_features = model_and_data
    cache_file = join(tmp_path, "cache.pkl")
    assert isinstance(dataset, pd.DataFrame)

    # Make sure explanations work
    tabular_lime = MegaExplainer(prediction_fn=model.predict_proba,
                                 data=dataset,
                                 cat_features=cat_features,
                                 cache_location=cache_file)

    ids = list(dataset.index)[:2]
    output = tabular_lime.summarize_explanations(dataset.loc[ids],
                                                 filtering_text="testing filtering description text")
    assert isinstance(output, tuple)


def test_tabular_mega_explainer(tmp_path, model_and_data):
    """Test the tabular mega explainer generations."""
    model, dataset, _, cat_features = model_and_data
    cache_file = join(tmp_path, "cache.pkl")
    assert isinstance(dataset, pd.DataFrame)

    # Make sure explanations work
    tabular_mexp = MegaExplainer(prediction_fn=model.predict_proba,
                                 data=dataset,
                                 cat_features=cat_features,
                                 cache_location=cache_file)

    assert isinstance(dataset, pd.DataFrame)
    assert len(tabular_mexp.cache) == 0

    tabular_mexp.update_cache_size(2)
    assert tabular_mexp.max_cache_size == 2

    np_dataset = dataset.to_numpy()
    ids = np.array(dataset.index[np.random.choice(len(dataset), size=2)])
    ids = list(ids)
    explanation = tabular_mexp.get_explanations(ids, dataset)
    recovered_ids = []
    for exp in explanation:
        recovered_ids.append(exp)
        current_explanation = explanation[exp]
        list_exp = current_explanation.list_exp
        assert len(list_exp) == np_dataset.shape[1]
    assert len(tabular_mexp.cache) == 2
    assert set(recovered_ids) == set(ids)

    # Do it again on same to make sure caching is ok
    new_explanation = tabular_mexp.get_explanations(ids, dataset)
    recovered_ids = []
    for exp in new_explanation:
        recovered_ids.append(exp)
        current_explanation = new_explanation[exp]
        list_exp = current_explanation.list_exp
        assert exp in explanation.keys()
        assert list_exp == explanation[exp].list_exp
        assert len(list_exp) == np_dataset.shape[1]
    assert len(tabular_mexp.cache) == 2
    assert set(recovered_ids) == set(ids)

    # Make sure cache is working well
    tabular_mexp.update_cache_size(3)
    ids = list(dataset.index)[5:8]
    explanation = tabular_mexp.get_explanations(ids, dataset)
    assert len(tabular_mexp.cache) == 3

    ids = list(dataset.index)[20:4]
    explanation = tabular_mexp.get_explanations(ids, dataset)
    assert len(tabular_mexp.cache) == 3

    tabular_mexp.update_cache_size(2)
    ids = list(dataset.index)[20:22]
    explanation = tabular_mexp.get_explanations(ids, dataset)
    assert len(tabular_mexp.cache) == 2

    # Make sure cat feature index conversion is working
    test_dataset = pd.DataFrame(np.random.rand(100, 10), columns=[f"c_{i}" for i in range(0, 10)])
    cat_inds = tabular_mexp.get_cat_features(test_dataset, cat_features=["c_1", "c_3", "c_9"])
    assert cat_inds == [1, 3, 9]
    cat_inds = tabular_mexp.get_cat_features(test_dataset, cat_features=[1, 3, 9])
    assert cat_inds == [1, 3, 9]
    cat_inds = tabular_mexp.get_cat_features(test_dataset, cat_features=[1, 2, 9])
    assert cat_inds == [1, 2, 9]

    assert isinstance(tabular_mexp.summarize_explanations(dataset.loc[ids]), tuple)

    output = tabular_mexp.summarize_explanations(dataset.loc[ids],
                                                 filtering_text="testing filtering description text")
    print(output)
    assert isinstance(output, tuple)
    assert isinstance(output[0], str)
    assert isinstance(output[1], str)


def test_cache(tmp_path):
    """Testing explanation cache."""
    filepath = join(tmp_path, "cache.pkl")

    explanation = Explanation(cache_location=filepath,
                              max_cache_size=2,
                              class_names={0: 'name', 1: 'other_name'})

    assert explanation._cache_size() == 0
    assert explanation.cache_loc == filepath
    assert explanation.max_cache_size == 2

    explanation.update_cache_size(new_cache_size=2)
    assert explanation._cache_size() == 0
    assert explanation.max_cache_size == 2

    ids = np.random.choice(list(range(1_000_000)), size=2)
    expls = {}
    for c_id in ids:
        expls[c_id] = np.random.rand(2, 2)

    explanation._write_to_cache(expls)
    assert explanation._cache_size() == 2
    assert collections.Counter(expls.keys()) == collections.Counter(ids)

    test_load = load_cache(filepath)
    for key in test_load.keys():
        assert key in explanation.cache
        assert np.allclose(test_load[key], explanation.cache[key])

    ids = np.random.choice(list(range(1_000_000)), size=2)
    new_expls = {}
    for c_id in ids:
        new_expls[c_id] = np.random.rand(2, 2)

    explanation._write_to_cache(expls)
    assert explanation._cache_size() == 2
