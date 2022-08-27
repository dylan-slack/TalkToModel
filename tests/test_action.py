"""Action tests."""
from os.path import dirname, abspath, join  # noqa: E402, F401
import sys

import gin
import pandas as pd
import pytest
import numpy as np

np.random.seed(0)
parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)

from explain.action import run_action  # noqa: E402, F401
from explain.explanation import MegaExplainer, TabularDice, Explanation, load_cache  # noqa: E402, F401
from explain.logic import load_sklearn_model  # noqa: E402, F401
from explain.conversation import Conversation  # noqa: E402, F401
from explain.utils import read_and_format_data  # noqa: E402, F401


gin.parse_config_file('./tests/tests-config.gin')


@pytest.fixture
def model_and_data():
    data = read_and_format_data()
    dataset, y_vals, cat_features, num_features = data[0], data[1], data[2], data[3]
    model = load_sklearn_model()
    return model, dataset, num_features, cat_features, y_vals


@pytest.fixture
def numerical_data():
    conv = Conversation()
    # Get values on the range -500 -> 500
    x_dim, y_dim = 1_000, 75
    data = (np.random.rand(x_dim, y_dim) * 1000.0) - 500.0
    numeric = list(range(y_dim))
    features = [f"f_{i}" for i in numeric]
    dataset = pd.DataFrame(data, columns=features)
    y_vals = np.random.choice([0, 1], size=x_dim)

    dataset['y'] = y_vals
    y_vals = dataset.pop('y')

    cat_num = list(range(y_dim, 100))
    cat_vals = np.random.choice(list(range(-100, 100)), size=(x_dim, 25))
    for i, c in enumerate(cat_num):
        dataset["f_" + str(c)] = cat_vals[:, i]

    features = list(dataset.columns)
    conv.add_dataset(dataset, y_vals, categorical=[], numeric=features)
    return conv


@pytest.fixture
def cat_data():
    conv = Conversation()
    # Get values on the range -500 -> 500
    x_dim, y_dim = 1_000, 75
    cat_num = list(range(y_dim))
    cat_vals = np.random.choice(list(range(-100, 100)), size=(x_dim, y_dim))
    dataset = pd.DataFrame()
    for i, c in enumerate(cat_num):
        dataset["f_" + str(c)] = cat_vals[:, i]
    y_vals = np.random.choice([0, 1], size=x_dim)
    dataset['y'] = y_vals
    y_vals = dataset.pop('y')
    conv.add_dataset(dataset, y_vals, categorical=list(
        dataset.columns), numeric=[])
    return conv


@pytest.fixture()
def cat_values():
    cat_values = [-100, -75, -25, 0, 1, 2, 50, 80, 100]
    return cat_values


def test_cat_filter(cat_data, cat_values):
    # Test categorical filtering
    features = ["f_0", "f_22", "f_31", "f_74", "f_18", "f_9"]
    for f in features:
        for num in cat_values:
            parse_string = f"filter {f} {str(num)} [e]"
            run_action(cat_data, None, parse_string)
            filtered_data = cat_data.temp_dataset.contents['X']
            assert np.sum(filtered_data[f] != num) == 0


def test_cat_compound_filter(cat_data, cat_values):
    # Test compound categorical filtering
    features = list(cat_data.get_var('dataset').contents['X'].columns)
    for _ in range(10):
        f1 = np.random.choice(features)
        f2 = np.random.choice(features)
        for num in cat_values:
            parse_string = f"filter {f1} {str(num)} and filter {f2} {str(num)} [e]"
            run_action(cat_data, None, parse_string)
            filtered_data = cat_data.temp_dataset.contents['X']
            assert np.sum((filtered_data[f1] != num) | (
                filtered_data[f2] != num)) == 0

        for num in cat_values:
            parse_string = f"filter {f1} {str(num)} or filter {f2} {str(num)} [e]"
            run_action(cat_data, None, parse_string)
            filtered_data = cat_data.temp_dataset.contents['X']
            assert np.sum((filtered_data[f1] != num) & (
                filtered_data[f2] != num)) == 0


@pytest.fixture()
def numerical_values():
    numerical_values = [-200.0,
                        -123.1,
                        -10000.0,
                        -1.0,
                        0.0,
                        23.22,
                        15.0,
                        2222.0,
                        10,
                        100,
                        150,
                        23,
                        0.00001,
                        -0.0000005,
                        -322.0,
                        150.0]
    return numerical_values


def test_numerical_filter(numerical_data, numerical_values):
    # Test numerical filtering
    features = ["f_0", "f_22", "f_31", "f_75", "f_80", "f_99"]
    original_data = numerical_data.get_var('dataset').contents['X']
    for f in features:
        for num in numerical_values:
            parse_string = f"filter {f} greater than {str(num)} [e]"
            run_action(numerical_data, None, parse_string)
            filtered_data_ids = set(
                numerical_data.temp_dataset.contents['X'].index)
            true_filter_ids = set(original_data[original_data[f] > num].index)
            assert filtered_data_ids == true_filter_ids

        for num in numerical_values:
            parse_string = f"filter {f} less than {str(num)} [e]"
            run_action(numerical_data, None, parse_string)
            filtered_data_ids = set(
                numerical_data.temp_dataset.contents['X'].index)
            true_filter_ids = set(original_data[original_data[f] < num].index)
            assert filtered_data_ids == true_filter_ids

        for num in numerical_values:
            parse_string = f"filter {f} equal to {str(num)} [e]"
            run_action(numerical_data, None, parse_string)
            filtered_data_ids = set(
                numerical_data.temp_dataset.contents['X'].index)
            true_filter_ids = set(original_data[original_data[f] == num].index)
            assert filtered_data_ids == true_filter_ids

        for num in numerical_values:
            parse_string = f"filter {f} not equal to {str(num)} [e]"
            run_action(numerical_data, None, parse_string)
            filtered_data_ids = set(
                numerical_data.temp_dataset.contents['X'].index)
            true_filter_ids = set(original_data[original_data[f] != num].index)
            assert filtered_data_ids == true_filter_ids

        for num in numerical_values:
            parse_string = f"filter {f} greater equal than {str(num)} [e]"
            run_action(numerical_data, None, parse_string)
            filtered_data_ids = set(
                numerical_data.temp_dataset.contents['X'].index)
            true_filter_ids = set(original_data[original_data[f] >= num].index)
            assert filtered_data_ids == true_filter_ids

        for num in numerical_values:
            parse_string = f"filter {f} less equal than {str(num)} [e]"
            run_action(numerical_data, None, parse_string)
            filtered_data_ids = set(
                numerical_data.temp_dataset.contents['X'].index)
            true_filter_ids = set(original_data[original_data[f] <= num].index)
            assert filtered_data_ids == true_filter_ids


def test_compound_numerical_filter(numerical_data, numerical_values):
    # Test compound numerical filtering
    features = list(numerical_data.get_var('dataset').contents['X'].columns)
    original_data = numerical_data.get_var('dataset').contents['X']
    for _ in range(10):
        f1 = np.random.choice(features)
        f2 = np.random.choice(features)
        for n1, n2 in zip(numerical_values, numerical_values[::-1]):
            parse_string = (f"filter {f1} greater than {str(n1)}"
                            f" and filter {f2} less than {str(n2)} [e]")
            run_action(numerical_data, None, parse_string)
            filtered_data_ids = set(
                numerical_data.temp_dataset.contents['X'].index)
            true_filter_ids = set(original_data[(original_data[f1] > n1)
                                                & (original_data[f2] < n2)].index)
            assert filtered_data_ids == true_filter_ids

        for n1, n2 in zip(numerical_values, numerical_values[::-1]):
            parse_string = (f"filter {f1} greater than {str(n1)} "
                            f"or filter {f2} less equal than {str(n2)} [e]")
            run_action(numerical_data, None, parse_string)
            filtered_data_ids = set(
                numerical_data.temp_dataset.contents['X'].index)
            true_filter_ids = set(original_data[(original_data[f1] > n1)
                                                | (original_data[f2] <= n2)].index)
            assert filtered_data_ids == true_filter_ids

        for num in numerical_values:
            parse_string = (f"filter {f1} greater than {str(num)} and "
                            f"filter {f2} less equal than {str(num)} [e]")
            run_action(numerical_data, None, parse_string)
            filtered_data_ids = set(
                numerical_data.temp_dataset.contents['X'].index)
            true_filter_ids = set(original_data[(original_data[f1] > num)
                                                & (original_data[f2] <= num)].index)
            assert filtered_data_ids == true_filter_ids

        for num in numerical_values:
            parse_string = (f"filter {f1} equal to {str(num)} or "
                            f"filter {f2} less equal than {str(num)} [e]")
            run_action(numerical_data, None, parse_string)
            filtered_data_ids = set(
                numerical_data.temp_dataset.contents['X'].index)
            true_filter_ids = set(original_data[(original_data[f1] == num)
                                                | (original_data[f2] <= num)].index)
            assert filtered_data_ids == true_filter_ids

        for num in numerical_values:
            parse_string = (f"filter {f1} equal to {str(num)} and "
                            f"filter {f2} less equal than {str(num)} [e]")
            run_action(numerical_data, None, parse_string)
            filtered_data_ids = set(
                numerical_data.temp_dataset.contents['X'].index)
            true_filter_ids = set(original_data[(original_data[f1] == num)
                                                & (original_data[f2] <= num)].index)
            assert filtered_data_ids == true_filter_ids


def test_strange_numerical_filter_calls(numerical_data, numerical_values):
    # make sure strange grammars fail

    bad_strings = [
        "filter something completely else",
        "filter another very strange string",
        "filter doesntexist equal to 10"
    ]
    for b in bad_strings:
        with pytest.raises(NameError):
            run_action(numerical_data, None, b)


def test_most_important_feature(model_and_data, tmp_path):
    """Test most important feature action"""
    model, dataset, _, cat_features, y_vals = model_and_data

    # down sample dataset
    all_ids = list(dataset.index)
    to_use = np.random.choice(all_ids, size=2)
    dataset = dataset.loc[to_use]
    y_vals = y_vals[to_use]

    cache_file = join(tmp_path, "cache.pkl")
    mega_explainer = MegaExplainer(prediction_fn=model.predict_proba,
                                   data=dataset,
                                   cat_features=cat_features,
                                   cache_location=cache_file,
                                   use_selection=False)
    conv = Conversation()
    conv.add_var('mega_explainer', mega_explainer, 'explanation')
    conv.add_dataset(dataset, y_vals, [], list(dataset.columns))

    for parse_string in ["important age",
                         "important black",
                         "important man",
                         "filter id 6173 and important age"]:
        output = run_action(conv, None, parse_string)
        print(output)

    for parse_string in ["important topk 3", "important topk 1"]:
        output = run_action(conv, None, parse_string)
        print(output)

    for parse_string in ["important all",
                         "filter age greater than 30 and important topk 5"]:
        output = run_action(conv, None, parse_string)
        print(output)

    assert isinstance(output, str)


def test_prediction_likelihood(model_and_data):
    """Test prediction likelihood"""
    model, dataset, _, cat_features, y_vals = model_and_data

    conv = Conversation()
    conv.add_dataset(dataset, y_vals, [], list(dataset.columns))
    conv.add_var('model', model, 'model')
    conv.add_var('model_prob_predict',
                 model.predict_proba,
                 'prediction_function')

    probas = model.predict_proba(dataset.loc[[6173]]).tolist()[0]
    class_0 = str(round(probas[0]*100, 3))
    class_1 = str(round(probas[1]*100, 3))

    outputs = [(class_0, class_1)]

    filtered_data = dataset[(dataset["age"] < 20)]
    predictions = model.predict(filtered_data)

    pct_0 = np.sum(predictions == 0) / predictions.shape[0]
    pct_1 = 1 - pct_0
    pct_0 = round(pct_0*100, 3)
    pct_1 = round(pct_1*100, 3)

    outputs += [(str(pct_0), str(pct_1))]

    for i, parse_string in enumerate(["filter id 6173 and likelihood",
                                      "filter age less than 20 and likelihood"]):
        output = run_action(conv, None, parse_string)
        for val in outputs[i]:
            assert val in output

    config = """Conversation.class_names = None"""
    gin.parse_config(config)

    outputs = [(class_0, class_1)]
    conv = Conversation()
    conv.add_dataset(dataset, y_vals, [], list(dataset.columns))
    conv.add_var('model', model, 'model')
    conv.add_var('model_prob_predict',
                 model.predict_proba,
                 'prediction_function')

    for i, parse_string in enumerate(["filter id 6173 and likelihood"]):
        output = run_action(conv, None, parse_string)
        for val in outputs[i]:
            assert val in output

    assert isinstance(output, str)


def test_show_data(model_and_data, tmp_path):
    """Test most important feature action"""
    model, dataset, _, cat_features, y_vals = model_and_data

    all_ids = list(dataset.index)
    conv = Conversation()
    conv.add_dataset(dataset, y_vals, [], list(dataset.columns))

    for parse_string in ["filter age greater than 50 and show",
                         "show",
                         f"filter id {all_ids[0]} and show"]:
        output = run_action(conv, None, parse_string)
        assert isinstance(output, str)


def test_explanation_action(model_and_data, tmp_path):
    """Test explanation actions."""
    model, dataset, _, cat_features, y_vals = model_and_data
    all_ids = list(dataset.index)
    to_select = np.random.choice(all_ids, size=2)
    dataset = dataset.loc[to_select]
    y_vals = y_vals.loc[to_select]

    cache_file = join(tmp_path, "cache.pkl")
    mega_explainer = MegaExplainer(prediction_fn=model.predict_proba,
                                   data=dataset,
                                   cat_features=cat_features,
                                   cache_location=cache_file,
                                   use_selection=False)
    conv = Conversation()
    conv.add_var('mega_explainer', mega_explainer, 'explanation')
    conv.add_dataset(dataset, y_vals, [], list(dataset.columns))

    ids = np.random.choice(list(dataset.index), size=2)
    parse_string = (f"filter id {ids[0]} or filter id {ids[1]} and "
                    "explain features")
    action_1 = run_action(conv, None, parse_string)
    assert isinstance(action_1, str)


def test_previous_filtering(model_and_data, tmp_path):
    """Tests the previous filtering operation."""
    model, dataset, _, cat_features, y_vals = model_and_data
    all_ids = list(dataset.index)
    to_select = np.random.choice(all_ids, size=2)
    dataset = dataset.loc[to_select]
    y_vals = y_vals.loc[to_select]

    cache_file = join(tmp_path, "cache.pkl")
    tabular_mega = MegaExplainer(prediction_fn=model.predict_proba,
                                 data=dataset,
                                 cat_features=cat_features,
                                 cache_location=cache_file,
                                 use_selection=False)
    conv = Conversation()

    # using this to write explanations to cache so that we
    # can assert equality on the explanations later on
    tabular_mega.summarize_explanations(dataset, save_to_cache=True)

    conv.add_var('mega_explainer', tabular_mega, 'explanation')
    conv.add_dataset(dataset, y_vals, [], list(dataset.columns))

    # First operation
    ids = np.random.choice(list(dataset.index), size=2)

    parse_string = (f"filter id {ids[0]} or filter id {ids[1]} and "
                    "explain features")
    action_1 = run_action(conv, None, parse_string)
    assert f"id equal to {ids[0]} or id equal to {ids[1]}" in action_1

    # Second filtering operation
    parse_string_2 = "previousfilter and explain features"
    action_2 = run_action(conv, None, parse_string_2)
    assert f"id equal to {ids[0]} or id equal to {ids[1]}" in action_2
    assert f"id equal to {ids[0]} and id equal to {ids[1]}" not in action_2


def test_what_if(model_and_data, tmp_path):  # noqa: C901
    """Tests the what if operation."""
    model, dataset, _, cat_features, y_vals = model_and_data

    cache_file = join(tmp_path, "cache.pkl")
    tabular_mega = MegaExplainer(prediction_fn=model.predict_proba,
                                 data=dataset,
                                 cat_features=cat_features,
                                 cache_location=cache_file)
    conv = Conversation()
    conv.add_var('tabular_mega', tabular_mega, 'explanation')
    conv.add_dataset(dataset,
                     y_vals,
                     ['recidivated', 'felony', 'misdemeanor', 'woman', 'man', 'black'],
                     ['age', 'numberofpriorcrimes', 'lengthofstay'])

    ids = np.random.choice(list(dataset.index), size=2)
    parse_string = f"filter id {ids[0]} and change age increase 2"
    run_action(conv, None, parse_string)
    assert (conv.temp_dataset.contents['X'].loc[[ids[0]]]['age'].values[0] ==
            dataset.loc[[ids[0]]]['age'].values[0] + 2)

    parse_string = "filter numberofpriorcrimes less than 10 and change numberofpriorcrimes set -40"
    run_action(conv, None, parse_string)
    for cid in conv.temp_dataset.contents['X'].index:
        assert (conv.temp_dataset.contents['X'].loc[[
                cid]]['numberofpriorcrimes'].values[0] == -40)

    parse_string = "filter numberofpriorcrimes less than 10 and change man true"
    run_action(conv, None, parse_string)
    for cid in conv.temp_dataset.contents['X'].index:
        assert (conv.temp_dataset.contents['X'].loc[[
                cid]]['man'].values[0] == 1)

    parse_string = "change man false"
    run_action(conv, None, parse_string)
    for cid in conv.temp_dataset.contents['X'].index:
        assert (conv.temp_dataset.contents['X'].loc[[
                cid]]['man'].values[0] == 0)

    parse_string = "filter age greater than 30 and change age increase 2"
    run_action(conv, None, parse_string)
    for cid in conv.temp_dataset.contents['X'].index:
        assert (conv.temp_dataset.contents['X'].loc[[cid]]['age'].values[0] ==
                dataset.loc[[cid]]['age'].values[0] + 2)

    parse_string = "filter age greater than 30 and change age increase -2"
    run_action(conv, None, parse_string)
    for cid in conv.temp_dataset.contents['X'].index:
        assert (conv.temp_dataset.contents['X'].loc[[cid]]['age'].values[0] ==
                dataset.loc[[cid]]['age'].values[0] - 2)

    parse_string = "filter age greater than 30 and change age decrease 2"
    run_action(conv, None, parse_string)
    for cid in conv.temp_dataset.contents['X'].index:
        assert (conv.temp_dataset.contents['X'].loc[[cid]]['age'].values[0] !=
                dataset.loc[[cid]]['age'].values[0] + 2)

    parse_string = "filter age greater than 30 and change age decrease 40"
    run_action(conv, None, parse_string)
    for cid in conv.temp_dataset.contents['X'].index:
        assert (conv.temp_dataset.contents['X'].loc[[cid]]['age'].values[0] ==
                dataset.loc[[cid]]['age'].values[0] - 40)

    parse_string = "filter age greater than 25 and change age set 10"
    run_action(conv, None, parse_string)
    for cid in conv.temp_dataset.contents['X'].index:
        assert (conv.temp_dataset.contents['X'].loc[[
                cid]]['age'].values[0] == 10)

    parse_string = "filter numberofpriorcrimes less than 10 and change numberofpriorcrimes set 40"
    run_action(conv, None, parse_string)
    for cid in conv.temp_dataset.contents['X'].index:
        assert (conv.temp_dataset.contents['X'].loc[[
                cid]]['numberofpriorcrimes'].values[0] == 40)

    parse_string = "filter numberofpriorcrimes less than 10 and change numberofpriorcrimes set 40"
    run_action(conv, None, parse_string)
    for cid in conv.temp_dataset.contents['X'].index:
        assert (conv.temp_dataset.contents['X'].loc[[
                cid]]['numberofpriorcrimes'].values[0] == 40)


def test_feature_interactions(model_and_data, tmp_path):
    """Tests the feature interaction filtering operation."""
    model, dataset, _, cat_features, y_vals = model_and_data
    all_ids = list(dataset.index)
    to_select = np.random.choice(all_ids, size=2)
    dataset = dataset.loc[to_select]
    y_vals = y_vals.loc[to_select]

    cache_file = join(tmp_path, "cache.pkl")
    tabular_mega = MegaExplainer(prediction_fn=model.predict_proba,
                                 data=dataset,
                                 cat_features=cat_features,
                                 cache_location=cache_file,
                                 use_selection=False)
    conv = Conversation()

    # using this to write explanations to cache so that we
    # can assert equality on the explanations later on
    tabular_mega.summarize_explanations(dataset, save_to_cache=True)

    conv.add_var('mega_explainer', tabular_mega, 'explanation')
    conv.add_var('model_prob_predict', model.predict_proba, 'model_proba')
    conv.add_dataset(dataset, y_vals, [], list(dataset.columns))

    parse_string = "interact"
    action_1 = run_action(conv, None, parse_string)
    print(action_1)
