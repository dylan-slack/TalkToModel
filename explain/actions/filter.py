"""The filtering action.

This action filters some data according to different filtering criteria, e.g., less than or equal
to, greater than, etc. It modifies the temporary dataset in the conversation object, updating that dataset to yield the
correct filtering based on the parse.
"""
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def filter_dataset(dataset, bools):
    """Selects x and y of dataset by booleans."""
    dataset['X'] = dataset['X'][bools]
    dataset['y'] = dataset['y'][bools]
    return dataset


def format_parse_string(feature_name, feature_value, operation):
    """Formats a string that describes the filtering parse."""
    return f"{feature_name} {operation} {str(feature_value)}"


def numerical_filter(parse_text, temp_dataset, i, feature_name):
    """Performs numerical filtering.

    All this routine does (though it looks a bit clunky) is look at
    the parse_text and decide which filtering operation to do (e.g.,
    greater than, equal to, etc.) and then performs the operation.
    """
    # Greater than or equal to
    if parse_text[i+2] == 'greater' and parse_text[i+3] == 'equal':
        print(parse_text)
        feature_value = float(parse_text[i+5])
        bools = temp_dataset['X'][feature_name] >= feature_value
        updated_dset = filter_dataset(temp_dataset, bools)
        interpretable_parse_text = format_parse_string(
            feature_name, feature_value, "greater than or equal to")
    # Greater than
    elif parse_text[i+2] == 'greater':
        feature_value = float(parse_text[i+4])
        bools = temp_dataset['X'][feature_name] > feature_value
        updated_dset = filter_dataset(temp_dataset, bools)
        interpretable_parse_text = format_parse_string(
            feature_name, feature_value, "greater than")
    # Less than or equal to
    elif parse_text[i+2] == 'less' and parse_text[i+3] == 'equal':
        feature_value = float(parse_text[i+5])
        bools = temp_dataset['X'][feature_name] <= feature_value
        updated_dset = filter_dataset(temp_dataset, bools)
        interpretable_parse_text = format_parse_string(
            feature_name, feature_value, "less than or equal to")
    # Less than
    elif parse_text[i+2] == 'less':
        feature_value = float(parse_text[i+4])
        bools = temp_dataset['X'][feature_name] < feature_value
        updated_dset = filter_dataset(temp_dataset, bools)
        interpretable_parse_text = format_parse_string(
            feature_name, feature_value, "less than")
    # Equal to
    elif parse_text[i+2] == 'equal':
        feature_value = float(parse_text[i+4])
        bools = temp_dataset['X'][feature_name] == feature_value
        updated_dset = filter_dataset(temp_dataset, bools)
        interpretable_parse_text = format_parse_string(
            feature_name, feature_value, "equal to")
    # Not equal to
    elif parse_text[i+2] == 'not':
        feature_value = float(parse_text[i+5])
        bools = temp_dataset['X'][feature_name] != feature_value
        updated_dset = filter_dataset(temp_dataset, bools)
        interpretable_parse_text = format_parse_string(
            feature_name, feature_value, "not equal to")
    else:
        raise NameError(f"Uh oh, looks like something is wrong with {parse_text}")
    return updated_dset, interpretable_parse_text


def categorical_filter(parse_text, temp_dataset, conversation, i, feature_name):
    """Perform categorical filtering of a data set."""
    feature_value = parse_text[i+2]

    interpretable_parse_text = f"{feature_name} equal to {str(feature_value)}"

    if feature_name == "incorrect":
        # In the case the user asks for the incorrect predictions
        data = temp_dataset['X']
        y_values = temp_dataset['y']

        # compute model predictions
        model = conversation.get_var('model').contents
        y_pred = model.predict(data)

        # set bools to when the predictions are not the same
        # this with parse out to when incorrect is true, filter by
        # predictions != ground truth
        if feature_value == "true":
            bools = y_values != y_pred
        else:
            bools = y_values == y_pred
    else:
        if is_numeric_dtype(temp_dataset['X'][feature_name]):
            if feature_value == 'true':
                feature_value = 1
            elif feature_value == 'false':
                feature_value = 0
            else:
                feature_value = float(feature_value)
        bools = temp_dataset['X'][feature_name] == feature_value
    updated_dset = filter_dataset(temp_dataset, bools)
    return updated_dset, interpretable_parse_text


def prediction_filter(temp_dataset, conversation, feature_name):
    """filters based on the model's prediction"""
    model = conversation.get_var('model').contents
    x_values = temp_dataset['X']

    # compute model predictions
    predictions = model.predict(x_values)

    # feature name is given as string from grammar, bind predictions to str
    # to get correct equivalence
    str_predictions = np.array([str(p) for p in predictions])
    bools = feature_name == str_predictions

    updated_dset = filter_dataset(temp_dataset, bools)
    class_text = conversation.get_class_name_from_label(int(feature_name))
    interpretable_parse_text = f"the model predicts {class_text}"

    return updated_dset, interpretable_parse_text


def label_filter(temp_dataset, conversation, feature_name):
    """Filters based on the labels in the data"""""
    y_values = temp_dataset['y']
    str_y_values = np.array([str(y) for y in y_values])
    bools = feature_name == str_y_values

    updated_dset = filter_dataset(temp_dataset, bools)
    class_text = conversation.get_class_name_from_label(int(feature_name))
    interpretable_parse_text = f"the ground truth label is {class_text}"

    return updated_dset, interpretable_parse_text


def filter_operation(conversation, parse_text, i, is_or=False, **kwargs):
    """The filtering operation.

    This function performs filtering on a data set.
    It updates the temp_dataset attribute in the conversation
    object.

    Arguments:
        is_or:
        conversation: The conversation object.
        parse_text: The grammatical text string.
        i: The index of the parse_text that filtering is called.
    """
    if is_or:
        # construct a new temp data set to or with
        temp_dataset = conversation.build_temp_dataset(save=False).contents
    else:
        temp_dataset = conversation.temp_dataset.contents

    operation = parse_text[i]
    feature_name = parse_text[i+1]
    if feature_name == 'id':
        # Id isn't included as a categorical feature in the data,
        # so get this by using .loc
        feature_value = int(parse_text[i+2])
        updated_dset = temp_dataset

        # If id never appears in index, set the data to empty
        # other functions will handle appropriately and indicate
        # they can't do anything.
        if feature_value not in list(updated_dset['X'].index):
            updated_dset['X'] = []
            updated_dset['y'] = []
        else:
            updated_dset['X'] = updated_dset['X'].loc[[feature_value]]
            updated_dset['y'] = updated_dset['y'].loc[[feature_value]]
        interp_parse_text = f"id equal to {feature_value}"
    elif operation == "predictionfilter":
        updated_dset, interp_parse_text = prediction_filter(temp_dataset, conversation, feature_name)
    elif operation == "labelfilter":
        updated_dset, interp_parse_text = label_filter(temp_dataset, conversation, feature_name)
    elif feature_name in temp_dataset['cat'] or feature_name == "incorrect":
        updated_dset, interp_parse_text = categorical_filter(parse_text, temp_dataset, conversation, i, feature_name)
    elif feature_name in temp_dataset['numeric']:
        updated_dset, interp_parse_text = numerical_filter(parse_text, temp_dataset, i, feature_name)
    else:
        raise NameError(f"Parsed unkown feature name {feature_name}")

    if is_or:
        current_dataset = conversation.temp_dataset.contents
        updated_dset['X'] = pd.concat([updated_dset['X'], current_dataset['X']]).drop_duplicates()
        updated_dset['y'] = pd.concat([updated_dset['y'], current_dataset['y']]).drop_duplicates()
        conversation.add_interpretable_parse_op("or")
    else:
        conversation.add_interpretable_parse_op("and")

    conversation.add_interpretable_parse_op(interp_parse_text)
    conversation.temp_dataset.contents = updated_dset

    return '', 1
