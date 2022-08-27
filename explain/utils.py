"""Utils"""
from os import listdir

import gin
import openai
import pandas as pd
from pandas import DataFrame, Series


@gin.configurable
def read_and_format_data(filepath,
                         index_col,
                         target_var_name,
                         cat_features,
                         num_features,
                         remove_underscores=True) -> tuple[DataFrame,
                                                           Series,
                                                           list[str],
                                                           list[str]]:
    """Reads and processes data.

    This routine loads and formats the data to be included in the
    conversation. There are a couple requirements on the data. First,
    it must be possible to load the data from filepath as a pandas df.
    The routine will attempt to use 0 as the index column.

    In addition, it will attempt to guess the split between categorical
    and numerical features if _both_ these are not specified manually.

    Finally, the routine will extract the target variable from the data.
    The target variable must be included in the original dataframe. The
    name of the target variable can be changed by modifying target_var_name.

    Arguments:
        filepath: The filepath of the data. Must be able to be loaded as a
                  pandas csv.
        index_col:
        target_var_name:
        cat_features:
        num_features:
        remove_underscores: Whether to remove underscores from variable names.
                            This can help decoding performance.
    Returns:
        dataset: The loaded + processed data (without the target var.)
        y: The target variable.
        cat_features: The categorical feature names.
        num_features: The numerical feature names.
    """
    # Load the dataset
    dataset = pd.read_csv(filepath, index_col=index_col)

    # Underscores hurt decoding performance in feature names, remove them
    # for better performance, if requested.
    if remove_underscores:
        dataset.columns = dataset.columns.str.replace('_', '')
        dataset.columns = dataset.columns.str.replace('-', '')
        dataset.columns = dataset.columns.str.lower()

    # Get the target variable from the data.
    y_values = dataset.pop(target_var_name)

    # Split the dataset into categorical + numeric, this is done by
    # guessing which features are which
    if not cat_features and not num_features:
        cat_features, num_features = get_numeric_categorical(dataset)

    return dataset, y_values, cat_features, num_features


def setup_gpt3():
    with open('openai_key.txt', 'r') as f:
        key = f.readline().strip()
        openai.api_key = key


def strip_ws(tok):
    """Removes whitespace off the beginning of a token."""
    if tok[0] == ' ' and len(tok) > 1:
        return tok[1:]
    else:
        return tok


def find_csv_filenames(path_to_dir, suffix=".csv"):
    """https://stackoverflow.com/questions/9234560/find-all-csv-files-in-a-directory-using-python"""
    filenames = listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]


def get_numeric_categorical(data, threshold=0.95, top_n=10):
    """Gets the numeric and categorical columns from a pandas dataset.

    This function uses the ratio of unique values to total values"""
    cat, num = [], []

    for var in data.columns:
        # From https://stackoverflow.com/questions/35826912/
        # what-is-a-good-heuristic-to-detect-if-a-column-in-a-pandas-dataframe-is-categori
        if 1. * data[var].value_counts(normalize=True).head(top_n).sum() > threshold:
            cat.append(var)
        else:
            num.append(var)
    return cat, num


def add_to_dict_lists(key, value, dictionary):
    """Stores values in list corresponding to key in place."""
    if key not in dictionary:
        dictionary[key] = [value]
    else:
        dictionary[key].append(value)
