"""Routines for processing data.

This code originates from one of my other projects:
https://github.com/dylan-slack/
Modeling-Uncertainty-Local-Explainability/blob/main/bayes/data_routines.py."""
import numpy as np
import pandas as pd

# The number of segments to use for the images
NSEGMENTS = 20
PARAMS = {
    'protected_class': 1,
    'unprotected_class': 0,
    'positive_outcome': 1,
    'negative_outcome': 0
}
IMAGENET_LABELS = {
    'french_bulldog': 245,
    'scuba_diver': 983,
    'corn': 987,
    'broccoli': 927
}


def get_and_preprocess_compas_data():
    """Handle processing of COMPAS according to: https://github.com/propublica/compas-analysis

    Parameters
    ----------
    params : Params
    Returns
    ----------
    Pandas data frame x_values of processed data, np.ndarray y_values, and list of column names
    """
    protected_class = PARAMS['protected_class']
    positive_outcome = PARAMS['positive_outcome']
    negative_outcome = PARAMS['negative_outcome']

    compas_df = pd.read_csv("data/compas.csv", index_col=0)
    compas_df = compas_df.loc[(compas_df['days_b_screening_arrest'] <= 30) &  # pylint: disable=E1136
                              (compas_df['days_b_screening_arrest'] >= -30) &  # pylint: disable=E1136
                              (compas_df['is_recid'] != -1) &   # pylint: disable=E1136
                              (compas_df['c_charge_degree'] != "O") &  # pylint: disable=E1136
                              (compas_df['score_text'] != "NA")]  # pylint: disable=E1136

    compas_df['length_of_stay'] = (pd.to_datetime(
        compas_df['c_jail_out']) - pd.to_datetime(compas_df['c_jail_in'])).dt.days
    x_values = compas_df[['age', 'two_year_recid', 'c_charge_degree',
                          'race', 'sex', 'priors_count', 'length_of_stay']]

    # if person has high score give them the _negative_ model outcome
    y_values = np.array([negative_outcome if score ==
                         'High' else positive_outcome for score in compas_df['score_text']])
    sens = x_values.pop('race')

    # assign African-American as the protected class
    x_values = pd.get_dummies(x_values)
    sensitive_attr = np.array(pd.get_dummies(sens).pop('African-American'))
    x_values['race'] = sensitive_attr

    # make sure everything is lining up
    assert all((sens == 'African-American') == (x_values['race'] == protected_class))

    # These are the categorical_features = [1, 4, 5, 6, 7, 8], if needed in the future

    x_values['y'] = y_values

    return x_values


def get_and_preprocess_german():
    """"Handle processing of German.  We use a preprocessed version of German from Ustun et. al.
    https://arxiv.org/abs/1809.06514.  Thanks Berk!
    Parameters:
    ----------
    params : Params
    Returns:
    ----------
    Pandas data frame x_values of processed data, np.ndarray y_values, and list of column names
    """
    positive_outcome = 1
    negative_outcome = 0

    x_values = pd.read_csv("./data/german_raw.csv")
    y_values = x_values["GoodCustomer"]
    loan_purpose = x_values["PurposeOfLoan"]

    unique_purposes = np.unique(loan_purpose.values)
    new_cols = np.zeros((len(loan_purpose), len(unique_purposes)))
    for i, purpose in enumerate(loan_purpose):
        indx = list(unique_purposes).index(purpose)
        new_cols[i, indx] = 1

    x_values = x_values.drop(["GoodCustomer", "PurposeOfLoan"], axis=1)

    for i, purpose in enumerate(unique_purposes):
        x_values["loanpurpose"+purpose] = new_cols[:, i]

    print(len(x_values.columns))

    x_values['Gender'] = [1 if v == "Male" else 0 for v in x_values['Gender'].values]

    y_values = np.array([positive_outcome if p ==
                         1 else negative_outcome for p in y_values.values])
    categorical_features = [0, 1, 2] + list(range(9, x_values.shape[1]))

    output = {
        "x_values": x_values,
        "y_values": y_values,
        "column_names": list(x_values.columns),
        "cat_indices": categorical_features,
    }

    return output


def get_dataset_by_name(name):
    """Gets a data set by name.

    Arguments:
        name: the name of the dataset.
    Returns:
        dataset: the dataset.
    """
    if name == "compas":
        dataset = get_and_preprocess_compas_data()
    elif name == "german":
        dataset = get_and_preprocess_german()
    else:
        message = f"Unkown dataset {name}"
        raise NameError(message)
    dataset['name'] = name
    return dataset
