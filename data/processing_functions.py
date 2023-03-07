"""Routines for processing data.

This code originates from one of my other projects:
https://github.com/dylan-slack/
Modeling-Uncertainty-Local-Explainability/blob/main/bayes/data_routines.py."""
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import re as re

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


def _save_column_id_to_value_index_mapping(data: np.ndarray,
                                           column_ids: [int]):
    """
    Uses LabelEncoder on the data to save the mapping from column id to value index.
    Takes in a dataset in numpy and a list of column indices to create a nested dict from the column indices to
    indexed unique values in a column of the dataset. {col_id: {value_id: unique_value}}
    Needed for XAI methods that handle categorical features this way.
    """
    categorical_names = {}
    for column_id in column_ids:
        if column_id != 2:  # Ignore column 'Job' because it is already in int.
            # As in LIME example "Categorical features"
            # https://marcotcr.github.io/lime/tutorials/Tutorial%20-%20continuous%20and%20categorical%20features.html
            le = LabelEncoder()
            le.fit(data[:, column_id])
            data[:, column_id] = le.transform(data[:, column_id])
            categorical_names[column_id] = le.classes_.tolist()

    with open("german_column_id_to_values_mapping.json", "w") as f:
        json.dump(categorical_names, f)

    return data, categorical_names


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

    x_values = pd.read_csv("german_raw.csv")
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


def get_and_preprocess_german_short():
    """Preprocess german_short with only 10 variables.
    Returns: Pandas data frame x_values of processed data, np.ndarray y_values, and categorical_mapping {col_id: {value_id: unique_value}}
    ----------
    """
    positive_outcome = 1
    negative_outcome = 0

    x_values = pd.read_csv("german_raw_short.csv", keep_default_na=False)
    y_values = x_values["Risk"]
    x_values = x_values.drop(["Risk"], axis=1)
    x_values = x_values.drop((["Unnamed: 0"]), axis=1)

    # Bin Age into 4 categories
    x_values['Age'] = pd.cut(x_values['Age'], bins=[18, 25, 35, 60, 120],
                             labels=['student', 'young', 'adult', 'senior'])

    col_names = list(x_values.columns)

    # Transform target label to 0 and 1.
    y_values = np.array([positive_outcome if p == "good" else negative_outcome for p in y_values.values])

    # Transform categorical values to int with LabelEncoder
    data_columns = list(x_values.columns)
    categorical_col_names = ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']

    categorical_col_ids = [data_columns.index(col) for col in categorical_col_names]

    x_values, categorical_mapping = _save_column_id_to_value_index_mapping(x_values.to_numpy(), categorical_col_ids)
    x_values = pd.DataFrame(x_values, columns=col_names, dtype=int)
    output = {
        "x_values": x_values,
        "y_values": y_values,
        "column_names": col_names
    }

    return output, categorical_mapping


def get_and_preprocess_titanic():
    """Load and preprocess Titanic dataset
    Returns: Pandas data frame x_values of processed data, np.ndarray y_values, and categorical_mapping {col_id: {value_id: unique_value}}
    ----------
    """
    positive_outcome = 1
    negative_outcome = 0

    raw_data = pd.read_csv("titanic_raw.csv", dtype={'Age': np.float64})

    # Split data in train and test
    train, test = train_test_split(raw_data, test_size=0.2)
    full_data = [train, test]

    # With the number of siblings/spouse and the number of children/parents we can create new feature called Family Size.
    for dataset in full_data:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    # Fare also has some missing value and we will replace it with the median. then we categorize it into 4 ranges.
    for dataset in full_data:
        dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

    # Age we have plenty of missing values in this feature. # generate random numbers between (mean - std) and
    # (mean + std). then we categorize age into 5 range.
    for dataset in full_data:
        age_avg = dataset['Age'].mean()
        age_std = dataset['Age'].std()
        age_null_count = dataset['Age'].isnull().sum()

        age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
        dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
        dataset['Age'] = dataset['Age'].astype(int)

    # Name: inside this feature we can find the title of people.
    def get_title(name):
        title_search = re.search(' ([A-Za-z]+)\.', name)
        # If the title exists, extract and return it.
        if title_search:
            return title_search.group(1)
        return ""

    # Fill embarked by mean values
    for dataset in full_data:
        dataset['Embarked'] = dataset['Embarked'].fillna('S')

    for dataset in full_data:
        dataset['Title'] = dataset['Name'].apply(get_title)
        dataset['Title'] = dataset['Title'].replace(
            ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    ### Data Cleaning ###
    for dataset in full_data:
        # Mapping Sex
        dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)

        # Mapping titles
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)

        # Mapping Embarked
        dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

        # Mapping Fare
        dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
        dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)

        # Mapping Age
        dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[dataset['Age'] > 64, 'Age'] = 4

    # Feature selecion
    drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'FamilySize']
    train = train.drop(drop_elements, axis=1)
    test = test.drop(drop_elements, axis=1)

    mapping_dict = {train.columns.get_loc("Sex") - 1: ["female", "male"],  # -1 because label in on 0 index
                    train.columns.get_loc("Embarked") - 1: ["S", "C", "Q"],
                    train.columns.get_loc("Title") - 1: ["Mr", "Miss", "Mrs", "Master", "Rare"],
                    train.columns.get_loc("Fare") - 1: ["0-7.91", "7.91-14.454", "14.454-31", "31+"],
                    train.columns.get_loc("Age") - 1: ["0-16", "16-32", "32-48", "48-64", "64+"]}

    # Save mapping dict to disk
    with open("titanic_column_id_to_values_mapping.json", "w") as f:
        json.dump(mapping_dict, f)
    col_names = list(train.columns[1:])

    output = {
        "train": train,
        "test": test,
        "column_names": col_names
    }

    return output, mapping_dict


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
    elif name == "titanic":
        dataset = get_and_preprocess_titanic()
    else:
        message = f"Unkown dataset {name}"
        raise NameError(message)
    dataset['name'] = name
    return dataset
