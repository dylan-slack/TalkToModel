"""Contains a representation of the conversation.

The file contains a representation of the conversation. The conversation
class contains routines to write variable to the conversation, read those
variables and print a representation of them.
"""
import copy
from typing import Union

import gin
import pandas as pd
from pandas import Series

from explain.dataset_description import DatasetDescription


class Variable:
    """A variable in the conversation."""

    def __init__(self, name: str, contents, kind: str):
        """Init.

        Arguments:
            name: The name of the variable
            contents: The data of the variable
            kind: the type of the variable
        """
        self.name = name
        self.contents = contents
        self.type = kind

    def update_name(self, name: str) -> None:
        """Updates name."""
        self.name = name

    def update_contents(self, contents) -> None:
        """Updates name."""
        self.contents = contents

    def update_type(self, new_type: str) -> None:
        """Updates name."""
        self.type = new_type

    def __repr__(self):
        message = f"{self.type} variable named {self.name}"
        return message


@gin.configurable
class Conversation:
    """The conversation object.

    This class defines the state of the conversation.
    """

    def __init__(self,
                 class_names: dict = None,
                 rounding_precision: int = 3,
                 index_col: int = 0,
                 target_var_name: str = "y",
                 default_metric: str = "accuracy",
                 eval_file_path: str = None,
                 feature_definitions: dict = None):
        """

        Args:
            data_folder:
            class_names:
            rounding_precision:
            index_col:
            target_var_name:
            default_metric:
            eval_file_path:
        """
        self.stored_vars = {}
        self.temp_dataset = None
        self.class_names = class_names

        self.rounding_precision = rounding_precision
        self.feature_definitions = feature_definitions

        # Unique per session
        self.parse_operation = []
        # Unique per session
        self.last_parse_string = []

        self.username = "unknown"

        # The description
        self.describe = DatasetDescription(index_col=index_col,
                                           target_var_name=target_var_name,
                                           eval_file_path=eval_file_path)

        # Set initial followup to brief description about the data and model
        self.followup = self.describe.get_text_description()

        self.default_metric = default_metric

    def get_feature_definition(self, feature_name):
        """Gets semantic feature definition."""
        if feature_name not in self.feature_definitions or self.feature_definitions is None:
            return None
        else:
            return self.feature_definitions[feature_name]

    def get_class_name_from_label(self, label: int):
        """Gets the class name from label"""
        if self.class_names is None:
            return str(label)
        else:
            return self.class_names[label]

    def set_user_name(self, username: str):
        self.username = username

    def _store_var(self, var: Variable):
        """Stores the variable."""
        self.stored_vars[var.name] = var

    def _update_temp_dataset(self, temp_dataset: dict):
        """Updates temp data set."""
        self.temp_dataset = temp_dataset

    def add_var(self, name: str, contents, kind: str):
        """Adds a variable with arbitrary contents."""
        var = Variable(name, contents, kind)
        self._store_var(var)
        return var

    def get_training_data_ids(self):
        """Gets the ids for the training data."""
        dataset = self.stored_vars["dataset"].contents["X"]
        return list(dataset.index)

    def add_dataset(self,
                    data: pd.DataFrame,
                    y_value: Series,
                    categorical: list[str],
                    numeric: list[str]):
        """Stores data as the dataset in the conversation."""
        dataset = {
            'X': data,
            'y': y_value,
            'cat': categorical,
            'numeric': numeric,
            'ids_to_regenerate': []
        }
        var = Variable(name='dataset', contents=dataset, kind='dataset')
        self._store_var(var)
        return var

    def build_temp_dataset(self, save=True):
        """Builds a temporary data set for filtering modifications."""
        temp_dataset = copy.deepcopy(self.get_var('dataset'))
        if save:
            self._update_temp_dataset(temp_dataset)
            self.parse_operation = []
        return temp_dataset

    def add_interpretable_parse_op(self, text):
        """Adds a parse operation to the text."""
        self.parse_operation.append(text)

    def clear_temp_dataset(self):
        """Clears the created temp data set."""
        del self.temp_dataset

    def get_var(self, name):
        """Gets a named variable from the variable store."""
        return self.stored_vars[name]

    def store_last_parse(self, string):
        """Stores the info on the last parse."""
        self.last_parse_string.append(string)

    def get_last_parse(self) -> Union[str, list]:
        """Gets the last parse string."""
        if len(self.last_parse_string) == 0:
            return ""
        return self.last_parse_string

    def store_followup_desc(self, string):
        """Store the last printout."""
        self.followup = string

    def get_followup_desc(self):
        """Gets the last print out."""
        return self.followup

    def __contains__(self, name):
        return name in self.stored_vars

    def __repr__(self):
        """Could be cool to change this to a graph representation."""
        out = '<p> Stored Variables:<br>'
        for var_name in self.stored_vars:
            out += f'&emsp;{var_name}<br>'
        out += '</p>'
        return out


def fork_conversation(conversation: Conversation, username: str):
    """Forks a conversation by doing a recursive copy.

    This operation may prove costly for larger models. We can think about setting up parameter
    sharing between models in the fork if this is an issue.
    """
    new_conversation = copy.deepcopy(conversation)
    new_conversation.set_user_name(username)
    return new_conversation
