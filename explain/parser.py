"""The parser object.

This file contains the parser class and the grammar.

TODO(dylan): Refactor this file a bit more for clarity.
"""
import copy
from typing import Any

import numpy as np
import pandas as pd

from explain.grammar import GRAMMAR, CAT_FEATURES, TARGET_VAR, NUM_FEATURES


def get_parse_tree(decoded_text: str) -> tuple[Any, str]:
    """Generates the parse tree given some decoded text."""

    # Trim off generation
    trimmed_response = decoded_text.split("[e]")[-2].split('parsed:')[-1]
    trimmed_response += "[e]"

    # text_parser = Lark(self.g, parser="lalr")
    # return text_parser.parse(trimmed_response), trimmed_response
    return None, trimmed_response


def add_terminal_or(name: str, cur_terminals: str) -> str:
    """Adds a terminal or to a string of terminals."""

    if len(cur_terminals) == 0:
        cur_terminals = " " + "\" " + name + "\""
    else:
        cur_terminals += " | " + "\" " + name + "\""
    return cur_terminals


def add_nonterminal_or(name: str, cur_terminals: str) -> str:
    """Adds a nonterminal or to a string of nonterminals."""

    if len(cur_terminals) == 0:
        cur_terminals = " " + " " + name
    else:
        cur_terminals += " | " + " " + name
    return cur_terminals


class Parser:
    """The parser object.

    The routine controls creating the grammar.
    """

    def __init__(self,
                 cat_features: list[str],
                 num_features: list[str],
                 dataset: pd.DataFrame,
                 target: list[str]):
        """
        Init.

        Arguments:
            cat_features: The categorical feature names
            num_features: The numeric feature names
            dataset: The pd.DataFrame dataset
            target: The class names in a list where the ith index corresponds that class's name
        """

        assert len(cat_features) > 0 or len(num_features) > 0, "There are no features!"

        # The nonterminal for the available feature types, i.e., categorical and numeric
        available_feature_types = ""

        self.absolute_number_of_features = len(cat_features) + len(num_features) - 1

        # Store features and available values
        self.features = {}

        # Add the categorical features to the grammar if they exist
        self.categorical_features_grammar = ""
        if len(cat_features) > 0:
            self.categorical_features_grammar = self.format_cat_features(cat_features, dataset)
            available_feature_types += "catnames"

        # Create a new nonterminal for numerical features
        self.numerical_features_grammar = ""
        if len(num_features) > 0:
            self.numerical_features_grammar = self.format_num_features(num_features, dataset)
            if len(available_feature_types) == 0:
                available_feature_types += "numnames"
            else:
                available_feature_types += " | numnames"

        # Add the target variable into the grammar
        target_str = ""
        for t_val in np.unique(target):
            target_str = add_terminal_or(str(t_val), target_str)

        # If adding categorical to binary true false
        # if len(np.unique(target)) == 2 and 1 in target and 0 in target:
        #     target_str = add_terminal_or("true", target_str)
        #     target_str = add_terminal_or("false", target_str)

        self.target_var_grammar = TARGET_VAR.format(classes=target_str)

        # update the grammar
        self.available_feature_types = available_feature_types

        # Add a nonterminal for all the feature names, in case we want to parse these
        numerical_feature_names = ""
        for feature in num_features:
            numerical_feature_names = add_terminal_or(feature, numerical_feature_names)

        if numerical_feature_names != "":
            self.numerical_feature_names = "\nnumfeaturenames: " + numerical_feature_names
        else:
            self.numerical_feature_names = ""

        # Add all the feature names as well, this could be condensed with above
        all_feature_names = ""
        for feature in num_features:
            all_feature_names = add_terminal_or(feature, all_feature_names)
        for feature in cat_features:
            all_feature_names = add_terminal_or(feature, all_feature_names)
        self.all_feature_names = "\nallfeaturenames: " + all_feature_names

    def format_num_features(self, num_features: list[str], dataset: pd.DataFrame) -> str:
        """Formats numerical features grammar."""

        # the numerical non-terminals
        num_names = ""

        # The terminals for the numerical values
        all_num_values = ""

        for nf_orig in num_features:
            nf_lower = nf_orig.lower()

            # add the non-terminal for the feature name to the grammar
            num_names = add_nonterminal_or(nf_lower, num_names)

            # add the numerical values to the grammar
            num_values = sorted(dataset[nf_orig].unique())
            num_terminal_values = ""
            # currently, selecting the numerical options as every value in the column
            num_values_numeric = []
            num_values_numeric.extend(num_values)
            for i, num_val in enumerate(num_values):
                num_terminal_values = add_terminal_or(str(num_val), num_terminal_values)

            # Add the feature name and possible values to the grammar
            all_num_values += "\n" + f"{nf_lower}: \" {nf_lower}\" equality" + \
                              "( adhocnumvalues )"

            self.features[nf_lower] = num_values_numeric

        num_features_grammar = NUM_FEATURES.format(numfeaturenames=num_names)
        num_features_grammar += all_num_values
        return num_features_grammar

    def format_cat_features(self, cat_features: Any, dataset: Any):
        """Formats the categorical features in the grammar.

        Note, typing dataset as pd.DataFrame causing warnings for some reason.
        Leaving this as typed Any.
        """

        # The non-terminals for each categorical feature
        cat_names = ""

        # The values for each categorical feature in a terminal
        all_cat_values = ""

        # create a new non-terminal for each categorical feature
        for cf_orig in cat_features:
            cat_f_lower = cf_orig.lower()

            # the new non-terminal
            cat_names = add_nonterminal_or(cat_f_lower, cat_names)

            cat_values = dataset[cf_orig].unique().tolist()

            new_terminal = cat_f_lower + ":"

            # if cat_f_lower != "incorrect":
            self.features[cat_f_lower] = cat_values

            new_terminal += f" \" {cat_f_lower}\"" + " ( "

            # Add true / false to boolean 1 / 0 columns
            if len(cat_values) == 2 and 1 in cat_values and 0 in cat_values:
                new_terminal += " \" true\" | \" false\" |"

            for i, cf_v in enumerate(cat_values):
                new_terminal += f" \" {str(cf_v)}\""

                if i != len(cat_values) - 1:
                    new_terminal += " |"

            new_terminal += " )"

            self.features[cat_f_lower].extend(['true', 'false'])
            all_cat_values += "\n" + new_terminal

        categorical_features = CAT_FEATURES.format(catfeaturenames=cat_names)
        categorical_features += all_cat_values
        return categorical_features

    def get_topk_grammar_text(self):
        """Gets text for the available top k features."""
        grammar_text = ""
        for num in range(1, self.absolute_number_of_features + 1):
            grammar_text = add_terminal_or(str(num), grammar_text)
        return grammar_text

    def get_grammar(self, adhoc_grammar_updates: dict = None):
        """Gets the grammar.

        This function returns the compiled grammar. It also allows
        specifying adhoc_grammar_updates or additional features beyond
        those included in the original grammar. These should be
        specified in dictionary format where keys correspond to the
        non-terminal in the grammar and values are the corresponding values
        in a list of terminals.

        Arguments:
            adhoc_grammar_updates: A dictionary containing "adhoc" updates to the grammar. This
                                   just means updates to the grammar that are specific to the
                                   input provided by the user. Mostly, this is used for storing
                                   the inferred set of numerical values in the user input.
        Returns:
            grammar: The fully compiled grammar that is given to the guided decoding procedure.
        """
        if adhoc_grammar_updates is not None:
            final_aval = copy.deepcopy(self.available_feature_types)
            final_aval_values = ""
            for feat in adhoc_grammar_updates:
                # add the adhoc grammar updates to the list of avaliable feature types in the
                # special case of the *id* feature. Other adhoc updates are not considered as
                # available feature types because they could be anything.
                if "id" in feat:
                    final_aval += f" | {feat}"
                final_aval_values += f"\n{feat}: {adhoc_grammar_updates[feat]}"
            grammar = GRAMMAR.format(avaliablefeaturetypes=final_aval,
                                     topkvalues=self.get_topk_grammar_text())
            grammar += final_aval_values
        else:
            grammar = GRAMMAR.format(avaliablefeaturetypes=self.available_feature_types,
                                     topkvalues=self.get_topk_grammar_text())
        grammar += self.categorical_features_grammar
        grammar += self.numerical_features_grammar

        # Clean up parts of the grammar that will break if there are no categorical or
        # numerical features.
        if self.categorical_features_grammar == "":
            grammar = grammar.replace("| catnames", "")
        if self.numerical_features_grammar == "":
            grammar = grammar.replace("( numfeaturenames numupdates adhocnumvalues ) |", "")

        grammar += self.target_var_grammar
        grammar += self.numerical_feature_names
        grammar += self.all_feature_names

        return grammar

