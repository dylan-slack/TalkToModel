"""Contains a function returning a dict mapping the key word for actions to the function.

This function is used to generate a dictionary of all the actions and the corresponding function.
This functionality is used later on to determine the set of allowable operations and what functions
to run when parsing the grammar.
"""
from explain.actions.data_summary import data_operation
from explain.actions.count_data_points import count_data_points
from explain.actions.define import define_operation
from explain.actions.explanation import explain_operation
from explain.actions.feature_stats import feature_stats
from explain.actions.important import important_operation
from explain.actions.filter import filter_operation
from explain.actions.followup import followup_operation
from explain.actions.function import function_operation
from explain.actions.interaction_effects import measure_interaction_effects
from explain.actions.labels import show_labels_operation
from explain.actions.last_turn_filter import last_turn_filter
from explain.actions.last_turn_operation import last_turn_operation
from explain.actions.mistakes import show_mistakes_operation
from explain.actions.model import model_operation
from explain.actions.predict import predict_operation
from explain.actions.prediction_likelihood import predict_likelihood
from explain.actions.score import score_operation
from explain.actions.self import self_operation
from explain.actions.show_data import show_operation
from explain.actions.what_if import what_if_operation


def get_all_action_functions_map():
    """Gets a dictionary mapping all the names of the actions in the parse tree to their functions."""
    actions = {
        'interact': measure_interaction_effects,
        'countdata': count_data_points,
        'filter': filter_operation,
        'explain': explain_operation,
        'predict': predict_operation,
        'self': self_operation,
        'previousfilter': last_turn_filter,
        'previousoperation': last_turn_operation,
        'data': data_operation,
        'followup': followup_operation,
        'important': important_operation,
        'show': show_operation,
        'change': what_if_operation,
        'likelihood': predict_likelihood,
        'model': model_operation,
        'function': function_operation,
        'score': score_operation,
        'label': show_labels_operation,
        'mistake': show_mistakes_operation,
        'statistic': feature_stats,
        'define': define_operation,
        'predictionfilter': filter_operation,
        'labelfilter': filter_operation
    }
    return actions
