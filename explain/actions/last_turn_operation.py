"""Last turn operation."""
from copy import deepcopy

from explain.actions.data_summary import data_operation
from explain.actions.define import define_operation
from explain.actions.explanation import explain_operation
from explain.actions.feature_stats import feature_stats
from explain.actions.important import important_operation
from explain.actions.followup import followup_operation
from explain.actions.interaction_effects import measure_interaction_effects
from explain.actions.labels import show_labels_operation
from explain.actions.function import function_operation
from explain.actions.mistakes import show_mistakes_operation
from explain.actions.model import model_operation
from explain.actions.predict import predict_operation
from explain.actions.prediction_likelihood import predict_likelihood
from explain.actions.self import self_operation
from explain.actions.show_data import show_operation
from explain.actions.what_if import what_if_operation
from explain.actions.score import score_operation


def get_most_recent_ops(operations, op_list):
    for op in operations:
        for word in op.split(' '):
            if word in op_list:
                return op
    return None


def last_turn_operation(conversation, parse_text, i, **kwargs):
    """Last turn operation.

    The function computes the last set of operations (excluding filtering) on the
    current temp_dataset. This feature enables things like doing filtering and then
    running whatever set of operations were run last.
    """

    # Just get the operations run in the last parse
    last_turn_operations = conversation.get_last_parse()[::-1]

    # Store the conversation
    last_turn_conversation = deepcopy(conversation)

    # Remove the filter operations so that only the actions
    # like explantions or predictions will be run
    # TODO(dylan): find a way to make this a bit cleaner... right now both this function
    # and the dictionary in get_action_function need to be updated with new functions. We can't
    # import from that file because of circular imports... should be a way to do this better though
    # so you don't have to update in both places.
    excluding_filter_ops = {
        'explain': explain_operation,
        'predict': predict_operation,
        'self': self_operation,
        'previousoperation': last_turn_operation,
        'data': data_operation,
        'followup': followup_operation,
        'important': important_operation,
        'show': show_operation,
        'likelihood': predict_likelihood,
        'model': model_operation,
        'function': function_operation,
        'score': score_operation,
        'interact': measure_interaction_effects,
        'label': show_labels_operation,
        'mistake': show_mistakes_operation,
        'statistic': feature_stats,
        'change': what_if_operation,
        'define': define_operation
    }

    most_recent_ops = get_most_recent_ops(last_turn_operations, excluding_filter_ops)

    # in case we can't find anything
    if most_recent_ops is None:
        return "", 1

    # Delete most recent op from prev parses so that we don't recurse to it again
    # if we don't do this, it may recurse indefinetly
    for i in range(len(last_turn_conversation.last_parse_string)):
        j = len(last_turn_conversation.last_parse_string) - i - 1
        if last_turn_conversation.last_parse_string[j] == most_recent_ops:
            del last_turn_conversation.last_parse_string[j]

    parsed_text = most_recent_ops.split(' ')
    return_statement = ''
    for i, p_text in enumerate(parsed_text):
        if parsed_text[i] in excluding_filter_ops:
            action_output, action_status = excluding_filter_ops[p_text](last_turn_conversation,
                                                                        parsed_text, i,
                                                                        is_or=False)
            return_statement += action_output
            if action_status == 0:
                break

    return return_statement, 1
