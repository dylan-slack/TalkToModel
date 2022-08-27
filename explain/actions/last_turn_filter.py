"""Last turn filtering operation."""
from copy import deepcopy

from explain.actions.filter import filter_operation
from explain.actions.what_if import what_if_operation


def get_most_recent_filter(operations):
    for op in operations:
        for word in op.split(' '):
            if "filter" == word or "previousfilter" == word or "predictionfilter" == word or "labelfilter" == word:
                return op
    return None


def last_turn_filter(conversation, parse_text, i, **kwargs):
    """Last turn operation for filtering.

    This function sets the dataset to the filtering operation performed in the last operation,
    overwriting any filtering that may have already occured.

    TODO(dylan): Maybe don't have it overwrite?
    """

    # Save a copy of the conversation and run the last turn operation
    # saving the last filtered dataset to the conversation and generating
    # the interpretable parse string for that conversation
    # TODO(dylan): find a way to remove this deepcopy. It already slows things
    # down a bit using COMPAS.
    last_turn_conversation = deepcopy(conversation)

    # Now newest operation is first
    last_turn_operations = last_turn_conversation.get_last_parse()[::-1]
    most_recent_filter = get_most_recent_filter(last_turn_operations)

    # If previousoperation is None, we haven't found any filters
    # just pass and do it on all the data
    if most_recent_filter is None:
        return "", 1

    # Remove the filter operations so that only the actions
    # like explanations or predictions will be run
    just_filter_op = {
        "filter": filter_operation,
        "change": what_if_operation,
        "previousfilter": last_turn_filter,
        "predictionfilter": filter_operation,
        "labelfilter": filter_operation
    }

    # Run action here
    parsed_text = most_recent_filter.split(" ")
    is_or = False
    return_statement = ""

    # Delete most recent filter from stack, so we don't keep looping
    for i in range(len(last_turn_conversation.last_parse_string)):
        j = len(last_turn_conversation.last_parse_string) - i - 1
        if last_turn_conversation.last_parse_string[j] == most_recent_filter:
            del last_turn_conversation.last_parse_string[j]

    for i, p_text in enumerate(parsed_text):
        if parsed_text[i] in just_filter_op:
            action_output, action_status = just_filter_op[p_text](
                last_turn_conversation, parsed_text, i, is_or=is_or)

            return_statement += action_output
            if action_status == 0:
                break

            # This is a bit ugly but basically if an or occurs
            # we hold onto the or until we need to flip it off, i.e. filtering
            # happens
            if is_or is True and just_filter_op[p_text] == "filter":
                is_or = False

        if p_text == "or":
            is_or = True

    # Store the previous filtered dataset as the current dataset and also
    # the interpretable parse text
    conversation.temp_dataset.contents = last_turn_conversation.temp_dataset.contents
    conversation.parse_operation = last_turn_conversation.parse_operation

    return "", 1
