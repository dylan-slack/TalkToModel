"""Executes actions for parsed canonical utterances.

This file implements routines to take actions in the conversation, returning
outputs to the user. Actions in the conversation are called `operations` and
are things like running an explanation or performing filtering.
"""
from flask import Flask

from explain.actions.explanation import explain_feature_importances
from explain.actions.filter import filter_operation
from explain.actions.interaction_effects import measure_interaction_effects
from explain.conversation import Conversation
from explain.actions.get_action_functions import get_all_action_functions_map

app = Flask(__name__)


def run_action(conversation: Conversation,
               parse_tree,
               parsed_string: str,
               actions=get_all_action_functions_map(),
               build_temp_dataset: bool = True) -> str:
    """Runs the action and updates the conversation object

    Arguments:
        build_temp_dataset: Whether to use the temporary dataset stored in the conversation
                            or to rebuild the temporary dataset from scratch.
        actions: The set of avaliable actions
        parsed_string: The grammatical text
        conversation: The conversation object, see `conversation.py`
        parse_tree: The parse tree of the canonical utterance. Note, currently, this is not used,
                    and we compute the actions from the parsed text.
    """
    if parse_tree:
        pretty_parse_tree = parse_tree.pretty()
        app.logger.info(f'Parse tree {pretty_parse_tree}')

    return_statement = ''

    # Will rebuilt the temporary dataset if requested (i.e, for filtering from scratch)
    if build_temp_dataset:
        conversation.build_temp_dataset()

    parsed_text = parsed_string.split(' ')
    is_or = False

    for i, p_text in enumerate(parsed_text):
        if parsed_text[i] in actions:
            action_return, action_status = actions[p_text](
                conversation, parsed_text, i, is_or=is_or)
            return_statement += action_return

            # If operation fails, return error output to user
            if action_status == 0:
                break

            # This is a bit ugly but basically if an or occurs
            # we hold onto the or until we need to flip it off, i.e. filtering
            # happens
            if is_or is True and actions[p_text] == 'filter':
                is_or = False

        if p_text == 'or':
            is_or = True

    # Store 1-turn parsing
    conversation.store_last_parse(parsed_string)

    while return_statement.endswith("<br>"):
        return_statement = return_statement[:-len("<br>")]

    return return_statement


def run_action_by_id(conversation: Conversation,
                     question_id: int,
                     instance_id: int,
                     feature_name: str = None,
                     build_temp_dataset: bool = True) -> str:
    """
    Runs the action selected by an ID instead of text parsing and updates the conversation object.

    conversation: Conversation, Conversation Object
    question_id: int, id of the question as defined in question_bank.csv
    instance_id: int, id of the instance that should be explained. Needed for local explanations
    feature_name: str, string of the feature name the question is about (if specified)
    build_temp_dataset: bool = True If building tmp_dataset is needed.
    """
    if build_temp_dataset:
        conversation.build_temp_dataset()

    # Create parse text as filter works with it
    parse_text = f"filter id {instance_id}".split(" ")
    _ = filter_operation(conversation, parse_text, 0)
    # Get tmp dataset to perform explanation on (here, single ID will be in tmp_dataset)
    data = conversation.temp_dataset.contents['X']
    regen = conversation.temp_dataset.contents['ids_to_regenerate']

    if question_id == 0:
        explanation = explain_lime(conversation, data, f"ID {instance_id}", regen)
        return explanation[0]
    if question_id == 1:
        # How important is each attribute to the model's predictions?
        explanation = measure_interaction_effects(conversation)
        return explanation[0]
    else:
        return f"This is a mocked answer to your question with id {question_id}."
