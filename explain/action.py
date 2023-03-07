"""Executes actions for parsed canonical utterances.

This file implements routines to take actions in the conversation, returning
outputs to the user. Actions in the conversation are called `operations` and
are things like running an explanation or performing filtering.
"""
from flask import Flask

from explain.actions.explanation import explain_feature_importances, explain_cfe, \
    get_feature_importance_by_feature_id, explain_cfe_by_given_features, \
    explain_anchor_changeable_attributes_without_effect
from explain.actions.filter import filter_operation
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
                     feature_id: int = None,
                     build_temp_dataset: bool = True) -> str:
    """
    Runs the action selected by an ID instead of text parsing and updates the conversation object.

    conversation: Conversation, Conversation Object
    question_id: int, id of the question as defined in question_bank.csv
    instance_id: int, id of the instance that should be explained. Needed for local explanations
    feature_id: int, id of the feature name the question is about (if specified)
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
    instance_predicted_label = 0
    if feature_id is None:  # TODO: Accept feature name from user
        feature_id = 1
    feature_name = data.columns[feature_id]
    parse_op = f"ID {instance_id}"

    if question_id == 0:
        # Which attributes does the model use to make predictions?
        return f"The model uses the following attributes to make predictions: {', '.join(list(data.columns))}."
    if question_id == 1:
        # Does the model include [feature X] when making the prediction?

        explanation = get_feature_importance_by_feature_id(conversation, data, regen, feature_id)
        answer = "Yes it does. "
        return answer + explanation[0]
    if question_id == 2:
        # How important is each attribute to the model's predictions?
        # Create full feature explanations
        explanation = explain_feature_importances(conversation, data, parse_op, regen,
                                                  return_full_summary=True)
        return explanation[0]
    if question_id == 3:
        # How strong does [feature X] affect the prediction?
        explanation = get_feature_importance_by_feature_id(conversation, data, regen, feature_id)
        return explanation[0]

    if question_id == 4:
        # What are the most important attributes for this prediction?
        explanation = explain_feature_importances(conversation, data, parse_op, regen,
                                                  return_full_summary=False)
        answer = "The most important attributes for this prediction are: "
        return answer + explanation[0]
    if question_id == 5:
        # Why did the model give this particular prediction for this person?
        explanation = explain_feature_importances(conversation, data, parse_op, regen,
                                                  return_full_summary=False)
        answer = "The prediction can be explained by looking at the most important attributes. <br>"
        answer += "Here are the three most important ones: <br>"
        return answer + explanation[0]
    if question_id == 6:
        # What attributes of this person led the model to make this prediction?
        explanation = explain_feature_importances(conversation, data, parse_op, regen,
                                                  return_full_summary=False)
        answer = "The following 3 attributes of the person were the most important for the prediction. "
        return answer + explanation[0]
    if question_id == 7:
        # What would happen to the prediction if we changed [feature] for this person?
        explanation = explain_cfe_by_given_features(conversation, data, [feature_name])
        if explanation[1] == 0:
            answer = explanation[0] + feature_name + "."
        else:
            answer = explanation[0]
        return answer
    if question_id == 8:
        # How should this person change to get a different prediction?
        explanation = explain_cfe(conversation, data, parse_op, regen)
        return explanation[0]
    if question_id == 9:
        # How should this attribute change to get a different prediction?
        explanation = explain_cfe_by_given_features(conversation, data, [feature_name])
        return explanation[0]
    if question_id == 10:
        # Which changes to this person would still get the same prediction?
        explanation = explain_anchor_changeable_attributes_without_effect(conversation, data, parse_op, regen)
        return explanation[0]
    if question_id == 11:
        # Which maximum changes would not influence the class prediction?
        pass
    if question_id == 12:
        # What attributes must be present or absent to guarantee this prediction?
        pass
    else:
        return f"This is a mocked answer to your question with id {question_id}."
