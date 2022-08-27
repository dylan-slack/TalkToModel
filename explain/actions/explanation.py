"""Explanation action.

This action controls the explanation generation operations.
"""
from explain.actions.utils import gen_parse_op_text


def explain_operation(conversation, parse_text, i, **kwargs):
    """The explanation operation."""
    # TODO(satya): replace explanation generation code here

    # Example code loading the model
    data = conversation.temp_dataset.contents['X']

    if len(conversation.temp_dataset.contents['X']) == 0:
        return 'There are no instances that meet this description!', 0

    regen = conversation.temp_dataset.contents['ids_to_regenerate']
    parse_op = gen_parse_op_text(conversation)

    # Note, do we want to remove parsing for lime -> mega_explainer here?
    if parse_text[i+1] == 'features' or parse_text[i+1] == 'lime':
        # mega explainer explanation case
        mega_explainer_exp = conversation.get_var('mega_explainer').contents
        full_summary, short_summary = mega_explainer_exp.summarize_explanations(data,
                                                                                filtering_text=parse_op,
                                                                                ids_to_regenerate=regen)
        conversation.store_followup_desc(full_summary)
        return short_summary, 1
    if parse_text[i+1] == 'cfe':
        dice_tabular = conversation.get_var('tabular_dice').contents
        out = dice_tabular.summarize_explanations(data,
                                                  filtering_text=parse_op,
                                                  ids_to_regenerate=regen)
        additional_options, short_summary = out
        conversation.store_followup_desc(additional_options)
        return short_summary, 1
    if parse_text[i+1] == 'shap':
        # This is when a user asks for a shap explanation
        raise NotImplementedError
    raise NameError(f"No explanation operation defined for {parse_text}")
