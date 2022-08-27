"""Measure interaction effects."""""
import numpy as np

from explain.actions.utils import gen_parse_op_text
from explain.feature_interaction import FeatureInteraction

NUM_FEATURES_TO_COMPUTE_INTERACTIONS = 4
NUM_TO_SAMPLE = 40


def get_text(interactions: list[tuple], feature_names: list, parse_op: str):
    """Gets the interaction text."""

    if len(parse_op) > 0:
        filtering_text = f"For the model's predictions on instances instances where <b>{parse_op}</b>,"
    else:
        filtering_text = "For the model's predictions on the data,"

    output = (f"{filtering_text} most significant feature interaction effects are as follows, "
              "where <em>higher</em> values correspond to <em>greater</em> interactions.<br><br>")

    for interaction in interactions:
        i, j, effect = interaction
        f1, f2 = feature_names[i], feature_names[j]
        effect = round(effect, 3)
        this_interaction = f"<b>{f1}</b>+<b>{f2}</b>: {effect}<br>"
        output += this_interaction

    return output


def measure_interaction_effects(conversation, parse_text, i, **kwargs):
    """Gets the interaction scores

    Arguments:
        conversation: The conversation object
        parse_text: The parse text for the question
        i: Index in the parse
        **kwargs: additional kwargs
    """

    # Filtering text
    parse_op = gen_parse_op_text(conversation)

    # The filtered dataset
    data = conversation.temp_dataset.contents['X']

    # Probability predicting func
    predict_proba = conversation.get_var('model_prob_predict').contents

    # Categorical features
    cat_features = conversation.get_var('dataset').contents['cat']
    cat_feature_names = [data.columns[i] for i in cat_features]

    interaction_explainer = FeatureInteraction(data=data,
                                               prediction_fn=predict_proba,
                                               cat_features=cat_feature_names)

    # Figure out which features to use by taking top k features
    ids = list(data.index)
    regen = conversation.temp_dataset.contents['ids_to_regenerate']
    mega_explainer_exp = conversation.get_var('mega_explainer').contents
    explanations = mega_explainer_exp.get_explanations(ids,
                                                       data,
                                                       ids_to_regenerate=regen,
                                                       save_to_cache=False)

    # Store the feature importance arrays
    feature_importances = []
    for i, current_id in enumerate(ids):
        list_exp = explanations[current_id].list_exp
        list_imps = [coef[1] for coef in list_exp]
        feature_importances.append(list_imps)
    feature_importances = np.array(feature_importances)
    mean_feature_importances = np.mean(np.abs(feature_importances), axis=0)

    # Get the names of the features
    topk_features = np.argsort(mean_feature_importances)[-NUM_FEATURES_TO_COMPUTE_INTERACTIONS:]
    topk_names = [data.columns[i] for i in topk_features]

    interactions = []
    for i, f1 in enumerate(topk_names):
        for j in range(i, len(topk_names)):
            if i == j:
                continue
            f2 = topk_names[j]
            i_score = interaction_explainer.feature_interaction(f1,
                                                                f2,
                                                                number_sub_samples=NUM_TO_SAMPLE)
            interactions.append((i, j, i_score))

    # Sort to the highest scoring first
    interactions = sorted(interactions, key=lambda x: x[2], reverse=True)

    return get_text(interactions, topk_names, parse_op), 1
