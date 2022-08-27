import numpy as np

from explain.actions.utils import gen_parse_op_text

SINGLE_INSTANCE_TEMPLATE = """
The model predicts the instance with <b>{filter_string}</b> as:
<b>
"""


def predict_likelihood(conversation, parse_text, i, **kwargs):
    """The prediction likelihood operation."""
    predict_proba = conversation.get_var('model_prob_predict').contents
    model = conversation.get_var('model').contents
    data = conversation.temp_dataset.contents['X'].values

    if len(conversation.temp_dataset.contents['X']) == 0:
        return 'There are no instances that meet this description!', 0

    model_prediction_probabilities = predict_proba(data)
    model_predictions = model.predict(data)
    num_classes = model_prediction_probabilities.shape[1]

    # Format return string
    return_s = ""

    filter_string = gen_parse_op_text(conversation)

    if model_prediction_probabilities.shape[0] == 1:
        return_s += f"The model predicts the instance with <b>{filter_string}</b> as:"
        return_s += "<ul>"
        for c in range(num_classes):
            proba = round(model_prediction_probabilities[0, c]*100, conversation.rounding_precision)
            return_s += "<li>"
            if conversation.class_names is None:
                return_s += f"class {str(c)}</b>"
            else:
                class_text = conversation.class_names[c]
                return_s += f"<b>{class_text}</b>"
            return_s += f" with <b>{str(proba)}%</b> probability"
            return_s += "</li>"
        return_s += "</ul>"
    else:
        if len(filter_string) > 0:
            filtering_text = f" where <b>{filter_string}</b>"
        else:
            filtering_text = ""
        return_s += f"Over {data.shape[0]} cases{filtering_text} in the data, the model predicts:"
        unique_preds = np.unique(model_predictions)
        return_s += "<ul>"
        for j, uniq_p in enumerate(unique_preds):
            return_s += "<li>"
            freq = np.sum(uniq_p == model_predictions) / len(model_predictions)
            round_freq = str(round(freq*100, conversation.rounding_precision))

            if conversation.class_names is None:
                return_s += f"<b>class {uniq_p}</b>, <b>{round_freq}%</b>"
            else:
                class_text = conversation.class_names[uniq_p]
                return_s += f"<b>{class_text}</b>, <b>{round_freq}%</b>"
            return_s += " of the time</li>"
        return_s += "</ul>"
    return_s += "<br>"
    return return_s, 1
