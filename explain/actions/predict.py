"""Prediction operation."""
import numpy as np

from explain.actions.utils import gen_parse_op_text, get_parse_filter_text


def predict_operation(conversation, parse_text, i, max_num_preds_to_print=1, **kwargs):
    """The prediction operation."""
    model = conversation.get_var('model').contents
    data = conversation.temp_dataset.contents['X']

    if len(conversation.temp_dataset.contents['X']) == 0:
        return 'There are no instances that meet this description!', 0

    model_predictions = model.predict(data)

    # Format return string
    return_s = ""

    filter_string = gen_parse_op_text(conversation)

    if len(model_predictions) == 1:
        return_s += f"The instance with <b>{filter_string}</b> is predicted "
        if conversation.class_names is None:
            prediction_class = str(model_predictions[0])
            return_s += f"<b>{prediction_class}</b>"
        else:
            class_text = conversation.class_names[model_predictions[0]]
            return_s += f"<b>{class_text}</b>."
    else:
        intro_text = get_parse_filter_text(conversation)
        return_s += f"{intro_text} the model predicts:"
        unique_preds = np.unique(model_predictions)
        return_s += "<ul>"
        for j, uniq_p in enumerate(unique_preds):
            return_s += "<li>"
            freq = np.sum(uniq_p == model_predictions) / len(model_predictions)
            round_freq = str(round(freq * 100, conversation.rounding_precision))

            if conversation.class_names is None:
                return_s += f"<b>class {uniq_p}</b>, {round_freq}%"
            else:
                class_text = conversation.class_names[uniq_p]
                return_s += f"<b>{class_text}</b>, {round_freq}%"
            return_s += "</li>"
        return_s += "</ul>"
    return_s += "<br>"
    return return_s, 1
