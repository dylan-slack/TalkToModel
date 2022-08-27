"""Score operation.

This operation computes a score metric on the data or the eval data.
"""
from explain.actions.utils import gen_parse_op_text


def score_operation(conversation, parse_text, i, **kwargs):
    """Self description."""

    # Get the name of the metric
    metric = parse_text[i+1]

    if metric == "default":
        metric = conversation.default_metric

    model = conversation.get_var('model').contents

    data = conversation.temp_dataset.contents['X']
    y_true = conversation.temp_dataset.contents['y']
    y_pred = model.predict(data)

    filter_string = gen_parse_op_text(conversation)
    if len(filter_string) <= 0:
        data_name = "the <b>all</b> the data"
    else:
        data_name = f"the data where <b>{filter_string}</b>"
    text = conversation.describe.get_score_text(y_true,
                                                y_pred,
                                                metric,
                                                conversation.rounding_precision,
                                                data_name)

    text += "<br><br>"
    return text, 1
