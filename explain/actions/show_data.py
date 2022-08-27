"""Function to show data instances.

For single instances, this function prints out the feature values. For many instances,
it returns the mean.
"""
import gin

from explain.actions.utils import gen_parse_op_text


@gin.configurable
def show_operation(conversation, parse_text, i, n_features_to_show=float("+inf"), **kwargs):
    """Generates text that shows an instance."""
    data = conversation.temp_dataset.contents['X']

    parse_op = gen_parse_op_text(conversation)
    if len(parse_op) > 0:
        intro_text = f"For the data with <b>{parse_op}</b>,"
    else:
        intro_text = "For all the instances in the data,"
    rest_of_info_string = "The rest of the features are<br><br>"
    init_len = len(rest_of_info_string)
    if len(data) == 0:
        return "There are no instances in the data that meet this description.", 0
    if len(data) == 1:
        return_string = f"{intro_text} the features are<br><br>"

        for i, feature_name in enumerate(data.columns):
            feature_value = data[feature_name].values[0]
            text = f"{feature_name}: {feature_value}<br>"
            if i < n_features_to_show:
                return_string += text
            else:
                rest_of_info_string += text
    else:
        """
        return_string = f"{intro_text} the feature values are on average:<br><br>"
        for i, feature_name in enumerate(data.columns):
            feature_value = round(data[feature_name].mean(), conversation.rounding_precision)
            text = f"{feature_name}: {feature_value}<br>"
            if i < n_features_to_show:
                return_string += text
            else:
                rest_of_info_string += text
        """
        instance_ids = str(list(data.index))
        return_string = f"{intro_text} the instance id's are:<br><br>"
        return_string += instance_ids
        return_string += "<br><br>Which one do you want to see?<br><br>"

    # If we've written additional info to this string
    if len(rest_of_info_string) > init_len:
        return_string += "<br><br>I've truncated this instance to be concise. Let me know if you"
        return_string += " want to see the rest of it.<br><br>"
        conversation.store_followup_desc(rest_of_info_string)
    return return_string, 1
