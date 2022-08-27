"""Show data labels"""
import gin

from explain.actions.utils import get_parse_filter_text


@gin.configurable
def show_labels_operation(conversation, parse_text, i, n_features_to_show=float("+inf"), **kwargs):
    """Generates text that shows labels."""
    y_values = conversation.temp_dataset.contents['y']
    intro_text = get_parse_filter_text(conversation)

    if len(y_values) == 0:
        return "There are no instances in the data that meet this description.", 0
    if len(y_values) == 1:
        label = y_values.item()
        label_text = conversation.get_class_name_from_label(label)
        return_string = f"{intro_text} the label is <b>{label_text}</b>."
    else:
        return_string = f"{intro_text} the labels are:<br><br>"
        for index, label in zip(list(y_values.index), y_values):
            label_text = conversation.get_class_name_from_label(label)
            return_string += f"id {index} is labeled {label_text}<br>"

    return return_string, 1
