"""Count the number of elements in the data."""
from explain.actions.utils import gen_parse_op_text


def count_data_points(conversation, parse_text, i, **kwargs):
    """Gets the number of elements in the data.

    Arguments:
        conversation: The conversation object
        parse_text: The parse text for the question
        i: Index in the parse
        **kwargs: additional kwargs
    """
    data = conversation.temp_dataset.contents['X']
    num_elements = len(data)

    parse_op = gen_parse_op_text(conversation)

    if len(parse_op) > 0:
        description_text = f" where <b>{parse_op}</b>"
    else:
        description_text = ""

    message = f"There are <b>{num_elements} items</b> in the data{description_text}."

    message += "<br><br>"
    message += "Let me know if you want to see their ids."
    ids = list(data.index)
    rest_of_text = str(ids)
    conversation.store_followup_desc(rest_of_text)
    return message, 1
