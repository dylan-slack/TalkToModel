"""Data summary operation."""


# Note, these are hardcode for compas!
def data_operation(conversation, parse_text, i, **kwargs):
    """Data summary operation."""
    description = conversation.describe.get_dataset_description()
    text = f"The data contains information related to <b>{description}</b>.<br><br>"

    # List out the feature names
    f_names = list(conversation.temp_dataset.contents['X'].columns)
    f_string = "<ul>"
    for fn in f_names:
        f_string += f"<li>{fn}</li>"
    f_string += "</ul>"
    df = conversation.temp_dataset.contents['X']
    text += f"The exact feature names in the data are listed as follows:{f_string}<br><br>"

    # Summarize performance
    model = conversation.get_var('model').contents
    score = conversation.describe.get_eval_performance(model, conversation.default_metric)

    # Note, if no eval data is specified this will return an empty string and nothing will happen.
    if score != "":
        text += score
        text += "<br><br>"

    # Create more in depth description of the data, summarizing a few statistics
    rest_of_text = ""
    rest_of_text += "Here's a more in depth summary of the data.<br><br>"

    for i, f in enumerate(f_names):
        mean = round(df[f].mean(), conversation.rounding_precision)
        std = round(df[f].std(), conversation.rounding_precision)
        min_v = round(df[f].min(), conversation.rounding_precision)
        max_v = round(df[f].max(), conversation.rounding_precision)
        new_feature = (f"{f}: The mean is {mean}, one standard deviation is {std},"
                       f" the minimum value is {min_v}, and the maximum value is {max_v}")
        new_feature += "<br><br>"

        rest_of_text += new_feature

    text += "Let me know if you want to see an in depth description of the dataset statistics.<br><br>"
    conversation.store_followup_desc(rest_of_text)

    return text, 1
