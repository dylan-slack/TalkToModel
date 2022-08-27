"""Describes the model."""


def model_operation(conversation, parse_text, i, **kwargs):
    """Model description."""

    objective = conversation.describe.get_dataset_objective()
    model = conversation.describe.get_model_description()
    text = f"I use a <em>{model}</em> model to {objective}.<br><br>"

    return text, 1
