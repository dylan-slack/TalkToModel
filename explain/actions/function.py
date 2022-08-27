"""Describes the system capabilities."""


def function_operation(conversation, parse_text, i, **kwargs):
    """function description."""

    text = ("I can answer questions about the model's predictions on the data."
            " For example, I could tell you why certain predictions were made,"
            " how likely different outcomes are, what needs to happen to get"
            " different predictions, or what would happen to the predictions"
            " if the features changed in a particular way.<br><br> I can perform"
            " these analyses on individual instances (i.e., using the instance id)"
            " or entire subgroups in the data.<br><br>If you want to see whether I"
            " can do something, try asking it!")

    return text, 1
