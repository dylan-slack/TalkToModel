"""Followup operation."""


def followup_operation(conversation, parse_text, i, **kwargs):
    """Follow up operation.

    If there's an explicit option to followup, this command deals with it.
    """
    follow_up_text = conversation.get_followup_desc()
    if follow_up_text == "":
        return "Sorry, I'm a bit unsure what you mean... try again?", 0
    else:
        return follow_up_text, 1
