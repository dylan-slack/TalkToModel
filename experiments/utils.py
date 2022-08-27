"""Eval utils"""


def check_correctness(parse_text, correct_parse):
    parse_text_no_spaces = parse_text.replace(" ", "")
    parse_text_no_spaces_no_dots = parse_text.replace(".", "")
    correct_parse_no_spaces = correct_parse.replace(" ", "")
    return (parse_text_no_spaces == correct_parse_no_spaces or
            parse_text_no_spaces_no_dots == correct_parse_no_spaces)


def load_test_data(location):
    """Loads the testing inputs and the appropriate response."""
    with open(location, "r") as testing_file:
        data = testing_file.read()
        # Each prompt is divided by new lines
        split_data = data.split("\n\n")

    testing_data_store = {}

    for item in split_data:
        pair = item.split("\n")
        pair[1] = pair[1].lower()
        # Check that data is properly formatted
        assert pair[1].endswith(" [e]")

        formatted_input = pair[0]
        formatted_response = pair[1]

        testing_data_store[formatted_input] = formatted_response

    return testing_data_store
