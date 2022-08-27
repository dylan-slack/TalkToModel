"""Tests performed booting up ExplainBot."""
from os import mkdir  # noqa: E402, F401
from os.path import dirname, abspath, join  # noqa: E402, F401
import sys

import gin
import numpy as np

np.random.seed(0)
parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)

from explain.logic import ExplainBot  # noqa: E402, F401
from explain.conversation import fork_conversation  # noqa: E402, F401

CONFIG = "./tests/data/diabetes-config-debug.gin"


def test_load(tmp_path):
    with open(CONFIG, "r") as gin_file:
        gin_config = gin_file.read()

    cache_path = join(tmp_path, "cache")
    mkdir(cache_path)
    gin_config = gin_config.replace("{prompt_cache}", cache_path)

    # Load formatted config
    gin.parse_config(gin_config)

    # Load the explainbot
    bot = ExplainBot()
    # Setup a conversation
    user_conversation = fork_conversation(bot.conversation, "testing")

    # To avoid querying API, setup desired outcome for the parse text
    def make_new_comp_parse_test_f(parse_text):
        """Generates a function with the desired parse text."""
        def overwrite_compute_parse_text(text):
            del text
            return None, parse_text
        return overwrite_compute_parse_text

    desired_parse_text = "score accuracy test"
    bot.compute_parse_text = make_new_comp_parse_test_f(desired_parse_text)
    _ = bot.update_state("", user_conversation)

    # Add link to testing data
    gin.parse_config(gin_config)

    # Reload the explainbot
    bot = ExplainBot()
    # Setup a conversation
    user_conversation = fork_conversation(bot.conversation, "testing")

    bot.compute_parse_text = make_new_comp_parse_test_f(desired_parse_text)
    result = bot.update_state("", user_conversation)
    # score results are tested in test_dataset_description, assume they're correct here
    correct = "The model scores <em>83.333% accuracy</em>"
    assert result.startswith(correct)
