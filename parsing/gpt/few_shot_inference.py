"""Few shot inference via gpt-j / neo series models."""
import sys
from os.path import dirname, abspath

import gin
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM, StoppingCriteriaList, MaxLengthCriteria)

parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)

from parsing.guided_decoding.gd_logits_processor import GuidedParser, GuidedDecodingLogitsProcessor


@gin.configurable
def get_few_shot_predict_f(model: str, device: str = "cpu", use_guided_decoding: bool = True):
    """Gets the few shot prediction model.

    Args:
        use_guided_decoding: whether to use guided decoding
        device: The device to load the model onto
        model: the name of the gpt series model for few shot prediction
    """

    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model)
    model = model.to(device)
    model.config.pad_token_id = model.config.eos_token_id

    def predict_f(text: str, grammar: str):
        """The function to get guided decoding."""
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        input_ids = input_ids.to(device)

        if use_guided_decoding:
            parser = GuidedParser(grammar, tokenizer, model="gpt")
            guided_preprocessor = GuidedDecodingLogitsProcessor(parser, input_ids.shape[1])
            generation = model.greedy_search(input_ids,
                                             logits_processor=guided_preprocessor,
                                             eos_token_id=parser.eos_token)
        else:
            stopping_criteria = MaxLengthCriteria(max_length=200)
            generation = model.greedy_search(input_ids,
                                             stopping_criteria=stopping_criteria)

        decoded_generation = tokenizer.decode(generation[0])
        return {"generation": decoded_generation}

    return predict_f
