"""Load model and do inference"""
import argparse
import sys
from functools import partial
from os.path import dirname, abspath
from typing import Any

import gin
import pandas as pd
import torch

from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration

parent = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(parent)

from parsing.guided_decoding.gd_logits_processor import GuidedParser, GuidedDecodingLogitsProcessor
from parsing.t5.start_fine_tuning import load_t5_params, T5Params
from parsing.t5.t5_dataset import T5Dataset


def load_pretrained(model_dir: str, device: str):
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    model = model.to(device)
    model.eval()
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    return model, tokenizer


def predict(sentences: list[str],
            tokenizer: T5Tokenizer,
            model: T5ForConditionalGeneration,
            t5_params: T5Params,
            guided_decoding: bool = False,
            explain_bot: Any = None,
            compute_grammar: bool = True,
            provided_grammar: str = None):
    """Computes predictions for the model"""
    sentences_with_instruction = [
        t5_params.append_text + sentence
        for sentence in sentences
    ]
    empty_targets = [""] * len(sentences_with_instruction)

    prediction_dict = {"source": sentences_with_instruction, "predict": empty_targets}
    prediction_df = pd.DataFrame(prediction_dict)

    prediction_data_set = T5Dataset(dataframe=prediction_df,
                                    tokenizer=tokenizer,
                                    source_len=t5_params.source_len,
                                    target_len=t5_params.target_len,
                                    source_text="source",
                                    target_text="predict")

    prediction_data_loader = DataLoader(prediction_data_set,
                                        batch_size=t5_params.val_batch_size,
                                        shuffle=False,
                                        num_workers=0)

    device = t5_params.device
    model.eval()
    generated_texts = []
    with torch.no_grad():
        for data in prediction_data_loader:
            ids = data["source_ids"].to(device, dtype=torch.long)
            mask = data["source_mask"].to(device, dtype=torch.long)
            texts = data["source_text"]

            if guided_decoding:
                generated_ids = []
                for i in range(len(ids)):
                    sampled_already = 1  # torch.sum(ids[i] != 0)
                    input_text = texts[i]
                    user_input_piece = input_text[len(t5_params.append_text):]
                    if not compute_grammar:
                        grammar = provided_grammar
                    else:
                        grammar, _ = explain_bot.compute_grammar(user_input_piece)

                    lark_grammar_parser = GuidedParser(grammar, tokenizer, model="t5")
                    guided_preprocessor = GuidedDecodingLogitsProcessor(lark_grammar_parser,
                                                                        sampled_already)
                    gen_toks = model.generate(
                        input_ids=ids[i:i + 1],
                        attention_mask=mask[i:i + 1],
                        logits_processor=[guided_preprocessor],
                        eos_token_id=lark_grammar_parser.eos_token,
                        max_length=256,
                    )

                    generated_ids.extend(gen_toks)
            else:
                generated_ids = model.generate(
                    input_ids=ids,
                    attention_mask=mask,
                    max_length=150,
                    early_stopping=True
                )
            generation_text = [
                tokenizer.decode(c_tokes, skip_special_tokens=True, clean_up_tokenization_spaces=True).lower()
                for c_tokes in generated_ids
            ]

            generated_texts.extend(generation_text)

    return generated_texts


def get_predict_func(t5_gin_file: str,
                     model_checkpoint_dir: str,
                     guided_decoding: bool,
                     compute_grammar: bool = True,
                     dataset_name: str = None,
                     bot_gin_file: str = None):
    """Gets the t5 model as a prediction function

    Note that if compute_grammar is set to True, the grammar is computed for the dataset by the prediction function.
    If it is set to false, the prediction function will excpect the grammar to be provided.

    Args:
        bot_gin_file:
        dataset_name:
        compute_grammar:
        guided_decoding:
        model_checkpoint_dir:
        t5_gin_file:
    Returns:
        predict_f: The prediction function
    """
    # Setup t5 params
    gin.parse_config_file(t5_gin_file)
    if dataset_name is not None:
        t5_params = load_t5_params(dataset_name=dataset_name)
    else:
        t5_params = load_t5_params()

    if bot_gin_file is None:
        if t5_params.dataset_name == "diabetes":
            bot_gin_file = "./configs/diabetes-config.gin"
        elif t5_params.dataset_name == "compas":
            bot_gin_file = "./configs/compas-config.gin"
        elif t5_params.dataset_name == "german":
            bot_gin_file = "./configs/german-config.gin"
        else:
            known_options = "diabetes, compas, or german"
            message = ("Please provide the gin file for the conversation in bot_gin_file, "
                       f"argument as the dataset {dataset_name} is unknown. Known dataset "
                       f"options are {known_options}.")
            raise NameError(message)

    if guided_decoding and compute_grammar:
        from explain.logic import ExplainBot
        gin.parse_config_file(bot_gin_file)
        bot = ExplainBot()
    else:
        bot = None

    # Get t5 model
    model, tokenizer = load_pretrained(model_checkpoint_dir, t5_params.device)

    if compute_grammar:
        # Case where we compute the grammar during prediction
        # This is easier for inference without loading explainbot
        partial_f = partial(predict,
                            model=model,
                            tokenizer=tokenizer,
                            t5_params=t5_params,
                            explain_bot=bot,
                            guided_decoding=guided_decoding,
                            compute_grammar=compute_grammar,
                            provided_grammar=None)
    else:
        partial_f = partial(predict,
                            model=model,
                            tokenizer=tokenizer,
                            t5_params=t5_params,
                            explain_bot=bot,
                            guided_decoding=guided_decoding,
                            compute_grammar=compute_grammar)

    return partial_f, t5_params


def main(parsed_args):
    # Load parameters
    predict_f, _ = get_predict_func(parsed_args.gin, parsed_args.model_dir, guided_decoding=True)
    print(predict_f([parsed_args.sentence]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--gin", type=str, required=True)
    parser.add_argument("--sentence", type=str, required=True)
    args = parser.parse_args()
    main(args)
