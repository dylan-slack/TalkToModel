"""Gets the error split by compositional and non-compositional parses."""
import argparse
import os
from os.path import dirname, abspath, join  # noqa: E402, F401
import sys

import gin
import numpy as np
import pandas as pd

np.random.seed(0)
parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)

from explain.logic import ExplainBot
from explain.actions.get_action_functions import get_all_action_functions_map


def get_config_loc(parsed_args: argparse.ArgumentParser()) -> str:
    """Gets the location of a dataset's configuration file"""
    if parsed_args.dataset == "diabetes" or parsed_args.dataset == "diabetes-nn":
        config_loc = "./configs/diabetes-config.gin"
    elif parsed_args.dataset == "compas" or parsed_args.dataset == "compas-nn":
        config_loc = "./configs/compas-config.gin"
    elif parsed_args.dataset == "german" or parsed_args.dataset == "german-nn":
        config_loc = "./configs/german-config.gin"
    else:
        raise NameError(f"Unknown dataset {parsed_args.dataset}")
    return config_loc


def do_load_templates(prompt_set: dict, program_only: bool = False):
    """Loads the prompt templates from the set"""
    all_templates = []
    for p_id in prompt_set:
        prompts = prompt_set[p_id]["prompts"]
        for cur_prompt in prompts:
            parse = cur_prompt.split("parsed: ")[1]
            cur_template = get_template_from_prompt(parse, program_only=program_only)
            all_templates.append(cur_template)
    return all_templates


def get_template_from_prompt(parse, program_only: bool = False):
    cur_template = []

    # add and & or to the programs, because we want these as well!
    programs = list(get_all_action_functions_map().keys())
    programs.append("and")
    programs.append("or")

    for word in parse.split(' '):

        # In this case we only look at the "program" words
        # things like FILTER
        if program_only:
            if word in programs:
                cur_template.append(word)
            continue

        # test for numeric
        try:
            float(word)
            continue
        except Exception as E:
            pass

        # test for true / false
        if word == "true" or word == "false":
            continue
        cur_template.append(word)
    return cur_template


def get_templates(load_templates, store_name, explain_bot=None, program_only: bool = False):
    # init explainbot
    if load_templates:
        try:
            templates_df = pd.read_csv(store_name, index_col=0)
            print("Got templates from file...")
            return templates_df
        except Exception as exp:
            print((f"Failed to load template store {load_templates}"
                   f" with error {exp}:"))

    print("Loading explainbot...")
    if explain_bot is None:
        explain_bot = ExplainBot()
    final_prompt_set = explain_bot.prompts.final_prompt_set

    # Log templates
    templates = do_load_templates(final_prompt_set, program_only=program_only)
    templates_dict = {"templates": templates}
    templates_df = pd.DataFrame(templates_dict)
    templates_df.to_csv(store_name)
    templates_df = pd.read_csv(store_name, index_col=0)
    return templates_df


def compute_error(keys, result_col, df=True):
    """Computes the accuracy on some keys"""
    incorrect, total = 0, 0
    num_incorrect_that_doesnt_have_all_keys = 0
    for key in keys:
        correct_text = f"correct_parse_{key}"
        parsed_text = f"parsed_text_{key}"

        if df:
            correct_item = result_col[correct_text].item().replace(" ", "")
            parsed_item = result_col[parsed_text].item().replace(" ", "")
        else:
            correct_item = result_col[correct_text].replace(" ", "")
            parsed_item = result_col[parsed_text].replace(" ", "")

        if correct_item != parsed_item:
            incorrect += 1

            has_all_parse_words = f"includes_all_words_{key}"
            if has_all_parse_words in result_col and not result_col[has_all_parse_words]:
                num_incorrect_that_doesnt_have_all_keys += 1

        total += 1

    if total == 0:
        return 0, 0, 1., 0

    error = 1. * incorrect / total

    return incorrect, total, error, num_incorrect_that_doesnt_have_all_keys / incorrect


def run_iid_compositional_accuracies(dataset, parse_keys, explain_bot, program_only: bool = False):
    """Computes the iid and compositional accuracies for a set of parses"""
    store_name = os.path.join("./experiments/results_store",
                              dataset + "_templates.csv")
    templates_df = get_templates(load_templates=False,
                                 store_name=store_name,
                                 explain_bot=explain_bot,
                                 program_only=program_only)
    templates_list = templates_df["templates"].to_list()

    eval_inds = {}
    for key in parse_keys:
        if key.startswith("correct_parse_"):
            eval_inds[int(key[len("correct_parse_"):])] = get_template_from_prompt(parse_keys[key],
                                                                                   program_only=program_only)

    in_domain_keys, compositional_keys = [], []
    for parse_key in eval_inds:
        parse = str(eval_inds[parse_key])
        if parse in templates_list:
            in_domain_keys.append(parse_key)
        else:
            compositional_keys.append(parse_key)

    in_domain_error = compute_error(in_domain_keys, parse_keys, df=False)
    compositional_error = compute_error(compositional_keys, parse_keys, df=False)
    all_keys = list(eval_inds.keys())
    overall_error = compute_error(all_keys, parse_keys, df=False)

    total_in_domain = in_domain_error[1]
    total_comp = compositional_error[1]

    return (1. - in_domain_error[2],
            1. - compositional_error[2],
            1. - overall_error[2],
            total_in_domain,
            total_comp,
            in_domain_error[3],
            compositional_error[3])


def main(parsed_args):
    """Main method"""
    # parse configuration
    print("Initializing configuration...")
    config_loc = get_config_loc(parsed_args)
    gin.parse_config_file(config_loc)

    store_name = os.path.join("./experiments/results_store",
                              parsed_args.dataset + "_templates.csv")
    templates_df = get_templates(parsed_args.load_templates, store_name, program_only=parsed_args.program_only)
    templates_list = templates_df["templates"].to_list()

    """Compute the splits"""
    parsing_results = pd.read_csv(parsed_args.parsing_results_location, index_col=0)
    for result_col_ind in parsing_results.index:
        eval_inds = {}
        result_col = parsing_results.loc[[result_col_ind]]
        for col in parsing_results.columns:
            if col.startswith("correct_parse_"):
                eval_inds[int(col[len("correct_parse_"):])] = get_template_from_prompt(result_col[col].item(),
                                                                                       parsed_args.program_only)

        in_domain_keys, compositional_keys = [], []
        for parse_key in eval_inds:
            parse = str(eval_inds[parse_key])

            if parse in templates_list:
                in_domain_keys.append(parse_key)
            else:
                compositional_keys.append(parse_key)

        in_domain_error = compute_error(in_domain_keys, result_col)
        compositional_error = compute_error(compositional_keys, result_col)
        all_keys = list(eval_inds.keys())
        overall_error = compute_error(all_keys, result_col)

        print(f"Ind: {result_col_ind}")
        print(f"In domain acc: {1 - in_domain_error[2]}")
        print(f"Compositional acc: {1 - compositional_error[2]}")
        print(f"Overall acc: {1 - overall_error[2]}")
        print(f"Num comp: {len(compositional_keys)}")
        print(f"Num iid: {len(in_domain_keys)}")
        print('-------------------------')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="the name of the dataset")
    parser.add_argument("--debug", action="store_true", help="whether to flag debugging mode")
    parser.add_argument("--parsing_results_location", type=str, required=True,
                        help="the location of the parsing results")
    parser.add_argument("--program_only", action="store_true")
    parser.add_argument("--load_templates", action="store_true", help="whether to try and get the templates from store")

    args = parser.parse_args()
    main(args)
