"""Computes the parsing accuracy of the model on a test suite."""
import argparse
import copy
import json
import os
from os.path import dirname, abspath, join  # noqa: E402, F401
import sys

from datetime import datetime
from typing import Any

from pytz import timezone

import gin
import numpy as np
import pandas as pd
import torch

np.random.seed(0)
torch.manual_seed(0)
parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)

from experiments.get_compositional_queries import run_iid_compositional_accuracies
from experiments.utils import load_test_data, check_correctness
from explain.logic import ExplainBot  # noqa: E402, F401
# Needed for gin configs
from parsing.t5.start_fine_tuning import load_t5_params  # noqa: E402, F401
from parsing.gpt.few_shot_inference import get_few_shot_predict_f  # noqa: E402, F401


def safe_name(model):
    return model.replace("/", "_")


def compute_accuracy(data, predict_func, verbose: bool = False, error_analysis: bool = False,
                     feature_names: Any = None):
    """Computes the parsing accuracy across some data."""
    nn_prompts = None
    misses, total = 0, 0
    parses_store = {}
    print(f"There are {len(data)} eval points", flush=True)

    for j, user_input in enumerate(data):
        has_all_parse_words = None
        if error_analysis:
            parse_text, nn_prompts = predict_func(user_input)
        else:
            parse_text = predict_func(user_input)

        # Get the gold label parse
        correct_parse = data[user_input]

        # Do this to make sure extra spaces are ignored around input
        is_correct = check_correctness(parse_text, correct_parse)

        if error_analysis:
            parses = []
            print(nn_prompts)

            if nn_prompts is None:
                parses = []
            else:
                for prompt in nn_prompts:
                    split_prompt = prompt.split("parsed: ")
                    nn_parse = split_prompt[1]
                    parses.append(nn_parse)
            parses = " ".join(parses)

            has_all_parse_words = True
            for word in correct_parse.split(" "):

                # Cases not to look at parse word, i.e., things like numbers,
                # bools, and the feature names
                try:
                    float(word)
                    continue
                except:
                    pass
                if word == "true" or word == "false":
                    continue
                if word in feature_names:
                    continue

                if word not in parses:
                    has_all_parse_words = False

        if not is_correct:
            misses += 1

            if verbose:
                print(">>>>>")
                print(f"User: {user_input}")
                print(f"Parsed: {parse_text}")
                print(f"Correct: {correct_parse}")
                print(">>>>>")

        parses_store[f"parsed_text_{j}"] = parse_text
        parses_store[f"correct_parse_{j}"] = correct_parse
        parses_store[f"user_input_{j}"] = user_input

        if error_analysis:
            parses_store[f"includes_all_words_{j}"] = has_all_parse_words

        total += 1

        if j % 25 == 0:
            print(f"Error Rate | it {j} |  {round(misses / total, 3)}", flush=True)

    error_stat = misses / total

    if verbose:
        print(f"Final Error Rate: {round(error_stat, 3)}", flush=True)

    return error_stat, parses_store


def set_t5_config(model_name):
    if "base" in model_name:
        set_config = "inference-t5-base.gin"
    elif "small" in model_name:
        set_config = "inference-t5-small.gin"
    elif "large" in model_name:
        set_config = "inference-t5-large.gin"
    else:
        raise NameError(f"Don't know {model_name}")
    return os.path.join("parsing/t5/gin_configs", set_config)


def wand_b_runs(model, dset):
    if dset == "diabetes":
        if model == "t5-small":
            return "95lif3eg"
        elif model == "t5-base":
            return "1wz6xw04"
        elif model == "t5-large":
            return "1rce37tm"
        elif model == "t5-3b":
            return "22sok5ix"
    elif dset == "compas":
        if model == "t5-small":
            return "1746dcw0"
        elif model == "t5-base":
            return "22zecolz"
        elif model == "t5-large":
            return "27igkwk7"
        elif model == "t5-3b":
            return "nlmkpypr"
    elif dset == "german":
        if model == "t5-small":
            return "ncbrsm89"
        elif model == "t5-base":
            return "22d11ute"
        elif model == "t5-large":
            return "2lw7i0l2"
    raise NameError(f"don't know model {model} and dataset {dset}")


def get_wandb_sub_folder(model, dset):
    wand_b_run_id = wand_b_runs(model, dset)
    for file in os.listdir("./wandb"):
        d = os.path.join("./wandb", file)
        if os.path.isdir(d) and wand_b_run_id in d:
            return d
    raise NameError(f"couldn't find directory for {model} and {dset}")


def retrieve_best_val_epoch(model, dset):
    wandb_sub_folder = get_wandb_sub_folder(model, dset)
    json_summary = os.path.join(wandb_sub_folder, "files/wandb-summary.json")
    with open(json_summary, 'r') as json_file:
        validation_data = json.load(json_file)
        best_val_epoch = validation_data['best_val_epoch']
    return best_val_epoch


def retrieve_t5_model_dir(model, dset, debug):
    """Retrieves the directory the t5 models are stored in"""
    if debug:
        return "./parsing/t5/models/diabetes_t5-small_epoch_9_lr_0.0003_batchsize_128"

    # Note, we used wandb to store the best validation epoch, but have since hard coded the best epochs
    # so as not to require keeping the wandb files in the release
    # best_val_epoch = retrieve_best_val_epoch(model, dset)
    # print(f"Best validation epoch {best_val_epoch}")

    if dset == "diabetes":
        if model == "t5-small":
            location = f"diabetes_t5-small_epoch_20_lr_0.0001_batchsize_32_optimizer_adamw"
        elif model == "t5-base":
            location = f"diabetes_t5-base_epoch_9_lr_0.0001_batchsize_32_optimizer_adamw"
        elif model == "t5-large":
            location = f"diabetes_t5-large_epoch_17_lr_0.0001_batchsize_32_optimizer_adamw"
        else:
            raise NameError(f"no t5 model for model type {model}")
    elif dset == "compas":
        if model == "t5-small":
            location = f"compas_t5-small_epoch_20_lr_0.0001_batchsize_32_optimizer_adamw"
        elif model == "t5-base":
            location = f"compas_t5-base_epoch_21_lr_0.0001_batchsize_32_optimizer_adamw"
        elif model == "t5-large":
            location = f"compas_t5-large_epoch_17_lr_0.0001_batchsize_32_optimizer_adamw"
        else:
            raise NameError(f"no t5 model for model type {model}")
    elif dset == "german":
        if model == "t5-small":
            location = f"german_t5-small_epoch_29_lr_0.0001_batchsize_32_optimizer_adamw"
        elif model == "t5-base":
            location = f"german_t5-base_epoch_15_lr_0.0001_batchsize_32_optimizer_adamw"
        elif model == "t5-large":
            location = f"german_t5-large_epoch_5_lr_0.0001_batchsize_32_optimizer_adamw"
        else:
            raise NameError(f"no t5 model for model type {model}")
    else:
        raise NameError(f"no t5 model for dataset {dset}")
    model_path = os.path.join("parsing/t5/models", location)
    return model_path


def main():
    results = {
        "dataset": [],
        "model": [],
        "num_prompts": [],
        "accuracy": [],
        "in_domain_accuracy": [],
        "compositional_accuracy": [],
        "overall_accuracy": [],
        "total_in_domain": [],
        "total_compositional": [],
        "guided_decoding": [],
        "iid_errors_pct_not_all_words": [],
        "comp_errors_pct_not_all_words": []
    }

    model = args.model
    guided_decoding = args.gd
    dset = args.dataset

    program_only_text = ""
    if args.program_only:
        program_only_text += "-program-only"
    results_location = (f"./experiments/results_store/{safe_name(model)}_{dset}_gd-{guided_decoding}"
                        f"_debug-{args.debug}{program_only_text}.csv")

    print(f"-----------------", flush=True)
    print("Debug:", args.debug, flush=True)
    print("Dataset:", dset, flush=True)
    print("Model:", model, flush=True)

    if dset == "diabetes":
        test_suite = "./experiments/parsing_accuracy/diabetes_test_suite.txt"
        config = "./configs/diabetes-config.gin"
    elif dset == "compas":
        test_suite = "./experiments/parsing_accuracy/compas_test_suite.txt"
        config = "./configs/compas-config.gin"
    elif dset == "german":
        test_suite = "./experiments/parsing_accuracy/german_test_suite.txt"
        config = "./configs/german-config.gin"
    else:
        raise NameError(f"Unknown dataset {dset}")

    # Parse config
    gin.parse_config_file(config)
    testing_data = load_test_data(test_suite)

    # load the model
    bot, get_parse_text = load_model(dset, guided_decoding, model)

    error_analysis = False
    if "t5" not in model:
        error_analysis = True

    # load the number of prompts to perform in the sweep
    n_prompts_configs = load_n_prompts(model)

    if args.debug:
        n_prompts_configs = [10, 2]

    feature_names = copy.deepcopy(list(bot.conversation.stored_vars["dataset"].contents["X"].columns))

    for num_prompts in n_prompts_configs:

        # Set the bot to the number of prompts
        bot.set_num_prompts(num_prompts)
        print("Num prompts:", bot.prompts.num_prompt_template)
        assert bot.prompts.num_prompt_template == num_prompts, "Prompt update failing"

        # Compute the accuracy
        error_rate, all_parses = compute_accuracy(testing_data,
                                                  get_parse_text,
                                                  args.verbose,
                                                  error_analysis=error_analysis,
                                                  feature_names=feature_names)

        # Add parses to results
        for key in all_parses:
            if key not in results:
                results[key] = [all_parses[key]]
            else:
                results[key].append(all_parses[key])

        # Compute the compositional / iid accuracy splits
        iid_comp_results = run_iid_compositional_accuracies(dset,
                                                            all_parses,
                                                            bot,
                                                            program_only=args.program_only)

        in_acc, comp_acc, ov_all, total_in, total_comp, iid_pct_keys, comp_pct_keys = iid_comp_results

        # Store metrics
        results["total_in_domain"].append(total_in)
        results["total_compositional"].append(total_comp)
        results["in_domain_accuracy"].append(in_acc)
        results["compositional_accuracy"].append(comp_acc)
        results["overall_accuracy"].append(ov_all)
        results["guided_decoding"].append(guided_decoding)
        results["model"].append(model)
        results["dataset"].append(dset)
        results["accuracy"].append(1 - error_rate)
        results["num_prompts"].append(num_prompts)
        results["iid_errors_pct_not_all_words"].append(iid_pct_keys)
        results["comp_errors_pct_not_all_words"].append(comp_pct_keys)

        # Write everything to dataframe
        final_results = results
        result_df = pd.DataFrame(final_results)
        result_df.to_csv(results_location)
        print("Saved locally...", flush=True)

        # optionally upload to wandb
        if args.wandb:
            import wandb
            results_table = wandb.Table(data=result_df)
            if args.debug:
                table_name = "parsing-accuracy-debug"
            else:
                table_name = "parsing-accuracy"
            run.log({table_name: results_table})
            print("Logged to wandb...", flush=True)

        print(f"Saved results to {results_location}")
        print(f"-----------------")


def load_n_prompts(model):
    n_prompts_configs = [
        30,
        20,
        10
    ]
    if model == "EleutherAI/gpt-j-6B":
        n_prompts_configs = [10, 5]
    if model == "EleutherAI/gpt-neo-2.7B":
        n_prompts_configs = [20, 10, 5]
    # doesn't matter if we draw many
    # when taking nn as result
    if model == "nearest-neighbor" or "t5" in model:
        n_prompts_configs = [1]
    return n_prompts_configs


def load_model(dset, guided_decoding, model):
    """Loads the model"""
    print("Initializing model...", flush=True)
    if "t5" not in model:
        gin.parse_config(f"ExplainBot.model_name = '{model}'")
        gin.parse_config(f"ExplainBot.use_guided_decoding = {guided_decoding}")

        if args.debug:
            gin.parse_config("get_few_shot_predict_f.device = 'cpu'")
        else:
            gin.parse_config("get_few_shot_predict_f.device = 'cuda'")

        # Case for NN and few shot gpt models
        bot = ExplainBot()

        def get_parse_text(user_input_to_parse):
            includes_all_words = None
            try:
                _, result_parse_text, includes_all_words = bot.compute_parse_text(user_input_to_parse,
                                                                                  error_analysis=True)
            except Exception as e:
                result_parse_text = f"Exception: {e}, likely OOM"
            return result_parse_text, includes_all_words
    else:
        t5_config = set_t5_config(model)
        t5_model_dir = retrieve_t5_model_dir(model, dset, args.debug)

        gin.parse_config(f"ExplainBot.model_name = '{t5_model_dir}'")
        gin.parse_config(f"ExplainBot.t5_config = '{t5_config}'")
        gin.parse_config(f"ExplainBot.use_guided_decoding = {guided_decoding}")

        # need to specify the dataset on load
        gin.parse_config(f"load_t5_params.dataset_name = '{dset}'")

        if args.debug:
            gin.parse_config("load_t5_params.device = 'cpu'")
        else:
            gin.parse_config("load_t5_params.device = 'cuda'")

        bot = ExplainBot()

        def get_parse_text(user_input_to_parse):
            try:
                _, result_parse_text = bot.compute_parse_text_t5(user_input_to_parse)
            except Exception as e:
                result_parse_text = f"Exception: {e}, likely OOM"
            return result_parse_text, None

    return bot, get_parse_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--gd", action="store_true")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--id", type=str, required=True, help="a unique id to associate with the run")
    parser.add_argument("--down_sample", action="store_true", help="this will break each run on 10 samples")
    parser.add_argument("--program_only", action="store_true", help="only uses the program name for templates")
    args = parser.parse_args()

    if args.wandb:
        import wandb
        run = wandb.init(project="project-ttm", entity="dslack")
    pst = timezone('US/Pacific')
    sa_time = datetime.now(pst)
    time = sa_time.strftime('%Y-%m-%d_%H-%M')
    if args.wandb:
        wandb.run.name = f"{args.id}-{safe_name(args.model)}_{args.dataset}_gd-{args.gd}"

    main()
