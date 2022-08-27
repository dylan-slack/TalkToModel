"""Evaluates the model performances on the sweeps"""
import argparse
import gin
from os.path import dirname, abspath
import sys

import pandas as pd

parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)

from experiments.get_compositional_queries import run_iid_compositional_accuracies
from experiments.compute_parsing_accuracy import set_t5_config, compute_accuracy
from experiments.utils import load_test_data
from explain.logic import ExplainBot

parser = argparse.ArgumentParser()
parser.add_argument("--pct", required=True, type=float)
args = parser.parse_args()


def get_results_locations(pct):
    return f"./experiments/sweep_results_{pct}.csv"


def get_best_models(pct):
    if pct == 0.2:
        return [12, 7, 15, 9, 18]
    elif pct == 0.4:
        return [18, 12, 11, 11, 19]
    elif pct == 0.6:
        return [15, 8, 7, 19, 14]
    elif pct == 0.8:
        return [12, 18, 18, 18, 13]
    elif pct == 1.0:
        return [11, 10, 9, 13, 17]
    else:
        raise NameError(f"Don't know pct {pct}")


def add_2_d(d, k, v):
    if k in d:
        d[k].append(v)
    else:
        d[k] = [v]


def main():
    test_suite = "./experiments/parsing_accuracy/diabetes_test_suite.txt"
    config = "./configs/diabetes-config.gin"

    gin.parse_config_file(config)
    testing_data = load_test_data(test_suite)

    results = {}
    for pct in [args.pct]:
        pcts = get_best_models(pct=pct)
        for i, epoch in enumerate(pcts):
            model = "t5-base"
            t5_config = set_t5_config(model)
            t5_model_dir = f"./parsing/t5/models/diabetes_t5-base_epoch_{epoch}_lr_0.0001_batchsize_32_optimizer_adamw_downsample_{pct}_model_id_{i+1}/"
            gin.parse_config(f"ExplainBot.model_name = '{t5_model_dir}'")
            gin.parse_config(f"ExplainBot.t5_config = '{t5_config}'")
            gin.parse_config(f"ExplainBot.use_guided_decoding = True")
            gin.parse_config(f"load_t5_params.dataset_name = 'diabetes'")
            gin.parse_config("load_t5_params.device = 'cuda'")
            bot = ExplainBot()

            def get_parse_text(user_input_to_parse):
                _, result_parse_text = bot.compute_parse_text_t5(user_input_to_parse)
                return result_parse_text

            error_rate, all_parses = compute_accuracy(testing_data, get_parse_text, verbose=False)

            in_acc, comp_acc, ov_all, total_in, total_comp = run_iid_compositional_accuracies("diabetes",
                                                                                              all_parses,
                                                                                              bot,
                                                                                              program_only=False)

            # NOTE(dylan): I forgot to save the pct and model id, so I will need to add these in later!
            add_2_d(results, "iid", in_acc)
            add_2_d(results, "comp", comp_acc)
            add_2_d(results, "total", ov_all)
            df = pd.DataFrame(results)
            df.to_csv(get_results_locations(str(pct)))

            print('=====')
            print(df)
            print('=====')


if __name__ == "__main__":
    main()
