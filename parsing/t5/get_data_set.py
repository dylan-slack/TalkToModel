"""Routines for getting datasets"""
from typing import Any, Union, Tuple

import gin
import os

import numpy as np
import pandas as pd
from pandas import DataFrame

from explain.logic import ExplainBot


def get_config_loc(dataset: str) -> str:
    """Gets the location of a dataset's configuration file"""
    if dataset == "diabetes":
        config_loc = "./configs/diabetes-config.gin"
    elif dataset == "compas":
        config_loc = "./configs/compas-config.gin"
    elif dataset == "german":
        config_loc = "./configs/german-config.gin"
    elif dataset.endswith(".gin"):
        config_loc = dataset
    else:
        raise NameError(f"Unknown dataset {dataset}")
    return config_loc


def add_to_dict_list(cur_dict: dict, key: str, value: Any):
    if key not in cur_dict:
        cur_dict[key] = [value]
    else:
        cur_dict[key].append(value)


def load_explainbot_dataset_pd(t5_params: Any,
                               split_validation_data: bool = True,
                               down_sample_pct: float = None,
                               seed: int = 0) -> tuple[DataFrame, Union[DataFrame, None]]:
    """Loads the explainbot dataset"""

    # Parse params
    config_loc = get_config_loc(t5_params.dataset_name)
    gin.parse_config_file(config_loc)

    if t5_params.dataset_name.endswith(".gin"):
        file_name = os.path.basename(t5_params.dataset_name)
        dataset_name = file_name[:len(".gin")]
    else:
        dataset_name = t5_params.dataset_name

    # Dataset filepath
    data_set_file_path = os.path.join("./parsing/t5/datasets",
                                      f"{dataset_name}_pandas.csv")

    # Load generated data
    explain_bot = ExplainBot()
    prompt_set = explain_bot.prompts.final_prompt_set

    # Set seed before choosing
    np.random.seed(seed)

    if down_sample_pct is not None:
        keys_to_remain = np.random.choice(list(prompt_set.keys()),
                                          size=int(len(prompt_set)*down_sample_pct),
                                          replace=False)
        prompt_set = {k: prompt_set[k] for k in keys_to_remain}

    dataset = {}
    for p_id in prompt_set:
        prompts = prompt_set[p_id]["prompts"]
        for cur_prompt in prompts:
            split_prompt = cur_prompt.split("\n")

            natural_language = split_prompt[0][len("user: "):]
            parsed_utterance = split_prompt[1][len("parsed: "):]

            add_to_dict_list(dataset, t5_params.source_text, natural_language)
            add_to_dict_list(dataset, t5_params.target_text, parsed_utterance)

    df = pd.DataFrame(dataset)
    df.to_csv(data_set_file_path)

    if split_validation_data:
        train_dataset = df.sample(frac=0.9)
        val_dataset = df.drop(train_dataset.index).reset_index(drop=True)
        train_dataset = train_dataset.reset_index(drop=True)
    else:
        train_dataset = df
        val_dataset = None

    return train_dataset, val_dataset
