"""T5 fine-tuning"""
import argparse
import sys
import random
from dataclasses import dataclass

import gin
from os.path import dirname, abspath
import numpy as np
import torch

parent = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(parent)

from parsing.t5.t5_trainer import t5_trainer


@dataclass
class T5Params:
    """Class for keeping track of t5 parameters"""
    model_name: str
    dataset_name: str
    regenerate_dataset: bool
    device: str
    seed: int
    verbose: bool
    source_len: int
    target_len: int
    source_text: str
    target_text: str
    train_pct: float
    append_text: str
    train_batch_size: int
    val_batch_size: int
    learning_rate: float
    optimizer: str
    n_epochs: int
    debug: bool
    model_dir: str
    decoder_only: bool


@gin.configurable
def load_t5_params(model_name: str,
                   dataset_name: str,
                   source_len: int,
                   target_len: int,
                   source_text: str,
                   target_text: str,
                   n_epochs: int,
                   optimizer: str,
                   learning_rate: float,
                   val_batch_size: int,
                   train_batch_size: int,
                   append_text: str,
                   train_pct: float,
                   regenerate_dataset: bool = False,
                   verbose: bool = False,
                   device: str = "cpu",
                   seed: int = 0,
                   model_dir: str = "./parsing/t5/models",
                   debug: bool = False,
                   decoder_only: bool = False):
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Load params object
    t5_params = T5Params(**locals())

    return t5_params


def main(parsed_args):

    t5_params = load_t5_params(dataset_name=parsed_args.dataset)

    if t5_params.verbose:
        print("Loaded configs...")

    if parsed_args.train_this_many is not None:
        seeds = list(range(parsed_args.train_this_many))
        for i in range(parsed_args.train_this_many):
            t5_trainer(t5_params,
                       down_sample_pct=parsed_args.down_sample_pct,
                       model_id=i+1,
                       seed=seeds[i])
    else:
        # Do training
        t5_trainer(t5_params,
                   down_sample_pct=parsed_args.down_sample_pct)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gin", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--down_sample_pct", type=float, default=1.0)
    parser.add_argument("--train_this_many", type=int, default=None)
    args = parser.parse_args()

    gin.parse_config_file(args.gin)
    main(args)
