"""T5 trainer

Modified from: https://github.com/Shivanandroy/T5-Finetuning-PyTorch
"""
import os
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
import wandb

from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration

from parsing.t5.get_data_set import load_explainbot_dataset_pd
from parsing.t5.t5_dataset import T5Dataset
from parsing.t5.t5_training_val_loops import train, validate


def make_decoder_only(model):
    """Make decoder only."""
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.shared.parameters():
        param.requires_grad = False
    print("Model decoder finetuning only...")


def load_model(name: str, device: str = "cpu", decoder_only: bool = False):
    """Loads t5 model"""
    tokenizer = T5Tokenizer.from_pretrained(name, device=device)
    model = T5ForConditionalGeneration.from_pretrained(name)
    model = model.to(device)

    if decoder_only:
        make_decoder_only(model)

    return tokenizer, model


def format_model_save_name(epoch: int, t5_params: Any, down_sample_pct: float = None, model_id: int = None):
    """Formats the name of the file used to save the model"""
    down_sample_string, model_id_string = "", ""
    if down_sample_pct is not None:
        down_sample_string += f"_downsample_{str(down_sample_pct)}"
    if model_id is not None:
        model_id_string += f"_model_id_{model_id}"
    name = (f"{t5_params.dataset_name}_{t5_params.model_name}_epoch_{epoch}_lr_"
            f"{t5_params.learning_rate}_batchsize_{t5_params.train_batch_size}_optimizer_{t5_params.optimizer}"
            f"{down_sample_string}{model_id_string}")
    return name


def load_dataset(t5_params: Any, tokenizer: T5Tokenizer, down_sample_pct: float = None, seed: int = 0):
    """Creates the dataloader for t5"""
    t5_df, t5_val_df = load_explainbot_dataset_pd(t5_params, down_sample_pct=down_sample_pct, seed=seed)

    # Append text, get source + target texts
    t5_df[t5_params.source_text] = t5_params.append_text + t5_df[t5_params.source_text]
    train_dataset = t5_df[[t5_params.source_text, t5_params.target_text]]

    if t5_val_df is not None:
        t5_val_df[t5_params.source_text] = t5_params.append_text + t5_val_df[t5_params.source_text]
        val_dataset = t5_val_df[[t5_params.source_text, t5_params.target_text]]
    else:
        val_dataset = None

    if t5_params.verbose:
        print("Train dataset shape:", train_dataset.shape)
        # print("Validation dataset shape:", val_dataset.shape)

    # Set up the dataset classes
    training_data_class = T5Dataset(train_dataset,
                                    tokenizer,
                                    source_len=t5_params.source_len,
                                    target_len=t5_params.target_len,
                                    source_text=t5_params.source_text,
                                    target_text=t5_params.target_text)

    if val_dataset is not None:
        val_data_class = T5Dataset(val_dataset,
                                   tokenizer,
                                   source_len=t5_params.source_len,
                                   target_len=t5_params.target_len,
                                   source_text=t5_params.source_text,
                                   target_text=t5_params.target_text)
    else:
        val_data_class = None

    return training_data_class, val_data_class


def make_safe(string):
    return string.replace("/", "")


def t5_trainer(t5_params, down_sample_pct: float = None, model_id: int = None, seed: int = 0):
    """T5 Training loop"""

    run = wandb.init(project="project-ttm", entity="dslack")
    wandb.run.name = f"{make_safe(t5_params.model_name)}-{make_safe(t5_params.dataset_name)}-training-debug-{t5_params.debug}"

    if t5_params.verbose:
        print("Loading model and tokenizer...", flush=True)

    # Loads the model and tokenizer
    tokenizer, model = load_model(t5_params.model_name,
                                  device=t5_params.device,
                                  decoder_only=t5_params.decoder_only)

    if t5_params.verbose:
        print("Loading train and val sets...")

    # Loading training and validation data
    training_set, validation_set = load_dataset(t5_params=t5_params,
                                                tokenizer=tokenizer,
                                                down_sample_pct=down_sample_pct,
                                                seed=seed)

    # Defining the parameters for creation of data loaders
    train_params = {
        "batch_size": t5_params.train_batch_size,
        "shuffle": True,
        "num_workers": 0,
    }

    val_params = {
        "batch_size": t5_params.val_batch_size,
        "shuffle": False,
        "num_workers": 0,
    }

    if t5_params.verbose:
        print("Creating data loaders...", flush=True)

    # Create data loaders
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(validation_set, **val_params)

    if t5_params.verbose:
        print("Setting up optimizer...", flush=True)

    # Setup optimizer
    if t5_params.optimizer == "adam":
        optimizer = torch.optim.Adam(
            params=model.parameters(), lr=t5_params.learning_rate
        )
    elif t5_params.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params=model.parameters(), lr=t5_params.learning_rate
        )
    elif t5_params.optimizer == "adam8bit":
        import bitsandbytes as bnb
        optimizer = bnb.optim.Adam8bit(
            params=model.parameters(), lr=t5_params.learning_rate
        )
    else:
        raise NameError(f"Unknown optimizer {t5_params.optimizer}")

    # Training loop
    if t5_params.verbose:
        print("Beginning fine-tuning...", flush=True)

    best_val_loss = float("+inf")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    for epoch in range(t5_params.n_epochs):
        # Break early on debug
        if epoch > 3 and t5_params.debug:
            break

        train(epoch,
              tokenizer,
              model,
              t5_params.device,
              training_loader,
              optimizer,
              t5_params.debug)

        # See predictions on validation data
        avg_val_loss, predictions, gold = validate(tokenizer,
                                                   model,
                                                   t5_params.device,
                                                   val_loader,
                                                   debug=t5_params.debug)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_epoch = epoch
            if model_id is not None:
                wandb.run.summary[f"best_val_loss_id_{model_id}"] = best_val_loss
                wandb.run.summary[f"best_val_epoch_id_{model_id}"] = best_val_epoch
            else:
                wandb.run.summary["best_val_loss"] = best_val_loss
                wandb.run.summary["best_val_epoch"] = best_val_epoch

        print(f"{epoch} | mean validation loss: {avg_val_loss}")
        to_show = np.random.choice(len(predictions), size=100)
        to_show_preds = [predictions[i] for i in to_show]
        to_show_gold = [gold[i] for i in to_show]

        to_show_dict = {"preds": to_show_preds, "gold": to_show_gold}
        to_show_df = pd.DataFrame(to_show_dict)

        print(f"{epoch} | generation samples")
        print(to_show_df)

        to_show_wandb_table = wandb.Table(data=to_show_df)

        wandb.log({
            "Epoch": epoch,
            "Val Loss": avg_val_loss,
            "Generation Table": to_show_wandb_table
        })

        if t5_params.verbose:
            print("Saving model...")

        # Saving the model after training
        save_name = format_model_save_name(epoch, t5_params, down_sample_pct=down_sample_pct, model_id=model_id)
        path = os.path.join(t5_params.model_dir, save_name)
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)

        if t5_params.verbose:
            print(f"Saved model model at {path}!")

    print("Done (≡^∇^≡)")
