
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


import os
from typing import List, Optional

import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
    
)

from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.loggers import Logger as LightopenningLoggerBase
import torch
from torch import nn
from lightning.pytorch.strategies import FSDPStrategy
from byprot import utils

log = utils.get_logger(__name__)


def load_mapped_state_dict(pl_module, checkpoint_path):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Get the current model state dict
    model_state_dict = pl_module.state_dict()

    # Initialize containers for matched and unmatched keys
    mapped_state_dict = {}
    matched_keys = []
    unmatched_keys = []

    # Total keys in the checkpoint
    total_keys = len(checkpoint['state_dict'])

    # Map the checkpoint keys to match the model's keys
    for ckpt_key, ckpt_value in checkpoint['state_dict'].items():
        # Add 'decoder.' prefix for all keys from the checkpoint
        # new_key = f"model.decoder.{ckpt_key}"
        new_key = f"{ckpt_key}"

        # If the new key exists in the model's state dict, map it
        if new_key in model_state_dict:
            mapped_state_dict[new_key] = ckpt_value
            matched_keys.append((ckpt_key, new_key))  # Track matched keys
        else:
            unmatched_keys.append(ckpt_key)  # Track unmatched keys

    # Load the mapped state dict into the model
    pl_module.load_state_dict(mapped_state_dict, strict=False)

    # Number of matched keys
    matched_count = len(matched_keys)

    # Print matched and unmatched keys for review
    # print(f"Matched keys: {matched_count}")
    # for ckpt_key, model_key in matched_keys:
    #     print(f"{ckpt_key} -> {model_key}")

    print(f"Unmatched keys: {len(unmatched_keys)}")
    for ckpt_key in unmatched_keys:
        print(f"{ckpt_key} not found in model")

    # Print summary of key updates
    print(f"\nSummary: {matched_count}/{total_keys} keys were successfully loaded.")


def train(config: DictConfig) -> Optional[float]:
    """Contains the training pipeline. Can additionally evaluate model on a testset, using best
    weights achieved during training.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    torch.cuda.set_per_process_memory_fraction(0.9)

    # Convert relative ckpt path to absolute path if necessary
    ckpt_path = not config.train.get("force_restart", False) and config.train.get("ckpt_path")
    if ckpt_path:
        ckpt_path = utils.resolve_ckpt_path(ckpt_dir=config.paths.ckpt_dir, ckpt_path=ckpt_path)
        if os.path.exists(ckpt_path):
            log.info(f"Resuming checkpoint from <{ckpt_path}>")
        else:
            log.info(f"Failed to resume checkpoint from <{ckpt_path}>: file not exists. Skip.")
            ckpt_path = None

    # loading pipeline
    datamodule, pl_module, logger, callbacks = utils.common_pipeline(config)

    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        log.info(f"World Size (Total GPUs): {world_size}, Rank: {rank}, Local Rank: {local_rank}")


    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")

    # logger is none
    logger = [WandbLogger(log_model="all", project="cfpgen_dplm2")]

    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        datamodule=datamodule,
        # model=model,
        model=pl_module,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    if config.get("train"):
        log.info("Starting training!")
        trainer.fit(model=pl_module, datamodule=datamodule, ckpt_path=ckpt_path)

        # load_mapped_state_dict(pl_module, ckpt_path)
        # trainer.fit(model=pl_module, datamodule=datamodule)

    # Get metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric and optimized_metric not in trainer.callback_metrics:
        raise Exception(
            "Metric for hyperparameter optimization not found! "
            "Make sure the `optimized_metric` in `hparams_search` config is correct!"
        )
    score = trainer.callback_metrics.get(optimized_metric)

    # Test the model
    if config.get("test"):
        log.info("Starting testing!")
        best_ckpt_path = os.path.join(config.paths.ckpt_dir, 'best.ckpt')
        trainer.test(model=pl_module, datamodule=datamodule, ckpt_path=best_ckpt_path)

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=pl_module,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    if not config.trainer.get("fast_dev_run") and config.get("train"):
        log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    return score
