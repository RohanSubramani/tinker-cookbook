"""Modular RL training script - set training configs at the top.

HYPERPARAMETERS: BATCH_SIZE=problems/batch, GROUP_SIZE=rollouts/problem
Total rollouts/batch=BATCH_SIZE*GROUP_SIZE. Batches=ceil(MAX_PROBLEMS/BATCH_SIZE)
"""

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.math_rl.math_env import get_math_dataset_builder
from tinker_cookbook.rl.train import AsyncConfig, Config, main
from tinker_cookbook.rl.types import RLDataset
from training_utils import LimitedRLDatasetBuilder, run_rl_evaluation

# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class TrainingConfig:
    model_name: str
    dataset_name: str  # "math", "polaris", "deepmath", or "gsm8k"
    max_problems: int | None = None  # Training problems (None=all)
    eval_problems: int | None = None  # Eval problems (None=all test)

TRAINING_CONFIGS = [
    TrainingConfig(model_name="moonshotai/Kimi-K2-Thinking", dataset_name="math", max_problems=50, eval_problems=5),
    TrainingConfig(model_name="meta-llama/Llama-3.1-70B", dataset_name="math", max_problems=50, eval_problems=5),
]


RUN_TRAINING = False
RUN_EVAL = True

BASE_LOG_PATH = Path(__file__).parent / "experiments"

SHUFFLE_SEED = 42
GROUP_SIZE = 4  # Rollouts per problem
BATCH_SIZE = 20  # Problems per batch
LEARNING_RATE = 1e-5
MAX_TOKENS = 2048
TEMPERATURE = 1.0
LORA_RANK = 32
SAVE_EVERY = 10
EVAL_EVERY = 10
WANDB_PROJECT = "tinkering-with-tinker-rl"
WANDB_NAME = None

async def extract_test_dataset(dataset_builder, eval_problems: int | None = None):
    """Extract test dataset, optionally limiting it."""
    _, test_dataset = await dataset_builder()
    if test_dataset is None or (eval_problems is None or len(test_dataset) == 0):
        return test_dataset
    batch_size = getattr(test_dataset, 'batch_size', 1)
    max_batches = (eval_problems + batch_size - 1) // batch_size
    class LimitedTestDataset(RLDataset):
        def __init__(self, base: RLDataset, max_b: int):
            self.base_dataset, self.max_batches = base, max_b
            if hasattr(base, 'batch_size'):
                self.batch_size = base.batch_size
        def get_batch(self, i: int):
            if i >= self.max_batches:
                raise IndexError(f"Batch {i} >= max {self.max_batches}")
            return self.base_dataset.get_batch(i)
        def __len__(self):
            return min(self.max_batches, len(self.base_dataset))
    return LimitedTestDataset(test_dataset, max_batches)

def build_config(model_name: str, dataset_name: str, log_path: str, max_problems: int | None = None) -> Config:
    """Build training configuration."""
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    base_builder = get_math_dataset_builder(dataset_name, BATCH_SIZE, model_name, renderer_name, GROUP_SIZE, SHUFFLE_SEED)
    dataset_builder = LimitedRLDatasetBuilder(base_builder=base_builder, max_problems=max_problems) if max_problems else base_builder
    return chz.Blueprint(Config).apply({
        "log_path": log_path,
        "model_name": model_name,
        "dataset_builder": dataset_builder,
        "learning_rate": LEARNING_RATE,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "lora_rank": LORA_RANK,
        "save_every": SAVE_EVERY,
        "eval_every": EVAL_EVERY,
        "wandb_project": WANDB_PROJECT,
        "wandb_name": WANDB_NAME,
        "loss_fn": "importance_sampling",
        "async_config": AsyncConfig(
            max_steps_off_policy=5,
            groups_per_batch=BATCH_SIZE,
        ),
    }).make()

def get_log_path(model_name: str, dataset_name: str, use_existing: bool = False) -> str:
    """Generate log path. If use_existing=True, find most recent existing log for this model/dataset."""
    if use_existing:
        model_slug = model_name.replace("/", "-").lower()
        pattern = f"{model_slug}_{dataset_name}_*"
        matching_dirs = sorted(BASE_LOG_PATH.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if matching_dirs:
            existing_path = str(matching_dirs[0])
            print(f"Using existing log path: {existing_path}")
            return existing_path
        print(f"Warning: No existing log found for {model_name} on {dataset_name}, creating new path")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(BASE_LOG_PATH / f"{model_name.replace('/', '-').lower()}_{dataset_name}_{timestamp}")

async def train_and_eval_model(model_name: str, dataset_name: str, max_problems: int | None = None, eval_problems: int | None = None):
    """Train and optionally evaluate a model."""
    log_path = get_log_path(model_name, dataset_name, use_existing=not RUN_TRAINING)
    config = build_config(model_name, dataset_name, log_path, max_problems)
    if RUN_TRAINING:
        cli_utils.check_log_dir(log_path, behavior_if_exists="ask")
        nb = "?" if max_problems is None else (max_problems + BATCH_SIZE - 1) // BATCH_SIZE
        tr = "?" if max_problems is None else nb * BATCH_SIZE * GROUP_SIZE
        print(f"\n{'='*80}\nTraining: {model_name} on {dataset_name}\nLog: {log_path}")
        print(f"BATCH_SIZE={BATCH_SIZE}, GROUP_SIZE={GROUP_SIZE}, max_problems={max_problems or 'all'}")
        print(f"Batches: {nb}, Total rollouts: {tr}, LR={LEARNING_RATE}, max_tokens={MAX_TOKENS}\n{'='*80}\n")
        await main(config)
    if RUN_EVAL:
        renderer_name = model_info.get_recommended_renderer_name(model_name)
        test_builder = get_math_dataset_builder(dataset_name, 1, model_name, renderer_name, 1, SHUFFLE_SEED)
        test_dataset = await extract_test_dataset(test_builder, eval_problems)
        if test_dataset:
            await run_rl_evaluation(config, test_dataset, model_name, dataset_name)

async def main_async():
    """Train all model configurations."""
    for config in TRAINING_CONFIGS:
        await train_and_eval_model(config.model_name, config.dataset_name, config.max_problems, config.eval_problems)

if __name__ == "__main__":
    asyncio.run(main_async())