"""Modular SFT training script - set training configs at the top.

IMPORTANT: Keep this file minimal! All helper functions and dataset builders
should be in training_utils.py. See training_utils.py for:
- LimitedConversationFileBuilder
- ModelComparisonEvaluator, run_model_comparison
"""

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import chz

from training_utils import LimitedConversationFileBuilder, run_model_comparison
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class TrainingConfig:
    model_name: str
    dataset_file: str
    train_size: int
    test_size: int

TRAINING_CONFIGS = [
    TrainingConfig(model_name="Qwen/Qwen3-30B-A3B", dataset_file="conversations.jsonl", train_size=50, test_size=5),
    TrainingConfig(model_name="Qwen/Qwen3-30B-A3B-Base", dataset_file="conversations.jsonl", train_size=50, test_size=5),
    TrainingConfig(model_name="Qwen/Qwen3-32B", dataset_file="conversations.jsonl", train_size=50, test_size=5),
    TrainingConfig(model_name="Qwen/Qwen3-8B", dataset_file="conversations.jsonl", train_size=50, test_size=5),
    TrainingConfig(model_name="Qwen/Qwen3-8B-Base", dataset_file="conversations.jsonl", train_size=50, test_size=5),
]

BASE_LOG_PATH = Path(__file__).parent / "experiments"
BASE_DATASET_DIR = Path(__file__).parent / "tinker_cookbook" / "example_data"

SHUFFLE_SEED = 42
MAX_LENGTH = 32768
BATCH_SIZE = 5
LEARNING_RATE = 2e-4
LR_SCHEDULE = "linear"
NUM_EPOCHS = 1 # train_size can override this
LORA_RANK = 32
SAVE_EVERY = 5 # This is a number of batches
EVAL_EVERY = 3 # This is a number of batches
WANDB_PROJECT = "tinkering-with-tinker"
WANDB_NAME = None
# Load wandb entity from environment variable, fallback to None (uses default/team account)
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", None)
RUN_TRAINING = True
RUN_EVAL = True

def build_config(model_name: str, dataset_file: Path, log_path: str, train_size: int, test_size: int) -> train.Config:
    """Build training configuration for a model-dataset pair."""
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    dataset_builder = LimitedConversationFileBuilder(
        common_config=common_config,
        file_path=str(dataset_file),
        train_size=train_size,
        test_size=test_size,
        shuffle_seed=SHUFFLE_SEED,
    )
    return chz.Blueprint(train.Config).apply({
        "log_path": log_path,
        "model_name": model_name,
        "dataset_builder": dataset_builder,
        "learning_rate": LEARNING_RATE,
        "lr_schedule": LR_SCHEDULE,
        "num_epochs": NUM_EPOCHS,
        "lora_rank": LORA_RANK,
        "save_every": SAVE_EVERY,
        "eval_every": EVAL_EVERY,
        "wandb_project": WANDB_PROJECT,
        "wandb_name": WANDB_NAME,
        "wandb_entity": WANDB_ENTITY,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_eps": 1e-8,
    }).make()

def extract_test_conversations(dataset_builder: LimitedConversationFileBuilder) -> list[dict]:
    """Extract raw test conversations from the dataset builder."""
    _, test_dataset = dataset_builder()
    return [] if test_dataset is None else test_dataset.hf_dataset.to_list()

def get_log_path(model_name: str, dataset_file: str) -> str:
    """Generate log path from model name, dataset, and timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model_name.replace("/", "-").lower()
    dataset_slug = Path(dataset_file).stem
    return str(BASE_LOG_PATH / f"{model_slug}_{dataset_slug}_{timestamp}")

async def train_and_eval_model(model_name: str, dataset_file: Path, train_size: int, test_size: int):
    """Train a single model on a dataset."""
    log_path = get_log_path(model_name, dataset_file.name)
    config = build_config(model_name, dataset_file, log_path, train_size, test_size)
    if RUN_TRAINING:
        cli_utils.check_log_dir(log_path, behavior_if_exists="ask")
        print(f"\n{'='*80}\nTraining: {model_name}\nLog: {log_path}\n{'='*80}\n")
        await train.main(config)
    if RUN_EVAL:
        test_conversations = extract_test_conversations(config.dataset_builder)
        if test_conversations:
            await run_model_comparison(
                config=config,
                test_conversations=test_conversations,
                model_name=model_name,
                renderer_name=model_info.get_recommended_renderer_name(model_name),
                dataset_name=dataset_file.name,
            )

async def main():
    """Train all model configurations."""
    for config in TRAINING_CONFIGS:
        dataset_file = BASE_DATASET_DIR / config.dataset_file
        await train_and_eval_model(config.model_name, dataset_file, config.train_size, config.test_size)

if __name__ == "__main__":
    asyncio.run(main())