"""
Simple SFT training script for testing with a small dataset.

This script demonstrates the core syntax for:
- Setting the model
- Setting the dataset  
- Setting hyperparameters

Run with:
    python test_sft_simple.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import blobfile
import chz
import datasets
import tinker
from tinker import types
from tinker_cookbook import checkpoint_utils, cli_utils, model_info, renderers
from tinker_cookbook.image_processing_utils import get_image_processor
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.renderers import Message, TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import (
    FromConversationFileBuilder,
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilder,
    ChatDatasetBuilderCommonConfig,
    SupervisedDataset,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer


@chz.chz
class LimitedConversationFileBuilder(ChatDatasetBuilder):
    """Dataset builder that limits the number of training examples."""
    file_path: str
    train_size: int  # Maximum number of training examples
    test_size: int = 0
    shuffle_seed: int = 0

    @property
    def renderer(self) -> renderers.Renderer:
        """Override renderer property to handle vision-language models."""
        tokenizer = self.tokenizer
        # Check if model is vision-language and needs image processor
        try:
            attributes = model_info.get_model_attributes(self.common_config.model_name_for_tokenizer)
            if attributes.is_vl:
                image_processor = get_image_processor(self.common_config.model_name_for_tokenizer)
                return renderers.get_renderer(
                    self.common_config.renderer_name,
                    tokenizer,
                    image_processor=image_processor,
                )
        except (ValueError, AttributeError):
            # If we can't determine model attributes, try without image processor first
            pass
        
        # Default: no image processor (for non-VL models)
        return renderers.get_renderer(
            self.common_config.renderer_name,
            tokenizer,
        )

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        # Load conversations from JSONL file
        conversations = []
        with blobfile.BlobFile(self.file_path, "r", streaming=False) as f:
            for line in f:
                data = json.loads(line.strip())
                if "messages" not in data:
                    raise ValueError(
                        f"Each line in the JSONL file must contain a 'messages' field. Got: {data.keys()}"
                    )
                conversations.append(data)

        # Create HuggingFace dataset from the loaded data
        dataset = datasets.Dataset.from_list(conversations)

        # Shuffle if seed is provided
        if self.shuffle_seed is not None:
            dataset = dataset.shuffle(seed=self.shuffle_seed)

        # Limit training dataset size
        train_ds = dataset.take(self.train_size)
        
        # Split test set if requested
        if self.test_size > 0 and len(dataset) > self.train_size + self.test_size:
            test_ds = dataset.skip(self.train_size).take(self.test_size)
        elif self.test_size > 0:
            # If not enough data, use remaining for test
            remaining = len(dataset) - self.train_size
            test_ds = dataset.skip(self.train_size).take(remaining) if remaining > 0 else None
        else:
            test_ds = None

        # Use train_on_what from common_config if provided, otherwise use default
        train_on_what = (
            TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )

        # Define mapping function
        def map_fn(row: dict) -> tinker.Datum:
            return conversation_to_datum(
                row["messages"], self.renderer, self.common_config.max_length, train_on_what
            )

        # Create supervised dataset
        supervised_dataset = SupervisedDatasetFromHFDataset(
            train_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
        )

        # Create evaluator if we have test data
        if test_ds is not None:
            test_dataset = SupervisedDatasetFromHFDataset(
                test_ds, batch_size=len(test_ds), map_fn=map_fn
            )
        else:
            test_dataset = None

        return supervised_dataset, test_dataset


@chz.chz
class ModelComparisonEvaluatorBuilder:
    """Builder for an evaluator that compares base and fine-tuned models."""
    test_conversations: list[dict[str, Any]]  # Raw test conversations
    model_name: str
    renderer_name: str
    log_path: str
    max_tokens: int = 512
    temperature: float = 0.7

    def __call__(self) -> "ModelComparisonEvaluator":
        return ModelComparisonEvaluator(
            test_conversations=self.test_conversations,
            model_name=self.model_name,
            renderer_name=self.renderer_name,
            log_path=self.log_path,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )


class ModelComparisonEvaluator(SamplingClientEvaluator):
    """Evaluator that compares base and fine-tuned model responses side-by-side."""

    def __init__(
        self,
        test_conversations: list[dict[str, Any]],
        model_name: str,
        renderer_name: str,
        log_path: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        dataset_name: str | None = None,
    ):
        self.test_conversations = test_conversations
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.log_path = log_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.dataset_name = dataset_name
        
        tokenizer = get_tokenizer(model_name)
        # Check if model is vision-language and needs image processor
        try:
            attributes = model_info.get_model_attributes(model_name)
            if attributes.is_vl:
                image_processor = get_image_processor(model_name)
                self.renderer = renderers.get_renderer(renderer_name, tokenizer, image_processor=image_processor)
            else:
                self.renderer = renderers.get_renderer(renderer_name, tokenizer)
        except (ValueError, AttributeError):
            # If we can't determine model attributes, try without image processor first
            self.renderer = renderers.get_renderer(renderer_name, tokenizer)

    async def __call__(self, fine_tuned_sampling_client: tinker.SamplingClient) -> dict[str, float]:
        """Compare base and fine-tuned models on test set."""
        import tinker
        from tinker_cookbook import renderers
        
        # Create base model sampling client
        service_client = tinker.ServiceClient()
        base_sampling_client = service_client.create_sampling_client(
            base_model=self.model_name
        )
        
        # Sampling parameters
        sampling_params = types.SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=1.0,
            stop=self.renderer.get_stop_sequences(),
        )
        
        # Collect all comparisons
        comparisons = []
        
        for idx, conv_data in enumerate(self.test_conversations):
            messages = conv_data["messages"]
            
            # Get the prompt (all messages except the last assistant message)
            # For generation, we want to generate the assistant's response
            prompt_messages = []
            for i, msg in enumerate(messages):
                # Include all messages except the last one if it's from assistant
                if i == len(messages) - 1 and msg["role"] == "assistant":
                    # This is the last assistant message - we'll generate this, so skip it
                    break
                prompt_messages.append(Message(role=msg["role"], content=msg["content"]))
            
            # Build the generation prompt
            model_input = self.renderer.build_generation_prompt(prompt_messages)
            
            # Generate from both models
            base_result = await base_sampling_client.sample_async(
                prompt=model_input,
                num_samples=1,
                sampling_params=sampling_params,
            )
            fine_tuned_result = await fine_tuned_sampling_client.sample_async(
                prompt=model_input,
                num_samples=1,
                sampling_params=sampling_params,
            )
            
            # Parse responses
            base_tokens = base_result.sequences[0].tokens
            fine_tuned_tokens = fine_tuned_result.sequences[0].tokens
            
            base_response = self.renderer.parse_response(base_tokens)[0]
            fine_tuned_response = self.renderer.parse_response(fine_tuned_tokens)[0]
            
            base_text = renderers.ensure_text(base_response["content"])
            fine_tuned_text = renderers.ensure_text(fine_tuned_response["content"])
            
            # Get expected response if available
            expected_text = ""
            if messages[-1]["role"] == "assistant":
                expected_text = messages[-1]["content"]
            
            comparisons.append({
                "example_id": idx + 1,
                "prompt": "\n".join([f"{msg['role']}: {msg['content']}" for msg in prompt_messages]),
                "expected": expected_text,
                "base_model": base_text,
                "fine_tuned_model": fine_tuned_text,
            })
        
        # Write comparison file
        output_file = os.path.join(self.log_path, "model_comparison.txt")
        self._write_comparison_file(comparisons, output_file)
        
        print(f"\n{'='*80}")
        print(f"Model comparison written to: {output_file}")
        print(f"{'='*80}\n")
        
        # Return empty metrics (this evaluator is for inspection, not metrics)
        return {}
    
    def _write_comparison_file(self, comparisons: list[dict], output_file: str):
        """Write side-by-side comparison to file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=" * 120 + "\n")
            f.write("MODEL COMPARISON: Base Model vs Fine-Tuned Model\n")
            f.write("=" * 120 + "\n")
            if self.model_name:
                f.write(f"Model: {self.model_name}\n")
            if self.dataset_name:
                f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Number of Evaluation Questions: {len(comparisons)}\n")
            f.write("=" * 120 + "\n\n")
            
            for comp in comparisons:
                f.write(f"\n{'='*120}\n")
                f.write(f"EXAMPLE {comp['example_id']}\n")
                f.write(f"{'='*120}\n\n")
                
                # Prompt
                f.write("PROMPT:\n")
                f.write("-" * 120 + "\n")
                f.write(comp["prompt"])
                f.write("\n\n")
                
                # Expected (if available)
                if comp["expected"]:
                    f.write("EXPECTED RESPONSE:\n")
                    f.write("-" * 120 + "\n")
                    f.write(comp["expected"])
                    f.write("\n\n")
                
                # Side-by-side comparison
                f.write("RESPONSES:\n")
                f.write("-" * 120 + "\n")
                f.write(f"{'BASE MODEL':<60} | {'FINE-TUNED MODEL':<60}\n")
                f.write("-" * 120 + "\n")
                
                # Split into lines for better formatting
                base_lines = comp["base_model"].split("\n")
                fine_tuned_lines = comp["fine_tuned_model"].split("\n")
                max_lines = max(len(base_lines), len(fine_tuned_lines))
                
                for i in range(max_lines):
                    base_line = base_lines[i] if i < len(base_lines) else ""
                    fine_tuned_line = fine_tuned_lines[i] if i < len(fine_tuned_lines) else ""
                    
                    # Truncate if too long
                    base_line = base_line[:58] if len(base_line) > 58 else base_line
                    fine_tuned_line = fine_tuned_line[:58] if len(fine_tuned_line) > 58 else fine_tuned_line
                    
                    f.write(f"{base_line:<60} | {fine_tuned_line:<60}\n")
                
                f.write("\n")
            
            f.write("\n" + "=" * 120 + "\n")
            f.write("END OF COMPARISON\n")
            f.write("=" * 120 + "\n")


def build_config_blueprint() -> chz.Blueprint[train.Config]:
    """
    Build the training configuration.
    
    This is where you configure:
    1. MODEL: Which base model to fine-tune
    2. DATASET: Where your training data lives
    3. HYPERPARAMETERS: Learning rate, batch size, epochs, etc.
    """
    
    # ============================================
    # 1. MODEL CONFIGURATION
    # ============================================
    # Set the base model you want to fine-tune
    # Available models: see docs/model-lineup.mdx
    model_name = "Qwen/Qwen3-VL-235B-A22B-Instruct" # "meta-llama/Llama-3.1-8B"
    
    # Get the recommended renderer for this model
    # The renderer converts messages to tokens in the format the model expects
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    # For Llama-3.1-8B, this returns "llama3"
    
    # ============================================
    # 2. DATASET CONFIGURATION
    # ============================================
    # Common config shared by all chat dataset builders
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,  # Used to load the tokenizer
        renderer_name=renderer_name,          # Which renderer to use
        max_length=32768,                     # Max sequence length (truncate if longer)
        batch_size=5,                         # 5 examples per batch = 10 batches for 50 examples
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,  # What tokens to train on
    )
    
    # Create dataset from a JSONL file
    # Format: each line is {"messages": [{"role": "user", "content": "..."}, ...]}
    example_data_path = Path(__file__).parent / "tinker_cookbook" / "example_data" / "conversations.jsonl"
    
    # Use limited dataset builder: 50 training examples, 10 batches of 5 each
    dataset_builder = LimitedConversationFileBuilder(
        common_config=common_config,
        file_path=str(example_data_path),
        train_size=50,  # Limit to 50 training examples
        test_size=5,    # Use 5 examples for test set
        shuffle_seed=42,
    )
    
    # ============================================
    # 3. HYPERPARAMETER CONFIGURATION
    # ============================================
    # Create the main training config
    log_path = "/tmp/tinker-test-sft-qwen3-vl-235b-a22b-instruct"
    return chz.Blueprint(train.Config).apply(
        {
            # Required: where to save logs and checkpoints
            "log_path": log_path,
            
            # Required: which model to fine-tune
            "model_name": model_name,
            
            # Required: dataset builder (configured above)
            "dataset_builder": dataset_builder,
            
            # Training hyperparameters
            "learning_rate": 2e-4,      # Learning rate (LoRA needs ~10x higher than full fine-tune)
            "lr_schedule": "linear",     # Learning rate schedule: "linear", "constant", or "cosine"
            "num_epochs": 1,            # Number of epochs
            
            # Model parameters
            "lora_rank": 32,            # LoRA rank (lower = fewer parameters, faster)
            
            # Checkpointing and evaluation
            "save_every": 5,            # Save checkpoint every N steps (0 = disabled)
            "eval_every": 3,            # Run evaluation every N steps (0 = disabled)
            
            # Note: Model comparison runs after training completes (see main function)
            
            # WandB logging (requires WANDB_API_KEY environment variable)
            "wandb_project": "tinkering-with-tinker",      # Set to "my-project-name" to enable wandb logging
            "wandb_name": None,         # Optional: custom run name (defaults to auto-generated)
            
            # Adam optimizer parameters (usually keep defaults)
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_eps": 1e-8,
        }
    )


async def run_model_comparison(
    config: train.Config,
    test_conversations: list[dict],
    model_name: str,
    renderer_name: str,
    dataset_name: str | None = None,
):
    """Run model comparison after training completes."""
    print("\n" + "="*80)
    print("Running model comparison on test set...")
    print("="*80 + "\n")
    
    # Get the final checkpoint path
    checkpoint_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if not checkpoint_info:
        print("Warning: No checkpoint found. Skipping model comparison.")
        return
    
    # Create service client and get the fine-tuned model
    service_client = tinker.ServiceClient(base_url=config.base_url)
    
    # Create fine-tuned sampling client from the final checkpoint
    # Must use sampler_path for sampling (not state_path)
    checkpoint_info = checkpoint_utils.get_last_checkpoint(config.log_path, required_key="sampler_path")
    if not checkpoint_info:
        print("Warning: No checkpoint with sampler_path found. Skipping model comparison.")
        return
    
    model_path = checkpoint_info.get("sampler_path")
    if not model_path:
        print("Warning: Could not find sampler_path. Skipping model comparison.")
        return
    
    fine_tuned_client = service_client.create_sampling_client(
        model_path=model_path,
        base_model=model_name,
    )
    
    # Create and run the evaluator
    evaluator = ModelComparisonEvaluator(
        test_conversations=test_conversations,
        model_name=model_name,
        renderer_name=renderer_name,
        log_path=config.log_path,
        max_tokens=512,
        temperature=0.7,
        dataset_name=dataset_name,
    )
    
    await evaluator(fine_tuned_client)


def main(config: train.Config):
    """Run the training."""
    # Check if log directory exists and ask what to do
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    
    # Extract test conversations from the dataset builder
    # Build the dataset to get the test dataset, then extract raw conversations
    dataset_builder = config.dataset_builder
    _, test_dataset = dataset_builder()
    
    # Extract raw test conversations from the test dataset if it exists
    test_conversations = []
    if test_dataset is not None:
        # Access the underlying HuggingFace dataset to get raw conversations
        test_conversations = test_dataset.hf_dataset.to_list()
    
    # Get model info
    model_name = config.model_name
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    
    # Run the async training loop
    asyncio.run(train.main(config))
    
    # Run model comparison after training
    if test_conversations:
        asyncio.run(run_model_comparison(
            config=config,
            test_conversations=test_conversations,
            model_name=model_name,
            renderer_name=renderer_name,
        ))


@chz.chz
class EvalOnlyConfig:
    """Configuration for running evaluation only (no training)."""
    log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))
    model_name: str
    model_path: str | None = None  # If None, uses last checkpoint from log_path
    dataset_file_path: str
    test_size: int = 5
    shuffle_seed: int = 42
    max_tokens: int = 512
    temperature: float = 0.7
    base_url: str | None = None


async def run_eval_only(config: EvalOnlyConfig):
    """Run model comparison evaluation without training."""
    from tinker_cookbook import checkpoint_utils
    
    print("="*80)
    print("Running Model Comparison Evaluation")
    print("="*80)
    print(f"Log path: {config.log_path}")
    print(f"Model: {config.model_name}")
    
    # Load test conversations
    conversations = []
    with blobfile.BlobFile(config.dataset_file_path, "r", streaming=False) as f:
        for line in f:
            data = json.loads(line.strip())
            if "messages" not in data:
                raise ValueError(
                    f"Each line in the JSONL file must contain a 'messages' field. Got: {data.keys()}"
                )
            conversations.append(data)
    
    # Create HuggingFace dataset and split
    dataset = datasets.Dataset.from_list(conversations)
    if config.shuffle_seed is not None:
        dataset = dataset.shuffle(seed=config.shuffle_seed)
    
    # Take test_size examples (or all if test_size is 0)
    if config.test_size > 0:
        test_conversations = dataset.take(config.test_size).to_list()
    else:
        test_conversations = dataset.to_list()
    
    print(f"Loaded {len(test_conversations)} test examples")
    
    # Get model path
    if config.model_path:
        model_path = config.model_path
        print(f"Using provided model path: {model_path}")
    else:
        # Get from last checkpoint - must use sampler_path for sampling
        checkpoint_info = checkpoint_utils.get_last_checkpoint(config.log_path, required_key="sampler_path")
        if not checkpoint_info:
            raise ValueError(
                f"No checkpoint with sampler_path found in {config.log_path}. "
                f"Provide model_path or run training first."
            )
        model_path = checkpoint_info.get("sampler_path")
        if not model_path:
            raise ValueError(f"No sampler_path found in checkpoint. Check {config.log_path}/checkpoints.jsonl")
        print(f"Using checkpoint sampler path: {model_path}")
    
    # Get renderer
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    
    # Create service client
    service_client = tinker.ServiceClient(base_url=config.base_url)
    
    # Create fine-tuned sampling client
    fine_tuned_client = service_client.create_sampling_client(
        model_path=model_path,
        base_model=config.model_name,
    )
    
    # Create and run evaluator
    evaluator = ModelComparisonEvaluator(
        test_conversations=test_conversations,
        model_name=config.model_name,
        renderer_name=renderer_name,
        log_path=config.log_path,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )
    
    await evaluator(fine_tuned_client)


def eval_main(config: EvalOnlyConfig):
    """Entry point for evaluation-only mode."""
    asyncio.run(run_eval_only(config))


if __name__ == "__main__":
    # Check if running in eval-only mode
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        # Remove "eval" from argv and parse as EvalOnlyConfig
        eval_argv = sys.argv[2:]
        blueprint = chz.Blueprint(EvalOnlyConfig)
        blueprint.make_from_argv(eval_argv, allow_hyphens=True)
        eval_main(blueprint.make())

        
    else:
        # Normal training mode
        blueprint = build_config_blueprint()
        blueprint.make_from_argv(sys.argv[1:], allow_hyphens=True)
        main(blueprint.make())

