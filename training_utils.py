"""Utility functions and classes for SFT and RL training and evaluation."""

import json
import os
from typing import Any

import blobfile
import chz
import datasets
import tinker
from tinker import types
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.image_processing_utils import get_image_processor
from tinker_cookbook.renderers import Message, TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import (
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilder,
    SupervisedDataset,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.rl.types import RLDatasetBuilder, RLDataset, EnvGroupBuilder
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator, dataset_to_env_group_builders
from tinker_cookbook.rl.train import Config as RLConfig
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.rl.problem_env import ProblemEnv
from tinker_cookbook.completers import TinkerTokenCompleter


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


@chz.chz
class LimitedRLDatasetBuilder(RLDatasetBuilder):
    """Wrapper that limits the number of training problems in an RL dataset.
    
    This wrapper limits the underlying dataset to max_problems before creating batches.
    For example, if max_problems=100 and batch_size=20, you'll get 5 batches.
    """
    base_builder: RLDatasetBuilder
    max_problems: int | None = None  # Maximum number of problems to train on (None = use all)
    
    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        train_dataset, test_dataset = await self.base_builder()
        
        if self.max_problems is not None:
            # Calculate how many batches we need for max_problems
            # batch_size is the number of problems per batch
            batch_size = train_dataset.batch_size if hasattr(train_dataset, 'batch_size') else 1
            max_batches = (self.max_problems + batch_size - 1) // batch_size
            
            class LimitedDataset(RLDataset):
                """Wrapper that limits the number of batches returned by the dataset."""
                def __init__(self, base_dataset: RLDataset, max_batches: int):
                    self.base_dataset = base_dataset
                    self.max_batches = max_batches
                    # Preserve batch_size attribute if it exists
                    if hasattr(base_dataset, 'batch_size'):
                        self.batch_size = base_dataset.batch_size
                
                def get_batch(self, index: int):
                    if index >= self.max_batches:
                        raise IndexError(f"Batch index {index} >= max_batches {self.max_batches}")
                    return self.base_dataset.get_batch(index)
                
                def __len__(self):
                    return min(self.max_batches, len(self.base_dataset))
            
            train_dataset = LimitedDataset(train_dataset, max_batches)
        
        return train_dataset, test_dataset


async def run_rl_evaluation(
    config: RLConfig,
    test_dataset: RLDataset | None,
    model_name: str,
    dataset_name: str | None = None,
):
    """Run RL evaluation on test set and save prompts/responses from both models.
    
    Similar to run_model_comparison but for RL - evaluates on test problems,
    saves metrics, and saves side-by-side comparison of base vs fine-tuned responses.
    """
    print("\n" + "="*80)
    print("Running RL evaluation on test set...")
    print("="*80 + "\n")
    
    if test_dataset is None:
        print("Warning: No test dataset available. Skipping evaluation.")
        return
    
    # Get the final checkpoint path
    checkpoint_info = checkpoint_utils.get_last_checkpoint(config.log_path, required_key="sampler_path")
    if not checkpoint_info:
        print("Warning: No checkpoint with sampler_path found. Skipping evaluation.")
        return
    
    model_path = checkpoint_info.get("sampler_path")
    if not model_path:
        print("Warning: Could not find sampler_path. Skipping evaluation.")
        return
    
    # Create service client and get both models
    service_client = tinker.ServiceClient(base_url=config.base_url)
    base_client = service_client.create_sampling_client(base_model=model_name)
    fine_tuned_client = service_client.create_sampling_client(
        model_path=model_path,
        base_model=model_name,
    )
    
    # Get renderer for parsing responses
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    
    # Get all env group builders from test dataset
    env_group_builders = dataset_to_env_group_builders(test_dataset)
    
    # Create policies
    base_policy = TinkerTokenCompleter(base_client, max_tokens=config.max_tokens)
    fine_tuned_policy = TinkerTokenCompleter(fine_tuned_client, max_tokens=config.max_tokens)
    
    # Run rollouts for both models
    print(f"Evaluating on {len(env_group_builders)} problems...")
    comparisons = []
    
    for idx, builder in enumerate(env_group_builders):
        # Get the problem/question by creating an env
        envs = await builder.make_envs()
        if not envs or not isinstance(envs[0], ProblemEnv):
            continue
        problem_env = envs[0]
        question = problem_env.get_question()
        reference_answer = problem_env.get_reference_answer()
        
        # Run rollouts
        base_traj_group = await do_group_rollout(builder, base_policy)
        fine_tuned_traj_group = await do_group_rollout(builder, fine_tuned_policy)
        
        # Extract responses (take first trajectory from each group)
        base_response = ""
        base_reward = 0.0
        if base_traj_group.trajectories_G:
            base_traj = base_traj_group.trajectories_G[0]
            if base_traj.transitions:
                base_tokens = base_traj.transitions[0].ac.tokens
                parsed = renderer.parse_response(base_tokens)
                base_response = renderers.ensure_text(parsed[0]["content"]) if parsed else ""
                base_reward = base_traj_group.get_total_rewards()[0]
        
        fine_tuned_response = ""
        fine_tuned_reward = 0.0
        if fine_tuned_traj_group.trajectories_G:
            fine_tuned_traj = fine_tuned_traj_group.trajectories_G[0]
            if fine_tuned_traj.transitions:
                fine_tuned_tokens = fine_tuned_traj.transitions[0].ac.tokens
                parsed = renderer.parse_response(fine_tuned_tokens)
                fine_tuned_response = renderers.ensure_text(parsed[0]["content"]) if parsed else ""
                fine_tuned_reward = fine_tuned_traj_group.get_total_rewards()[0]
        
        comparisons.append({
            "example_id": idx + 1,
            "question": question,
            "reference_answer": reference_answer,
            "base_model": base_response,
            "base_reward": base_reward,
            "fine_tuned_model": fine_tuned_response,
            "fine_tuned_reward": fine_tuned_reward,
        })
    
    # Compute metrics
    evaluator = RLTestSetEvaluator(
        dataset=test_dataset,
        max_tokens=config.max_tokens,
        name="test",
        num_groups_to_log=0,  # Don't log during metric computation
    )
    metrics = await evaluator(fine_tuned_client)
    
    # Save comparison file
    output_file = os.path.join(config.log_path, "model_comparison.txt")
    _write_rl_comparison_file(comparisons, metrics, output_file, model_name, dataset_name)
    
    print(f"\n{'='*80}")
    print(f"Model comparison written to: {output_file}")
    print(f"\nKey Metrics:")
    for key in ["test/reward/total", "test/env/all/reward/total"]:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.4f}")
    print(f"{'='*80}\n")


def _write_rl_comparison_file(
    comparisons: list[dict],
    metrics: dict[str, float],
    output_file: str,
    model_name: str,
    dataset_name: str | None,
):
    """Write side-by-side comparison to file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 120 + "\n")
        f.write("RL MODEL COMPARISON: Base Model vs Fine-Tuned Model\n")
        f.write("=" * 120 + "\n")
        f.write(f"Model: {model_name}\n")
        if dataset_name:
            f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Number of Evaluation Problems: {len(comparisons)}\n")
        f.write("=" * 120 + "\n\n")
        
        # Write metrics summary
        f.write("METRICS SUMMARY:\n")
        f.write("-" * 120 + "\n")
        for key in ["test/reward/total", "test/env/all/reward/total"]:
            if key in metrics:
                f.write(f"{key}: {metrics[key]:.4f}\n")
        f.write("\n" + "=" * 120 + "\n\n")
        
        # Write each comparison
        for comp in comparisons:
            f.write(f"\n{'='*120}\n")
            f.write(f"EXAMPLE {comp['example_id']}\n")
            f.write(f"{'='*120}\n\n")
            
            # Question
            f.write("PROBLEM:\n")
            f.write("-" * 120 + "\n")
            f.write(comp["question"])
            f.write("\n\n")
            
            # Reference answer
            if comp["reference_answer"]:
                f.write("REFERENCE ANSWER:\n")
                f.write("-" * 120 + "\n")
                f.write(comp["reference_answer"])
                f.write("\n\n")
            
            # Side-by-side comparison
            f.write("RESPONSES:\n")
            f.write("-" * 120 + "\n")
            f.write(f"{'BASE MODEL (reward: ' + str(comp['base_reward']) + ')':<60} | {'FINE-TUNED MODEL (reward: ' + str(comp['fine_tuned_reward']) + ')':<60}\n")
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

