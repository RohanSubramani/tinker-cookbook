"""
Chat evaluation script for comparing base and fine-tuned models side-by-side.
"""

import asyncio
from pathlib import Path
from datetime import datetime
from typing import List
import tinker
from tinker_cookbook import renderers, model_info, checkpoint_utils
from tinker_cookbook.completers import TinkerMessageCompleter, MessageCompleter
from tinker_cookbook.tokenizer_utils import get_tokenizer


def format_message(role: str, content: str, width: int = 70) -> List[str]:
    """Format a message with word wrapping."""
    lines = []
    header = f"[{role.upper()}]"
    lines.append(header)
    
    # Word wrap the content
    words = content.split()
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 <= width:
            current_line += (word + " ")
        else:
            if current_line:
                lines.append(current_line.rstrip())
            current_line = word + " "
    if current_line:
        lines.append(current_line.rstrip())
    
    return lines


def format_side_by_side(
    base_messages: List[dict],
    finetuned_messages: List[dict],
    width: int = 70
) -> str:
    """Format two conversation histories side-by-side."""
    separator = " | "
    total_width = width * 2 + len(separator)
    
    output = []
    output.append("=" * total_width)
    output.append(f"{'BASE MODEL':^{width}}{separator}{'FINE-TUNED MODEL':^{width}}")
    output.append("=" * total_width)
    output.append("")
    
    # Process messages in pairs (they should be synchronized)
    max_messages = max(len(base_messages), len(finetuned_messages))
    
    for i in range(max_messages):
        base_msg = base_messages[i] if i < len(base_messages) else None
        ft_msg = finetuned_messages[i] if i < len(finetuned_messages) else None
        
        # Format both messages
        base_lines = []
        if base_msg:
            base_lines = format_message(base_msg["role"], base_msg["content"], width)
        
        ft_lines = []
        if ft_msg:
            ft_lines = format_message(ft_msg["role"], ft_msg["content"], width)
        
        # Output side by side
        max_lines = max(len(base_lines), len(ft_lines))
        for j in range(max_lines):
            left = base_lines[j] if j < len(base_lines) else ""
            right = ft_lines[j] if j < len(ft_lines) else ""
            output.append(f"{left:<{width}}{separator}{right:<{width}}")
        
        output.append("")
        output.append("-" * total_width)
        output.append("")
    
    return "\n".join(output)


async def get_response(
    completer: MessageCompleter,
    messages: List[dict]
) -> str:
    """Get a response from a model given conversation history."""
    response = await completer(messages)
    return response["content"]


def find_recent_experiments(experiments_dir: Path, limit: int = 5) -> List[Path]:
    """Find the most recent experiment runs."""
    if not experiments_dir.exists():
        return []
    
    # Find all directories matching the pattern
    experiment_dirs = [
        d for d in experiments_dir.iterdir()
        if d.is_dir() and d.name.startswith("advised_rl_")
    ]
    
    # Sort by modification time (most recent first)
    experiment_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    
    return experiment_dirs[:limit]


def load_checkpoint_path(experiment_dir: Path) -> str:
    """Load the model path from checkpoint file."""
    checkpoint_info = checkpoint_utils.get_last_checkpoint(
        str(experiment_dir), 
        required_key="sampler_path"
    )
    
    if not checkpoint_info:
        raise ValueError(
            f"No checkpoint with sampler_path found in {experiment_dir}.\n"
            f"This might indicate an incomplete training run.\n"
            f"Make sure the training completed successfully."
        )
    
    model_path = checkpoint_info.get("sampler_path")
    if not model_path:
        raise ValueError(f"Could not find sampler_path in checkpoint: {checkpoint_info}")
    
    return model_path


async def main():
    """Main chat evaluation loop."""
    # Configuration
    BASE_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
    MAX_TOKENS = 1024
    
    print("=" * 80)
    print("CHAT EVALUATION: Base vs Fine-Tuned Model")
    print("=" * 80)
    print()
    
    # Select experiment directory
    experiments_dir = Path(__file__).parent.parent / "experiments"
    recent_experiments = find_recent_experiments(experiments_dir)
    
    selected_experiment_dir = None
    
    if recent_experiments:
        print("Recent experiment runs:")
        print()
        for i, exp_dir in enumerate(recent_experiments, 1):
            # Get timestamp from directory name
            timestamp = exp_dir.name.split("_")[-2:]
            timestamp_str = "_".join(timestamp) if len(timestamp) == 2 else "unknown"
            print(f"  {i}. {exp_dir.name}")
            print(f"     ({timestamp_str})")
        print()
        print("Enter a number (1-5) to select, or enter a custom experiment directory path:")
        user_input = input("> ").strip()
        
        # Check if input is a number
        if user_input.isdigit():
            selection = int(user_input)
            if 1 <= selection <= len(recent_experiments):
                selected_experiment_dir = recent_experiments[selection - 1]
                print(f"\nSelected: {selected_experiment_dir.name}")
            else:
                print(f"Invalid selection. Please enter a number between 1 and {len(recent_experiments)}.")
                return
        else:
            selected_experiment_dir = Path(user_input)
    else:
        print("No recent experiments found in experiments/ directory.")
        print("Enter the path to your experiment directory:")
        user_input = input("> ").strip()
        if user_input:
            selected_experiment_dir = Path(user_input)
    
    if not selected_experiment_dir:
        print("No experiment directory provided. Exiting.")
        return
    
    print(f"\nLoading checkpoint from: {selected_experiment_dir}")
    
    # Load the checkpoint path (tinker:// URI)
    try:
        finetuned_model_path = load_checkpoint_path(selected_experiment_dir)
        print(f"Checkpoint loaded: {finetuned_model_path}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    print("\nInitializing models...")
    
    # Get tokenizer and renderer
    tokenizer = get_tokenizer(BASE_MODEL_NAME)
    renderer_name = model_info.get_recommended_renderer_name(BASE_MODEL_NAME)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    
    # Create service client and sampling clients
    service_client = tinker.ServiceClient()
    base_sampling_client = service_client.create_sampling_client(base_model=BASE_MODEL_NAME)
    finetuned_sampling_client = service_client.create_sampling_client(
        model_path=finetuned_model_path,
        base_model=BASE_MODEL_NAME,
    )
    
    # Initialize completers
    base_completer = TinkerMessageCompleter(
        sampling_client=base_sampling_client,
        renderer=renderer,
        max_tokens=MAX_TOKENS
    )
    
    finetuned_completer = TinkerMessageCompleter(
        sampling_client=finetuned_sampling_client,
        renderer=renderer,
        max_tokens=MAX_TOKENS
    )
    
    # Initialize conversation histories
    base_messages = []
    finetuned_messages = []
    
    # Create output directory and file
    chat_evals_dir = Path(__file__).parent / "chat_evals"
    chat_evals_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = chat_evals_dir / f"chat_eval_{timestamp}.txt"
    
    print(f"\nConversation will be saved to: {output_file.absolute()}")
    print("\nYou can now start chatting. Type 'quit' or 'exit' to end the conversation.")
    print("=" * 80)
    print()
    
    turn = 0
    while True:
        # Get user input
        print(f"\n{'='*80}")
        print(f"Turn {turn + 1}")
        print(f"{'='*80}")
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nEnding conversation. Goodbye!")
            break
        
        if not user_input:
            print("Please enter a message.")
            continue
        
        # Add user message to both histories
        user_message = {"role": "user", "content": user_input}
        base_messages.append(user_message)
        finetuned_messages.append(user_message)
        
        print("\nGenerating responses...\n")
        
        # Get responses from both models in parallel
        base_response, ft_response = await asyncio.gather(
            get_response(base_completer, base_messages),
            get_response(finetuned_completer, finetuned_messages)
        )
        
        # Add responses to histories
        base_messages.append({"role": "assistant", "content": base_response})
        finetuned_messages.append({"role": "assistant", "content": ft_response})
        
        # Display responses
        print("─" * 80)
        print(f"{'BASE MODEL':^40} │ {'FINE-TUNED MODEL':^40}")
        print("─" * 80)
        
        # Display side by side in terminal
        base_lines = base_response.split('\n')
        ft_lines = ft_response.split('\n')
        max_lines = max(len(base_lines), len(ft_lines))
        
        for i in range(max_lines):
            left = base_lines[i] if i < len(base_lines) else ""
            right = ft_lines[i] if i < len(ft_lines) else ""
            # Truncate for terminal display
            left = left[:38] + ".." if len(left) > 40 else left
            right = right[:38] + ".." if len(right) > 40 else right
            print(f"{left:<40} │ {right:<40}")
        
        print("─" * 80)
        
        # Write to file
        with output_file.open('w') as f:
            f.write(f"Chat Evaluation Session\n")
            f.write(f"Started: {timestamp}\n")
            f.write(f"Base Model: {BASE_MODEL_NAME}\n")
            f.write(f"Fine-tuned Experiment: {selected_experiment_dir.name}\n")
            f.write(f"Fine-tuned Model Path: {finetuned_model_path}\n")
            f.write(f"\n")
            f.write(format_side_by_side(base_messages, finetuned_messages))
        
        turn += 1
    
    # Final save
    with output_file.open('w') as f:
        f.write(f"Chat Evaluation Session\n")
        f.write(f"Started: {timestamp}\n")
        f.write(f"Base Model: {BASE_MODEL_NAME}\n")
        f.write(f"Fine-tuned Experiment: {selected_experiment_dir.name}\n")
        f.write(f"Fine-tuned Model Path: {finetuned_model_path}\n")
        f.write(f"Total Turns: {turn}\n")
        f.write(f"\n")
        f.write(format_side_by_side(base_messages, finetuned_messages))
    
    print(f"\nFinal conversation saved to: {output_file.absolute()}")


if __name__ == "__main__":
    asyncio.run(main())

