<h1 align="center">Tinker Cookbook</h1>
<div align="center">
  <img src="assets/tinker-cover.png" width="60%" />
</div>

We provide two libraries for the broader community to customize their language models: `tinker` and `tinker-cookbook`.

## Quick Start (Rohan edits): Running Modular SFT Training

The `modular_sft_test.py` script provides a convenient way to train multiple models and set training datasets and hyperparameters. Here's how to use it:

### Prerequisites

1. **API Keys**: Set up your environment variables (add to `~/.zshrc` or `~/.bashrc`):
   ```bash
   export TINKER_API_KEY="your-tinker-api-key"
   export WANDB_API_KEY="your-wandb-api-key"
   export WANDB_ENTITY="your-wandb-username"  # Optional: defaults to team account if not set
   ```

2. **Installation**: 
   - Install `uv` if you haven't already: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Install dependencies: `uv sync` (this creates a virtual environment and installs all dependencies)
   - The project uses `uv` for dependency management - see the Installation section below for more details

### Running the Script

1. **Configure your training**: Edit the configuration section at the top of `modular_sft_test.py`:
   - `TRAINING_CONFIGS`: List of model-dataset pairs to train
   - `WANDB_PROJECT`: Your wandb project name
   - `WANDB_ENTITY`: Loaded from `WANDB_ENTITY` environment variable (set in your shell config)
   - Other hyperparameters: learning rate, batch size, epochs, etc.

2. **Run the script**:
   ```bash
   uv run python modular_sft_test.py
   ```
   
   **Note**: We use `uv run` instead of `python` directly. This ensures the script runs with the correct virtual environment and dependencies. `uv run` automatically activates the project's virtual environment and runs the command with the right Python version and packages.

The script will:
- Train each model configuration sequentially
- Save checkpoints and logs to `experiments/` directory
- Run model comparisons on test sets after training
- Log metrics to wandb (if configured)

### Customization

- Set `RUN_TRAINING = False` to skip training and only run evaluation
- Set `RUN_EVAL = False` to skip evaluation
- Modify `TRAINING_CONFIGS` to add/remove model configurations
- Adjust hyperparameters in the configuration section

## Installation

1. Sign up for Tinker [here](https://auth.thinkingmachines.ai/sign-up).
2. Once you have access, create an API key from the [console](https://tinker-console.thinkingmachines.ai) and export it as environment variable `TINKER_API_KEY`.
3. **Install `uv`** (recommended): 
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   `uv` is a fast Python package installer and resolver written in Rust. It manages virtual environments and dependencies automatically.

4. **Install dependencies with `uv`**:
   ```bash
   uv sync
   ```
   This command:
   - Creates a virtual environment (if it doesn't exist)
   - Installs all dependencies from `pyproject.toml` and `uv.lock`
   - Ensures reproducible builds with locked dependency versions

5. **Alternative: Install with pip** (if you prefer not to use `uv`):
   ```bash
   pip install tinker
   pip install -e .
   ```

### Using `uv` for Running Scripts

This project uses `uv` for dependency management. When running scripts, use `uv run` instead of `python` directly:

```bash
# Instead of: python script.py
uv run python script.py

# Or activate the environment manually:
source .venv/bin/activate  # uv creates .venv by default
python script.py
```

`uv run` automatically:
- Activates the project's virtual environment
- Uses the correct Python version
- Ensures all dependencies are available

## Tinker

Refer to the [docs](https://tinker-docs.thinkingmachines.ai/training-sampling) to start from basics.
Here we introduce a few Tinker primitives - the basic components to fine-tune LLMs:

```python
import tinker
service_client = tinker.ServiceClient()
training_client = service_client.create_lora_training_client(
  base_model="meta-llama/Llama-3.2-1B", rank=32,
)
training_client.forward_backward(...)
training_client.optim_step(...)
training_client.save_state(...)
training_client.load_state(...)

sampling_client = training_client.save_weights_and_get_sampling_client(name="my_model")
sampling_client.sample(...)
```

See [tinker_cookbook/recipes/sl_loop.py](tinker_cookbook/recipes/sl_loop.py) and [tinker_cookbook/recipes/rl_loop.py](tinker_cookbook/recipes/rl_loop.py) for minimal examples of using these primitives to fine-tune LLMs.

To download the weights of any model:
```python
rest_client = service_client.create_rest_client()
future = rest_client.get_checkpoint_archive_url_from_tinker_path(sampling_client.model_path)
with open(f"model-checkpoint.tar.gz", "wb") as f:
    f.write(future.result())
```

### Tinker Cookbook

Besides these primitives, we also offer **Tinker Cookbook** (a.k.a. this repo), a library of a wide range of abstractions to help you customize training environments.
[`tinker_cookbook/recipes/sl_basic.py`](tinker_cookbook/recipes/sl_basic.py) and [`tinker_cookbook/recipes/rl_basic.py`](tinker_cookbook/recipes/rl_basic.py) contain minimal examples to configure supervised learning and reinforcement learning.

We also include a wide range of more sophisticated examples in the [`tinker_cookbook/recipes/`](tinker_cookbook/recipes/) folder:
1. **[Chat supervised learning](tinker_cookbook/recipes/chat_sl/)**: supervised fine-tuning on conversational datasets like Tulu3.
2. **[Math reasoning](tinker_cookbook/recipes/math_rl/)**: improve LLM reasoning capability by rewarding it for answering math questions correctly.
3. **[Preference learning](tinker_cookbook/recipes/preference/)**: showcase a three-stage RLHF pipeline: 1) supervised fine-tuning, 2) learning a reward model, 3) RL against the reward model.
4. **[Tool use](tinker_cookbook/recipes/tool_use/)**: train LLMs to better use retrieval tools to answer questions more accurately.
5. **[Prompt distillation](tinker_cookbook/recipes/prompt_distillation/)**: internalize long and complex instructions into LLMs.
6. **[Multi-Agent](tinker_cookbook/recipes/multiplayer_rl/)**: optimize LLMs to play against another LLM or themselves.

These examples are located in each subfolder, and their `README.md` files will walk you through the key implementation details, the commands to run them, and the expected performance.

### Documentation

The `docs/` directory contains a mirror of the Tinker documentation. These files are synced from our internal documentation site.

**Note:** The documentation files use MDX format (Markdown with JSX), which includes some syntax that isn't standard Markdown. You may see things like `import` statements, `<Callout>` components, or curly-brace expressions. These are artifacts of our documentation framework - the actual content should still be readable as Markdown.

If you find errors or want to improve the documentation, feel free to submit a PR editing files in `docs/`. We'll sync the changes back to our documentation site.

For the rendered documentation, visit [tinker-docs.thinkingmachines.ai](https://tinker-docs.thinkingmachines.ai).

### Import our utilities

Tinker cookbook includes several utilities. Here's a quick overview:
- [`renderers`](tinker_cookbook/renderers.py) converts tokens from/to structured chat message objects
- [`hyperparam_utils`](tinker_cookbook/hyperparam_utils.py) helps calculate hyperparameters suitable for LoRAs
- [`evaluation`](tinker_cookbook/eval/evaluators.py) provides abstractions for evaluating Tinker models and [`inspect_evaluation`](tinker_cookbook/eval/inspect_evaluators.py) shows how to integrate with InspectAI to make evaluating on standard benchmarks easy.

## Contributing

This project is built in the spirit of open science and collaborative development. We believe that the best tools emerge through community involvement and shared learning.

We welcome PR contributions after our private beta is over. If you have any feedback, please email us at tinker@thinkingmachines.ai.

## Citation
If you use Tinker for your research, please cite it as:
```
Thinking Machines Lab, 2025. Tinker. https://thinkingmachines.ai/tinker/.
```

Or use this BibTeX citation:
```
@misc{tml2025tinker,
  author = {Thinking Machines Lab},
  title = {Tinker},
  year = {2025},
  url = {https://thinkingmachines.ai/tinker/},
}
```
