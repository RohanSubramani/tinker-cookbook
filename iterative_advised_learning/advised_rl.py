"""Iterative advised RL training script - config at top, easy to edit.

IMPORTANT: Keep this file minimal! All helper functions should be in
iterative_advised_learning/training_helpers.py. See training_helpers.py for:
- setup_clients_and_dataset, load_and_split_data
- run_phase1_training, run_phase2_training
- run_quick_test_eval, run_test_evaluation
- finalize_training, find_log_path, load_eval_checkpoint
"""

import asyncio
import os
from pathlib import Path

from tinker_cookbook.utils import ml_log

from iterative_advised_learning.training_helpers import (
    finalize_training,
    find_log_path,
    load_and_split_data,
    load_eval_checkpoint,
    run_phase1_training,
    run_phase2_training,
    run_quick_test_eval,
    run_test_evaluation,
    setup_clients_and_dataset,
)

# ============================================================================
# CONFIGURATION - Edit these parameters
# ============================================================================
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
GRADER_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
ADVISOR_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

NUM_TRAIN_QUESTIONS = 400  # Number of questions to train on
NUM_TEST_QUESTIONS = 40  # Number of questions for test set
NUM_EPOCHS_WITH_ADVICE = 5  # Number of epochs to train with advice (after initial epoch without)
# Total groups = 1 epoch without advice + NUM_EPOCHS_WITH_ADVICE epochs with advice
TOTAL_TRAIN_GROUPS = (1 + NUM_EPOCHS_WITH_ADVICE) * NUM_TRAIN_QUESTIONS
GROUP_SIZE = 8  # Rollouts per problem
BATCH_SIZE = 10  # Problems per batch

TOTAL_ROLLOUTS = TOTAL_TRAIN_GROUPS * GROUP_SIZE

LEARNING_RATE = 1e-4
MAX_TOKENS = 2048
TEMPERATURE = 1.0
LORA_RANK = 32

SCORE_THRESHOLD_FOR_ADVICE = 100.0  # Get advice if score < this
RECONTEXTUALIZE_BY_REMOVING_ADVICE = True  # Remove advice from prompts before computing gradients
SAVE_RECONTEXTUALIZATION_EXAMPLE = True  # Save example showing generation vs training prompts

RUN_TRAIN = True  # Whether to run training
RUN_EVAL = True  # Whether to run final evaluation (requires checkpoint if RUN_TRAIN=False)

EQUALLY_SPACED_SAVES_EXCLUDING_FINAL = 2  # Number of equally spaced checkpoints during training (final save always happens)
EQUALLY_SPACED_EVALS_EXCLUDING_FINAL = 2  # Number of equally spaced test set evals during training (final eval always happens)

WANDB_PROJECT = "tinkering-with-tinker-rl"  # Set to enable wandb logging (e.g., "iterative-advised-rl")
WANDB_NAME = None  # Optional wandb run name

BASE_LOG_PATH = Path(__file__).parent.parent / "experiments"
DATA_PATH = Path(__file__).parent / "data" / "moral_philosophy.jsonl"
# If RUN_TRAIN=False, specify the log_path to evaluate (or None to use most recent)
EVAL_LOG_PATH = None  # e.g., "experiments/advised_rl_qwen-qwen3-4b-instruct-2507_20251229_173400"

# ============================================================================
# TRAINING LOGIC
# ============================================================================


async def run_advised_training():
    """Run iterative advised training."""
    # Setup
    log_path = find_log_path(RUN_TRAIN, BASE_LOG_PATH, MODEL_NAME, EVAL_LOG_PATH)
    ml_logger = ml_log.setup_logging(
        log_path, WANDB_PROJECT, WANDB_NAME,
        config={
            "model_name": MODEL_NAME,
            "num_train_questions": NUM_TRAIN_QUESTIONS,
            "total_train_groups": TOTAL_TRAIN_GROUPS,
            "batch_size": BATCH_SIZE,
            "group_size": GROUP_SIZE,
            "learning_rate": LEARNING_RATE,
            "score_threshold_for_advice": SCORE_THRESHOLD_FOR_ADVICE,
        }
    )
    
    train_questions, test_questions = load_and_split_data(
        DATA_PATH, NUM_TRAIN_QUESTIONS, NUM_TEST_QUESTIONS
    )
    
    service_client, training_client, sampling_client, policy, train_dataset, renderer, advisor = (
        await setup_clients_and_dataset(
            MODEL_NAME, GRADER_MODEL_NAME, ADVISOR_MODEL_NAME, DATA_PATH,
            BATCH_SIZE, GROUP_SIZE, SCORE_THRESHOLD_FOR_ADVICE, LORA_RANK, RUN_TRAIN
        )
    )
    
    if RUN_TRAIN:
        question_states = {}
        
        # Print training plan
        print(f"\n{'🎯 '*40}")
        print(f"TRAINING PLAN")
        print(f"{'─'*80}")
        print(f"  Training Questions: {NUM_TRAIN_QUESTIONS}")
        print(f"  Test Questions: {NUM_TEST_QUESTIONS}")
        print(f"  Group Size: {GROUP_SIZE} rollouts/question")
        print(f"  Batch Size: {BATCH_SIZE} questions/batch")
        print(f"  Total Groups: {TOTAL_TRAIN_GROUPS}")
        print(f"  Total Rollouts: {TOTAL_ROLLOUTS}")
        print(f"  Learning Rate: {LEARNING_RATE}")
        print(f"{'─'*80}")
        print(f"  Phase 1: {NUM_TRAIN_QUESTIONS} groups (no advice)")
        print(f"  Phase 2: {TOTAL_TRAIN_GROUPS - NUM_TRAIN_QUESTIONS} groups ({NUM_EPOCHS_WITH_ADVICE} epochs with advice)")
        print(f"{'─'*80}")
        print(f"  Checkpoints: {EQUALLY_SPACED_SAVES_EXCLUDING_FINAL} intermediate + 1 final")
        print(f"  Evaluations: {EQUALLY_SPACED_EVALS_EXCLUDING_FINAL} intermediate + 1 final")
        print(f"{'🎯 '*40}\n")
        
        # Initial baseline evaluation before training
        print(f"\n{'='*80}")
        print(f"BASELINE EVALUATION (before training)")
        print(f"{'='*80}\n")
        initial_sampling_client = await training_client.save_weights_and_get_sampling_client_async()
        await run_quick_test_eval(
            test_questions=test_questions,
            sampling_client=initial_sampling_client,
            renderer=renderer,
            grader=train_dataset.grader,
            ml_logger=ml_logger,
            groups_trained=0,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        
        # Phase 1: Process all questions without advice
        groups_trained, policy, details = await run_phase1_training(
            NUM_TRAIN_QUESTIONS, TOTAL_TRAIN_GROUPS, BATCH_SIZE,
            train_dataset, train_questions, question_states, policy, renderer, advisor,
            SCORE_THRESHOLD_FOR_ADVICE, training_client, LEARNING_RATE,
            RECONTEXTUALIZE_BY_REMOVING_ADVICE, ml_logger, test_questions,
            train_dataset.grader, log_path, EQUALLY_SPACED_SAVES_EXCLUDING_FINAL, EQUALLY_SPACED_EVALS_EXCLUDING_FINAL,
            MAX_TOKENS, TEMPERATURE,
        )
        
        # Phase 2: Process all questions with advice for multiple epochs
        groups_trained, policy, details = await run_phase2_training(
            NUM_TRAIN_QUESTIONS, TOTAL_TRAIN_GROUPS, BATCH_SIZE, NUM_EPOCHS_WITH_ADVICE,
            train_dataset, train_questions, question_states, policy, renderer, advisor,
            SCORE_THRESHOLD_FOR_ADVICE, training_client, LEARNING_RATE,
            RECONTEXTUALIZE_BY_REMOVING_ADVICE, ml_logger, test_questions,
            train_dataset.grader, log_path, EQUALLY_SPACED_SAVES_EXCLUDING_FINAL, EQUALLY_SPACED_EVALS_EXCLUDING_FINAL,
            MAX_TOKENS, TEMPERATURE, groups_trained, details,
            SAVE_RECONTEXTUALIZATION_EXAMPLE,
        )
        
        # Finalize training
        details_path, transcript_path, final_sampling_client = await finalize_training(
            training_client, groups_trained, log_path, details, ml_logger, EQUALLY_SPACED_SAVES_EXCLUDING_FINAL
        )
    else:
        # Evaluation-only mode: load checkpoint
        final_sampling_client = await load_eval_checkpoint(log_path, service_client, MODEL_NAME)
        details_path = os.path.join(log_path, "training_details.json")
        transcript_path = os.path.join(log_path, "transcripts")
    
    # Run model comparison on test set
    if RUN_EVAL:
        await run_test_evaluation(
            test_questions=test_questions,
            base_sampling_client=service_client.create_sampling_client(base_model=MODEL_NAME),
            fine_tuned_sampling_client=final_sampling_client,
            renderer=renderer,
            grader=train_dataset.grader,
            log_path=log_path,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
    
    ml_logger.close()
    print(f"\n{'='*80}\nTraining complete!\nLog: {log_path}\nDetails: {details_path}\nTranscripts: {transcript_path}\n{'='*80}\n")


if __name__ == "__main__":
    asyncio.run(run_advised_training())
