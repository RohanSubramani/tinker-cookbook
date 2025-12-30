"""Helper functions for iterative advised RL training."""

import asyncio
import json
import os
import random
from pathlib import Path
from typing import Any

import tinker
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup, Transition

from iterative_advised_learning.env import IterativeAdvisedDatasetBuilder, IterativeAdvisedGroupBuilder
from iterative_advised_learning.prompts import build_advisor_prompt
from iterative_advised_learning.utils import parse_advisor_output, write_all_transcripts


# ============================================================================
# Logging helpers
# ============================================================================

def log_batch_start(batch_question_indices: list[int], batch_types: list[str] | None = None, phase: str = ""):
    """Log the start of processing a batch."""
    if batch_types:
        types_str = ", ".join(batch_types)
        print(f"\n{'─'*80}")
        print(f"📦 BATCH START {phase}: Questions {batch_question_indices}")
        print(f"   Types: {types_str}")
        print(f"{'─'*80}")
    else:
        print(f"\n{'─'*80}")
        print(f"📦 BATCH START {phase}: Questions {batch_question_indices}")
        print(f"{'─'*80}")


def log_advice_start(question_idx: int):
    """Log when advice generation starts."""
    print(f"  💡 Q{question_idx}: Generating advice...")


def log_batch_complete(batch_question_indices: list[int], scores: list[float]):
    """Log the completion of a batch."""
    avg_score = sum(scores) / len(scores) if scores else 0.0
    print(f"{'─'*80}")
    print(f"✅ BATCH COMPLETE: Questions {batch_question_indices}")
    print(f"   Scores: {[f'{s:.1f}' for s in scores]} | Avg: {avg_score:.1f}/100")
    print(f"{'─'*80}\n")


async def do_group_rollout_with_logging(
    builder: IterativeAdvisedGroupBuilder,
    policy: TinkerTokenCompleter,
    question_idx: int,
) -> TrajectoryGroup:
    """Wrapper around do_group_rollout that adds detailed logging."""
    group_size = builder.group_size
    has_advice = builder.advice is not None
    
    # Log start of all rollouts in the group
    if has_advice:
        num_with_advice = group_size // 2
        advice_str = f" ({num_with_advice} with advice, {group_size - num_with_advice} without)"
    else:
        advice_str = ""
    print(f"  🎯 Q{question_idx}: Generating {group_size} responses in parallel{advice_str}...")
    
    # Execute the rollouts (do_group_rollout handles all rollouts in parallel internally)
    traj_group = await do_group_rollout(builder, policy)
    
    # Log completion of each rollout
    if traj_group.metrics_G:
        scores = []
        num_with_advice = group_size // 2 if has_advice else 0
        for rollout_idx, metrics in enumerate(traj_group.metrics_G):
            score = metrics.get("score", 0.0)
            scores.append(score)
            # Show which rollouts have advice
            advice_marker = " 💡" if has_advice and rollout_idx < num_with_advice else ""
            print(f"    ✅ Q{question_idx} Rollout {rollout_idx+1}/{group_size}{advice_marker}: Score = {score:.1f}/100")
        
        if scores:
            max_score = max(scores)
            avg_score = sum(scores) / len(scores)
            # If advice is used, show separate stats
            if has_advice and len(scores) >= 2:
                advised_scores = scores[:num_with_advice]
                unadvised_scores = scores[num_with_advice:]
                avg_advised = sum(advised_scores) / len(advised_scores) if advised_scores else 0.0
                avg_unadvised = sum(unadvised_scores) / len(unadvised_scores) if unadvised_scores else 0.0
                print(f"    📊 Q{question_idx} Group summary: Max={max_score:.1f}, Avg={avg_score:.1f} | 💡Advised: {avg_advised:.1f}, No advice: {avg_unadvised:.1f}")
            else:
                print(f"    📊 Q{question_idx} Group summary: Max={max_score:.1f}, Avg={avg_score:.1f}")
    
    return traj_group


# ============================================================================
# Batch processing functions
# ============================================================================

async def process_batch_without_advice(
    batch_question_indices: list[int],
    train_dataset: Any,
    train_questions: list[str],
    question_states: dict,
    policy: TinkerTokenCompleter,
    renderer: Any,
    advisor: Any,
    score_threshold: float,
    batch_size: int,
) -> tuple[list[Any], list[dict], list[Any], int]:
    """Process a batch of questions without advice. Returns (traj_groups, details, batch_builders, groups_trained)."""
    train_dataset.advice_map = {}
    batch_builders = [train_dataset.get_builder_for_question(i) for i in batch_question_indices]

    log_batch_start(batch_question_indices, batch_types=["without advice"] * len(batch_question_indices), phase="Phase 1")
    
    # Start all rollouts with logging
    print(f"  🚀 Starting rollouts for {len(batch_question_indices)} questions in parallel...")
    traj_groups = await asyncio.gather(*[
        do_group_rollout_with_logging(b, policy, q_idx) 
        for q_idx, b in zip(batch_question_indices, batch_builders)
    ])

    details = []
    advice_tasks = []  # Track advice generation tasks
    
    async def get_advice_for_question(q_idx: int, question: str, resp_text: str, reasoning: str, max_score: float):
        """Helper to get advice for a single question."""
        adv_output = parse_advisor_output(renderers.ensure_text(
            (await advisor(build_advisor_prompt(question, resp_text, reasoning, max_score)))["content"]
        ))
        return q_idx, adv_output
    
    # Process each question as data becomes available and start advice generation immediately
    for i, (q_idx, tg) in enumerate(zip(batch_question_indices, traj_groups)):
        if tg.trajectories_G and tg.trajectories_G[0].transitions:
            traj = tg.trajectories_G[0]
            resp_text = renderers.ensure_text(renderer.parse_response(traj.transitions[0].ac.tokens)[0]["content"])
            score = tg.metrics_G[0].get("score", 0.0)
            reasoning = tg.metrics_G[0].get("grader_reasoning", "")
            state = question_states[q_idx]
            
            # Extract all scores from the group (across all rollouts)
            all_scores = [tg.metrics_G[j].get("score", 0.0) for j in range(len(tg.metrics_G))]
            max_score = max(all_scores, default=0.0)
            state["score"] = max_score
            state["done_original"] = True

            detail = {
                "question_index": q_idx,
                "question": train_questions[q_idx],
                "iteration": 0,
                "prompt": train_questions[q_idx],
                "response": resp_text,
                "score": score,
                "scores_in_group": all_scores,
                "max_score_in_group": max_score,
                "grader_reasoning": reasoning,
                "advice": None,
            }
            details.append(detail)

            # Start advice generation immediately if needed (don't wait for other questions)
            if max_score < score_threshold and advisor:
                print(f"  ⚠️  Q{q_idx}: Max score {max_score:.1f} < {score_threshold}, requesting advice immediately...")
                task = asyncio.create_task(
                    get_advice_for_question(q_idx, train_questions[q_idx], resp_text, reasoning, max_score)
                )
                advice_tasks.append((q_idx, task, detail))
    
    # Wait for all advice generation tasks to complete
    if advice_tasks:
        print(f"  💡 Waiting for {len(advice_tasks)} advice generation tasks to complete...")
        for q_idx, task, detail in advice_tasks:
            result_q_idx, adv_output = await task
            assert q_idx == result_q_idx
            question_states[q_idx]["advice"] = adv_output.advice
            detail.update({"advice": adv_output.advice, "advice_reasoning": adv_output.reasoning})
            print(f"    ✅ Q{q_idx}: Advice received")

    # Log batch completion
    batch_scores = [max(tg.metrics_G[j].get("score", 0.0) for j in range(len(tg.metrics_G))) if tg.metrics_G else 0.0 for tg in traj_groups]
    log_batch_complete(batch_question_indices, batch_scores)

    return traj_groups, details, batch_builders, len(batch_question_indices)


async def process_batch_with_advice(
    batch_question_indices: list[int],
    batch_has_advice: list[bool],
    train_dataset: Any,
    train_questions: list[str],
    question_states: dict,
    questions_needing_advice: list[int],
    policy: TinkerTokenCompleter,
    renderer: Any,
    advisor: Any,
    score_threshold: float,
    batch_size: int,
) -> tuple[list[Any], list[dict], list[Any], int]:
    """Process a batch with mix of advised and unadvised questions. Returns (traj_groups, details, batch_builders, groups_trained)."""
    train_dataset.advice_map = {q_idx: question_states[q_idx]["advice"] for q_idx, has_adv in zip(batch_question_indices, batch_has_advice) if has_adv}
    batch_builders = [train_dataset.get_builder_for_question(i) for i in batch_question_indices]

    batch_types = ["with advice" if has_adv else "without advice" for has_adv in batch_has_advice]
    log_batch_start(batch_question_indices, batch_types, phase="Phase 2")
    
    # Start all rollouts with logging
    print(f"  🚀 Starting rollouts for {len(batch_question_indices)} questions in parallel...")
    traj_groups = await asyncio.gather(*[
        do_group_rollout_with_logging(b, policy, q_idx) 
        for q_idx, b in zip(batch_question_indices, batch_builders)
    ])

    details = []
    advice_tasks = []  # Track advice generation tasks
    extracted_data = []  # Store extracted data for building details later
    
    async def get_advice_for_question(q_idx: int, question: str, resp_text: str, reasoning: str, max_score: float):
        """Helper to get advice for a single question."""
        adv_output = parse_advisor_output(renderers.ensure_text(
            (await advisor(build_advisor_prompt(question, resp_text, reasoning, max_score)))["content"]
        ))
        return q_idx, adv_output
    
    # Process each question and start advice generation immediately when needed
    for i, (q_idx, tg, has_adv) in enumerate(zip(batch_question_indices, traj_groups, batch_has_advice)):
        if tg.trajectories_G and tg.trajectories_G[0].transitions:
            traj = tg.trajectories_G[0]
            resp_text = renderers.ensure_text(renderer.parse_response(traj.transitions[0].ac.tokens)[0]["content"])
            score = tg.metrics_G[0].get("score", 0.0)
            reasoning = tg.metrics_G[0].get("grader_reasoning", "")
            state = question_states[q_idx]

            # Extract all scores from the group (across all rollouts)
            all_scores = [tg.metrics_G[j].get("score", 0.0) for j in range(len(tg.metrics_G))]
            max_score = max(all_scores, default=0.0)
            
            if has_adv:
                state["done_enhanced"] = True
                if q_idx in questions_needing_advice:
                    questions_needing_advice.remove(q_idx)
                iteration = 1
            else:
                state["score"] = max_score
                state["done_original"] = True
                iteration = 0
                # Start advice generation immediately if needed (don't wait for other questions)
                if max_score < score_threshold and advisor and state["advice"] is None:
                    print(f"  ⚠️  Q{q_idx}: Max score {max_score:.1f} < {score_threshold}, requesting advice immediately...")
                    task = asyncio.create_task(
                        get_advice_for_question(q_idx, train_questions[q_idx], resp_text, reasoning, max_score)
                    )
                    advice_tasks.append((q_idx, task, state))
            
            # Store extracted data for building details after advice is generated
            extracted_data.append((i, q_idx, tg, has_adv, resp_text, score, all_scores, max_score, reasoning, state, iteration))
    
    # Wait for all advice generation tasks to complete
    if advice_tasks:
        print(f"  💡 Waiting for {len(advice_tasks)} advice generation tasks to complete...")
        for q_idx, task, state in advice_tasks:
            result_q_idx, adv_output = await task
            assert q_idx == result_q_idx
            state["advice"] = adv_output.advice
            if q_idx not in questions_needing_advice:
                questions_needing_advice.append(q_idx)
            print(f"    ✅ Q{q_idx}: Advice received")
    
    # Build details (now that advice is available)
    for i, q_idx, tg, has_adv, resp_text, score, all_scores, max_score, reasoning, state, iteration in extracted_data:
        # Get the actual prompt used (with advice if present)
        from iterative_advised_learning.prompts import build_enhanced_prompt
        actual_prompt = build_enhanced_prompt(batch_builders[i].question, state["advice"]) if (has_adv and state["advice"]) else batch_builders[i].question
        
        detail = {
            "question_index": q_idx,
            "question": train_questions[q_idx],
            "iteration": iteration,
            "prompt": actual_prompt,
            "response": resp_text,
            "score": score,
            "scores_in_group": all_scores,
            "max_score_in_group": max_score,
            "grader_reasoning": reasoning,
            "advice": state["advice"] if has_adv else None,
        }
        if has_adv and state["advice"]:
            detail["advice_reasoning"] = "From previous iteration"
        details.append(detail)

    # Log batch completion
    batch_scores = [max(tg.metrics_G[j].get("score", 0.0) for j in range(len(tg.metrics_G))) if tg.metrics_G else 0.0 for tg in traj_groups]
    log_batch_complete(batch_question_indices, batch_scores)

    return traj_groups, details, batch_builders, len(batch_question_indices)


def recontextualize_trajectories(
    traj_groups: list[TrajectoryGroup],
    group_builders: list[Any],
    recontextualize: bool,
) -> list[TrajectoryGroup]:
    """Remove advice from observations if recontextualize is True.
    
    This implements recontextualization: use advice during generation, but remove it
    before computing gradients so the model learns to generate good responses without advice.
    
    Args:
        traj_groups: List of trajectory groups
        group_builders: List of corresponding group builders
        recontextualize: If True, remove advice from observations
    
    Returns:
        Modified trajectory groups with advice removed from observations
    """
    if not recontextualize:
        return traj_groups
    
    modified_groups = []
    for traj_group, builder in zip(traj_groups, group_builders):
        # Only modify if advice was available (even if only some rollouts used it)
        if builder.advice is None:
            modified_groups.append(traj_group)
            continue
        
        # Rebuild observations from original question (without advice)
        original_ob = builder.renderer.build_generation_prompt([
            {"role": "user", "content": builder.original_question}
        ])
        
        # When advice is available, only the first half of rollouts use it
        # So we only recontextualize those (remove advice from their observations)
        num_with_advice = len(traj_group.trajectories_G) // 2
        
        # Create modified trajectories
        modified_trajectories = []
        for rollout_idx, traj in enumerate(traj_group.trajectories_G):
            if traj.transitions:
                # Only recontextualize if this rollout had advice (first half)
                if rollout_idx < num_with_advice:
                    # Replace the observation in the first transition with original (no advice)
                    first_transition = traj.transitions[0]
                    modified_transition = Transition(
                        ob=original_ob,
                        ac=first_transition.ac,
                        reward=first_transition.reward,
                        episode_done=first_transition.episode_done,
                        metrics=first_transition.metrics,
                    )
                    # Keep other transitions unchanged (if any)
                    modified_transitions = [modified_transition] + traj.transitions[1:]
                    modified_traj = Trajectory(
                        transitions=modified_transitions,
                        final_ob=traj.final_ob,
                    )
                    modified_trajectories.append(modified_traj)
                else:
                    # This rollout didn't have advice, keep it unchanged
                    modified_trajectories.append(traj)
            else:
                modified_trajectories.append(traj)
        
        modified_group = TrajectoryGroup(
            trajectories_G=modified_trajectories,
            final_rewards_G=traj_group.final_rewards_G,
            metrics_G=traj_group.metrics_G,
        )
        modified_groups.append(modified_group)
    
    return modified_groups


async def train_on_batch(
    traj_groups: list[Any],
    group_builders: list[Any],
    training_client: tinker.TrainingClient,
    learning_rate: float,
    recontextualize_by_removing_advice: bool = True,
    capture_example: bool = False,
) -> tuple[tinker.SamplingClient | None, dict | None]:
    """Train on a batch of trajectory groups. Returns (updated sampling_client, example_data)."""
    num_groups = len(traj_groups)
    recontext_str = " (recontextualized)" if recontextualize_by_removing_advice else ""
    print(f"  🎓 Training on {num_groups} groups{recontext_str}...")
    
    # Recontextualize: remove advice from observations before training
    modified_traj_groups = recontextualize_trajectories(
        traj_groups, group_builders, recontextualize_by_removing_advice
    )
    
    # Capture example data if requested (for the first trajectory with advice)
    example_data = None
    if capture_example:
        for i, (traj_group, builder, modified_group) in enumerate(zip(traj_groups, group_builders, modified_traj_groups)):
            if builder.advice is not None and traj_group.trajectories_G and traj_group.trajectories_G[0].transitions:
                # Get the prompts from the transitions
                generation_transition = traj_group.trajectories_G[0].transitions[0]
                training_transition = modified_group.trajectories_G[0].transitions[0]
                
                # Extract tokens from ModelInput chunks
                def extract_tokens_from_model_input(model_input: tinker.ModelInput) -> list[int]:
                    tokens = []
                    for chunk in model_input.chunks:
                        if isinstance(chunk, tinker.EncodedTextChunk):
                            tokens.extend(chunk.tokens)
                    return tokens
                
                generation_tokens = extract_tokens_from_model_input(generation_transition.ob)
                training_tokens = extract_tokens_from_model_input(training_transition.ob)
                
                # Decode using the tokenizer
                tokenizer = builder.renderer.tokenizer
                generation_prompt = tokenizer.decode(generation_tokens)
                training_prompt = tokenizer.decode(training_tokens)
                
                # Get response text
                response_tokens = generation_transition.ac.tokens
                response = tokenizer.decode(response_tokens)
                
                # Get scores from metrics
                scores = [traj_group.metrics_G[j].get("score", 0.0) for j in range(len(traj_group.metrics_G))]
                
                example_data = {
                    "question": builder.question,
                    "advice": builder.advice,
                    "generation_prompt": generation_prompt,
                    "response": response,
                    "training_prompt": training_prompt,
                    "scores_in_group": scores,
                    "recontextualized": recontextualize_by_removing_advice,
                }
                break
    
    if data_D := assemble_training_data(modified_traj_groups, compute_advantages(modified_traj_groups))[0]:
        print(f"  ⚙️  Running forward/backward pass...")
        await training_client.forward_backward_async(data_D, loss_fn="importance_sampling")
        print(f"  ⚙️  Running optimizer step (LR={learning_rate})...")
        await training_client.optim_step_async(tinker.AdamParams(learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8))
        print(f"  ✅ Training step complete, saving weights...")
        return await training_client.save_weights_and_get_sampling_client_async(), example_data
    print(f"  ⚠️  No training data generated, skipping training step")
    return None, example_data


def log_batch_metrics(ml_logger: Any, batch_details: list[dict], step: int):
    """Log metrics for a batch."""
    if batch_details:
        avg_score = sum(d["score"] for d in batch_details) / len(batch_details)
        ml_logger.log_metrics({"train/avg_score": avg_score, "train/groups": step}, step=step)


def write_recontextualization_example(
    log_path: str,
    question: str,
    advice: str,
    generation_prompt: str,
    response: str,
    training_prompt: str,
    scores_in_group: list[float],
    recontextualized: bool,
) -> None:
    """Save an example showing generation vs training prompts."""
    example_path = os.path.join(log_path, "recontextualization_example.txt")
    
    with open(example_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("RECONTEXTUALIZATION EXAMPLE\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("ORIGINAL QUESTION:\n")
        f.write("-" * 80 + "\n")
        f.write(question + "\n\n")
        
        f.write("ADVICE PROVIDED:\n")
        f.write("-" * 80 + "\n")
        f.write(advice + "\n\n")
        
        f.write("PROMPT USED FOR GENERATION (with advice):\n")
        f.write("-" * 80 + "\n")
        f.write(generation_prompt + "\n\n")
        
        f.write("MODEL RESPONSE:\n")
        f.write("-" * 80 + "\n")
        f.write(response + "\n\n")
        
        f.write(f"SCORES IN GROUP (n={len(scores_in_group)}):\n")
        f.write("-" * 80 + "\n")
        f.write(f"Scores: {scores_in_group}\n")
        f.write(f"Max: {max(scores_in_group):.2f}, Min: {min(scores_in_group):.2f}, Avg: {sum(scores_in_group)/len(scores_in_group):.2f}\n\n")
        
        f.write(f"PROMPT USED FOR COMPUTING GRADIENTS ({'without advice' if recontextualized else 'with advice'}):\n")
        f.write("-" * 80 + "\n")
        f.write(training_prompt + "\n\n")
        
        if recontextualized:
            f.write("NOTE: Recontextualization is ENABLED. The model generated with advice but\n")
            f.write("gradients are computed using the prompt without advice. This teaches the\n")
            f.write("model to produce high-quality responses even without advice.\n")
        else:
            f.write("NOTE: Recontextualization is DISABLED. Gradients are computed using the\n")
            f.write("same prompt (with advice) that was used for generation.\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"Saved recontextualization example to: {example_path}")


def log_final_metrics(ml_logger: Any, details: list[dict], groups_trained: int):
    """Log final summary metrics."""
    if details:
        final_scores = [d["score"] for d in details if d.get("iteration", 0) == 0]
        enhanced_scores = [d["score"] for d in details if d.get("iteration", 0) == 1]
        ml_logger.log_metrics({
            "final/original_avg_score": sum(final_scores) / len(final_scores) if final_scores else 0.0,
            "final/enhanced_avg_score": sum(enhanced_scores) / len(enhanced_scores) if enhanced_scores else 0.0,
            "final/total_groups": groups_trained,
        }, step=groups_trained)


async def run_quick_test_eval(
    test_questions: list[str],
    sampling_client: tinker.SamplingClient,
    renderer: Any,
    grader: Any,
    ml_logger: Any,
    groups_trained: int,
    max_tokens: int,
    temperature: float,
):
    """Run quick evaluation on test set and log average score."""
    if not test_questions:
        return
    
    print(f"  📊 Running quick test evaluation on {len(test_questions)} questions...")
    policy = TinkerTokenCompleter(sampling_client, max_tokens=max_tokens, temperature=temperature)
    
    # Create all builders
    builders = [
        IterativeAdvisedGroupBuilder(
            question=question,
            renderer=renderer,
            grader=grader,
            advisor=None,
            advice=None,
            score_threshold=100.0,
            group_size=1,
        )
        for question in test_questions
    ]
    
    # Run all rollouts in parallel
    print(f"  🚀 Generating responses for {len(test_questions)} test questions in parallel...")
    traj_groups = await asyncio.gather(*[do_group_rollout(builder, policy) for builder in builders])
    
    # Extract scores
    scores = []
    for i, traj_group in enumerate(traj_groups):
        if traj_group.trajectories_G and traj_group.trajectories_G[0].transitions:
            score = traj_group.metrics_G[0].get("score", 0.0)
            scores.append(score)
            print(f"  ✅ Test Q{i+1}: Score = {score:.1f}/100")
    
    if scores:
        avg_score = sum(scores) / len(scores)
        ml_logger.log_metrics({"test/avg_score": avg_score}, step=groups_trained)
        print(f"  📊 Test evaluation (group {groups_trained}): Average score = {avg_score:.2f}/100")


async def run_test_evaluation(
    test_questions: list[str],
    base_sampling_client: tinker.SamplingClient,
    fine_tuned_sampling_client: tinker.SamplingClient,
    renderer: Any,
    grader: Any,
    log_path: str,
    max_tokens: int,
    temperature: float,
):
    """Run evaluation on test set comparing base and fine-tuned models."""
    print(f"\n{'='*80}\nRunning test set evaluation\n{'='*80}\n")
    
    if not test_questions:
        print("No test questions available. Skipping evaluation.")
        return
    
    base_policy = TinkerTokenCompleter(base_sampling_client, max_tokens=max_tokens, temperature=temperature)
    fine_tuned_policy = TinkerTokenCompleter(fine_tuned_sampling_client, max_tokens=max_tokens, temperature=temperature)
    
    # Create all builders upfront
    builders_for_questions = []
    for idx, question in enumerate(test_questions):
        base_builder = IterativeAdvisedGroupBuilder(
            question=question,
            renderer=renderer,
            grader=grader,
            advisor=None,
            advice=None,
            score_threshold=100.0,
            group_size=1,
        )
        fine_tuned_builder = IterativeAdvisedGroupBuilder(
            question=question,
            renderer=renderer,
            grader=grader,
            advisor=None,
            advice=None,
            score_threshold=100.0,
            group_size=1,
        )
        builders_for_questions.append((idx, question, base_builder, fine_tuned_builder))
    
    # Run all rollouts in parallel
    print(f"  🚀 Generating responses for {len(test_questions)} test questions (base + fine-tuned) in parallel...")
    print(f"     Total rollouts: {len(test_questions) * 2} (2 models × {len(test_questions)} questions)")
    all_rollout_tasks = []
    for idx, question, base_builder, fine_tuned_builder in builders_for_questions:
        all_rollout_tasks.append(do_group_rollout(base_builder, base_policy))
        all_rollout_tasks.append(do_group_rollout(fine_tuned_builder, fine_tuned_policy))
    
    all_traj_groups = await asyncio.gather(*all_rollout_tasks)
    print(f"  ✅ All rollouts complete, processing results...")
    
    # Process results
    comparisons = []
    base_scores = []
    fine_tuned_scores = []
    
    for i, (idx, question, _, _) in enumerate(builders_for_questions):
        base_traj_group = all_traj_groups[i * 2]
        fine_tuned_traj_group = all_traj_groups[i * 2 + 1]
        
        # Extract base model response and score
        base_response = ""
        base_score = 0.0
        if base_traj_group.trajectories_G and base_traj_group.trajectories_G[0].transitions:
            base_traj = base_traj_group.trajectories_G[0]
            base_tokens = base_traj.transitions[0].ac.tokens
            parsed = renderer.parse_response(base_tokens)
            base_response = renderers.ensure_text(parsed[0]["content"]) if parsed else ""
            base_score = base_traj_group.metrics_G[0].get("score", 0.0)
            base_scores.append(base_score)
            print(f"  📊 Test Q{idx+1} (base): Score = {base_score:.1f}/100")
        
        # Extract fine-tuned model response and score
        fine_tuned_response = ""
        fine_tuned_score = 0.0
        if fine_tuned_traj_group.trajectories_G and fine_tuned_traj_group.trajectories_G[0].transitions:
            fine_tuned_traj = fine_tuned_traj_group.trajectories_G[0]
            fine_tuned_tokens = fine_tuned_traj.transitions[0].ac.tokens
            parsed = renderer.parse_response(fine_tuned_tokens)
            fine_tuned_response = renderers.ensure_text(parsed[0]["content"]) if parsed else ""
            fine_tuned_score = fine_tuned_traj_group.metrics_G[0].get("score", 0.0)
            fine_tuned_scores.append(fine_tuned_score)
            print(f"  📊 Test Q{idx+1} (fine-tuned): Score = {fine_tuned_score:.1f}/100")
        
        comparisons.append({
            "question_id": idx + 1,
            "question": question,
            "base_response": base_response,
            "base_score": base_score,
            "fine_tuned_response": fine_tuned_response,
            "fine_tuned_score": fine_tuned_score,
        })
    
    # Compute averages
    avg_base_score = sum(base_scores) / len(base_scores) if base_scores else 0.0
    avg_fine_tuned_score = sum(fine_tuned_scores) / len(fine_tuned_scores) if fine_tuned_scores else 0.0
    
    # Save comparison file
    comparison_path = os.path.join(log_path, "test_comparison.json")
    with open(comparison_path, "w") as f:
        json.dump({
            "summary": {
                "num_test_questions": len(test_questions),
                "avg_base_score": avg_base_score,
                "avg_fine_tuned_score": avg_fine_tuned_score,
                "improvement": avg_fine_tuned_score - avg_base_score,
            },
            "comparisons": comparisons,
        }, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n{'='*80}")
    print("Test Set Evaluation Results")
    print(f"{'='*80}")
    print(f"Number of test questions: {len(test_questions)}")
    print(f"Average Base Model Score: {avg_base_score:.2f}/100")
    print(f"Average Fine-Tuned Model Score: {avg_fine_tuned_score:.2f}/100")
    print(f"Improvement: {avg_fine_tuned_score - avg_base_score:+.2f} points")
    print(f"{'='*80}\n")
    print(f"Detailed comparison saved to: {comparison_path}\n")


# ============================================================================
# Setup and initialization helpers
# ============================================================================

def find_log_path(
    run_train: bool,
    base_log_path: Path,
    model_name: str,
    eval_log_path: str | None = None,
) -> str:
    """Find or create log path for training or evaluation."""
    if run_train:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = str(base_log_path / f"advised_rl_{model_name.replace('/', '-').lower()}_{timestamp}")
        from tinker_cookbook import cli_utils
        cli_utils.check_log_dir(log_path, behavior_if_exists="ask")
        os.makedirs(log_path, exist_ok=True)
        return log_path
    else:
        # Evaluation-only mode: find log path
        if eval_log_path:
            log_path = str(base_log_path / eval_log_path) if not os.path.isabs(eval_log_path) else eval_log_path
        else:
            # Find most recent experiment directory
            pattern = f"advised_rl_{model_name.replace('/', '-').lower()}_*"
            matching_dirs = sorted(base_log_path.glob(pattern), key=os.path.getmtime, reverse=True)
            if not matching_dirs:
                raise ValueError(f"No experiment directories found matching {pattern}")
            log_path = str(matching_dirs[0])
            print(f"Using most recent experiment: {log_path}")
        if not os.path.exists(log_path):
            raise ValueError(f"Log path does not exist: {log_path}")
        return log_path


def load_and_split_data(
    data_path: Path,
    num_train: int,
    num_test: int,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Load questions from JSONL and split into train/test sets."""
    with open(data_path, "r") as f:
        questions = [json.loads(line)["problem"] for line in f if line.strip()]
    random.seed(seed)
    random.shuffle(questions)
    train_questions = questions[:num_train]
    test_questions = questions[num_train:num_train + num_test]
    print(f"Train set: {len(train_questions)} questions, Test set: {len(test_questions)} questions")
    return train_questions, test_questions


async def setup_clients_and_dataset(
    model_name: str,
    grader_model_name: str,
    advisor_model_name: str,
    data_path: Path,
    batch_size: int,
    group_size: int,
    score_threshold: float,
    lora_rank: int,
    run_train: bool,
) -> tuple[Any, Any | None, Any | None, Any | None, Any, Any, Any]:
    """Set up service client, training client (if needed), dataset, renderer, and advisor.
    
    Returns: (service_client, training_client, sampling_client, policy, train_dataset, renderer, advisor)
    When run_train=False, training_client, sampling_client, and policy are None.
    """
    service_client = tinker.ServiceClient()
    
    dataset_builder = IterativeAdvisedDatasetBuilder(
        jsonl_path=str(data_path),
        batch_size=batch_size,
        group_size=group_size,
        renderer_name=model_info.get_recommended_renderer_name(model_name),
        model_name_for_tokenizer=model_name,
        grader_model_name=grader_model_name,
        advisor_model_name=advisor_model_name,
        score_threshold=score_threshold,
    )
    train_dataset, _ = await dataset_builder()
    renderer, advisor = train_dataset.renderer, train_dataset.advisor
    
    if run_train:
        training_client = await service_client.create_lora_training_client_async(
            base_model=model_name, rank=lora_rank
        )
        sampling_client = service_client.create_sampling_client(base_model=model_name)
        policy = TinkerTokenCompleter(sampling_client, max_tokens=2048, temperature=1.0)
        return service_client, training_client, sampling_client, policy, train_dataset, renderer, advisor
    else:
        return service_client, None, None, None, train_dataset, renderer, advisor


# ============================================================================
# Training loop helpers
# ============================================================================

def _calculate_equally_spaced_intervals(
    total_groups: int,
    num_intervals_excluding_final: int,
) -> list[int]:
    """Calculate equally spaced intervals for saves/evals.
    
    Returns list of group numbers at which to save/eval.
    Excludes the final checkpoint (at total_groups).
    """
    if num_intervals_excluding_final <= 0:
        return []
    
    intervals = []
    # Divide total into (num_intervals_excluding_final + 1) segments
    # Save at the end of each segment except the last one
    for i in range(1, num_intervals_excluding_final + 1):
        checkpoint_group = int(total_groups * i / (num_intervals_excluding_final + 1))
        if checkpoint_group > 0:
            intervals.append(checkpoint_group)
    
    return intervals


async def maybe_save_checkpoint(
    training_client: Any,
    groups_trained: int,
    log_path: str,
    total_train_groups: int,
    num_saves_excluding_final: int,
) -> None:
    """Save checkpoint if it's time for an equally spaced save."""
    if num_saves_excluding_final <= 0:
        return
    
    save_intervals = _calculate_equally_spaced_intervals(total_train_groups, num_saves_excluding_final)
    
    if groups_trained in save_intervals:
        await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name=f"{groups_trained:06d}",
            log_path=log_path,
            loop_state={"groups_trained": groups_trained},
            kind="both",
        )


async def maybe_run_quick_eval(
    test_questions: list[str],
    training_client: Any,
    renderer: Any,
    grader: Any,
    ml_logger: Any,
    groups_trained: int,
    max_tokens: int,
    temperature: float,
    total_train_groups: int,
    num_evals_excluding_final: int,
) -> None:
    """Run quick test set evaluation if it's time for an equally spaced eval."""
    if num_evals_excluding_final <= 0 or not test_questions:
        return
    
    eval_intervals = _calculate_equally_spaced_intervals(total_train_groups, num_evals_excluding_final)
    
    if groups_trained in eval_intervals:
        current_sampling_client = await training_client.save_weights_and_get_sampling_client_async()
        await run_quick_test_eval(
            test_questions=test_questions,
            sampling_client=current_sampling_client,
            renderer=renderer,
            grader=grader,
            ml_logger=ml_logger,
            groups_trained=groups_trained,
            max_tokens=max_tokens,
            temperature=temperature,
        )


async def run_phase1_training(
    num_train_questions: int,
    total_train_groups: int,
    batch_size: int,
    train_dataset: Any,
    train_questions: list[str],
    question_states: dict,
    policy: TinkerTokenCompleter,
    renderer: Any,
    advisor: Any,
    score_threshold: float,
    training_client: Any,
    learning_rate: float,
    recontextualize_by_removing_advice: bool,
    ml_logger: Any,
    test_questions: list[str],
    grader: Any,
    log_path: str,
    num_saves_excluding_final: int,
    num_evals_excluding_final: int,
    max_tokens: int,
    temperature: float,
) -> tuple[int, TinkerTokenCompleter, list[dict]]:
    """Run Phase 1: Process all questions without advice. Returns (groups_trained, updated_policy, details)."""
    details = []
    groups_trained = 0
    next_question_idx = 0
    
    print(f"\n{'='*80}\nPhase 1: Processing all questions without advice\n{'='*80}\n")
    while next_question_idx < num_train_questions and groups_trained < total_train_groups:
        batch_question_indices = [
            next_question_idx + i for i in range(batch_size)
            if next_question_idx + i < num_train_questions
        ]
        if not batch_question_indices:
            break
        for q_idx in batch_question_indices:
            if q_idx not in question_states:
                question_states[q_idx] = {
                    "advice": None, "score": 0.0, "done_original": False, "done_enhanced": False
                }
        next_question_idx += len(batch_question_indices)
        
        traj_groups, batch_details, batch_builders, batch_groups = await process_batch_without_advice(
            batch_question_indices, train_dataset, train_questions, question_states,
            policy, renderer, advisor, score_threshold, batch_size
        )
        for i, d in enumerate(batch_details):
            d["group"] = groups_trained + i
            details.append(d)
        
        sampling_client, _ = await train_on_batch(
            traj_groups, batch_builders, training_client, learning_rate,
            recontextualize_by_removing_advice, capture_example=False
        )
        if sampling_client:
            policy = TinkerTokenCompleter(sampling_client, max_tokens=max_tokens, temperature=temperature)
        
        log_batch_metrics(ml_logger, batch_details, groups_trained + batch_groups)
        groups_trained += batch_groups
        
        await maybe_save_checkpoint(
            training_client, groups_trained, log_path, total_train_groups, num_saves_excluding_final
        )
        await maybe_run_quick_eval(
            test_questions, training_client, renderer, grader, ml_logger,
            groups_trained, max_tokens, temperature, total_train_groups, num_evals_excluding_final
        )
    
    return groups_trained, policy, details


async def run_phase2_training(
    num_train_questions: int,
    total_train_groups: int,
    batch_size: int,
    num_epochs_with_advice: int,
    train_dataset: Any,
    train_questions: list[str],
    question_states: dict,
    policy: TinkerTokenCompleter,
    renderer: Any,
    advisor: Any,
    score_threshold: float,
    training_client: Any,
    learning_rate: float,
    recontextualize_by_removing_advice: bool,
    ml_logger: Any,
    test_questions: list[str],
    grader: Any,
    log_path: str,
    num_saves_excluding_final: int,
    num_evals_excluding_final: int,
    max_tokens: int,
    temperature: float,
    groups_trained: int,
    details: list[dict],
    save_recontextualization_example: bool = True,
) -> tuple[int, TinkerTokenCompleter, list[dict]]:
    """Run Phase 2: Process all questions with advice for multiple epochs."""
    example_saved = False
    
    # Ensure all questions have advice if they scored below threshold
    # (should already be set from phase 1, but this is a safety check)
    questions_with_advice = [
        q_idx for q_idx in range(num_train_questions)
        if q_idx in question_states and question_states[q_idx].get("advice") is not None
    ]
    
    print(f"\n{'='*80}\nPhase 2: Training with advice for {num_epochs_with_advice} epochs\n")
    print(f"Questions with advice: {len(questions_with_advice)}/{num_train_questions}\n{'='*80}\n")
    
    # Run multiple epochs with advice
    for epoch in range(num_epochs_with_advice):
        if groups_trained >= total_train_groups:
            break
            
        print(f"\n{'─'*80}\nEpoch {epoch + 1}/{num_epochs_with_advice} (with advice)\n{'─'*80}\n")
        
        # Process all questions in batches
        for batch_start in range(0, num_train_questions, batch_size):
            if groups_trained >= total_train_groups:
                break
                
            batch_end = min(batch_start + batch_size, num_train_questions)
            batch_question_indices = list(range(batch_start, batch_end))
            
            # All questions use advice in this phase (if they have it)
            batch_has_advice = [
                q_idx in questions_with_advice for q_idx in batch_question_indices
            ]
            
            # Skip this batch if no questions have advice
            if not any(batch_has_advice):
                continue
            
            traj_groups, batch_details, batch_builders, batch_groups = await process_batch_with_advice(
                batch_question_indices, batch_has_advice, train_dataset, train_questions,
                question_states, [], policy, renderer, advisor,  # Empty list for questions_needing_advice since we're not modifying it
                score_threshold, batch_size
            )
            for i, d in enumerate(batch_details):
                d["group"] = groups_trained + i
                d["epoch"] = epoch + 2  # Epoch 2+ (phase 1 was epoch 1)
                details.append(d)
            
            # Capture example on first batch with advice
            capture_example = save_recontextualization_example and not example_saved and any(batch_has_advice)
            
            sampling_client, example_data = await train_on_batch(
                traj_groups, batch_builders, training_client, learning_rate,
                recontextualize_by_removing_advice, capture_example=capture_example
            )
            if sampling_client:
                policy = TinkerTokenCompleter(sampling_client, max_tokens=max_tokens, temperature=temperature)
            
            # Save the example if captured
            if example_data and not example_saved:
                write_recontextualization_example(
                    log_path=log_path,
                    question=example_data["question"],
                    advice=example_data["advice"],
                    generation_prompt=example_data["generation_prompt"],
                    response=example_data["response"],
                    training_prompt=example_data["training_prompt"],
                    scores_in_group=example_data["scores_in_group"],
                    recontextualized=example_data["recontextualized"],
                )
                example_saved = True
            
            log_batch_metrics(ml_logger, batch_details, groups_trained + batch_groups)
            groups_trained += batch_groups
            
            await maybe_save_checkpoint(
                training_client, groups_trained, log_path, total_train_groups, num_saves_excluding_final
            )
            await maybe_run_quick_eval(
                test_questions, training_client, renderer, grader, ml_logger,
                groups_trained, max_tokens, temperature, total_train_groups, num_evals_excluding_final
            )
    
    return groups_trained, policy, details


# ============================================================================
# Finalization helpers
# ============================================================================

async def finalize_training(
    training_client: Any,
    groups_trained: int,
    log_path: str,
    details: list[dict],
    ml_logger: Any,
    save_every: int,
) -> tuple[str, str, Any]:
    """Save training details, transcripts, final checkpoint. Returns (details_path, transcript_path, final_sampling_client)."""
    details_path = os.path.join(log_path, "training_details.json")
    with open(details_path, "w") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)
    transcript_path = os.path.join(log_path, "transcripts")
    write_all_transcripts(transcript_path, details)
    log_final_metrics(ml_logger, details, groups_trained)
    
    # Always save final checkpoint (regardless of SAVE_EVERY setting)
    await checkpoint_utils.save_checkpoint_async(
        training_client=training_client,
        name=f"{groups_trained:06d}_final",
        log_path=log_path,
        loop_state={"groups_trained": groups_trained, "final": True},
        kind="both",
    )
    
    # Get final sampling client from training client (always up-to-date)
    final_sampling_client = await training_client.save_weights_and_get_sampling_client_async()
    return details_path, transcript_path, final_sampling_client


async def load_eval_checkpoint(
    log_path: str,
    service_client: Any,
    model_name: str,
) -> Any:
    """Load checkpoint for evaluation-only mode. Returns sampling client."""
    checkpoint_info = checkpoint_utils.get_last_checkpoint(log_path, required_key="sampler_path")
    if not checkpoint_info:
        checkpoints_file = os.path.join(log_path, "checkpoints.jsonl")
        if os.path.exists(checkpoints_file):
            raise ValueError(
                f"No checkpoint with sampler_path found in {log_path}.\n"
                f"Checkpoints file exists but doesn't contain a checkpoint with sampler_path.\n"
                f"This might indicate an incomplete training run or a checkpoint save failure.\n"
                f"To fix: run training again, or specify a different EVAL_LOG_PATH that contains valid checkpoints."
            )
        else:
            raise ValueError(
                f"No checkpoints found in {log_path}.\n"
                f"This likely means training was never completed (no final checkpoint was saved).\n"
                f"To fix:\n"
                f"  1. Run training to completion (a final checkpoint is always saved at the end), or\n"
                f"  2. Specify a different EVAL_LOG_PATH that contains checkpoints (checkpoints.jsonl file).\n"
                f"     You can find experiment directories with checkpoints in: {os.path.dirname(log_path)}"
            )
    model_path = checkpoint_info.get("sampler_path")
    if not model_path:
        raise ValueError(f"Could not find sampler_path in checkpoint: {checkpoint_info}")
    print(f"Loading checkpoint from: {model_path}")
    return service_client.create_sampling_client(
        model_path=model_path,
        base_model=model_name,
    )

