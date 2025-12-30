"""
Utility functions for iterative advised learning.

This module provides helper functions for:
- Extracting scores from grader responses
- Parsing advisor outputs
- Creating LLM completers for grader and advisor
- Logging and formatting utilities
"""

import json
import re
from typing import Any

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.completers import MessageCompleter, TinkerMessageCompleter
from tinker_cookbook.tokenizer_utils import get_tokenizer

from iterative_advised_learning.types import AdvisorOutput, GradedResponse


# ============================================================================
# SCORE EXTRACTION
# ============================================================================

def extract_score_from_grader_response(response_text: str) -> tuple[str, float]:
    """Extract reasoning and score from grader response.
    
    Expected format:
    <reasoning>[reasoning text]</reasoning>
    <score>[numerical score]</score>
    
    Args:
        response_text: The full text response from the grader
    
    Returns:
        Tuple of (reasoning, score) where reasoning is the text in <reasoning> tags
        and score is a float 0-100 from <score> tags
    """
    # Primary method: Extract from XML tags
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", response_text, re.IGNORECASE | re.DOTALL)
    score_match = re.search(r"<score>(\d+(?:\.\d+)?)</score>", response_text, re.IGNORECASE | re.DOTALL)
    
    if reasoning_match and score_match:
        reasoning = reasoning_match.group(1).strip()
        score = float(score_match.group(1))
        # Clamp score to valid range
        score = max(0.0, min(100.0, score))
        return reasoning, score
    
    # Fallback: Try to find score tag even if reasoning tag is missing
    if score_match:
        score = float(score_match.group(1))
        score = max(0.0, min(100.0, score))
        # Try to extract reasoning from before the score tag
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        else:
            # Use everything before <score> as reasoning
            score_start = score_match.start()
            reasoning = response_text[:score_start].strip()
            # Remove any other XML tags that might be there
            reasoning = re.sub(r"<[^>]+>", "", reasoning).strip()
        return reasoning, score
    
    # Fallback: Try old format with "SCORE:" marker
    score_marker_match = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)", response_text, re.IGNORECASE | re.MULTILINE)
    if score_marker_match:
        score = float(score_marker_match.group(1))
        score = max(0.0, min(100.0, score))
        reasoning_end = score_marker_match.start()
        reasoning = response_text[:reasoning_end].strip()
        reasoning = re.sub(r"^REASONING:\s*", "", reasoning, flags=re.IGNORECASE | re.MULTILINE).strip()
        return reasoning, score
    
    # Last resort: Try to find any number 0-100 near the end
    numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", response_text)
    if numbers:
        for num_str in reversed(numbers):
            num = float(num_str)
            if 0 <= num <= 100:
                reasoning = response_text.replace(num_str, "", 1).strip()
                return reasoning, num
    
    # Final fallback: return full text as reasoning, score 0
    print(f"Warning: Could not extract score from grader response. First 200 chars: {response_text[:200]}...")
    return response_text, 0.0


# ============================================================================
# ADVISOR OUTPUT PARSING
# ============================================================================

def parse_advisor_output(response_text: str) -> AdvisorOutput:
    """Parse structured output from advisor LLM.
    
    Expected format:
    <reasoning>[reasoning text]</reasoning>
    <advice>[advice text]</advice>
    
    Falls back to JSON and other formats if XML parsing fails.
    
    Args:
        response_text: The response from the advisor LLM
    
    Returns:
        AdvisorOutput with reasoning and advice
    """
    # Primary method: Extract from XML tags
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", response_text, re.IGNORECASE | re.DOTALL)
    advice_match = re.search(r"<advice>(.*?)</advice>", response_text, re.IGNORECASE | re.DOTALL)
    
    if reasoning_match and advice_match:
        return AdvisorOutput(
            reasoning=reasoning_match.group(1).strip(),
            advice=advice_match.group(1).strip(),
        )
    
    # Fallback: Try to parse as JSON
    try:
        # Try parsing the whole response as JSON
        data = json.loads(response_text.strip())
        # Handle case where advice might be an array (convert to string)
        advice = data.get("advice", "")
        if isinstance(advice, list):
            advice = " ".join(str(item) for item in advice)
        elif not isinstance(advice, str):
            advice = str(advice)
        
        return AdvisorOutput(
            reasoning=str(data.get("reasoning", "")),
            advice=advice,
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        # Try to find JSON object in the response
        start_idx = response_text.find("{")
        if start_idx != -1:
            brace_count = 0
            for i in range(start_idx, len(response_text)):
                if response_text[i] == "{":
                    brace_count += 1
                elif response_text[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = response_text[start_idx : i + 1]
                        try:
                            data = json.loads(json_str)
                            advice = data.get("advice", "")
                            if isinstance(advice, list):
                                advice = " ".join(str(item) for item in advice)
                            elif not isinstance(advice, str):
                                advice = str(advice)
                            
                            if "reasoning" in data and "advice" in data:
                                return AdvisorOutput(
                                    reasoning=str(data.get("reasoning", "")),
                                    advice=advice,
                                )
                        except (json.JSONDecodeError, KeyError, TypeError):
                            pass
                        break
    
    # Fallback: try to extract from structured text patterns
    reasoning_text_match = re.search(r'"reasoning":\s*"([^"]+)"', response_text)
    advice_text_match = re.search(r'"advice":\s*"([^"]+)"', response_text)
    
    if reasoning_text_match and advice_text_match:
        return AdvisorOutput(
            reasoning=reasoning_text_match.group(1),
            advice=advice_text_match.group(1),
        )
    
    # Last resort: try to split on common markers
    if "REASONING:" in response_text.upper() and "ADVICE:" in response_text.upper():
        parts = re.split(r"REASONING:|ADVICE:", response_text, flags=re.IGNORECASE)
        if len(parts) >= 3:
            return AdvisorOutput(
                reasoning=parts[1].strip(),
                advice=parts[2].strip(),
            )
    
    # Final fallback: use entire response as advice, empty reasoning
    print(f"Warning: Could not parse structured advisor output. First 200 chars: {response_text[:200]}...")
    return AdvisorOutput(
        reasoning="",
        advice=response_text.strip(),
    )
    return AdvisorOutput(
        reasoning="Unable to parse structured output",
        advice=response_text.strip(),
    )


# ============================================================================
# LLM COMPLETER CREATION
# ============================================================================

def create_grader_completer(
    model_name: str,
    max_tokens: int = 2048,
    base_url: str | None = None,
) -> MessageCompleter:
    """Create a MessageCompleter for the grader LLM.
    
    Args:
        model_name: Name of the model to use for grading
        max_tokens: Maximum tokens for grader responses
        base_url: Optional base URL for Tinker service
    
    Returns:
        MessageCompleter configured for grading
    """
    tokenizer = get_tokenizer(model_name)
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    
    service_client = tinker.ServiceClient(base_url=base_url)
    sampling_client = service_client.create_sampling_client(base_model=model_name)
    
    return TinkerMessageCompleter(
        sampling_client=sampling_client,
        renderer=renderer,
        max_tokens=max_tokens,
    )


def create_advisor_completer(
    model_name: str,
    max_tokens: int = 2048,
    base_url: str | None = None,
) -> MessageCompleter:
    """Create a MessageCompleter for the advisor LLM.
    
    Args:
        model_name: Name of the model to use for advising
        max_tokens: Maximum tokens for advisor responses
        base_url: Optional base URL for Tinker service
    
    Returns:
        MessageCompleter configured for advising
    """
    tokenizer = get_tokenizer(model_name)
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    
    service_client = tinker.ServiceClient(base_url=base_url)
    sampling_client = service_client.create_sampling_client(base_model=model_name)
    
    return TinkerMessageCompleter(
        sampling_client=sampling_client,
        renderer=renderer,
        max_tokens=max_tokens,
    )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_graded_response(
    prompt: str,
    response: str,
    grader_response_text: str,
) -> GradedResponse:
    """Create a GradedResponse from grader output.
    
    Args:
        prompt: The prompt that was used
        response: The model's response
        grader_response_text: The full text response from the grader
    
    Returns:
        GradedResponse with extracted reasoning and score
    """
    reasoning, score = extract_score_from_grader_response(grader_response_text)
    
    # Determine if advice is needed (score below threshold)
    from iterative_advised_learning.prompts import SCORE_THRESHOLD
    needs_advice = score < SCORE_THRESHOLD
    
    return GradedResponse(
        prompt=prompt,
        response=response,
        grader_reasoning=reasoning,
        score=score,
        needs_advice=needs_advice,
    )


def format_training_example_summary(example: Any) -> str:
    """Format a TrainingExample for logging/display.
    
    Args:
        example: A TrainingExample object
    
    Returns:
        Formatted string summary
    """
    if not hasattr(example, "attempts") or not example.attempts:
        return f"Example {getattr(example, 'example_id', 'unknown')}: No attempts"
    
    lines = [f"Example {getattr(example, 'example_id', 'unknown')}:"]
    lines.append(f"  Original prompt: {getattr(example, 'original_prompt', '')[:100]}...")
    lines.append(f"  Number of attempts: {len(example.attempts)}")
    
    for i, attempt in enumerate(example.attempts):
        lines.append(f"  Attempt {i}:")
        if hasattr(attempt, "response") and hasattr(attempt.response, "score"):
            lines.append(f"    Score: {attempt.response.score:.1f}/100")
            if hasattr(attempt.response, "grader_reasoning"):
                reasoning_preview = attempt.response.grader_reasoning[:100]
                lines.append(f"    Reasoning: {reasoning_preview}...")
        if hasattr(attempt, "advice") and attempt.advice:
            advice_preview = attempt.advice.advice[:100] if hasattr(attempt.advice, "advice") else str(attempt.advice)[:100]
            lines.append(f"    Advice given: {advice_preview}...")
    
    return "\n".join(lines)


# ============================================================================
# TRANSCRIPT UTILITIES
# ============================================================================

def write_transcript(
    transcript_path: str,
    question_idx: int,
    question: str,
    iteration: int,
    prompt: str,
    model_response: str,
    grader_reasoning: str,
    score: float,
    max_score_in_group: float,
    advice: str | None = None,
    advice_reasoning: str | None = None,
):
    """Write a single iteration to the transcript file."""
    import os
    transcript_file = os.path.join(transcript_path, f"question_{question_idx:03d}.txt")
    mode = "a" if os.path.exists(transcript_file) else "w"
    with open(transcript_file, mode, encoding="utf-8") as f:
        if mode == "w":
            f.write("=" * 120 + "\n")
            f.write(f"QUESTION {question_idx}\n")
            f.write("=" * 120 + "\n\n")
            f.write(f"Question: {question}\n\n")
        f.write("-" * 120 + "\n")
        f.write(f"ITERATION {iteration}\n")
        f.write("-" * 120 + "\n\n")
        f.write("PROMPT:\n")
        f.write(prompt + "\n\n")
        f.write("MODEL RESPONSE:\n")
        f.write(model_response + "\n\n")
        f.write("GRADER REASONING:\n")
        f.write(grader_reasoning + "\n\n")
        f.write(f"SCORE: {score:.1f}/100\n")
        f.write(f"MAX SCORE IN GROUP: {max_score_in_group:.1f}/100\n\n")
        if advice:
            f.write("ADVISOR REASONING:\n")
            f.write((advice_reasoning or "(From previous iteration)") + "\n\n")
            f.write("ADVICE:\n")
            f.write(advice + "\n\n")
        f.write("\n")


def write_all_transcripts(transcript_path: str, details: list[dict]):
    """Write all training details to organized transcript files."""
    import os
    os.makedirs(transcript_path, exist_ok=True)
    by_question = {}
    for detail in details:
        q_idx = detail["question_index"]
        if q_idx not in by_question:
            by_question[q_idx] = []
        by_question[q_idx].append(detail)
    for q_idx, question_details in sorted(by_question.items()):
        question_details.sort(key=lambda d: d.get("iteration", 0))
        for detail in question_details:
            write_transcript(
                transcript_path=transcript_path,
                question_idx=q_idx,
                question=detail["question"],
                iteration=detail.get("iteration", 0),
                prompt=detail["prompt"],
                model_response=detail["response"],
                grader_reasoning=detail.get("grader_reasoning", ""),
                score=detail["score"],
                max_score_in_group=detail.get("max_score_in_group", detail["score"]),
                advice=detail.get("advice"),
                advice_reasoning=detail.get("advice_reasoning"),
            )

