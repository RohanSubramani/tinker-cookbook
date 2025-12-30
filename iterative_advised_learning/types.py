"""
Data structures for iterative advised learning.

This module defines the core types used in the iterative advised learning system,
where a model receives feedback from a grader and advice from an advisor LLM to
improve its responses over multiple iterations.
"""

from dataclasses import dataclass


@dataclass
class GradedResponse:
    """A model response that has been graded by an LLM grader.
    
    The grader provides reasoning first, then a score. This structure ensures
    reasoning always precedes the score in any context where both exist.
    """
    
    prompt: str
    """The prompt that was used (may include advice from previous iterations)."""
    
    response: str
    """The model's response to the prompt."""
    
    grader_reasoning: str
    """The grader's explanation for the score (comes before score)."""
    
    score: float
    """The grade assigned by the grader (0-100 scale)."""
    
    needs_advice: bool
    """True if score is below the threshold and advice should be requested."""


@dataclass
class AdvisorOutput:
    """Structured advice from the advisor LLM.
    
    The advisor provides reasoning first, then the actual advice text. Each
    iteration gets the full advice string (not cumulative from previous iterations).
    """
    
    reasoning: str
    """Why this advice was given (comes before advice)."""
    
    advice: str
    """The actual advice text to inject into the prompt for the next iteration."""
    
    # Could add additional fields like:
    # focus_areas: list[str]
    # examples: list[str]
    # confidence: float


@dataclass
class Attempt:
    """One iteration of the prompt → response → grade → (maybe advice) cycle.
    
    Each attempt represents a single round where:
    1. A prompt (possibly with advice) is given to the model
    2. The model generates a response
    3. The response is graded
    4. If the score is below threshold, advice is generated for the next iteration
    """
    
    iteration: int
    """Which attempt this is (0-indexed)."""
    
    prompt: str
    """The prompt used (with advice if iteration > 0)."""
    
    response: GradedResponse
    """The graded response from this attempt."""
    
    advice: AdvisorOutput | None
    """Advice for the next iteration, or None if score was good enough or this is the last attempt."""


@dataclass
class TrainingExample:
    """Tracks the full lifecycle of a training example across multiple iterations.
    
    This structure supports variable numbers of iterations - there's no hardcoded
    limit on how many attempts can be made. Each attempt contains the full advice
    string used in that iteration (not cumulative).
    """
    
    example_id: str
    """Unique identifier for this training example."""
    
    original_prompt: str
    """The base prompt without any advice."""
    
    attempts: list[Attempt]
    """Variable number of attempts, each with its own prompt, response, grade, and advice."""
    
    # Could add computed properties for easy querying:
    # @property
    # def final_score(self) -> float:
    #     """The score from the last attempt."""
    #     return self.attempts[-1].response.score if self.attempts else 0.0
    #
    # @property
    # def num_iterations(self) -> int:
    #     """Number of iterations attempted."""
    #     return len(self.attempts)
    #
    # @property
    # def improvement(self) -> float:
    #     """Score improvement from first to last attempt."""
    #     if len(self.attempts) < 2:
    #         return 0.0
    #     return self.attempts[-1].response.score - self.attempts[0].response.score

