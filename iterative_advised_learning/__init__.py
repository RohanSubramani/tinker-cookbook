"""
Iterative Advised Learning

A system for RL training where models receive iterative feedback:
1. Model generates responses to prompts
2. LLM grader scores responses with reasoning
3. Advisor LLM provides structured advice for low-scoring responses
4. Advice is injected into prompts for subsequent iterations
"""

from iterative_advised_learning.types import (
    AdvisorOutput,
    Attempt,
    GradedResponse,
    TrainingExample,
)

# Import prompts and utils for easy access
from iterative_advised_learning import prompts, utils

__all__ = [
    "AdvisorOutput",
    "Attempt",
    "GradedResponse",
    "TrainingExample",
    "prompts",
    "utils",
]

