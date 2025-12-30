"""Environment for iterative advised learning."""

import json
from dataclasses import dataclass
from functools import partial
from typing import Sequence

import chz
import tinker
from tinker.types import ModelInput

from tinker_cookbook import renderers
from tinker_cookbook.completers import MessageCompleter, StopCondition
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import (
    Action,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)

from iterative_advised_learning.prompts import build_enhanced_prompt
from iterative_advised_learning.utils import (
    create_graded_response,
    parse_advisor_output,
)


class IterativeAdvisedEnv(ProblemEnv):
    """Environment that handles iterative advice for moral philosophy questions."""

    def __init__(
        self,
        renderer: renderers.Renderer,
        question: str,
        grader: MessageCompleter,
        advisor: MessageCompleter | None,
        advice: str | None = None,
        score_threshold: float = 100.0,
    ):
        super().__init__(renderer, convo_prefix=None)
        self.question = question
        self.grader = grader
        self.advisor = advisor
        self.advice = advice
        self.score_threshold = score_threshold

    def get_question(self) -> str:
        if self.advice:
            return build_enhanced_prompt(self.question, self.advice)
        return self.question

    def check_format(self, sample_str: str) -> bool:
        # No strict format requirement for moral philosophy
        return True

    def check_answer(self, sample_str: str) -> bool:
        # Grading is done by LLM, not boolean check
        return True

    def get_reference_answer(self) -> str:
        return "N/A - Graded by LLM"

    async def step(self, action: Action) -> StepResult:
        # Parse the response
        message, parse_success = self.renderer.parse_response(action)
        response_text = renderers.ensure_text(message["content"])

        # Grade the response
        from iterative_advised_learning.prompts import build_grader_prompt

        grader_messages = build_grader_prompt(self.question, response_text)
        grader_response = await self.grader(grader_messages)
        grader_text = renderers.ensure_text(grader_response["content"])

        graded_response = create_graded_response(
            self.get_question(), response_text, grader_text
        )

        # Reward is the score (0-100 scale, but we might want to normalize)
        reward = graded_response.score / 100.0

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "score": graded_response.score,
                "needs_advice": graded_response.needs_advice,
                "parse_success": float(parse_success),
                "grader_reasoning": graded_response.grader_reasoning,
            },
        )


@dataclass(frozen=True)
class IterativeAdvisedGroupBuilder(EnvGroupBuilder):
    question: str
    renderer: renderers.Renderer
    grader: MessageCompleter
    advisor: MessageCompleter | None
    advice: str | None
    score_threshold: float
    group_size: int
    
    @property
    def original_question(self) -> str:
        """The original question without advice. Same as question when no advice is present."""
        return self.question

    async def make_envs(self) -> Sequence[IterativeAdvisedEnv]:
        # If advice is available, only give it to half the environments in the group
        # This allows comparison between advised and unadvised responses
        envs = []
        for i in range(self.group_size):
            # First half get advice (if available), second half don't
            use_advice = self.advice is not None and i < (self.group_size // 2)
            envs.append(
                IterativeAdvisedEnv(
                    renderer=self.renderer,
                    question=self.question,
                    grader=self.grader,
                    advisor=self.advisor,
                    advice=self.advice if use_advice else None,
                    score_threshold=self.score_threshold,
                )
            )
        return envs

    async def compute_group_rewards(
        self, trajectory_group: list, env_group: Sequence
    ) -> list[tuple[float, dict]]:
        """Extract metrics from the final transition of each trajectory."""
        results = []
        for traj in trajectory_group:
            if traj.transitions:
                # Get metrics from the final transition (which has the StepResult from env.step)
                final_metrics = traj.transitions[-1].metrics
                # Final reward is 0 (we use per-step rewards)
                results.append((0.0, final_metrics))
            else:
                results.append((0.0, {}))
        return results


class IterativeAdvisedDataset(RLDataset):
    """Dataset for iterative advised learning."""

    def __init__(
        self,
        questions: list[str],
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        grader: MessageCompleter,
        advisor: MessageCompleter | None,
        advice_map: dict[int, str],  # Maps question index to advice
        score_threshold: float,
    ):
        self.questions = questions
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.grader = grader
        self.advisor = advisor
        self.advice_map = advice_map
        self.score_threshold = score_threshold

    def get_batch(self, index: int) -> Sequence[IterativeAdvisedGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.questions))
        return [
            IterativeAdvisedGroupBuilder(
                question=self.questions[i],
                renderer=self.renderer,
                grader=self.grader,
                advisor=self.advisor,
                advice=self.advice_map.get(i),
                score_threshold=self.score_threshold,
                group_size=self.group_size,
            )
            for i in range(batch_start, batch_end)
        ]
    
    def get_builder_for_question(self, question_idx: int) -> IterativeAdvisedGroupBuilder:
        """Create a builder for a specific question index."""
        return IterativeAdvisedGroupBuilder(
            question=self.questions[question_idx],
            renderer=self.renderer,
            grader=self.grader,
            advisor=self.advisor,
            advice=self.advice_map.get(question_idx),
            score_threshold=self.score_threshold,
            group_size=self.group_size,
        )

    def __len__(self) -> int:
        return (len(self.questions) + self.batch_size - 1) // self.batch_size


@chz.chz
class IterativeAdvisedDatasetBuilder(RLDatasetBuilder):
    """Builder for iterative advised learning dataset."""

    jsonl_path: str
    batch_size: int
    group_size: int
    renderer_name: str
    model_name_for_tokenizer: str
    grader_model_name: str
    advisor_model_name: str | None = None
    score_threshold: float = 100.0
    base_url: str | None = None

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        # Load questions from JSONL
        questions = []
        with open(self.jsonl_path, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    questions.append(data["problem"])

        # Create renderer
        from tinker_cookbook.tokenizer_utils import get_tokenizer
        from tinker_cookbook.renderers import get_renderer

        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = get_renderer(self.renderer_name, tokenizer)

        # Create grader and advisor
        from iterative_advised_learning.utils import (
            create_advisor_completer,
            create_grader_completer,
        )

        grader = create_grader_completer(self.grader_model_name, base_url=self.base_url)
        advisor = (
            create_advisor_completer(self.advisor_model_name, base_url=self.base_url)
            if self.advisor_model_name
            else None
        )

        # No advice for initial dataset (advice_map is empty)
        train_dataset = IterativeAdvisedDataset(
            questions=questions,
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            grader=grader,
            advisor=advisor,
            advice_map={},
            score_threshold=self.score_threshold,
        )

        return train_dataset, None

