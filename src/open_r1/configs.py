# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import trl


# TODO: add the shared options with a mixin to reduce code duplication
@dataclass
class GRPOConfig(trl.GRPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use."},
    )
    hub_model_revision: Optional[str] = field(
        default="main", metadata={"help": "The Hub model branch to push the model to."}
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    wandb_run_group: Optional[str] = field(
        default=None,
        metadata={"help": ("The group to store runs under.")},
    )


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    wandb_run_group: Optional[str] = field(
        default=None,
        metadata={"help": ("The group to store runs under.")},
    )


@dataclass
class GRPOScriptArguments(trl.ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', 'tag_count', 'code', 'ioi_code', 'code_format'.
        cosine_min_value_wrong (`float`):
            Minimum reward for cosine scaling for wrong answers.
        cosine_max_value_wrong (`float`):
            Maximum reward for cosine scaling for wrong answers.
        cosine_min_value_correct (`float`):
            Minimum reward for cosine scaling for correct answers.
        cosine_max_value_correct (`float`):
            Maximum reward for cosine scaling for correct answers.
        cosine_max_len (`int`):
            Maximum length for cosine scaling.
        code_language (`str`):
            Language for code format reward.
    """
    kg_path: str = field(
        default_factory=lambda: str(Path(__file__).with_name("knowledge_base_graph57.json")),
        metadata={"help": "Path to the HKG JSON file for curriculum-difficulty reward."},
    )
    difficulty_reward_temperature: float = field(
        default=10.0,
        metadata={"help": "Temperature tau used by sigmoid((d_target - d_model) / tau)."},
    )
    difficulty_evaluator_url: str = field(
        default="http://localhost:5000/generate",
        metadata={"help": "Schema-following fallback endpoint used only for ambiguity-routed candidates."},
    )
    difficulty_gate_scorer_url: str = field(
        default="http://localhost:5001/score",
        metadata={"help": "Local MiniVerifier-Gate endpoint that scores every reasoning-unit/node pair."},
    )
    difficulty_evaluator_temperature: float = field(
        default=0.0,
        metadata={"help": "Sampling temperature for deterministic evaluator calls."},
    )
    difficulty_evaluator_max_tokens: int = field(
        default=4096,
        metadata={"help": "Maximum tokens returned by the evaluator endpoint."},
    )
    difficulty_cav_prompt_path: str = field(
        default="reproducibility_package/prompts/hkg_fallback_prompt.txt",
        metadata={"help": "Prompt template for bounded RBA-CAV fallback extraction."},
    )
    difficulty_gate_thresholds_path: str = field(
        default="reproducibility_package/verifier/ucdg_thresholds_weights.csv",
        metadata={"help": "CSV manifest containing role/evidence/node/stage gate thresholds."},
    )
    eval_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Optional local/HF evaluation dataset. For local files, parquet/json/jsonl/csv are supported."},
    )
    eval_dataset_prompt_column: Optional[str] = field(
        default=None,
        metadata={"help": "Optional prompt column for the evaluation dataset; defaults to dataset_prompt_column."},
    )

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'difficulty', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', 'tag_count', 'code', 'code_format'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )
    code_language: str = field(
        default="python",
        metadata={
            "help": "Language for code format reward. Based on E2B supported languages https://e2b.dev/docs/code-interpreting/supported-languages",
            "choices": ["python", "javascript", "r", "java", "bash", "cpp"],
        },
    )
    code_eval_test_batch_size: int = field(
        default=1,
        metadata={
            "help": "for each generation, evaluate these many test cases in parallel, then check if any of them failed (0 score): if so stop evaluating; otherwise continue with the next batch of test cases. Useful to avoid overloading the eval server + save time on wrong solutions"
        },
    )
    parallel_code_exec_per_proc: int = field(
        default=2,
        metadata={
            "help": "Number of parallel E2B code executions per process. Default of 2 is suitable for the Free Hobby tier of E2B with 8 GPUs used for training."
        },
    )

    dataset_prompt_column: str = field(
        default="prompt",
        metadata={"help": "Column to use as prompts for training."},
    )

    e2b_router_url: Optional[str] = field(
        default=None,
        metadata={"help": "URL for the E2B route. See scripts/e2b_router.py"},
    )
