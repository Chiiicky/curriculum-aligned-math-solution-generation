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

"""Reward functions for GRPO training."""

import asyncio
from collections import Counter, OrderedDict
import json
import math
import re
import unicodedata
from difflib import SequenceMatcher
from functools import partial, update_wrapper
from pathlib import Path
from typing import Callable, Dict, Optional

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from .utils import is_e2b_available
from .utils.ioi import SubtaskResult, add_includes, get_piston_client_from_env, score_subtask


if is_e2b_available():
    from dotenv import load_dotenv
    from e2b_code_interpreter import AsyncSandbox

    from .utils.routed_sandbox import RoutedSandbox

    load_dotenv()
else:
    AsyncSandbox = None


def accuracy_reward(
    completions: list[list[dict[str, str]]],
    solution: Optional[list[str]] = None,
    answer: Optional[list[str]] = None,
    **kwargs,
) -> list[Optional[float]]:
    """Reward function that checks if the completion is the same as the ground truth."""
    solution = solution if solution is not None else answer
    if solution is None:
        raise ValueError("accuracy_reward requires a `solution` or `answer` field from the dataset.")

    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Compute binary rewards if verifiable, `None` otherwise to skip this example
            try:
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = None
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            reward = None
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """Reward valid <think>/<answer> structure with a boxed final answer for reliable extraction."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\\boxed\{.+?\}.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def tag_count_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.25
        if text.count("\n</think>\n") == 1:
            count += 0.25
        if text.count("\n<answer>\n") == 1:
            count += 0.25
        if text.count("\n</answer>") == 1:
            count += 0.25
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic number 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def len_reward(completions: list[Dict[str, str]], solution: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward


def _init_event_loop():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def ioi_code_reward(completions, test_batch_size: int = 1, **kwargs) -> list[float]:
    """Reward function that evaluates IOI problems using Piston+our IOI package.

    Assumes the dataset has the same format as hf.co/datasets/open-r1/ioi

    test_batch_size: evaluate these many test cases in parallel, then check if any of them failed (0 score): if so stop evaluating; otherwise continue with the next batch of test cases.
    """
    # for info on setting up piston workers, see slurm/piston/README.md
    piston_client = get_piston_client_from_env()

    code_snippets = [
        # note: grading is automatically skipped if no code is extracted
        add_includes(extract_code(completion[-1]["content"], "cpp"), problem_id)
        for completion, problem_id in zip(completions, kwargs["id"])
    ]

    async def run_catch_exceptions(task):
        try:
            return await task
        except Exception as e:
            print(f"Error from Piston worker: {e}")
            return SubtaskResult()  # score 0.0

    # load problem data. undo separating kwargs by column
    problems_data = [dict(zip(kwargs.keys(), values)) for values in zip(*kwargs.values())]

    loop = _init_event_loop()
    evals = [
        loop.create_task(
            run_catch_exceptions(score_subtask(piston_client, problem_data, code, test_batch_size=test_batch_size))
        )
        for problem_data, code in zip(problems_data, code_snippets)
    ]
    results = loop.run_until_complete(asyncio.gather(*evals))

    return [result.score for result in results]


def extract_code(completion: str, language: str = "python") -> str:
    pattern = re.compile(rf"```{language}\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer


def binary_code_reward(completions, num_parallel: int = 2, e2b_router_url=None, **kwargs) -> list[float]:
    rewards = code_reward(completions, num_parallel=num_parallel, e2b_router_url=e2b_router_url, **kwargs)
    BINARY_THRESHOLD = 0.99

    output = []
    for reward in rewards:
        if reward is None:
            output.append(None)
        else:
            output.append(1.0 if reward > BINARY_THRESHOLD else 0.0)

    return output


def code_reward(completions, num_parallel: int = 2, e2b_router_url=None, **kwargs) -> list[float]:
    """Reward function that evaluates code snippets using the E2B code interpreter.

    Assumes the dataset contains a `verification_info` column with test cases.
    """
    if not is_e2b_available():
        raise ImportError(
            "E2B is not available and required for this reward function. Please install E2B with "
            "`pip install e2b-code-interpreter` and add an API key to a `.env` file."
        )

    # TODO: add support for other languages in E2B: https://e2b.dev/docs/code-interpreting/supported-languages
    """Returns a reward function that evaluates code snippets in a sandbox."""
    evaluation_script_template = """
    import subprocess
    import json

    def evaluate_code(code, test_cases):
        passed = 0
        total = len(test_cases)
        exec_timeout = 5

        for case in test_cases:
            process = subprocess.run(
                ["python3", "-c", code],
                input=case["input"],
                text=True,
                capture_output=True,
                timeout=exec_timeout
            )

            if process.returncode != 0:  # Error in execution
                continue

            output = process.stdout.strip()

            # TODO: implement a proper validator to compare against ground truth. For now we just check for exact string match on each line of stdout.
            all_correct = True
            for line1, line2 in zip(output.split('\\n'), case['output'].split('\\n')):
                all_correct = all_correct and line1.strip() == line2.strip()

            if all_correct:
                passed += 1

        success_rate = (passed / total)
        return success_rate

    code_snippet = {code}
    test_cases = json.loads({test_cases})

    evaluate_code(code_snippet, test_cases)
    """
    code_snippets = [extract_code(completion[-1]["content"]) for completion in completions]
    verification_info = kwargs["verification_info"]
    scripts = [
        evaluation_script_template.format(code=json.dumps(code), test_cases=json.dumps(json.dumps(info["test_cases"])))
        for code, info in zip(code_snippets, verification_info)
    ]

    language = verification_info[0]["language"]
    if not all(v["language"] == language for v in verification_info):
        raise ValueError("All verification_info must have the same language", verification_info)

    if e2b_router_url is not None:
        routed_sandbox = RoutedSandbox(router_url=e2b_router_url)

        executions = routed_sandbox.run_code(
            scripts=scripts,
            language=language,
            timeout=30,
            request_timeout=28,
        )

        rewards = []
        for execution in executions:
            try:
                reward = float(execution.text)
                rewards.append(reward)
            except Exception:
                rewards.append(None)
        return rewards

    try:
        rewards = run_async_from_sync(scripts, language, num_parallel)
    except Exception as e:
        print(f"Error from E2B executor: {e}")
        rewards = [0.0] * len(completions)

    return rewards


def get_code_format_reward(language: str = "python"):
    """Format reward function specifically for code responses.

    Args:
        language: Programming language supported by E2B https://e2b.dev/docs/code-interpreting/supported-languages
    """
    pattern = rf"^<think>\n.*?\n</think>\n<answer>\n.*?```{language}.*?```.*?\n</answer>$"

    def code_format_reward(completions, **kwargs):
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]

    return code_format_reward


def run_async_from_sync(scripts: list[str], language: str, num_parallel: int) -> list[float]:
    """Function wrapping the `run_async` function."""
    # Create a new event loop and set it
    try:
        # Run the async function and get the result
        rewards = asyncio.run(run_async(scripts, language, num_parallel))
    except Exception as e:
        print(f"Error from E2B executor async: {e}")
        raise e

    return rewards


async def run_async(scripts: list[str], language: str, num_parallel: int) -> list[float]:
    # Limit the number of concurrent tasks
    semaphore = asyncio.Semaphore(num_parallel)

    # Create a list of tasks for running scripts concurrently
    tasks = [run_script(script, language, semaphore) for script in scripts]

    # Wait for all tasks to complete and gather their results as they finish
    results = await asyncio.gather(*tasks)
    rewards = list(results)  # collect results

    return rewards


async def run_script(script: str, language: str, semaphore: asyncio.Semaphore) -> float:
    # We set a timeout margin, as the AsyncSandbox timeout does not seem to work
    # These values are based on running 256 examples with the gold solution
    # from open-r1/verifiable-coding-problems-python_decontaminated
    # see scripts/benchmark_e2b.py

    SANDBOX_TIMEOUT = 30
    MARGIN = 2
    REQUEST_TIMEOUT = SANDBOX_TIMEOUT - MARGIN
    ASYNCIO_TIMEOUT = SANDBOX_TIMEOUT + MARGIN

    async with semaphore:
        try:
            sandbox = await AsyncSandbox.create(timeout=SANDBOX_TIMEOUT, request_timeout=REQUEST_TIMEOUT)
            execution = await asyncio.wait_for(sandbox.run_code(script, language=language), timeout=ASYNCIO_TIMEOUT)
            return float(execution.text)
        except (TypeError, ValueError):
            return 0.0
        except asyncio.TimeoutError:
            print("Operation timed out")
            return 0.0
        except Exception as e:
            print(f"Error in `run_script` from E2B sandbox ID {sandbox.sandbox_id} : {e}")
            return 0.0
        finally:
            try:
                await sandbox.kill()
            except Exception as e:
                print(f"Error from E2B executor kill with sandbox ID {sandbox.sandbox_id} : {e}")


import requests
import os
import csv
import json
import networkx as nx
import logging
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log
)
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import hashlib
# 设置日志格式
logging.basicConfig(level=logging.WARNING)

DEFAULT_KG_PATH = Path(__file__).with_name("knowledge_base_graph57.json")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAV_PROMPT_PATH = REPO_ROOT / "reproducibility_package" / "prompts" / "hkg_fallback_prompt.txt"
DEFAULT_GATE_THRESHOLDS_PATH = REPO_ROOT / "reproducibility_package" / "verifier" / "ucdg_thresholds_weights.csv"
DEFAULT_HKG_NODES_PATH = REPO_ROOT / "hkg" / "hkg_nodes.csv"
DEFAULT_HKG_SYNONYMS_PATH = REPO_ROOT / "hkg" / "hkg_synonyms.csv"
DEFAULT_DIFFICULTY_TEMPERATURE = 10.0
MIN_GRADE_VALUE = 1.0
MAX_GRADE_VALUE = 12.5
NUM_CURRICULUM_STAGES = 24
DEFAULT_GATE_THRESHOLDS = {
    "role": 0.62,
    "node": 0.60,
    "context": 0.60,
    "stage": 0.58,
    "evidence": 0.64,
    "accept": 0.80,
    "reject": 0.45,
}


def _chinese_number_to_int(text: str) -> int:
    digit_map = {
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    if text in digit_map:
        return digit_map[text]
    if text.startswith("十"):
        return 10 + digit_map.get(text[1:], 0)
    if text.endswith("十"):
        return digit_map.get(text[:-1], 0) * 10
    if "十" in text:
        left, right = text.split("十", 1)
        return digit_map.get(left, 0) * 10 + digit_map.get(right, 0)
    return 0


def grade_value_to_difficulty_score(grade_value: float) -> float:
    """Map a semester-level curriculum stage to a normalized 0-100 difficulty score."""
    if grade_value <= 0:
        return 0.0
    grade_value = max(MIN_GRADE_VALUE, min(MAX_GRADE_VALUE, float(grade_value)))
    whole_grade = int(math.floor(grade_value))
    is_lower_semester = grade_value - whole_grade >= 0.5
    stage_index = (whole_grade - 1) * 2 + (2 if is_lower_semester else 1)
    return ((stage_index - 1) / (NUM_CURRICULUM_STAGES - 1)) * 100.0


def difficulty_gap_to_reward(difficulty_gap: float, temperature: float = DEFAULT_DIFFICULTY_TEMPERATURE) -> float:
    """Paper reward: R_difficulty = sigmoid((d_target - d_model) / tau)."""
    if temperature <= 0:
        raise ValueError("difficulty reward temperature must be positive")
    return 1.0 / (1.0 + math.exp(-float(difficulty_gap) / temperature))


class MathDifficulty:
    def __init__(
        self,
        kg_path: str | os.PathLike | None = None,
        evaluator_url: str = "http://localhost:5000/generate",
        gate_scorer_url: str = "http://localhost:5001/score",
        evaluator_temperature: float = 0.0,
        evaluator_max_tokens: int = 4096,
        cav_prompt_path: str | os.PathLike | None = None,
        gate_thresholds_path: str | os.PathLike | None = None,
        hkg_nodes_path: str | os.PathLike | None = None,
        hkg_synonyms_path: str | os.PathLike | None = None,
    ):
        # 初始化日志
        self.LOGGER = logging.getLogger(__name__)

        self.kg_path = str(kg_path or DEFAULT_KG_PATH)
        self.evaluator_url = evaluator_url
        self.gate_scorer_url = gate_scorer_url
        self.evaluator_temperature = evaluator_temperature
        self.evaluator_max_tokens = evaluator_max_tokens
        self.cav_prompt_path = Path(cav_prompt_path or DEFAULT_CAV_PROMPT_PATH)
        self.gate_thresholds_path = Path(gate_thresholds_path or DEFAULT_GATE_THRESHOLDS_PATH)
        self.hkg_nodes_path = Path(hkg_nodes_path or DEFAULT_HKG_NODES_PATH)
        self.hkg_synonyms_path = Path(hkg_synonyms_path or DEFAULT_HKG_SYNONYMS_PATH)

        # 缓存初始化
        self.cache = {}
        self.kp_cache = OrderedDict()  # 知识点缓存
        self.MAX_CACHE_SIZE = 2000       # 保持最近500条知识点

        # 常量配置
        self.MAX_RETRIES = 3
        self.RETRY_DELAY = 1
        self.MAX_WORKERS = 50  # 并发线程数，可以根据您的服务器性能进行调整

        # 知识图谱初始化
        self.KNOWLEDGE_GRAPH = None
        self.kg = self.init_knowledge_graph(self.kg_path)
        self.hkg_node_records = self.build_hkg_node_records()
        self.hkg_node_records_by_id = {record["node_id"]: record for record in self.hkg_node_records}
        self.hkg_node_records_by_name = {
            self._normalize_knowledge_text(record["knowledge_point"]): record for record in self.hkg_node_records
        }
        self.gate_thresholds = self.load_gate_thresholds(self.gate_thresholds_path)
        self.router_weights = self.load_router_weights(self.gate_thresholds_path)
        self.retrieval_vocabulary = self.build_retrieval_vocabulary(self.hkg_node_records)
        self.cav_prompt_template = self.load_cav_prompt_template(self.cav_prompt_path)

    MAX_RETRIES = 3  # 将 MAX_RETRIES 定义为类属性
    LOGGER = logging.getLogger(__name__)  # 将 LOGGER 定义为类属性

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(MAX_RETRIES),
        reraise=True,
        before_sleep=before_sleep_log(LOGGER, logging.WARNING)
    )
    def __call__(self, problems, generated_solutions, grades, target_is_score: bool = False):
        """处理每个问题，计算难度得分差值。"""
        # 使用缓存避免重复计算
        cache_payload = {
            "problems": problems,
            "generated_solutions": generated_solutions,
            "grades": grades,
            "target_is_score": target_is_score,
        }
        cache_key = hashlib.md5(json.dumps(cache_payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if cache_key in self.cache:
            return self.cache[cache_key]

        results = []
        try:
            with torch.no_grad():
                results = self.calculate_difficulties(problems, generated_solutions, grades, target_is_score)
            self.cache[cache_key] = results
            return results
        except Exception as e:
            self.LOGGER.error(f"计算难度得分时出现异常，超过最大重试次数：{e}")
            raise

    def init_knowledge_graph(self, path):
        """以单例模式初始化知识图谱。"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"知识图谱文件路径错误: {path}")

        if self.KNOWLEDGE_GRAPH is None:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.KNOWLEDGE_GRAPH = nx.node_link_graph(data, edges="links")
            for _, attrs in self.KNOWLEDGE_GRAPH.nodes(data=True):
                if "grade" in attrs and "difficulty_score" not in attrs:
                    attrs["difficulty_score"] = self.grade_to_difficulty_score(attrs["grade"])
        return self.KNOWLEDGE_GRAPH

    def load_gate_thresholds(self, path: Path) -> dict[str, float]:
        """Load the fixed MiniVerifier-Gate thresholds reported in the manuscript."""
        thresholds = dict(DEFAULT_GATE_THRESHOLDS)
        if not path.exists():
            self.LOGGER.warning("CAV gate threshold file not found: %s; using defaults.", path)
            return thresholds

        parameter_map = {
            "tau_role": "role",
            "tau_node": "node",
            "tau_ctx": "context",
            "tau_stage": "stage",
            "tau_evid": "evidence",
            "tau_accept": "accept",
            "tau_reject": "reject",
        }
        loaded_keys = set()
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                key = parameter_map.get(row.get("parameter", ""))
                if key is None:
                    continue
                try:
                    thresholds[key] = float(row["value"])
                    loaded_keys.add(key)
                except (KeyError, TypeError, ValueError):
                    self.LOGGER.warning("Skipping malformed threshold row: %s", row)

        if "node" in loaded_keys and "context" not in loaded_keys:
            thresholds["context"] = thresholds["node"]
        elif "context" in loaded_keys and "node" not in loaded_keys:
            thresholds["node"] = thresholds["context"]
        return thresholds

    def load_router_weights(self, path: Path) -> dict[str, float]:
        weights = {
            "lexical": 0.10,
            "synonym": 0.12,
            "signature": 0.14,
            "scope_cosine": 0.13,
            "role": 0.11,
            "node_identity": 0.10,
            "stage": 0.11,
            "evidence": 0.17,
            "entropy": 0.08,
        }
        parameter_map = {
            "w_lex": "lexical",
            "w_syn": "synonym",
            "w_sig": "signature",
            "w_emb": "scope_cosine",
            "w_sem": "scope_cosine",
            "w_role": "role",
            "w_ctx": "node_identity",
            "w_node": "node_identity",
            "w_stage": "stage",
            "w_evid": "evidence",
            "w_entropy": "entropy",
        }
        if path.exists():
            with path.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    key = parameter_map.get(row.get("parameter", ""))
                    if key is None:
                        continue
                    try:
                        weights[key] = float(row["value"])
                    except (KeyError, TypeError, ValueError):
                        self.LOGGER.warning("Skipping malformed router-weight row: %s", row)
        return weights

    def route_score(self, retrieval_features: dict, gate_scores: dict) -> float:
        w = self.router_weights
        s0 = (
            w["lexical"] * retrieval_features.get("lexical", 0.0)
            + w["synonym"] * retrieval_features.get("synonym", 0.0)
            + w["signature"] * retrieval_features.get("signature", 0.0)
            + w["scope_cosine"] * retrieval_features.get("scope_cosine", 0.0)
            + w["role"] * gate_scores.get("role", 0.0)
            + w["node_identity"] * gate_scores.get("node_identity", 0.0)
            + w["stage"] * gate_scores.get("stage", 0.0)
            + w["evidence"] * gate_scores.get("evidence", 0.0)
        )
        clipped = min(max(s0, 1e-12), 1.0 - 1e-12)
        entropy = -(clipped * math.log(clipped) + (1.0 - clipped) * math.log(1.0 - clipped)) / math.log(2.0)
        return min(max(s0 - w["entropy"] * entropy, 0.0), 1.0)

    def load_cav_prompt_template(self, path: Path) -> str:
        if path.exists():
            return path.read_text(encoding="utf-8")
        self.LOGGER.warning("CAV prompt file not found: %s; using built-in fallback prompt.", path)
        return (
            "You are a mathematics curriculum annotator. Identify admitted HKG nodes used in the solution. "
            "Apply role, evidence, node-identity, and stage-scope gates before returning used_nodes. "
            "Return valid JSON only with keys used_nodes, unmatched_concepts, and requires_manual_review.\n\n"
            "problem: {problem}\n"
            "solution: {solution}\n"
            "target_stage: {target_stage}\n"
            "candidate_lattice: {candidate_lattice}\n"
        )

    def build_hkg_node_records(self) -> list[dict]:
        """Load the source-backed public HKG records used by paper-aligned retrieval."""
        if self.hkg_nodes_path.exists():
            synonym_rows: dict[str, list[dict]] = {}
            if self.hkg_synonyms_path.exists():
                with self.hkg_synonyms_path.open("r", encoding="utf-8-sig", newline="") as f:
                    for row in csv.DictReader(f):
                        synonym_rows.setdefault(row.get("node_id", ""), []).append(row)

            records = []
            with self.hkg_nodes_path.open("r", encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    node_id = row.get("node_id", "").strip()
                    if not node_id:
                        continue
                    linked_rows = synonym_rows.get(node_id, [])
                    inline_synonyms = [s.strip() for s in row.get("synonyms", "").split(";") if s.strip()]
                    synonyms = list(inline_synonyms)
                    signatures = []
                    for synonym_row in linked_rows:
                        surface = synonym_row.get("surface_form", "").strip()
                        if not surface:
                            continue
                        if synonym_row.get("surface_type", "").lower() in {"formula", "operator", "signature"}:
                            signatures.append(surface)
                        else:
                            synonyms.append(surface)
                    records.append(
                        {
                            "node_id": node_id,
                            "domain": row.get("domain", "").strip(),
                            "subfield": row.get("subfield", "").strip(),
                            "knowledge_point": row.get("knowledge_point", "").strip(),
                            "stage_label": row.get("stage_label", "").strip(),
                            "difficulty_score": self._safe_float(row.get("normalized_difficulty"), 0.0),
                            "source_reference": row.get("source_reference", "").strip(),
                            "scope_note": row.get("scope_note", "").strip(),
                            "synonyms": tuple(dict.fromkeys(synonyms)),
                            "operator_signatures": tuple(dict.fromkeys(signatures)),
                            "path": ["release_schema", row.get("domain", ""), row.get("subfield", ""), node_id],
                        }
                    )
            if records:
                return records

        # Compatibility fallback for older local graphs that predate the public tables.
        records = []
        for node, attrs in self.kg.nodes(data=True):
            is_runtime_node = "grade" in attrs
            is_release_node = attrs.get("type") == "knowledge_point" or "normalized_difficulty" in attrs
            if not is_runtime_node and not is_release_node:
                continue

            domain = attrs.get("domain", "")
            subfield = attrs.get("subfield", "")
            if not domain or not subfield:
                try:
                    path = nx.shortest_path(self.kg, "root", node)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    path = ["root", "", "", node]
                domain = domain or (path[1] if len(path) > 1 else "")
                subfield = subfield or (path[2] if len(path) > 2 else "")
            else:
                path = ["release_schema", domain, subfield, node]

            stage_label = attrs.get("grade", attrs.get("stage_label", ""))
            difficulty_score = attrs.get("difficulty_score", attrs.get("normalized_difficulty"))
            if difficulty_score is None and "grade" in attrs:
                difficulty_score = self.grade_to_difficulty_score(attrs["grade"])
            difficulty_score = self._safe_float(difficulty_score, 0.0)
            knowledge_point = attrs.get("knowledge_point", attrs.get("label", node))
            records.append(
                {
                    "node_id": node,
                    "domain": domain,
                    "subfield": subfield,
                    "knowledge_point": knowledge_point,
                    "stage_label": stage_label,
                    "difficulty_score": difficulty_score,
                    "source_reference": "",
                    "scope_note": "",
                    "synonyms": (),
                    "operator_signatures": (),
                    "path": path,
                }
            )
        return records

    @staticmethod
    def _retrieval_tokens(text: str) -> list[str]:
        normalized = re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(text)).lower())
        characters = [ch for ch in normalized if ch.isalnum() or "\u4e00" <= ch <= "\u9fff"]
        bigrams = [f"bg:{characters[i]}{characters[i + 1]}" for i in range(max(0, len(characters) - 1))]
        operators = re.findall(r"[a-z]+|\d+(?:\.\d+)?|[+\-*/=^√∑∫△∠⊥∥≠≤≥]", normalized)
        return bigrams + [f"op:{token}" for token in operators]

    def build_retrieval_vocabulary(self, records: list[dict]) -> frozenset[str]:
        vocabulary = set()
        for record in records:
            node_text = " ".join(
                [
                    record.get("knowledge_point", ""),
                    *record.get("synonyms", ()),
                    *record.get("operator_signatures", ()),
                    record.get("scope_note", ""),
                ]
            )
            vocabulary.update(self._retrieval_tokens(node_text))
        return frozenset(vocabulary)

    def _scope_cosine(self, reasoning_unit: str, record: dict) -> float:
        if not record.get("scope_note"):
            return 0.0
        node_text = " ".join(
            [
                record.get("knowledge_point", ""),
                *record.get("synonyms", ()),
                *record.get("operator_signatures", ()),
                record.get("scope_note", ""),
            ]
        )
        unit_counts = Counter(t for t in self._retrieval_tokens(reasoning_unit) if t in self.retrieval_vocabulary)
        node_counts = Counter(t for t in self._retrieval_tokens(node_text) if t in self.retrieval_vocabulary)
        if not unit_counts or not node_counts:
            return 0.0
        dot = sum(value * node_counts.get(token, 0) for token, value in unit_counts.items())
        unit_norm = math.sqrt(sum(value * value for value in unit_counts.values()))
        node_norm = math.sqrt(sum(value * value for value in node_counts.values()))
        return dot / (unit_norm * node_norm) if unit_norm and node_norm else 0.0

    @staticmethod
    def segment_reasoning_units(solution: str) -> list[str]:
        solution_text = str(solution)
        think_match = re.search(r"<think>\s*(.*?)\s*</think>", solution_text, flags=re.DOTALL | re.IGNORECASE)
        reasoning_text = think_match.group(1) if think_match else solution_text
        reasoning_text = re.sub(r"(\$\$.*?\$\$|\\\[.*?\\\])", r"\n\1\n", reasoning_text, flags=re.DOTALL)
        fragments = re.split(
            r"(?:\r?\n)+|(?<=[。！？.!?;；])\s+|(?=因此|所以|于是|从而|由此|then|therefore|thus)",
            reasoning_text,
            flags=re.IGNORECASE,
        )
        units = []
        for fragment in fragments:
            fragment = fragment.strip()
            if not fragment:
                continue
            if units and len(fragment) < 8:
                units[-1] = f"{units[-1]} {fragment}".strip()
            else:
                units.append(fragment)
        return units or [reasoning_text.strip()]

    def build_candidate_lattice(self, problem: str, reasoning_unit: str) -> list[dict]:
        """Paper-aligned Retrieve: exact/synonym/signature union, with no cutoff or top-k."""
        unit_key = self._normalize_knowledge_text(reasoning_unit)
        problem_key = self._normalize_knowledge_text(problem)
        candidates = []
        provenance_order = {"exact": 0, "synonym": 1, "signature": 2}
        for record in self.hkg_node_records:
            canonical_key = self._normalize_knowledge_text(record.get("knowledge_point", ""))
            synonym_match = any(
                (surface_key := self._normalize_knowledge_text(surface)) and surface_key in unit_key
                for surface in record.get("synonyms", ())
            )
            signature_match = any(
                (signature_key := self._normalize_knowledge_text(signature)) and signature_key in unit_key
                for signature in record.get("operator_signatures", ())
            )
            exact_match = bool(canonical_key and canonical_key in unit_key)
            if not (exact_match or synonym_match or signature_match):
                continue
            provenance = "exact" if exact_match else "synonym" if synonym_match else "signature"
            domain_key = self._normalize_knowledge_text(record.get("domain", ""))
            subfield_key = self._normalize_knowledge_text(record.get("subfield", ""))
            candidates.append(
                {
                    **record,
                    "difficulty_score": round(record["difficulty_score"], 6),
                    "retrieval_features": {
                        "lexical": float(exact_match),
                        "synonym": float(synonym_match),
                        "signature": float(signature_match),
                        "scope_cosine": self._scope_cosine(reasoning_unit, record),
                        "domain_prior": float(
                            bool((domain_key and domain_key in problem_key) or (subfield_key and subfield_key in problem_key))
                        ),
                    },
                    "provenance": provenance,
                    "reasoning_span": reasoning_unit,
                }
            )
        candidates.sort(key=lambda item: (provenance_order[item["provenance"]], item["node_id"]))
        return candidates

    @staticmethod
    def _extract_response_content(response) -> str:
        if response is None:
            return ""
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            if "message" in response and isinstance(response["message"], dict):
                return str(response["message"].get("content", ""))
            return str(response.get("content", response.get("text", "")))
        return str(response)

    def parse_cav_response(self, response) -> dict:
        """Parse the bounded fallback evaluator JSON for RBA-CAV."""
        content = self._extract_response_content(response).strip()
        if "</think>" in content:
            content = content.split("</think>")[-1].strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?", "", content.strip(), flags=re.IGNORECASE).strip()
            content = re.sub(r"```$", "", content).strip()

        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("CAV evaluator did not return a JSON object")
        return json.loads(content[start : end + 1])

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def normalize_gate_scores(self, item: dict) -> dict[str, float]:
        scores = item.get("gate_scores") or {}
        role = str(item.get("role", "")).lower()
        confidence = self._safe_float(item.get("confidence"), 0.0)
        return {
            "role": self._safe_float(scores.get("role"), 1.0 if role in {"central", "supporting"} else confidence),
            "evidence": self._safe_float(scores.get("evidence", scores.get("evid")), confidence),
            "node_identity": self._safe_float(
                scores.get("node_identity", scores.get("node", scores.get("context"))), confidence
            ),
            "stage": self._safe_float(scores.get("stage"), confidence),
        }

    def resolve_cav_node(self, item: dict) -> Optional[dict]:
        for key in ("node_id", "knowledge_point", "canonical_name", "node"):
            value = item.get(key)
            if not value:
                continue
            if value in self.hkg_node_records_by_id:
                return self.hkg_node_records_by_id[value]
            normalized = self._normalize_knowledge_text(value)
            if normalized in self.hkg_node_records_by_name:
                return self.hkg_node_records_by_name[normalized]
        return None

    def admitted_nodes_from_cav_payload(self, payload: dict) -> tuple[list[dict], list[dict]]:
        admitted_nodes = []
        audit_log = []
        for item in payload.get("used_nodes", []) or []:
            role = str(item.get("role", "")).lower()
            record = self.resolve_cav_node(item)
            gate_scores = self.normalize_gate_scores(item)
            passes = (
                role in {"central", "supporting"}
                and record is not None
                and bool(record.get("source_reference"))
                and gate_scores["role"] >= self.gate_thresholds["role"]
                and gate_scores["evidence"] >= self.gate_thresholds["evidence"]
                and gate_scores["node_identity"] >= self.gate_thresholds["node"]
                and gate_scores["stage"] >= self.gate_thresholds["stage"]
            )
            audit_entry = {
                "candidate": item,
                "resolved_node_id": record["node_id"] if record else None,
                "gate_scores": gate_scores,
                "admitted": bool(passes),
            }
            audit_log.append(audit_entry)
            if passes:
                admitted_nodes.append(
                    {
                        **record,
                        "evidence": item.get("evidence", ""),
                        "role": role,
                        "confidence": self._safe_float(item.get("confidence"), min(gate_scores.values())),
                    }
                )

        for item in payload.get("unmatched_concepts", []) or []:
            audit_log.append({"candidate": item, "resolved_node_id": None, "admitted": False})
        return admitted_nodes, audit_log

    def get_local_gate_scores(self, candidate_records: list[dict]) -> list[dict]:
        """Score every reasoning-unit/node pair with the local four-head MiniVerifier service."""
        if not candidate_records:
            return []
        response = requests.post(
            self.gate_scorer_url,
            json={"records": candidate_records},
            timeout=300,
        )
        response.raise_for_status()
        payload = response.json()
        outputs = payload.get("scores", payload) if isinstance(payload, dict) else payload
        if not isinstance(outputs, list) or len(outputs) != len(candidate_records):
            raise ValueError("MiniVerifier-Gate must return one four-score record per candidate pair")
        return [self.normalize_gate_scores(output) for output in outputs]

    def _fallback_prompt(self, problem: str, reasoning_unit: str, grade, lattice: list[dict]) -> str:
        public_lattice = [
            {
                key: candidate.get(key)
                for key in (
                    "node_id",
                    "domain",
                    "subfield",
                    "knowledge_point",
                    "stage_label",
                    "difficulty_score",
                    "source_reference",
                    "scope_note",
                    "synonyms",
                    "operator_signatures",
                    "retrieval_features",
                )
            }
            for candidate in lattice
        ]
        prompt_values = {
            "{problem}": str(problem),
            "{solution}": str(reasoning_unit),
            "{target_stage}": str(grade),
            "{candidate_lattice}": json.dumps(public_lattice, ensure_ascii=False),
        }
        prompt = self.cav_prompt_template
        for placeholder, value in prompt_values.items():
            prompt = prompt.replace(placeholder, value)
        return prompt

    def extract_cav_admissions(self, problems, generated_solutions, grades, target_is_score=False):
        """Apply Retrieve -> local gates -> uncertainty routing -> conjunction -> aggregation."""
        records = []
        for problem, solution, grade in zip(problems, generated_solutions, grades):
            target_score = self.target_to_difficulty_score(grade, target_is_score)
            units = self.segment_reasoning_units(solution)
            unit_lattices = [self.build_candidate_lattice(problem, unit) for unit in units]
            scorer_records = []
            candidate_refs = []
            for unit_index, (unit, lattice) in enumerate(zip(units, unit_lattices)):
                for candidate in lattice:
                    scorer_records.append(
                        {
                            "problem_text": str(problem),
                            "reasoning_span": unit,
                            "surface_mention": candidate.get("knowledge_point", ""),
                            "node_id": candidate["node_id"],
                            "node_name": candidate.get("knowledge_point", ""),
                            "domain": candidate.get("domain", ""),
                            "subfield": candidate.get("subfield", ""),
                            "stage_label": candidate.get("stage_label", ""),
                            "normalized_difficulty": candidate.get("difficulty_score", 0.0),
                            "synonyms": candidate.get("synonyms", ()),
                            "scope_note": candidate.get("scope_note", ""),
                            "retrieval_features": candidate.get("retrieval_features", {}),
                            "symbolic_gate_flags": {
                                "source_identity_resolved": bool(
                                    candidate.get("node_id") and candidate.get("source_reference")
                                ),
                                "canonical_match": bool(
                                    candidate.get("retrieval_features", {}).get("lexical", 0.0)
                                ),
                                "synonym_match": bool(
                                    candidate.get("retrieval_features", {}).get("synonym", 0.0)
                                ),
                                "signature_match": bool(
                                    candidate.get("retrieval_features", {}).get("signature", 0.0)
                                ),
                                "domain_or_subfield_prior": bool(
                                    candidate.get("retrieval_features", {}).get("domain_prior", 0.0)
                                ),
                                "scope_note_available": bool(candidate.get("scope_note")),
                            },
                        }
                    )
                    candidate_refs.append((unit_index, candidate))

            try:
                local_scores = self.get_local_gate_scores(scorer_records)
            except Exception as exc:
                records.append(
                    {
                        "parse_success": False,
                        "error": f"MiniVerifier-Gate failure: {exc}",
                        "admitted_nodes": [],
                        "audit_log": [],
                        "candidate_lattice": unit_lattices,
                        "target_difficulty_score": target_score,
                        "requires_manual_review": True,
                        "model_response": "",
                    }
                )
                continue

            routed = []
            fallback_prompts = []
            fallback_indices = []
            for index, ((unit_index, candidate), gate_scores) in enumerate(zip(candidate_refs, local_scores)):
                score = self.route_score(candidate.get("retrieval_features", {}), gate_scores)
                decision_source = "deterministic_reject" if score <= self.gate_thresholds["reject"] else "local_gate"
                routed.append(
                    {
                        "unit_index": unit_index,
                        "candidate": candidate,
                        "gate_scores": gate_scores,
                        "route_score": score,
                        "decision_source": decision_source,
                    }
                )
                if self.gate_thresholds["reject"] < score < self.gate_thresholds["accept"]:
                    fallback_indices.append(index)
                    fallback_prompts.append(self._fallback_prompt(problem, units[unit_index], grade, unit_lattices[unit_index]))

            if fallback_prompts:
                for routed_index, response in zip(fallback_indices, self.get_model_responses(fallback_prompts)):
                    candidate = routed[routed_index]["candidate"]
                    try:
                        payload = self.parse_cav_response(response)
                        replacement = next(
                            (
                                item
                                for item in payload.get("used_nodes", [])
                                if self.resolve_cav_node(item)
                                and self.resolve_cav_node(item)["node_id"] == candidate["node_id"]
                            ),
                            None,
                        )
                        routed[routed_index]["gate_scores"] = (
                            self.normalize_gate_scores(replacement) if replacement else {key: 0.0 for key in ("role", "node_identity", "stage", "evidence")}
                        )
                        routed[routed_index]["decision_source"] = "llm_fallback"
                    except Exception as exc:
                        routed[routed_index]["gate_scores"] = {key: 0.0 for key in ("role", "node_identity", "stage", "evidence")}
                        routed[routed_index]["decision_source"] = "fallback_parse_failure"
                        routed[routed_index]["fallback_error"] = str(exc)

            admitted_by_id = {}
            audit_log = []
            for item in routed:
                candidate = item["candidate"]
                scores = item["gate_scores"]
                source_identity_resolved = bool(candidate.get("node_id") and candidate.get("source_reference"))
                passes = (
                    item["route_score"] > self.gate_thresholds["reject"]
                    and source_identity_resolved
                    and scores["role"] >= self.gate_thresholds["role"]
                    and scores["evidence"] >= self.gate_thresholds["evidence"]
                    and scores["node_identity"] >= self.gate_thresholds["node"]
                    and scores["stage"] >= self.gate_thresholds["stage"]
                )
                # The stage gate verifies the source-backed operator interpretation in
                # this local span; it never tests candidate_stage <= target_stage.
                audit_log.append({**item, "admitted": bool(passes)})
                if passes:
                    admitted_by_id.setdefault(candidate["node_id"], candidate)

            records.append(
                {
                    "parse_success": True,
                    "admitted_nodes": list(admitted_by_id.values()),
                    "audit_log": audit_log,
                    "candidate_lattice": unit_lattices,
                    "target_difficulty_score": target_score,
                    "requires_manual_review": not bool(admitted_by_id),
                    "model_response": "",
                }
            )
        return records

    def get_model_responses(self, prompts):
        """通过调用模型服务的 API 批量获取模型响应。"""
        # 定义用于发送单个请求的函数
        @retry(
            retry=retry_if_exception_type(Exception),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            stop=stop_after_attempt(self.MAX_RETRIES),
            reraise=True
        )
        def send_request(prompt, index):
            try:
                response = requests.post(
                    self.evaluator_url,
                    json={
                        'prompt': prompt,
                        'temperature': self.evaluator_temperature,
                        'max_tokens': self.evaluator_max_tokens
                    },
                    timeout=300  # 根据需要调整超时时间
                )
                response.raise_for_status()

                generated_text = response.json()['text']
                if isinstance(generated_text, list):
                    generated_text = '\n'.join(generated_text)

                return {"message": {"content": generated_text}}
            except requests.exceptions.RequestException as e:
                self.LOGGER.error(f"请求 {index+1} 模型服务请求失败: {str(e)}")
                raise
            except KeyError as e:
                self.LOGGER.error(f"请求 {index+1} 响应格式错误，缺少键：{e}")
                self.LOGGER.error(f"响应内容：{response.text}")
                raise

        # 使用线程池并发发送请求，并维持响应顺序
        total_requests = len(prompts)
        responses = [None] * total_requests  # 预先分配响应列表

        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            futures = {executor.submit(send_request, prompt, i): i for i, prompt in enumerate(prompts)}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    response = future.result()
                    responses[index] = response
                except Exception as e:
                    self.LOGGER.error(f"请求 {index+1} 失败: {str(e)}")
                    raise

        return responses


    @retry(
        retry=retry_if_exception_type(ValueError),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(5),
        reraise=True
    )
    def summarize_knowledge_points(self, problems, answers):

        # 递归展平嵌套结构
        def deep_flatten(nested_item):
            """将多层嵌套结构展平为字符串"""
            if isinstance(nested_item, (list, tuple)):
                return ' '.join(deep_flatten(item) for item in nested_item)
            elif isinstance(nested_item, dict):
                return deep_flatten(nested_item.get('content', nested_item.get('text', '')))
            return str(nested_item).strip()

        """使用模型服务对知识点进行总结，并验证格式。支持批量处理。"""
        prompts = []
        marker="### 请在此处开始总结 ###\n"

        for problem, answer in zip(problems, answers):
            # 添加长度截断逻辑（处理中文字符）
            answer = answer.split('</think>')[-1].strip()
            if len(answer) > 1000:
                # 计算前500和后500字符的索引（确保不越界）
                start = max(0, 500)
                end = max(len(answer)-500, 500)
                truncated_answer = (
                    f"{answer[:500]}\n"
                    "[...中间内容过长已省略...]\n"
                    f"{answer[-500:]}"
                )
            else:
                truncated_answer = answer

            prompt = (
                "请根据以下提供的题目和答案，总结出其中直接考察到的最主要的1-3个知识点，不要有重复，不要重新解题，思维过程要简洁。\n"
                "输出格式：每行一条知识点，使用 # 符号开头。示例：\n"
                "# 一百以内不进位不退位的加减法\n"
                "# 区分长方体、正方体、圆柱体、球\n"
                f"题目：{problem}\n答案：{truncated_answer}"
                f"其中直接考察到的最主要的1-3个知识点是什么？{marker}"  # 添加分隔符
            )
            prompts.append(prompt)

            print("当前答案是：", (lambda s: s[:100] if len(s) > 100 else s)(truncated_answer))


        # 获取模型响应
        responses = self.get_model_responses(prompts)

        # 处理响应，验证格式
        knowledge_points_list = []
        model_responses_list = []

        for response in responses:
            try:
                # 防御式获取内容
                raw_content = response.get('message', {}) if response else {}

                # 递归展平所有嵌套结构
                content = deep_flatten(raw_content)

                # 类型安全处理（确保最终为字符串）
                if isinstance(content, (list, tuple)):
                    content = ' '.join(map(str, content))
                content = str(content).strip()

                # 记录原始响应（调试用）
                self.LOGGER.debug(f"原始响应内容类型: {type(raw_content)}，内容片段: {content[:200]}...")

                # 处理</think>标签
                think_tag = "</think>"
                if think_tag in content:
                    # 分割出</think>标签后的内容
                    content_parts = content.split(think_tag)
                    if len(content_parts) > 1:
                        content = content_parts[-1].strip()
                        self.LOGGER.debug("检测到</think>标签，使用标签后内容进行解析")
                    else:
                        self.LOGGER.warning("</think>标签存在但未找到有效分割内容")
                # 原有分隔符处理
                elif marker in content:
                    content = content.split(marker, 1)[1].strip()
                else:
                    self.LOGGER.warning("未找到预期的分隔符或</think>标签，完整内容解析")


                # 提取有效知识点
                knowledge_points = []

                ### 修改开始：调整解析逻辑，支持更多格式的知识点前缀 ###
                for line in content.split('\n'):
                    line = line.strip()
                    # 有效性校验
                    if 2 < len(line) < 50:

                        # 匹配可能的前缀格式
                        # 包括：#, 知识点, 知识点1/一, 数字序号（1. or 一、）, 项目符号（- 或 •）
                        prefix_pattern = r'^(#|知识点[\s\d一二三四五六七八九十]*[：:、.．．\s]?|[\d一二三四五六七八九十]+[）)\.、:：．．\s]?|[-•●•·])'

                        # 检查行是否以指定的前缀开头
                        if re.match(prefix_pattern, line):
                            # 移除前缀
                            kp = re.sub(prefix_pattern, '', line).strip()

                            # 过滤无效内容
                            if kp and not re.search(r'示例|注意|格式|题目|答案|提示', kp):
                                knowledge_points.append(kp)
                    else:
                        # 如果行长度不符合要求，跳过
                        continue
                ### 修改结束 ###

                # 去重处理
                seen = set()
                unique_points = []
                for kp in knowledge_points:
                    key = re.sub(r'\s+', '', kp).lower()
                    if key not in seen and len(unique_points) < 3:
                        seen.add(key)
                        unique_points.append(kp)

                # 结果验证
                if not unique_points:
                    self.LOGGER.warning(f"未提取到有效知识点，响应内容：{content[-200:]}...")
                    unique_points = ["综合应用"]  # 兜底值

                knowledge_points_list.append(unique_points)
                model_responses_list.append(content[:500])

            except Exception as e:
                self.LOGGER.error(f"响应处理异常：{str(e)}", exc_info=True)
                knowledge_points_list.append(["解析异常"])
                model_responses_list.append("")

        # 最终校验
        if all(not kp_list or kp_list == ["解析异常"] for kp_list in knowledge_points_list):
            raise ValueError("无法从任何模型响应中提取到知识点")

        print("知识点列表：", knowledge_points_list)

        return knowledge_points_list, model_responses_list


    def convert_grade(self, grade_str):
        """将年级字符串转换为数值，支持中文数字和阿拉伯数字。"""
        if grade_str is None:
            return 0
        if isinstance(grade_str, (int, float)):
            return float(grade_str)
        grade_str = str(grade_str).strip()
        if not grade_str:
            return 0
        try:
            numeric_grade = float(grade_str)
            return numeric_grade
        except ValueError:
            pass

        # 匹配模式允许数字或中文数字的年级，例如：8年级下册、八年级下册、高一下册
        pattern = r'^(?:(\d+)|([一二三四五六七八九十]+)|(高[一二三]))\s*年?级?\s*([上下]册?)?$'
        match = re.match(pattern, grade_str)
        if not match:
            self.LOGGER.warning(f"年级格式无法解析: {grade_str}")
            return 0  # 无法匹配时返回0

        # 解析匹配组
        num_grade, cn_grade, high_grade, semester = match.groups()
        grade_value = 0

        # 处理普通年级（数字或中文）
        if num_grade:
            grade_value = int(num_grade)
        elif cn_grade:
            grade_value = _chinese_number_to_int(cn_grade)
        elif high_grade:
            # 处理高中年级（高一、高二、高三）
            high_map = {'高一':10, '高二':11, '高三':12}
            grade_value = high_map.get(high_grade, 0)

        # 处理学期（下册加0.5）
        if semester and '下' in semester and grade_value > 0:
            grade_value += 0.5

        return grade_value

    def grade_to_difficulty_score(self, grade_label):
        """Convert grade labels or raw difficulty values to normalized curriculum difficulty scores."""
        if isinstance(grade_label, (int, float)):
            value = float(grade_label)
            if value > MAX_GRADE_VALUE:
                return max(0.0, min(100.0, value))
            return grade_value_to_difficulty_score(value)

        label = str(grade_label).strip()
        try:
            value = float(label)
            if value > MAX_GRADE_VALUE:
                return max(0.0, min(100.0, value))
            return grade_value_to_difficulty_score(value)
        except ValueError:
            pass

        return grade_value_to_difficulty_score(self.convert_grade(label))

    @staticmethod
    def clamp_difficulty_score(value):
        return max(0.0, min(100.0, float(value)))

    def target_to_difficulty_score(self, target_label, target_is_score=False):
        if target_is_score:
            return self.clamp_difficulty_score(target_label)
        return self.grade_to_difficulty_score(target_label)


    def extract_selected_number(self, text):
        """提取文本中的数字编号。"""
        pattern = r'(\\{0,2}boxed)\s*\{+\s*(\d+)\s*\}+'
        matches = list(re.finditer(pattern, text))
        if matches:
            match = matches[-1]  # 取最后一个匹配项
            selected_num = match.group(2)
            return True, selected_num
        else:
            return False, "未找到正确的格式"

    @retry(
        retry=retry_if_exception_type(ValueError),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True
    )
    def select_category_node(self, kg, kp, problem):
        """根据知识点和题目，选择最合适的分类节点。"""
        # 缓存命中检查
        if kp in self.kp_cache:
            self.LOGGER.info(f"知识点缓存命中: {kp}")
            # 更新访问顺序
            value = self.kp_cache.pop(kp)
            self.kp_cache[kp] = value
            return value  # 返回缓存的(current_node, grade)

        current_node = 'root'
        path = ['root']
        grade = None

        # 三层选择提示语（可以单独修改）
        level_prompts = [
            {  # 第一级选择提示
                "prompt": (
                    "请选择最适合描述 \"{kp}\" 的分类编号：\n{options}\n\n"
                    "请直接返回数字编号，不要包含其他内容，并将数字编号用\\boxed{{}}包裹。示例：\\boxed{{1}}。"
                ),
                "error_msg": "未匹配到一级知识点"
            },
            {  # 第二级选择提示
                "prompt": (
                    "请判断知识点 \"{kp}\" 属于以下哪个类别：\n{options}\n\n"
                    "请直接返回类别的数字编号，不要包含其他内容，并将数字编号用\\boxed{{}}包裹。示例：\\boxed{{1}}。"
                ),
                "error_msg": "未匹配到二级知识点"
            },
            {  # 第三级选择提示
                "prompt": (
                    "请根据以下题目内容，从知识点目录中选择和题目以及其可能涉及的知识点 \"{kp}\" 最相似的知识点编号：\n"
                    "题目：{question}\n"
                    "选项：\n{options}\n\n"
                    "请直接返回数字编号，不要包含其他内容，并将数字编号用\\boxed{{}}包裹。示例：\\boxed{{1}}。"
                ),
                "error_msg": "未匹配到三级知识点"
            }
        ]

        for level in range(3):
            # 获取当前节点的子节点
            children = list(kg.successors(current_node))
            if not children:
                return None, None  # 没有子节点提前终止

            # 生成选项列表
            options = []
            node_map = {}
            for i, child in enumerate(children, 1):
                node_map[str(i)] = child
                options.append(f"{i}. {child}")

            # 获取当前层级的提示语
            prompt_config = level_prompts[level]
            prompt = prompt_config["prompt"].format(
                kp=kp,
                question=problem,  # 插入题目内容
                options="\n".join(options)
            )

            # 获取模型响应
            response = self.get_model_responses([prompt])[0]
            if not response or 'message' not in response or 'content' not in response['message']:
                self.LOGGER.error("获取模型响应失败")
                return None, None

            response_content = response['message']['content']

            # 记录交互过程（INFO级别）
            self.LOGGER.info(f"[模型交互] Level {level+1}\nPrompt: {prompt}\nResponse: {response_content}")

            # 解析选择的数字编号
            success, selected_num = self.extract_selected_number(response_content.strip())

            if not success or selected_num not in node_map:
                self.LOGGER.warning(f"{prompt_config['error_msg']}: {selected_num}")
                # 如果这是可重试的异常，可以抛出异常以触发重试
                raise ValueError(prompt_config['error_msg'])

            # 更新当前节点和路径
            current_node = node_map[selected_num]
            path.append(current_node)

            # 第三级时检查年级属性
            if level == 2:
                grade = kg.nodes[current_node].get('grade')

        # 验证结果有效性后更新缓存
        if current_node != 'root' and grade:
            # 维护缓存大小
            if len(self.kp_cache) >= self.MAX_CACHE_SIZE:
                self.kp_cache.popitem(last=False)  # 移除最久未使用的条目
            self.kp_cache[kp] = (current_node, grade)
            self.LOGGER.debug(f"新增缓存条目: {kp} -> {current_node}|{grade}")

        return current_node, grade

    @staticmethod
    def _normalize_knowledge_text(text):
        normalized = unicodedata.normalize("NFKC", str(text)).lower()
        return re.sub(r"[\s，。、“”‘’（）()《》<>:：；;,.]+", "", normalized)

    def match_knowledge_point_locally(self, kp):
        """Match an extracted knowledge point to a graded HKG node by keyword and lexical similarity."""
        kp_key = self._normalize_knowledge_text(kp)
        if not kp_key:
            return None, None

        best_node = None
        best_grade = None
        best_score = 0.0

        for node, attrs in self.kg.nodes(data=True):
            grade = attrs.get("grade")
            if not grade:
                continue
            node_key = self._normalize_knowledge_text(node)
            if not node_key:
                continue
            if node_key == kp_key:
                return node, grade
            if node_key in kp_key or kp_key in node_key:
                score = 0.92
            else:
                score = SequenceMatcher(None, kp_key, node_key).ratio()
            if score > best_score:
                best_node = node
                best_grade = grade
                best_score = score

        if best_score >= 0.86:
            return best_node, best_grade
        return None, None

    def get_max_difficulty_score(self, problem, knowledge_points):
        """Return d_model: the maximum normalized HKG difficulty score among extracted knowledge points."""
        kg = self.kg
        max_difficulty_score = 0.0
        for kp in knowledge_points:
            # 优先检查缓存
            if kp in self.kp_cache:
                cached_node, cached_grade = self.kp_cache[kp]
                difficulty_score = self.grade_to_difficulty_score(cached_grade)
                max_difficulty_score = max(max_difficulty_score, difficulty_score)
                continue

            current_node, grade = self.match_knowledge_point_locally(kp)
            if current_node and grade:
                if len(self.kp_cache) >= self.MAX_CACHE_SIZE:
                    self.kp_cache.popitem(last=False)
                self.kp_cache[kp] = (current_node, grade)
            else:
                current_node, grade = self.select_category_node(kg, kp, problem)

            if current_node and grade:
                difficulty_score = self.grade_to_difficulty_score(grade)
                if difficulty_score > max_difficulty_score:
                    max_difficulty_score = difficulty_score
            else:
                self.LOGGER.error("未能匹配到有效的知识点分类节点和年级")
                continue

        if max_difficulty_score == 0:
            self.LOGGER.info(f"题目 \"{problem}\" 未能计算出有效的难度，采用最低课程难度 0.0")

        return max_difficulty_score

    def get_max_grade_value(self, problem, knowledge_points):
        """Backward-compatible alias for older evaluation scripts."""
        return self.get_max_difficulty_score(problem, knowledge_points)

    def calculate_difficulties(self, problems, generated_solutions, grades, target_is_score=False):
        """Return CAV-based difficulty gaps for generated solutions.

        The path follows the paper: the verifier returns candidate
        reasoning-unit/HKG-node evidence with role, evidence, node-identity,
        and stage gate scores; only admitted nodes enter max difficulty
        aggregation. A local gate-service failure is handled as an empty
        admitted path and marked for review; it is not replaced by an
        unbounded knowledge-point extraction path.
        """
        results = []
        try:
            cav_records = self.extract_cav_admissions(problems, generated_solutions, grades, target_is_score)
            for i, cav_record in enumerate(cav_records):
                problem = problems[i]
                grade_str = grades[i]
                target_difficulty_score = cav_record["target_difficulty_score"]

                admitted_nodes = cav_record["admitted_nodes"]
                audit_log = cav_record["audit_log"]
                knowledge_points = [node["knowledge_point"] for node in admitted_nodes]
                extraction_mode = "rba_cav" if cav_record["parse_success"] else "rba_cav_gate_failure"
                model_response = cav_record["model_response"]
                empty_admitted_path = len(admitted_nodes) == 0
                if empty_admitted_path:
                    # Algorithm 1: if K_a is empty, set d_model to d_target
                    # and retain the manual-review flag in the audit record.
                    model_difficulty_score = target_difficulty_score
                else:
                    model_difficulty_score = max(node["difficulty_score"] for node in admitted_nodes)

                difficulty_difference = target_difficulty_score - model_difficulty_score

                self.LOGGER.debug(
                    "mode=%s admitted=%s target=%.2f model=%.2f diff=%.2f",
                    extraction_mode,
                    knowledge_points,
                    target_difficulty_score,
                    model_difficulty_score,
                    difficulty_difference,
                )

                result = {
                    "problem": problem,
                    "generated_solution": generated_solutions[i],
                    "grade": grade_str,
                    "grade_value": target_difficulty_score,
                    "target_difficulty_score": target_difficulty_score,
                    "extracted_knowledge_points": knowledge_points,
                    "admitted_nodes": admitted_nodes,
                    "audit_log": audit_log,
                    "candidate_lattice": cav_record["candidate_lattice"],
                    "requires_manual_review": cav_record["requires_manual_review"] or empty_admitted_path,
                    "empty_admitted_path": empty_admitted_path,
                    "extraction_mode": extraction_mode,
                    "extracted_grade_value": model_difficulty_score,
                    "model_difficulty_score": model_difficulty_score,
                    "difficulty_difference": difficulty_difference,
                    "model_response": model_response,
                }
                results.append(result)

            return results

        except Exception as e:
            self.LOGGER.error(f"最终计算失败: {str(e)}")
            # 如果发生异常，抛出异常
            raise

def difficulty_reward(completions, **kwargs) -> list[float]:
    """HKG curriculum-alignment reward: sigmoid((d_target - d_model) / tau)."""
    logger = logging.getLogger(__name__)

    def deep_flatten(nested_iterable):
        for item in nested_iterable:
            if isinstance(item, (list, tuple)):
                yield from deep_flatten(item)
            else:
                yield item

    def ensure_clean_string_list(input_data):
        flattened = list(deep_flatten([input_data]))
        return [str(item) for item in flattened if not isinstance(item, (dict, type(None)))]

    def get_field(field_names):
        for field_name in field_names:
            raw_data = kwargs.get(field_name)
            if raw_data is not None:
                return ensure_clean_string_list(raw_data), field_name
        return [], field_names[0]

    problems, _ = get_field(["problem", "prompt"])
    target_values, target_field = get_field(["target_difficulty", "difficulty", "grade"])
    generated_solutions = [completion[0]["content"] for completion in completions]

    if len(problems) != len(generated_solutions) or len(problems) != len(target_values):
        raise ValueError(
            f"Difficulty reward data length mismatch: problems={len(problems)}, "
            f"solutions={len(generated_solutions)}, {target_field}={len(target_values)}"
        )

    script_args = kwargs.get("script_args")
    kg_path = kwargs.get("kg_path") or getattr(script_args, "kg_path", None) or str(DEFAULT_KG_PATH)
    evaluator_url = (
        kwargs.get("difficulty_evaluator_url")
        or getattr(script_args, "difficulty_evaluator_url", None)
        or "http://localhost:5000/generate"
    )
    gate_scorer_url = (
        kwargs.get("difficulty_gate_scorer_url")
        or getattr(script_args, "difficulty_gate_scorer_url", None)
        or "http://localhost:5001/score"
    )
    evaluator_temperature = float(
        kwargs.get(
            "difficulty_evaluator_temperature",
            getattr(script_args, "difficulty_evaluator_temperature", 0.0),
        )
    )
    evaluator_max_tokens = int(
        kwargs.get(
            "difficulty_evaluator_max_tokens",
            getattr(script_args, "difficulty_evaluator_max_tokens", 4096),
        )
    )
    cav_prompt_path = kwargs.get("difficulty_cav_prompt_path") or getattr(
        script_args, "difficulty_cav_prompt_path", str(DEFAULT_CAV_PROMPT_PATH)
    )
    gate_thresholds_path = kwargs.get("difficulty_gate_thresholds_path") or getattr(
        script_args, "difficulty_gate_thresholds_path", str(DEFAULT_GATE_THRESHOLDS_PATH)
    )
    temperature = float(
        kwargs.get(
            "difficulty_reward_temperature",
            getattr(script_args, "difficulty_reward_temperature", DEFAULT_DIFFICULTY_TEMPERATURE),
        )
    )

    calculator_settings = (
        kg_path,
        evaluator_url,
        gate_scorer_url,
        evaluator_temperature,
        evaluator_max_tokens,
        cav_prompt_path,
        gate_thresholds_path,
    )
    try:
        if (
            not hasattr(difficulty_reward, "calculator")
            or getattr(difficulty_reward, "calculator_settings", None) != calculator_settings
        ):
            difficulty_reward.calculator = MathDifficulty(
                kg_path=kg_path,
                evaluator_url=evaluator_url,
                gate_scorer_url=gate_scorer_url,
                evaluator_temperature=evaluator_temperature,
                evaluator_max_tokens=evaluator_max_tokens,
                cav_prompt_path=cav_prompt_path,
                gate_thresholds_path=gate_thresholds_path,
            )
            difficulty_reward.calculator_settings = calculator_settings
    except Exception as e:
        logger.error("Failed to initialize MathDifficulty: %s", e)
        return [0.0] * len(problems)

    try:
        results = difficulty_reward.calculator(
            problems=problems,
            generated_solutions=generated_solutions,
            grades=target_values,
            target_is_score=target_field in {"target_difficulty", "difficulty"},
        )
        if not results or len(results) != len(problems):
            logger.warning("Difficulty result count mismatch: got %s expected %s", len(results), len(problems))
            return [0.0] * len(problems)

        return [difficulty_gap_to_reward(r["difficulty_difference"], temperature) for r in results]

    except Exception:
        logger.exception("Failed to compute difficulty reward")
        return [0.0] * len(problems)

def get_reward_funcs(script_args) -> list[Callable]:
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "code": update_wrapper(
            partial(
                code_reward,
                num_parallel=script_args.parallel_code_exec_per_proc,
                e2b_router_url=script_args.e2b_router_url,
            ),
            code_reward,
        ),
        "binary_code": update_wrapper(
            partial(
                binary_code_reward,
                num_parallel=script_args.parallel_code_exec_per_proc,
                e2b_router_url=script_args.e2b_router_url,
            ),
            binary_code_reward,
        ),
        "ioi_code": update_wrapper(
            partial(ioi_code_reward, test_batch_size=script_args.code_eval_test_batch_size), ioi_code_reward
        ),
        "code_format": get_code_format_reward(language=script_args.code_language),
        "tag_count": tag_count_reward,
        # "difficulty": lambda completions, **kwargs: difficulty_reward(
        #     completions,
        #     solution=kwargs.get("solution", []),  # 显式传递solution
        #     **kwargs  # 透传所有其他参数
        # ),
        "difficulty": difficulty_reward,
        # "difficulty": lambda completions, **kwargs: difficulty_reward(
        #     completions,
        #     **kwargs
        # ),
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    return reward_funcs









# # test
# import unittest
# from unittest.mock import patch, MagicMock
# import math

# class TestMathDifficultyReward(unittest.TestCase):
#     @patch('requests.post')  # 拦截所有API请求
#     @patch('logging.Logger.error')  # 捕获日志错误
#     def test_normal_operation(self, mock_log, mock_post):
#         """测试标准流程：正常响应、有效知识点提取、正确年级转换"""
#         # 配置模拟响应链
#         mock_responses = [
#             # 知识点提取响应
#             MagicMock(json=lambda: {"text": "# 勾股定理\n# 直角三角形性质"}),
#             # 三级分类选择响应
#             MagicMock(json=lambda: {"text": "应选择\\boxed{1}"}),
#             MagicMock(json=lambda: {"text": "正确答案是\\boxed{2}"}),
#             MagicMock(json=lambda: {"text": "最合适的是\\boxed{3}"}),
#         ]
#         mock_post.side_effect = mock_responses

#         # 构造测试数据
#         test_data = {
#             "problem": ["已知直角三角形三边长为3,4,5，求面积"],
#             "answer": ["6"],
#             "grade": ["八年级下册"]  # 对应8.5年级值
#         }

#         # 执行测试
#         rewards = difficulty_reward([[[{"content": "6"}]]], **test_data)

#         # 验证结果
#         self.assertAlmostEqual(rewards[0], 1/(1+math.exp(-0.5)), delta=0.1,  # 假设提取到九年级(9.0)
#                               msg="八年级题目匹配九年级知识点应有正向奖励")
#         mock_log.assert_not_called()

#     @patch('requests.post')
#     def test_list_response_handling(self, mock_post):
#         """测试模型返回列表类型响应的处理能力"""
#         # 模拟返回嵌套列表结构
#         mock_post.return_value = MagicMock(
#             json=lambda: {"text": ["# 代数运算", {"content": "# 方程求解"}, ["# 无效知识点", 123]]}
#         )

#         test_data = {
#             "problem": ["解方程2x+5=15"],
#             "answer": ["x=5"],
#             "grade": ["七年级上册"]
#         }

#         rewards = difficulty_reward([[[{"content": "x=5"}]]], **test_data)
#         self.assertTrue(0 < rewards[0] < 1, "列表响应应成功解析并计算奖励")

#     @patch('requests.post')
#     def test_error_handling(self, mock_post):
#         """测试异常流程：连续3次API失败"""
#         mock_post.side_effect = Exception("API服务不可用")

#         with self.assertLogs(level='ERROR') as log_context:
#             rewards = difficulty_reward(
#                 [[[{"content": "测试答案"}]]],
#                 problem=["测试问题"],
#                 answer=["测试答案"],
#                 grade=["六年级上册"]
#             )

#         self.assertEqual(rewards, [0.0], "API完全失败时应返回0值")
#         self.assertIn("超过最大重试次数", str(log_context.output))

#     def test_grade_conversion_logic(self):
#         """验证年级字符串到数值的转换准确性"""
#         test_cases = [
#             ("一年级上册", 1.0),
#             ("三年级下册", 3.5),
#             ("九年级上册", 9.0),
#             ("无效格式", 0.0)
#         ]

#         calculator = MathDifficulty()
#         for input_str, expected in test_cases:
#             with self.subTest(input_str=input_str):
#                 self.assertEqual(calculator.convert_grade(input_str), expected,
#                                f"{input_str} 转换错误")

#     @patch('requests.post')
#     def test_multi_level_selection(self, mock_post):
#         """测试三级分类选择的完整流程"""
#         # 模拟三级选择响应
#         select_responses = [
#             MagicMock(json=lambda: {"text": "\\boxed{1}"}),  # 第一级选择
#             MagicMock(json=lambda: {"text": "\\boxed{2}"}),  # 第二级选择
#             MagicMock(json=lambda: {"text": "\\boxed{1}"}),  # 第三级选择
#         ]
#         # 知识点提取响应
#         mock_post.side_effect = [MagicMock(json=lambda: {"text": "# 几何图形"})] + select_responses

#         test_data = {
#             "problem": ["计算圆的面积"],
#             "answer": ["πr²"],
#             "grade": ["六年级下册"]
#         }

#         rewards = difficulty_reward([[[{"content": "πr²"}]]], **test_data)
#         self.assertGreater(rewards[0], 0, "完整三级选择应产生有效奖励")

# if __name__ == '__main__':
#     unittest.main(verbosity=2)
