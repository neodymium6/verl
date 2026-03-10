#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable

from experiment_paths import EXPERIMENT_PATHS
from tqdm import tqdm

try:
    import orjson as _orjson
except Exception:
    _orjson = None


def iter_jsonl(path: Path):
    loads = _orjson.loads if _orjson is not None else json.loads
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield loads(line)


@dataclass
class Record:
    experiment: str
    step: int
    input: str | None
    response_token_length: int | None


@dataclass(frozen=True)
class Settings:
    prompt_bs: int
    n_resp: int


@dataclass(frozen=True)
class StepTask:
    exp_name: str
    exp_path: str
    step: int
    settings: Settings


def _max_n_for_experiment(exp_name: str) -> int | None:
    if "Qwen" in exp_name:
        return 640
    if "DSD" in exp_name or "DS-" in exp_name:
        return 480
    return None


def process_step(
    task: StepTask,
) -> tuple[str, int, dict[str, dict[str, list[int] | list[bool]]] | None, int, str | None]:
    jsonl_path = Path(task.exp_path) / "train" / f"{task.step}.jsonl"
    if not jsonl_path.exists():
        raise RuntimeError(f"missing file: {jsonl_path}")

    batch_size = task.settings.prompt_bs * task.settings.n_resp
    count = 0
    prompts: dict[str, dict[str, list[int] | list[bool]]] = {}
    for obj in iter_jsonl(jsonl_path):
        prompt = obj.get("input")
        resp_len = obj.get("response_token_length")
        acc = obj.get("acc")
        if prompt is None or resp_len is None:
            raise RuntimeError(f"missing input/response_token_length: {jsonl_path}")
        if acc is None:
            raise RuntimeError(f"missing acc: {jsonl_path}")
        entry = prompts.setdefault(prompt, {"lengths": [], "accs": []})
        entry["lengths"].append(resp_len)
        entry["accs"].append(bool(acc))
        count += 1
    if count != batch_size:
        raise RuntimeError(
            f"invalid batch size: {jsonl_path} has {count}, expected {batch_size}"
        )
    if len(prompts) != task.settings.prompt_bs:
        raise RuntimeError(
            f"invalid prompt batch size: {jsonl_path} has {len(prompts)}, expected {task.settings.prompt_bs}"
        )
    for prompt, entry in prompts.items():
        lengths = entry["lengths"]
        if len(lengths) != task.settings.n_resp:
            raise RuntimeError(
                f"invalid responses per prompt: {jsonl_path} has {len(lengths)} for prompt={repr(prompt)}, expected {task.settings.n_resp}"
            )
    return task.exp_name, task.step, prompts, count, None


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        raise RuntimeError("mean on empty list")
    return sum(vals) / len(vals)


def variance(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        raise RuntimeError("variance on empty list")
    mu = mean(vals)
    return sum((x - mu) ** 2 for x in vals) / len(vals)


def main() -> int:
    settings = Settings(prompt_bs=64, n_resp=16)
    grouped: dict[str, dict[str, dict[str, dict[str, list[int] | list[bool]]]]] = {}
    tasks: list[StepTask] = []
    for exp_name, exp_path in EXPERIMENT_PATHS.items():
        max_n = _max_n_for_experiment(exp_name)
        if max_n is None:
            raise RuntimeError(f"unknown experiment type: {exp_name}")
        for step in range(1, max_n + 1):
            tasks.append(
                StepTask(exp_name=exp_name, exp_path=exp_path, step=step, settings=settings)
            )

    max_workers = min(len(tasks), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for exp_name, step, prompts, count, missing in tqdm(
            ex.map(process_step, tasks, chunksize=1),
            total=len(tasks),
            desc="steps",
        ):
            grouped.setdefault(exp_name, {})[str(step)] = prompts

    output_path = Path("scripts/analysis/output") / "length_var_by_prompt.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(grouped, f, ensure_ascii=True)
    print(f"saved\t{output_path}")

    stats: dict[str, dict[str, dict[str, float]]] = {}
    B = settings.prompt_bs
    G = settings.n_resp
    for exp_name, steps in grouped.items():
        for step_str, prompt_map in steps.items():
            all_vals: list[float] = []
            prompt_cvs: list[float] = []
            prompt_means: list[float] = []
            s_list: list[float] = []
            abs_log_r_sample_list: list[float] = []
            for prompt, entry in prompt_map.items():
                lengths = entry["lengths"]
                float_lengths = [float(x) for x in lengths]
                all_vals.extend(float_lengths)
                mu = mean(float_lengths)
                var = variance(float_lengths)
                if mu == 0:
                    raise RuntimeError(f"zero mean for prompt in {exp_name} step={step_str}")
                prompt_cvs.append((var ** 0.5) / mu)
                prompt_means.append(mu)
                s_j = sum(float_lengths)
                if s_j == 0:
                    raise RuntimeError(f"zero S_j for prompt in {exp_name} step={step_str}")
                s_list.append(s_j)
                for y in float_lengths:
                    r_s = (G * y) / s_j
                    abs_log_r_sample_list.append(abs(math.log(r_s)))
            t_total = sum(s_list)
            if t_total == 0:
                raise RuntimeError(f"zero T in {exp_name} step={step_str}")
            abs_log_r_token_list: list[float] = []
            target = t_total / B
            for s_j in s_list:
                r_t = target / s_j
                abs_log_r_token_list.append(abs(math.log(r_t)))
            abs_log_r_token_sample_list: list[float] = []
            target_ts = t_total / (B * G)
            for y in all_vals:
                r_ts = y / target_ts
                abs_log_r_token_sample_list.append(abs(math.log(r_ts)))
            stats.setdefault(exp_name, {})[step_str] = {
                "cv_all_outputs": (variance(all_vals) ** 0.5) / mean(all_vals),
                "mean_prompt_cv": sum(prompt_cvs) / len(prompt_cvs),
                "cv_prompt_mean": (variance(prompt_means) ** 0.5) / mean(prompt_means),
                "mean_abs_log_r_token": mean(abs_log_r_token_list),
                "max_abs_log_r_token": max(abs_log_r_token_list),
                "mean_abs_log_r_sample": mean(abs_log_r_sample_list),
                "max_abs_log_r_sample": max(abs_log_r_sample_list),
                "mean_abs_log_r_token_sample": mean(abs_log_r_token_sample_list),
                "max_abs_log_r_token_sample": max(abs_log_r_token_sample_list),
                "rel_prompt_token_minus_sample": mean(abs_log_r_token_list)
                - mean(abs_log_r_sample_list),
                "rel_prompt_token_ratio": mean(abs_log_r_token_list)
                / (mean(abs_log_r_token_list) + mean(abs_log_r_sample_list) + 1e-12),
            }

    stats_path = Path("scripts/analysis/output") / "length_var_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=True)
    print(f"saved\t{stats_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
