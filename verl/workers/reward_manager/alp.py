# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from collections import defaultdict

import numpy as np
import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


def compute_token_ngram_repetition(
    tokens: list[int],
    n: int,
) -> float:
    if len(tokens) < n:
        return 0.0

    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if len(ngrams) == 0:
        return 0.0
    unique_ngrams = set(ngrams)
    repetition_rate = 1.0 - (len(unique_ngrams) / len(ngrams))
    return repetition_rate


def compute_token_ngram_topk_ratio(
    tokens: list[int],
    n: int,
    top_ks: list[int],
) -> dict[str, float]:
    if len(tokens) < n:
        return {f"topk{k}_{n}gram": 0.0 for k in top_ks}

    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    from collections import Counter

    ngram_counts = Counter(ngrams)
    total_ngrams = len(ngrams)

    # Get top k most common ngrams and their counts
    most_common = ngram_counts.most_common()

    result = {}
    for k in top_ks:
        if k <= len(most_common):
            topk_count = sum(count for _, count in most_common[:k])
            topk_ratio = topk_count / total_ngrams
        else:
            topk_ratio = 1.0  # If k exceeds unique ngrams, all ngrams are included
        result[f"topk{k}_{n}gram"] = topk_ratio

    return result


def analyze_response_repetition(
    token_ids: torch.Tensor,
    n_values: list[int],
    top_ks: list[int],
) -> dict:
    tokens = token_ids.tolist()  # torch.Tensor -> list[int]
    res = {}
    for n in n_values:
        res[f"repetition_{n}gram"] = compute_token_ngram_repetition(tokens, n)
        topk_metrics = compute_token_ngram_topk_ratio(
            tokens,
            n,
            top_ks,
        )
        res.update(topk_metrics)
    return res


@register("alp")
class ALPRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        alp_cfg=None,
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.max_resp_len = max_resp_len
        self.alp_cfg = alp_cfg
        assert self.alp_cfg is not None, "alp_cfg must be provided"
        self.alp_beta = self.alp_cfg.get("beta", 1e-8)
        # TODO: assert using with roo advantage estimator

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        scores_list = []
        uid2is_correct: dict[str, list[bool]] = defaultdict(list)
        valid_response_length_list: list[int] = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            valid_response_length_list.append(valid_response_length.item())

            n_values = [3, 5, 10]
            top_ks = [1, 5, 10]
            repetition_metrics = analyze_response_repetition(
                valid_response_ids,
                n_values,
                top_ks,
            )
            for key, value in repetition_metrics.items():
                reward_extra_info[key].append(value)

            reward_extra_info["prompt_length"].append(valid_prompt_length.item())
            reward_extra_info["response_token_length"].append(valid_response_length.item())
            reward_extra_info["total_token_length"].append((valid_prompt_length + valid_response_length).item())

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            result = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            score: float
            if isinstance(result, dict):
                score = result["score"]
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result
                reward_extra_info["acc"].append(score)

            scores_list.append(score)
            uid = data_item.non_tensor_batch.get("uid", None)
            assert uid is not None, "uid must be provided"
            uid2is_correct[uid].append(score == 1)
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        for i in range(len(data)):
            score = scores_list[i]
            data_item = data[i]  # DataProtoItem
            uid = data_item.non_tensor_batch.get("uid", None)
            assert uid is not None, "uid must be provided"
            pass_rate = float(np.mean(uid2is_correct[uid]))
            maxed_pass_rate = max(pass_rate, 1 / len(uid2is_correct[uid]))
            overlong_reward = -self.alp_beta * valid_response_length_list[i] * maxed_pass_rate
            reward_extra_info["overlong_reward"].append(overlong_reward)
            reward = score + overlong_reward
            reward_extra_info["total_reward"].append(reward)
            reward_tensor[i, valid_response_length_list[i] - 1] = reward

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
