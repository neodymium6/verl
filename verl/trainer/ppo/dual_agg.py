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

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch


def _compute_scale_factor_stats(
    scale_factors: np.ndarray,
    mask: np.ndarray,
    clip_factor: float | None = None,
) -> dict[str, float]:
    scale_factors_filtered = scale_factors[mask]
    count = len(scale_factors_filtered)

    if count == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "q25": 0.0,
            "q50": 0.0,
            "q75": 0.0,
            "up_clipped_ratio": 0.0,
            "down_clipped_ratio": 0.0,
            "clipped_ratio": 0.0,
        }

    stats = {
        "count": int(count),
        "mean": float(np.mean(scale_factors_filtered)),
        "std": float(np.std(scale_factors_filtered)),
        "min": float(np.min(scale_factors_filtered)),
        "max": float(np.max(scale_factors_filtered)),
        "q25": float(np.quantile(scale_factors_filtered, 0.25)),
        "q50": float(np.quantile(scale_factors_filtered, 0.50)),
        "q75": float(np.quantile(scale_factors_filtered, 0.75)),
    }

    if clip_factor is not None:
        up_clipped = np.sum(scale_factors_filtered > clip_factor)
        down_clipped = np.sum(scale_factors_filtered < 1 / clip_factor)
        stats.update(
            {
                "up_clipped_ratio": float(up_clipped / count),
                "down_clipped_ratio": float(down_clipped / count),
                "clipped_ratio": float((up_clipped + down_clipped) / count),
            }
        )
    else:
        stats.update(
            {
                "up_clipped_ratio": 0.0,
                "down_clipped_ratio": 0.0,
                "clipped_ratio": 0.0,
            }
        )

    return stats


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def _g_rloo_lp(lengths: np.ndarray, custom_adv_config: dict[str, Any], **_kwargs: Any) -> np.ndarray:
    alpha = float(custom_adv_config.get("dual_agg_rloo_alpha", 0.1))
    mu = float(np.mean(lengths))
    sigma = float(np.std(lengths))
    eps = 1e-6
    if not np.isfinite(sigma) or sigma < eps:
        sigma = eps
    z = (lengths - mu) / sigma
    return 1.0 - alpha * _sigmoid(z)


def _g_alp_linear(
    lengths: np.ndarray,
    custom_adv_config: dict[str, Any],
    **kwargs: Any,
) -> np.ndarray:
    acc = kwargs.get("acc")
    group_size = kwargs.get("group_size")
    if acc is None or group_size is None:
        raise ValueError("alp_linear requires acc and group_size")
    beta = float(custom_adv_config.get("dual_agg_alp_beta", 1e-3))
    eps = float(custom_adv_config.get("dual_agg_alp_eps", 0.05))
    s = max(float(acc), 1.0 / max(int(group_size), 1))
    g = 1.0 - beta * lengths * s
    return np.maximum(g, eps).astype(np.float32)


def _compute_g(
    lengths: np.ndarray,
    g_type: str | None,
    custom_adv_config: dict[str, Any],
    **kwargs: Any,
) -> np.ndarray:
    if g_type is None:
        raise ValueError("dual_agg_g_type must be set for general dual-agg")
    g_funcs = {
        "rloo_lp": _g_rloo_lp,
        "alp_linear": _g_alp_linear,
    }
    g_func = g_funcs.get(g_type)
    if g_func is None:
        raise ValueError(f"unsupported dual_agg_g_type: {g_type}")
    return g_func(lengths, custom_adv_config, **kwargs)


def _apply_scale_factors(
    scores: torch.Tensor,
    raw_scale_factors: np.ndarray,
    clip_factor: float | None,
) -> torch.Tensor:
    for i in range(scores.shape[0]):
        scale_factor = raw_scale_factors[i]
        if clip_factor is not None:
            scale_factor = float(np.clip(scale_factor, 1 / clip_factor, clip_factor))
        scores[i] = scores[i] * scale_factor
    return scores


def _update_dual_agg_stats(
    metrics: dict[str, Any],
    raw_scale_factors: np.ndarray,
    is_pos_score: np.ndarray,
    clip_factor: float | None,
) -> None:
    all_mask = np.ones_like(is_pos_score, dtype=bool)
    metrics.update(
        {
            f"dual_agg/{k}": v
            for k, v in _compute_scale_factor_stats(
                raw_scale_factors,
                all_mask,
                clip_factor=clip_factor,
            ).items()
        }
    )
    metrics.update(
        {
            f"dual_agg/pos_{k}": v
            for k, v in _compute_scale_factor_stats(
                raw_scale_factors,
                is_pos_score,
                clip_factor=clip_factor,
            ).items()
        }
    )
    metrics.update(
        {
            f"dual_agg/neg_{k}": v
            for k, v in _compute_scale_factor_stats(
                raw_scale_factors,
                ~is_pos_score,
                clip_factor=clip_factor,
            ).items()
        }
    )


def apply_dual_agg_general(
    *,
    scores: torch.Tensor,
    response_mask: torch.Tensor,
    index,
    custom_adv_config: dict[str, Any],
    metrics: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply generalized DualAgg reweighting to per-sequence scores.

    Current implementation supports the RLOO-LP style weighting on correct samples only.
    """
    if custom_adv_config.get("dual_agg_batch_mean", False):
        raise NotImplementedError("dual_agg_batch_mean is not supported in general dual-agg")

    g_type = custom_adv_config.get("dual_agg_g_type", None)
    if g_type is None:
        raise ValueError("dual_agg_g_type must be set for general dual-agg")
    if g_type not in ("rloo_lp", "alp_linear"):
        raise ValueError(f"unsupported dual_agg_g_type: {g_type}")

    clip_factor = custom_adv_config.get("dual_agg_clip_factor", None)
    bsz = scores.shape[0]
    valid_length = response_mask.sum(dim=1).detach().cpu().numpy().astype(np.float32)
    is_pos_score = (scores > 0).detach().cpu().numpy()

    id2indices: dict[Any, list[int]] = defaultdict(list)
    for i in range(bsz):
        id2indices[index[i]].append(i)

    raw_scale_factors = np.ones((bsz,), dtype=np.float32)

    for _, indices in id2indices.items():
        pos_indices = [i for i in indices if is_pos_score[i]]
        if len(pos_indices) == 0:
            continue

        pos_lengths = valid_length[pos_indices]
        pos_len_sum = float(np.sum(pos_lengths))
        if pos_len_sum <= 0.0 or not np.isfinite(pos_len_sum):
            continue

        acc = float(len(pos_indices) / max(len(indices), 1))
        g = _compute_g(
            pos_lengths,
            g_type,
            custom_adv_config,
            acc=acc,
            group_size=len(indices),
        )
        if not np.all(np.isfinite(g)):
            continue

        g_len_sum = float(np.sum(g * pos_lengths))
        if g_len_sum <= 0.0 or not np.isfinite(g_len_sum):
            continue

        z_norm = g_len_sum / pos_len_sum
        if z_norm <= 0.0 or not np.isfinite(z_norm):
            continue

        for pos_i, sample_idx in enumerate(pos_indices):
            raw_scale_factors[sample_idx] = float(g[pos_i] / z_norm)

    scores = _apply_scale_factors(scores, raw_scale_factors, clip_factor)
    _update_dual_agg_stats(metrics, raw_scale_factors, is_pos_score, clip_factor)

    return scores, metrics
