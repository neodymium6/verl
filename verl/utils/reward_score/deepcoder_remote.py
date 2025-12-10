import json
from typing import Any

import requests

from verl.utils.reward_score.deepcoder.utils import extract_code_from_model


def compute_score(
    solution_str: str,
    ground_truth: Any,
) -> dict[str, Any]:
    model_response = solution_str
    tests = ground_truth

    if tests is None:
        print("No tests found in task_info")
        return {"score": 0.0, "acc": False, "pred": "[NO TESTS]"}

    model_code = extract_code_from_model(model_response)
    if model_code is None:
        return {"score": 0.0, "acc": False, "pred": "[NO CODE]"}

    is_correct = False

    # Handle case where tests is a JSON string
    if isinstance(tests, str):
        tests = json.loads(tests)
    is_correct, _test_details = lcb_check_correctness_remote(
        tests,
        model_code,
        timeout=5,
    )

    if is_correct:
        return {"score": 1.0, "acc": True, "pred": "[CORRECT]"}
    else:
        return {"score": 0.0, "acc": False, "pred": "[INCORRECT]"}


def lcb_check_correctness_remote(sample, generation, timeout=5):
    try:
        r = requests.post(
            "http://localhost:12244/check_lcb",
            json={
                "sample": sample,
                "generation": generation,
                "timeout": timeout,
                "debug": False,
            },
            timeout=timeout + 10,
        )
    except requests.RequestException as e:
        print(f"Request to judge server failed: {e}")
        return False, {"error": f"request_failed: {e}"}

    if r.status_code != 200:
        print(f"Judge server returned error: {r.status_code}, {r.text}")
        return False, {
            "error": f"judge_http_error: {r.status_code}",
            "detail": r.text,
        }

    data = r.json()
    return data["all_passed"], data["detailed_results"]
