import json
from typing import Any

from verl.utils.reward_score.deepcoder.utils import extract_code_from_model, lcb_check_correctness_v2


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
    is_correct, test_details = lcb_check_correctness_v2(
        tests,
        model_code,
        debug=False,
    )

    if is_correct:
        return {"score": 1.0, "acc": True, "pred": "[CORRECT]"}
    else:
        return {"score": 0.0, "acc": False, "pred": "[INCORRECT]"}
