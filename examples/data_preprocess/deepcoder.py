import json
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset
from tqdm import tqdm

LCB_SYSTEM_MESSAGE_GENERIC = "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."

LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."

LCB_FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."


def fetch_live_code_bench_system_prompt(prompt: str, starter_code: str | None = None):
    # https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/prompts/code_generation.py
    prompt = LCB_SYSTEM_MESSAGE_GENERIC + "\n\n" + prompt
    if starter_code:
        prompt += f"### Format: {LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {LCB_FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def load_deepcoder_dataset() -> tuple[Dataset, Dataset]:
    primeintellect_train = load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="primeintellect", split="train")
    taco_train = load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="taco", split="train")
    lcbv5_train = load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="lcbv5", split="train")
    assert isinstance(primeintellect_train, Dataset)
    assert isinstance(taco_train, Dataset)
    assert isinstance(lcbv5_train, Dataset)
    train_dataset = concatenate_datasets([primeintellect_train, taco_train, lcbv5_train])
    codeforces_test = load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="codeforces", split="test")
    lcbv5_test = load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="lcbv5", split="test")
    assert isinstance(codeforces_test, Dataset)
    assert isinstance(lcbv5_test, Dataset)
    test_dataset = concatenate_datasets([codeforces_test, lcbv5_test])
    return train_dataset, test_dataset


def prepare_deepcoder_data(
    train_size: int | None = None,
    test_size: int | None = None,
):
    train_dataset, test_dataset = load_deepcoder_dataset()

    def preprocess_fn(example, idx):
        starter_code = example.get("starter_code", "")
        question = fetch_live_code_bench_system_prompt(example["problem"], starter_code if starter_code else None)

        tests_raw = example["tests"]
        # Handle different test formats
        if isinstance(tests_raw, str):
            tests = json.loads(tests_raw)
        else:
            tests = tests_raw
        metadata = example.get("metadata", {})

        # Convert TACO format to standard format
        if isinstance(tests, dict) and "inputs" in tests and "outputs" in tests:
            normalized_tests = []
            for input_val, output_val in zip(tests["inputs"], tests["outputs"], strict=False):
                normalized_tests.append({"input": input_val, "output": output_val, "testtype": "stdin_stdout"})
            tests = normalized_tests

        # Ensure tests is always a list
        if not isinstance(tests, list):
            tests = [tests] if tests else []

        for test in tests:
            if test.get("testtype") == "functional" and metadata.get("func_name") is not None:
                test["metadata"] = {"func_name": str(metadata["func_name"])}
            else:
                test["metadata"] = {"func_name": None}

        data = {
            "data_source": "livecodebench_dc",
            "prompt": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "ability": "code",
            "reward_model": {"style": "rule", "ground_truth": json.dumps(tests)},
            "extra_info": {
                "uid": f"deepcoder_{idx}",
            },
        }
        return {"data": json.dumps(data)}

    def map_fn(
        dataset: Dataset,
    ) -> Dataset:
        processed_examples = []
        import signal

        def handler(_signum, _frame):
            raise TimeoutError("Preprocessing timed out")

        for i, example in tqdm(enumerate(dataset), total=len(dataset), desc="Preprocessing"):
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(10)  # 10 seconds timeout per example
            try:
                processed_example = preprocess_fn(example, i)
            except TimeoutError:
                print(f"  - Timeout processing example {i}, skipping.")
                print(example)
                continue
            finally:
                signal.alarm(0)  # Disable the alarm
            processed_examples.append(processed_example)
        del dataset
        return Dataset.from_list(processed_examples)

    if train_size:
        train_dataset = train_dataset.select(range(min(train_size, len(train_dataset))))
    if test_size:
        test_dataset = test_dataset.select(range(min(test_size, len(test_dataset))))

    train_dataset = map_fn(train_dataset)
    test_dataset = map_fn(test_dataset)

    save_path = Path("./data/deepcoder")
    save_path.mkdir(parents=True, exist_ok=True)
    chunk_size = 10000
    for split_name, dataset in [("train", train_dataset), ("test", test_dataset)]:
        num_chunks = (len(dataset) + chunk_size - 1) // chunk_size
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(dataset))
            chunk = dataset.select(range(start_idx, end_idx))
            chunk.to_parquet(save_path / f"{split_name}_part{chunk_idx}.parquet")

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_deepcoder_data()
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print("Sample train example:")
    import pprint

    pprint.pprint(train_dataset[0])
