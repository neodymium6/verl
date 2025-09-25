from pathlib import Path

import datasets
import pandas as pd


def prepare_dsr(
    save_path: Path,
) -> pd.DataFrame:
    data_source = "deepscaler"
    dataset = datasets.load_dataset(
        "agentica-org/DeepScaleR-Preview-Dataset",
        split="train",
    )

    def process_fn(example, idx):
        problem = example.pop("problem")
        answer = example.pop("answer")
        solution = example.pop("solution")
        instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": f"{problem} {instruction_following}",
                }
            ],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer,
            },
            "extra_info": {
                "split": "train",
                "index": str(idx),
                "solution": solution,
                "problem": problem,
            },
        }
        return data

    dataset = dataset.map(function=process_fn, with_indices=True)
    save_path.mkdir(parents=True, exist_ok=True)
    assert type(dataset) is datasets.Dataset, "Dataset should be of type datasets.Dataset"
    dataset.to_parquet(save_path / "train.parquet")
    df = dataset.to_pandas()
    assert type(df) is pd.DataFrame, "Dataset should be converted to pandas DataFrame"

    return df


if __name__ == "__main__":
    local_dir = Path("./data/dsr")
    df = prepare_dsr(
        local_dir,
    )

    print(df.head())
    df.info()
