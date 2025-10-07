import argparse
from pathlib import Path

import datasets
import pandas as pd


def prepare_dapo(
    save_path: Path,
    dedup: bool,
    boxed: bool,
) -> pd.DataFrame:
    dataset = datasets.load_dataset("BytedTsinghua-SIA/AIME-2024", split="train")
    dataset = dataset.map(
        lambda x: {
            "prompt_for_dedup": x["prompt"][0]["content"],
        },
    )
    assert type(dataset) is datasets.Dataset, "Dataset should be of type datasets.Dataset"
    df = dataset.to_pandas()
    assert type(df) is pd.DataFrame, "Dataset should be converted to pandas DataFrame"
    if dedup:
        df = df.drop_duplicates("prompt_for_dedup")
    df = df.reset_index(drop=True)
    df = df.drop(columns=["prompt_for_dedup"])
    if boxed:
        df["data_source"] = df["data_source"] + "_boxed"
        df = build_boxed_prompt(df)

    save_path.mkdir(parents=True, exist_ok=True)
    file_name = "aime-2024"
    if dedup:
        file_name += "_dedup"
    if boxed:
        file_name += "_boxed"
    df.to_parquet(save_path / (file_name + ".parquet"))
    return df


def build_boxed_prompt(
    df: pd.DataFrame,
) -> pd.DataFrame:
    def process_data(prompt_list):
        full_prompt = prompt_list[0]["content"]
        lines = full_prompt.split("\n")
        problem_lines = lines[2:-2]
        problem = "\n".join(problem_lines).strip()
        instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
        new_prompt = f"{problem} {instruction_following}"
        return [{"role": "user", "content": new_prompt}]

    df["prompt"] = df["prompt"].apply(process_data)
    for idx in range(len(df)):
        prompt = df.loc[idx, "prompt"][0]["content"]
        problem = prompt.split(" Let's think step by step")[0]
        raw_problem = df.loc[idx, "extra_info"]["raw_problem"]
        assert problem == raw_problem, f"Problem mismatch at index {idx}"
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dedup", action="store_true", help="Whether to deduplicate the dataset")
    parser.add_argument("--boxed", action="store_true", help="Whether to use boxed version")
    args = parser.parse_args()
    local_dir = Path("./data/dapo")
    df = prepare_dapo(
        local_dir,
        args.dedup,
        args.boxed,
    )
    print(df.head())
    df.info()
