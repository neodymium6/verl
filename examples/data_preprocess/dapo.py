import argparse
from pathlib import Path

import datasets
import pandas as pd


def prepare_dapo(
    save_path: Path,
    dedup: bool,
    rename_to_dapo01: bool,
) -> pd.DataFrame:
    dataset = datasets.load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
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
    df = df.drop(columns=["prompt_for_dedup"])
    if rename_to_dapo01:
        df["data_source"] = df["data_source"] + "01"
    df = df.reset_index(drop=True)

    save_path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(save_path / ("dapo-math-17k_dedup.parquet" if dedup else "dapo-math-17k.parquet"))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dedup", action="store_true", help="Whether to deduplicate the dataset")
    parser.add_argument("--r01", action="store_true", help="Whether to rename the dataset to dapo01")
    args = parser.parse_args()
    local_dir = Path("./data/dapo01" if args.r01 else "./data/dapo")
    df = prepare_dapo(
        local_dir,
        args.dedup,
        args.r01,
    )
    print(df.head())
    df.info()
