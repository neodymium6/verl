import os
import sys

import tqdm

from verl.model_merger.base_model_merger import ModelMergerConfig


class SuppressOutput:
    """Context manager to suppress stdout, stderr, and warnings."""

    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._devnull = open(os.devnull, "w")
        sys.stdout = self._devnull
        sys.stderr = self._devnull
        # warnings.filterwarnings("ignore")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        self._devnull.close()
        # warnings.filterwarnings("default")


def merge(
    backend: str,
    local_dir: str,
    target_dir: str = "tmp",
    tie_word_embedding: bool = False,
    trust_remote_code: bool = False,
    is_value_model: bool = False,
    use_cpu_initialization: bool = False,
    hf_upload_path: str | None = None,
    private: bool = False,
) -> None:
    """Merge model checkpoints to HuggingFace format."""
    config = ModelMergerConfig(
        operation="merge",
        backend=backend,
        local_dir=local_dir,
        target_dir=target_dir,
        tie_word_embedding=tie_word_embedding,
        trust_remote_code=trust_remote_code,
        is_value_model=is_value_model,
        use_cpu_initialization=use_cpu_initialization,
        hf_upload_path=hf_upload_path,
        private=private,
        hf_model_config_path=os.path.join(local_dir, "huggingface"),
    )

    assert config.target_dir is not None
    os.makedirs(config.target_dir, exist_ok=True)
    print(f"config: {config}")

    if config.backend == "fsdp":
        from verl.model_merger.fsdp_model_merger import FSDPModelMerger

        merger = FSDPModelMerger(config)
    elif config.backend == "megatron":
        from verl.model_merger.megatron_model_merger import MegatronModelMerger

        merger = MegatronModelMerger(config)
    else:
        raise NotImplementedError(f"Unknown backend: {config.backend}")

    merger.merge_and_save()
    merger.cleanup()


def get_checkpoint_dir(
    checkpoints_base_dir: str,
    project_name: str,
    exp_name: str,
    step: int | None = None,
) -> tuple[str, int]:
    """Get checkpoint actor directory.

    Returns:
        tuple of (actor_dir_path, step_number)
    """
    exp_dir = os.path.join(checkpoints_base_dir, project_name, exp_name)

    if not os.path.exists(exp_dir):
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    if step is None:
        latest_file = os.path.join(exp_dir, "latest_checkpointed_iteration.txt")
        if not os.path.exists(latest_file):
            raise FileNotFoundError(f"latest_checkpointed_iteration.txt not found in {exp_dir}")

        with open(latest_file) as f:
            step = int(f.read().strip())

    actor_dir = os.path.join(exp_dir, f"global_step_{step}", "actor")

    if not os.path.exists(actor_dir):
        raise FileNotFoundError(f"Actor directory not found: {actor_dir}")

    return actor_dir, step


def merge_latest_batch(
    checkpoints_base_dir: str,
    experiments: list[tuple[str, str, int | None]],
    backend: str = "fsdp",
    output_base_dir: str = "merged_models",
    tie_word_embedding: bool = False,
    trust_remote_code: bool = False,
    is_value_model: bool = False,
    use_cpu_initialization: bool = False,
    verbose: bool = False,
) -> None:
    """Merge checkpoints for multiple experiments.

    Args:
        checkpoints_base_dir: Base directory (e.g., "checkpoints")
        experiments: List of tuples (project_name, exp_name, step)
            e.g., [("DAPO_la", "DS-1.5B_fp16", None), ("DAPO_la", "Qwen3-1.7B-Base", 800)]
            step=None uses latest_checkpointed_iteration.txt
        backend: Backend type ('fsdp' or 'megatron')
        output_base_dir: Base directory for merged models
        verbose: If False, suppress stdout during merge
    """
    pbar = tqdm.tqdm(experiments)
    for project_name, exp_name, step in pbar:
        pbar.set_description(f"Merging {project_name}/{exp_name}" + (f" step={step}" if step else " (latest)"))

        try:
            local_dir, actual_step = get_checkpoint_dir(checkpoints_base_dir, project_name, exp_name, step)
            target_dir = os.path.join(output_base_dir, project_name, exp_name, f"step_{actual_step}")

            if verbose:
                merge(
                    backend=backend,
                    local_dir=local_dir,
                    target_dir=target_dir,
                    tie_word_embedding=tie_word_embedding,
                    trust_remote_code=trust_remote_code,
                    is_value_model=is_value_model,
                    use_cpu_initialization=use_cpu_initialization,
                )
            else:
                with SuppressOutput():
                    merge(
                        backend=backend,
                        local_dir=local_dir,
                        target_dir=target_dir,
                        tie_word_embedding=tie_word_embedding,
                        trust_remote_code=trust_remote_code,
                        is_value_model=is_value_model,
                        use_cpu_initialization=use_cpu_initialization,
                    )

            # print(f"Successfully merged to {target_dir}")
            pbar.write(f"Successfully merged {project_name}/{exp_name} to {target_dir}")

        except Exception as e:
            # print(f"Failed to merge {project_name}/{exp_name}: {e}")
            pbar.write(f"Failed to merge {project_name}/{exp_name}: {e}")
            continue


def main():
    """Example main function to merge multiple experiments."""
    # checkpoints_base_dir = "checkpoints"
    checkpoints_base_dir = "/home/9/um05219/bs-nii-llm/verl_checkpoints"
    output_base_dir = "/home/9/um05219/bs-nii-llm/verl_converted_checkpoints"
    # experiments: list[tuple[str, str, int | None]] = [
    #     ("DAPO_la", "DSD-1.5B-DAPO", None),
    #     ("DAPO_la", "DSD-1.5B-GRPO", None),
    #     ("DAPO_la", "DSD-1.5B-PromptAvg", None),
    #     ("DAPO_la", "DSD-1.5B-DualAgg", None),
    #     ("DAPO_la", "DSD-1.5B-DualAggAdaptive", None),
    #     ("DAPO_la", "DSD-1.5B-DualAgg1.5", None),
    #     ("DAPO_la", "DSD-1.5B-DAPO-ROOLP0.2", None),
    # ]
    # same step 480
    # experiments: list[tuple[str, str, int | None]] = [
    #     ("DAPO_la", "DSD-1.5B-DAPO", 480),
    #     # ("DAPO_la", "DSD-1.5B-GRPO", None),
    #     # ("DAPO_la", "DSD-1.5B-PromptAvg", None),
    #     ("DAPO_la", "DSD-1.5B-DualAgg", 480),
    #     ("DAPO_la", "DSD-1.5B-DualAggAdaptive", 480),
    #     ("DAPO_la", "DSD-1.5B-DualAgg1.5", 480),
    #     ("DAPO_la", "DSD-1.5B-DAPO-ROOLP0.2", 480),
    # ]
    # experiments: list[tuple[str, str, int | None]] = [
    #     ("DAPO_la", "DSD-1.5B-DAPO-ALP1e-4", None),
    #     ("DAPO_la", "DSD-1.5B-DAPO-ALP1e-4", 480),
    # ]
    # experiments: list[tuple[str, str, int | None]] = [
    #     ("DAPO_la", "Q3B-1.7B-DAPO-ALP1e-4", None),
    #     ("DAPO_la", "Q3B-1.7B-DAPO-ALP1e-4", 640),
    # ]
    # experiments: list[tuple[str, str, int | None]] = [
    #     ("DAPO_la", "Q3B-1.7B-DAPO-ROOLP0.2", None),
    #     ("DAPO_la", "Q3B-1.7B-DAPO-ROOLP0.2", 640),
    #     ("DAPO_la", "Q3B-1.7B-DRPO0.1-fixed", None),
    #     ("DAPO_la", "Q3B-1.7B-DRPO0.1-fixed", 640),
    #     ("DAPO_la", "Qwen3-1.7B-Base_da2.0", None),
    #     ("DAPO_la", "Qwen3-1.7B-Base_da2.0", 640),
    #     ("DAPO_la", "Qwen3-1.7B-Base_da4.0", None),
    #     ("DAPO_la", "Qwen3-1.7B-Base_da4.0", 640),
    #     ("DAPO_la", "Qwen3-1.7B-Base_da8.0", None),
    #     ("DAPO_la", "Qwen3-1.7B-Base_da8.0", 640),
    # ]
    # experiments: list[tuple[str, str, int | None]] = [
    # ("DAPO_la", "DSD-1.5B-DAPO-ALP1e-4-v2", None),
    # ("DAPO_la", "DSD-1.5B-DAPO-ALP1e-4-v2", 480),
    # ("DAPO_la", "Q3B-1.7B-DRPO0.5-fixed", None),
    # ("DAPO_la", "Q3B-1.7B-DRPO0.5-fixed", 640),
    # ("DAPO_la", "Q3B-1.7B-DAPO-ROOLP0.1", None),
    # ("DAPO_la", "Q3B-1.7B-DAPO-ROOLP0.1", 640),
    # ("DAPO_la", "Q3B-1.7B-DAPO-ROOLP0.4", None),
    # ("DAPO_la", "Q3B-1.7B-DAPO-ROOLP0.4", 640),
    # ("DAPO_la", "DSD-1.5B-DualAgg2.0", None),
    # ("DAPO_la", "DSD-1.5B-DualAgg2.0", 480),
    # ("DAPO_la", "DSD-1.5B-DAPO-ALP1e-6", None),
    # ("DAPO_la", "DSD-1.5B-DAPO-ALP1e-6", 480),
    # ("DAPO_la", "DSD-1.5B-DAPO-ALP3e-4", None),
    # ("DAPO_la", "DSD-1.5B-DAPO-ALP3e-4", 480),
    # ("DAPO_la", "DSD-1.5B-DRPO0.05-fixed", None),
    # ("DAPO_la", "DSD-1.5B-DRPO0.05-fixed", 480),
    # ("DAPO_la", "DSD-1.5B-DRPO0.02-fixed", None),
    # ("DAPO_la", "DSD-1.5B-DRPO0.02-fixed", 480),
    # ("DAPO_la", "DSD-1.5B-DAPO-ROOLP1.0", 480),
    # ("DAPO_la", "DSD-1.5B-DAPO-ROOLP1.0", None),
    # ("DAPO_la", "DSD-1.5B-DRPO0.1-fixed", None),
    # ("DAPO_la", "DSD-1.5B-DRPO0.1-fixed", 480),
    # ("DAPO_la", "DSD-1.5B-DRPO0.2-fixed", None),
    # ("DAPO_la", "DSD-1.5B-DRPO0.2-fixed", 480),
    # ("DAPO_la", "DSD-1.5B-DRPO0.5-fixed", None),
    # ("DAPO_la", "DSD-1.5B-DRPO0.5-fixed", 480),
    # ("DAPO_la", "Q3B-1.7B-DAPO-ALP1e-5", None),
    # ("DAPO_la", "Q3B-1.7B-DAPO-ALP1e-5", 640),
    # ("DAPO_la", "Q3B-1.7B-DAPO-ALP1e-6", None),
    # ("DAPO_la", "Q3B-1.7B-DAPO-ALP1e-6", 640),
    # ("DAPO_la", "Q3B-1.7B-DRPO0.2-fixed", None),
    # ("DAPO_la", "Q3B-1.7B-DRPO0.2-fixed", 640),
    # ("DAPO_la", "DSD-1.5B-DAPO-ALP1e-5", None),
    # ("DAPO_la", "DSD-1.5B-DAPO-ALP1e-5", 480),
    # ("DAPO_la", "DSD-1.5B-DAPO-ROOLP0.1", 480),
    # ("DAPO_la", "DSD-1.5B-DAPO-ROOLP0.1", None),
    # ("DAPO_la", "DSD-1.5B-DAPO-ROOLP0.4", 480),
    # ("DAPO_la", "DSD-1.5B-DAPO-ROOLP0.4", None),
    # ("DAPO_la", "DSD-1.5B-DAPO-ALP1e-4-8k", None),
    # ("DAPO_la", "DSD-1.5B-DAPO-8k", None),
    # ("DAPO_la", "DSD-1.5B-DAPO-8k", 480),
    # ("DAPO_la", "DSD-1.5B-DualAggAdaptive-8k", None),
    # ("DAPO_la", "DSD-1.5B-DualAggAdaptive-8k", 480),
    # ("DAPO_la", "DSD-1.5B-DAPO-ALP1e-4-8k", 480),
    # ("DAPO_la", "DSD-1.5B-DAPO-ALP1e-5-8k", 480),
    # ("DAPO_la", "DSD-1.5B-DAPO-ALP1e-6-8k", 480),
    # ("DAPO_la", "DSD-1.5B-DAPO-ALP1e-5-8k", None),
    # ("DAPO_la", "DSD-1.5B-DAPO-ALP1e-6-8k", None),
    # ("DAPO_la", "DSD-1.5B-DAPO-ROOLP0.2-8k", 480),
    # ("DAPO_la", "DSD-1.5B-DAPO-ROOLP0.2-8k", None),
    # ("DAPO_la", "DSD-1.5B-DAPO-8k", 480),
    # ("DAPO_la", "DSD-1.5B-DAPO-ALP1e-3", 480),
    # ("DAPO_la", "DSD-1.5B-DAPO-ALP1e-3", None),
    # ("DAPO_la", "DSD-1.5B-DAPO-ROOLP0.8", 480),
    # ("DAPO_la", "DSD-1.5B-DAPO-ROOLP0.8", None),
    # ("DAPO_la", "DSD-1.5B-DRPO0.1-fixed-8k", None),
    # ("DAPO_la", "DSD-1.5B-DRPO0.1-fixed-8k", 480),
    # ("DAPO_la", "DSD-1.5B-DRPO0.2-fixed-8k", None),
    # ("DAPO_la", "DSD-1.5B-DRPO0.2-fixed-8k", 480),
    # ]

    # experiments: list[tuple[str, str, int | None]] = [
    #     ("DAPO_la", "Qwen3-1.7B-Base_64", 640),
    #     ("DAPO_la", "Qwen3-1.7B-Base_sm2tm", 720),
    #     ("DAPO_la", "Qwen3-1.7B-Base_pm2", None),
    #     ("DAPO_la", "Qwen3-1.7B-Base_DAA", None),
    #     ("DAPO_la", "Qwen3-1.7B-Base_da1.5", 1040),
    # ]
    # same step 640
    # experiments: list[tuple[str, str, int | None]] = [
    #     # ("DAPO_la", "Qwen3-1.7B-Base_64", 640),
    #     ("DAPO_la", "Qwen3-1.7B-Base_sm2tm", 640),
    #     # ("DAPO_la", "Qwen3-1.7B-Base_pm2", None),
    #     ("DAPO_la", "Qwen3-1.7B-Base_DAA", 640),
    #     ("DAPO_la", "Qwen3-1.7B-Base_da1.5", 640),
    # ]
    experiments: list[tuple[str, str, int | None]] = [
        ("DAPO_la", "Q3-1.7B-ALP1e-3", 180),
        ("DAPO_la", "Q3-1.7B-ALP1e-3", 240),
        ("DAPO_la", "Q3-1.7B-ALP1e-3", 320),
        ("DAPO_la", "Q3-1.7B-ALP3e-4", 180),
        ("DAPO_la", "Q3-1.7B-ALP3e-4", 240),
        ("DAPO_la", "Q3-1.7B-ALP3e-4", 320),
        ("DAPO_la", "Q3-1.7B-ALP1e-4", 180),
        ("DAPO_la", "Q3-1.7B-ALP1e-4", 240),
        ("DAPO_la", "Q3-1.7B-ALP1e-4", 320),
        ("DAPO_la", "Q3-1.7B-ALP1e-6", 180),
        ("DAPO_la", "Q3-1.7B-ALP1e-6", 240),
        ("DAPO_la", "Q3-1.7B-DAPO", 180),
        ("DAPO_la", "Q3-1.7B-DAPO", 240),
        ("DAPO_la", "Q3-1.7B-DAPO", 320),
        ("DAPO_la", "Q3-1.7B-GRPO", 180),
        ("DAPO_la", "Q3-1.7B-GRPO", 240),
    ]

    merge_latest_batch(
        checkpoints_base_dir=checkpoints_base_dir,
        output_base_dir=output_base_dir,
        experiments=experiments,
        backend="fsdp",
    )


if __name__ == "__main__":
    main()
