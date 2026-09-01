"""Train DM05 with sparse visual memory on episode-oriented JSONL data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import tyro

from opendm.constants.robot import ActionMode
from opendm.data.augmentations import (
    NoAugmentationPipeline,
    TrainingTransformPipeline,
)
from opendm.data.collator import TrainingCollator
from opendm.data.transforms import (
    ChatTokenization,
    LoadImages,
    Normalize,
    PadAction,
    Pipeline,
    PixelTransform,
)
from opendm.dataset.memory_dataset import (
    EpisodeJsonlMemoryDataset,
    LoadCurrentAndMemoryVideoFrames,
    LoadMemoryImages,
)
from opendm.exp.dm05_exp import DM05DataConfig as _DM05DataConfig
from opendm.exp.dm05_exp import DM05Exp as _DM05Exp
from opendm.exp.dm05_exp import DM05InferenceConfig as _DM05InferenceConfig
from opendm.exp.dm05_exp import DM05ModelConfig as _DM05ModelConfig
from opendm.exp.dm05_exp import DM05OptimizerConfig as _DM05OptimizerConfig
from opendm.exp.dm05_exp import DM05TrainerConfig as _DM05TrainerConfig


def _csv(value: str, field_name: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{field_name} must contain at least one value")
    return values


@dataclass
class DM05DataConfig(_DM05DataConfig):
    """Describe any DM05 JSONL dataset and its sparse visual memory."""

    dataset_name: str = field(default="memory_sft")
    jsonl_dir: str | None = field(default=None)
    image_dir: str | None = field(default=None)
    image_keys: str = field(default="images_1,images_2,images_3")
    image_prompts: str = field(default="Head,Left wrist,Right wrist")
    history_image_keys: str = field(default="images_1")
    history_frames: int = field(default=5)
    history_stride: int = field(default=16)
    max_history_images: int = field(default=5)
    left_pad_history: bool = field(default=False)
    augmentation_probability: float = field(default=0.5)
    video_backend: Literal["pyav", "torchcodec"] = field(default="pyav")
    video_decoder_cache_size: int = field(default=32)
    video_seek_mode: Literal["exact", "approximate"] = field(default="exact")
    video_decoder_threads: int = field(default=1)
    robot_type: str = field(default="generic")
    state_desc: str = field(default="")
    action_mode: ActionMode = field(default=ActionMode.RELATIVE)
    is_history: bool = field(default=True)

    def _dataset_info(self) -> dict:
        if not self.jsonl_dir:
            raise ValueError("--data-config.jsonl-dir is required")
        if not self.image_dir:
            raise ValueError("--data-config.image-dir is required")
        image_keys = _csv(self.image_keys, "image_keys")
        image_prompts = _csv(self.image_prompts, "image_prompts")
        if len(image_keys) != len(image_prompts):
            raise ValueError(
                "image_keys and image_prompts must have the same number of values"
            )
        state_desc = _csv(self.state_desc, "state_desc")
        if not 0.0 <= self.augmentation_probability <= 1.0:
            raise ValueError("augmentation_probability must be in [0, 1]")
        return {
            "jsonl_dir": self.jsonl_dir,
            "image_dir": self.image_dir,
            "image_keys": image_keys,
            "image_prompts": image_prompts,
            "history_image_keys": _csv(
                self.history_image_keys, "history_image_keys"
            ),
            "robot_type": self.robot_type,
            "state_desc": state_desc,
        }

    def build_dataset(
        self,
        processor,
        action_horizon: int,
        tokenizer_max_length: int = 1024,
    ) -> tuple:
        dataset_info = self._dataset_info()
        if self.video_backend == "torchcodec":
            image_transforms = [
                LoadCurrentAndMemoryVideoFrames(
                    current_image_keys=dataset_info["image_keys"],
                    history_image_keys=dataset_info["history_image_keys"],
                    image_dir=dataset_info["image_dir"],
                    frames=self.history_frames,
                    stride=self.history_stride,
                    max_history_images=self.max_history_images,
                    left_pad=self.left_pad_history,
                    decoder_cache_size=self.video_decoder_cache_size,
                    seek_mode=self.video_seek_mode,
                    decoder_threads=self.video_decoder_threads,
                )
            ]
        else:
            image_transforms = [
                LoadImages(
                    image_keys=dataset_info["image_keys"],
                    image_dir=dataset_info["image_dir"],
                ),
                LoadMemoryImages(
                    image_keys=dataset_info["history_image_keys"],
                    image_dir=dataset_info["image_dir"],
                    frames=self.history_frames,
                    stride=self.history_stride,
                    max_history_images=self.max_history_images,
                    left_pad=self.left_pad_history,
                ),
            ]
        pipeline = Pipeline(
            [
                self._action_transform(action_horizon),
                *image_transforms,
                PixelTransform(
                    transform_pipeline=(
                        NoAugmentationPipeline()
                        if self.augmentation_probability == 0.0
                        else TrainingTransformPipeline(
                            p=self.augmentation_probability
                        )
                    )
                ),
                Normalize(
                    norm_stats_path=str(self.norm_stats_path(action_horizon)),
                    norm_keys=["state", "action"],
                    use_quantiles=True,
                ),
                ChatTokenization(
                    processor=processor,
                    n_bins=self.n_bins,
                    max_length=tokenizer_max_length,
                    image_prompts=dataset_info["image_prompts"],
                    add_state=self.add_state,
                    is_history=self.is_history,
                ),
                PadAction(32),
            ]
        )
        dataset = EpisodeJsonlMemoryDataset(
            jsonl_dir=dataset_info["jsonl_dir"],
            transforms=pipeline,
            dataset_name=self.dataset_name,
            dataset_meta=self._dataset_meta(dataset_info),
        )
        collator = TrainingCollator(
            pad_token_id=processor.tokenizer.pad_token_id,
            max_length=tokenizer_max_length,
        )
        return dataset, collator

    def build_norm_stats_dataset(self, action_horizon: int):
        dataset_info = self._dataset_info()
        return EpisodeJsonlMemoryDataset(
            jsonl_dir=dataset_info["jsonl_dir"],
            transforms=Pipeline([self._action_transform(action_horizon)]),
            dataset_name=self.dataset_name,
            dataset_meta=self._dataset_meta(dataset_info),
        )


@dataclass
class DM05ModelConfig(_DM05ModelConfig):
    chunk_size: int = field(default=50)
    llm_attn_implementation: str = field(default="sdpa")
    vision_attn_implementation: str = field(default="sdpa")
    action_attn_implementation: str = field(default="sdpa")
    liger_kernel: bool = field(default=False)
    vlm_gradient_checkpointing: bool = field(default=True)
    ae_gradient_checkpointing: bool = field(default=True)


@dataclass
class DM05OptimizerConfig(_DM05OptimizerConfig):
    optim: Literal["adamw", "muon_adamw"] = field(default="muon_adamw")
    base_lr: float = field(default=2e-5)
    warmup_steps: int = field(default=1000)


@dataclass
class DM05TrainerConfig(_DM05TrainerConfig):
    output_dir: str = field(default="artifacts/checkpoints/dm05_memory_sft")
    fsdp1: bool | None = field(default=True)
    per_device_train_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=2)
    num_train_steps: int = field(default=30000)
    save_steps: int = field(default=5000)
    save_total_limit: int = field(default=2)
    dataloader_num_workers: int = field(default=4)
    model_max_length: int = field(default=1024)
    save_only_model: bool = field(default=False)
    seed: int = field(default=42)


@dataclass
class DM05InferenceConfig(_DM05InferenceConfig):
    output_action_dim: int = field(default=14)


@dataclass
class DM05Exp(_DM05Exp):
    use_lora: bool | None = field(default=False)
    model_config: DM05ModelConfig = field(default_factory=DM05ModelConfig)
    optimizer_config: DM05OptimizerConfig = field(default_factory=DM05OptimizerConfig)
    trainer_config: DM05TrainerConfig = field(default_factory=DM05TrainerConfig)
    data_config: DM05DataConfig = field(default_factory=DM05DataConfig)
    inference_config: DM05InferenceConfig = field(default_factory=DM05InferenceConfig)


def main() -> None:
    experiment = tyro.cli(DM05Exp)
    if experiment.task == "train":
        experiment.train()
    elif experiment.task == "inference":
        experiment.inference()
    else:
        raise ValueError(f"Invalid task: {experiment.task}")


if __name__ == "__main__":
    main()
