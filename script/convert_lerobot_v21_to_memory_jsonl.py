#!/usr/bin/env python3
"""Convert a local LeRobot v2.1 dataset to DM05 memory-training JSONL.

LeRobot v2.1 stores one Parquet file and one MP4 per camera and episode. This
converter keeps those videos in place and writes lightweight episode JSONL
files that reference the original MP4 files by paths relative to the LeRobot
dataset root.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "pyarrow is required. Install the OpenDM environment before conversion."
    ) from None


@dataclass(frozen=True)
class ConversionConfig:
    """User-selected field mapping for LeRobot v2.1 conversion."""

    input_root: Path
    output_dir: Path
    state_key: str = "observation.state"
    action_key: str = "action"
    camera_keys: tuple[str, ...] = ()
    output_image_keys: tuple[str, ...] = ()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required metadata file does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return records


def _to_json_value(value: Any) -> Any:
    """Convert Arrow and NumPy-like values to JSON-compatible Python values."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, tuple):
        return [_to_json_value(item) for item in value]
    if isinstance(value, list):
        return [_to_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_json_value(item) for key, item in value.items()}
    return value


def _format_path(template: str, **values: Any) -> Path:
    try:
        return Path(template.format(**values))
    except KeyError as exc:
        raise ValueError(
            f"Unsupported path template {template!r}; missing field {exc.args[0]!r}"
        ) from exc


def _load_task_lookup(meta_root: Path) -> dict[int, str]:
    task_records = _read_jsonl(meta_root / "tasks.jsonl")
    return {
        int(record["task_index"]): str(record["task"])
        for record in task_records
        if "task_index" in record and "task" in record
    }


def _episode_prompt(
    row: dict[str, Any],
    episode_record: dict[str, Any],
    task_lookup: dict[int, str],
) -> str:
    for key in ("task", "prompt", "instruction"):
        if row.get(key):
            return str(row[key])
    if row.get("task_index") is not None:
        prompt = task_lookup.get(int(row["task_index"]))
        if prompt:
            return prompt
    tasks = episode_record.get("tasks") or []
    if tasks:
        return str(tasks[0])
    task_indices = episode_record.get("task_indices") or []
    if task_indices:
        prompt = task_lookup.get(int(task_indices[0]))
        if prompt:
            return prompt
    raise ValueError(
        f"No language task found for episode {episode_record.get('episode_index')}"
    )


def _resolve_camera_mapping(
    info: dict[str, Any],
    requested_camera_keys: tuple[str, ...],
    requested_output_keys: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    video_keys = tuple(
        key
        for key, feature in info.get("features", {}).items()
        if feature.get("dtype") == "video"
    )
    camera_keys = requested_camera_keys or video_keys
    if not camera_keys:
        raise ValueError(
            "No video camera features found. Convert image-backed LeRobot data "
            "to video-backed v2.1 before running this converter."
        )
    missing = sorted(set(camera_keys) - set(video_keys))
    if missing:
        raise KeyError(f"Requested camera keys are not video features: {missing}")

    output_keys = requested_output_keys or tuple(
        f"images_{index}" for index in range(1, len(camera_keys) + 1)
    )
    if len(camera_keys) != len(output_keys):
        raise ValueError("--camera-keys and --output-image-keys must have equal lengths")
    if len(set(output_keys)) != len(output_keys):
        raise ValueError("--output-image-keys must be unique")
    return camera_keys, output_keys


def convert_dataset(config: ConversionConfig) -> dict[str, Any]:
    """Convert every v2.1 episode and return a conversion summary."""
    input_root = config.input_root.expanduser().resolve()
    output_dir = config.output_dir.expanduser().resolve()
    info = _read_json(input_root / "meta" / "info.json")
    version = str(info.get("codebase_version", "unknown"))
    if version not in {"v2.0", "v2.1", "2.0", "2.1"}:
        raise ValueError(
            f"Expected a LeRobot v2.1 dataset, found codebase_version={version!r}. "
            "Convert v3.0 with convert_lerobot_v3_to_v21.py first."
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Choose a new directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_keys, output_keys = _resolve_camera_mapping(
        info, config.camera_keys, config.output_image_keys
    )
    chunks_size = int(info.get("chunks_size", 1000))
    data_template = info.get(
        "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    video_template = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    if not video_template:
        raise ValueError("Dataset metadata does not define a video_path template")

    episode_records = _read_jsonl(input_root / "meta" / "episodes.jsonl")
    if not episode_records:
        episode_records = [
            {"episode_index": index}
            for index in range(int(info.get("total_episodes", 0)))
        ]
    task_lookup = _load_task_lookup(input_root / "meta")

    total_frames = 0
    for episode_record in episode_records:
        episode_index = int(episode_record["episode_index"])
        episode_chunk = episode_index // chunks_size
        parquet_path = input_root / _format_path(
            data_template,
            episode_chunk=episode_chunk,
            episode_index=episode_index,
        )
        if not parquet_path.is_file():
            raise FileNotFoundError(f"Episode parquet does not exist: {parquet_path}")
        rows = pq.read_table(parquet_path).to_pylist()
        jsonl_path = output_dir / f"episode_{episode_index:06d}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as output_file:
            for position, row in enumerate(rows):
                if config.state_key not in row:
                    raise KeyError(f"Missing state key {config.state_key!r} in {parquet_path}")
                if config.action_key not in row:
                    raise KeyError(f"Missing action key {config.action_key!r} in {parquet_path}")
                frame_index = int(row.get("frame_index", position))
                converted = {
                    "state": _to_json_value(row[config.state_key]),
                    "action": _to_json_value(row[config.action_key]),
                    "prompt": _episode_prompt(row, episode_record, task_lookup),
                }
                for camera_key, output_key in zip(camera_keys, output_keys):
                    video_path = _format_path(
                        video_template,
                        episode_chunk=episode_chunk,
                        episode_index=episode_index,
                        video_key=camera_key,
                    )
                    if not (input_root / video_path).is_file():
                        raise FileNotFoundError(
                            f"Episode video does not exist: {input_root / video_path}"
                        )
                    converted[output_key] = {
                        "type": "video",
                        "url": video_path.as_posix(),
                        "frame_idx": frame_index,
                    }
                output_file.write(json.dumps(converted, ensure_ascii=False) + "\n")
        total_frames += len(rows)

    summary = {
        "source_format": version,
        "episodes": len(episode_records),
        "frames": total_frames,
        "jsonl_dir": str(output_dir),
        "image_dir": str(input_root),
        "camera_mapping": dict(zip(output_keys, camera_keys)),
        "state_key": config.state_key,
        "action_key": config.action_key,
    }
    with (output_dir / "conversion_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return summary


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_args() -> ConversionConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--camera-keys", default="")
    parser.add_argument("--output-image-keys", default="")
    args = parser.parse_args()
    return ConversionConfig(
        input_root=args.input_root,
        output_dir=args.output_dir,
        state_key=args.state_key,
        action_key=args.action_key,
        camera_keys=_csv(args.camera_keys),
        output_image_keys=_csv(args.output_image_keys),
    )


def main() -> None:
    summary = convert_dataset(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
