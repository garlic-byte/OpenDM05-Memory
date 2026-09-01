"""Generic visual-memory sampling for episode-oriented DM05 JSONL datasets."""

from __future__ import annotations

import os

import megfile
import orjson

from opendm.data.dataset import JsonlDataset, _load_jsonl
from opendm.data.transforms import LoadImages, _load_image
from opendm.data.video_reader import TorchCodecVideoReader


def past_history_indices(step_idx: int, frames: int, stride: int) -> list[int]:
    """Return strictly-past indices, ordered from oldest to newest."""
    if frames <= 0:
        raise ValueError(f"history_frames must be positive, got {frames}")
    if stride <= 0:
        raise ValueError(f"history_stride must be positive, got {stride}")
    return [
        index
        for index in range(step_idx - frames * stride, step_idx, stride)
        if index >= 0
    ]


class EpisodeJsonlMemoryDataset(JsonlDataset):
    """JSONL dataset with a relocatable, absolute in-memory episode index.

    Existing DM05 index caches may contain paths relative to the directory from
    which they were generated. A training job in another checkout would then
    fail despite receiving an absolute ``jsonl_dir``. Scanning here keeps the
    source dataset untouched and makes a single training entry portable.
    """

    def _get_index_cache(self, jsonl_dir: str) -> dict:
        if not megfile.smart_isdir(jsonl_dir):
            raise FileNotFoundError(f"Dataset directory does not exist: {jsonl_dir}")
        jsonl_files = sorted(
            megfile.smart_glob(os.path.join(jsonl_dir, "**", "*.jsonl"))
        )
        if not jsonl_files:
            raise FileNotFoundError(f"Dataset contains no JSONL files: {jsonl_dir}")
        return {
            "data": {
                os.path.abspath(path): len(_load_jsonl(path)) for path in jsonl_files
            }
        }


class LoadMemoryImages:
    """Load sparse history images from earlier rows in the current episode.

    The standard ``JsonlDataset`` exposes all rows from the current episode as
    ``raw_lines``. This transform samples those rows without creating another
    dataset format or duplicating video files.
    """

    def __init__(
        self,
        image_keys: list[str],
        image_dir: str,
        *,
        frames: int = 5,
        stride: int = 16,
        max_history_images: int = 5,
        left_pad: bool = False,
    ):
        if not image_keys:
            raise ValueError("history_image_keys must contain at least one key")
        if frames * len(image_keys) > max_history_images:
            raise ValueError(
                "The configured history stack exceeds inference capacity: "
                f"{frames} frames x {len(image_keys)} cameras > "
                f"{max_history_images} images"
            )
        self.image_keys = list(image_keys)
        self.frames = int(frames)
        self.stride = int(stride)
        self.max_history_images = int(max_history_images)
        self.left_pad = bool(left_pad)
        self.image_loader = LoadImages(self.image_keys, image_dir=image_dir)

        # Validate values at construction time, including datasets whose first
        # sampled row has no available history.
        past_history_indices(0, self.frames, self.stride)

    def __call__(self, data: dict) -> dict:
        if "raw_lines" not in data or "meta_data" not in data:
            raise ValueError(
                "LoadMemoryImages requires JsonlDataset raw_lines and meta_data"
            )
        frame_index = int(data["meta_data"]["frame_index"])
        history_indices = past_history_indices(
            frame_index, self.frames, self.stride
        )
        history_images = []
        if self.left_pad:
            missing_frames = self.frames - len(history_indices)
            history_images.extend(
                [None] * (missing_frames * len(self.image_keys))
            )
        for history_index in history_indices:
            history_row = orjson.loads(data["raw_lines"][history_index])
            missing = [key for key in self.image_keys if key not in history_row]
            if missing:
                raise KeyError(
                    f"History row {history_index} is missing image keys: {missing}"
                )
            self.image_loader(history_row)
            history_images.extend(history_row["images"])
        expected_images = self.frames * len(self.image_keys)
        if self.left_pad and len(history_images) != expected_images:
            raise RuntimeError(
                "Fixed-layout history produced an unexpected number of slots: "
                f"expected {expected_images}, got {len(history_images)}"
            )
        data["history_images"] = history_images
        return data


class LoadCurrentAndMemoryVideoFrames:
    """Batch current and sparse-history video requests with TorchCodec.

    All requested indices for the same MP4 are submitted in one
    ``get_frames_at`` call. This combines the head-camera current frame with
    its history and avoids reopening the video for every requested frame.
    """

    def __init__(
        self,
        current_image_keys: list[str],
        history_image_keys: list[str],
        image_dir: str,
        *,
        frames: int = 5,
        stride: int = 16,
        max_history_images: int = 5,
        left_pad: bool = False,
        decoder_cache_size: int = 32,
        seek_mode: str = "exact",
        decoder_threads: int = 1,
        video_reader=None,
    ):
        if not current_image_keys:
            raise ValueError("image_keys must contain at least one key")
        if not history_image_keys:
            raise ValueError("history_image_keys must contain at least one key")
        if frames * len(history_image_keys) > max_history_images:
            raise ValueError(
                "The configured history stack exceeds inference capacity: "
                f"{frames} frames x {len(history_image_keys)} cameras > "
                f"{max_history_images} images"
            )
        self.current_image_keys = list(current_image_keys)
        self.history_image_keys = list(history_image_keys)
        self.image_dir = image_dir
        self.frames = int(frames)
        self.stride = int(stride)
        self.max_history_images = int(max_history_images)
        self.left_pad = bool(left_pad)
        self.video_reader = video_reader or TorchCodecVideoReader(
            cache_size=decoder_cache_size,
            seek_mode=seek_mode,
            decoder_threads=decoder_threads,
        )
        past_history_indices(0, self.frames, self.stride)

    def _absolute_path(self, metadata: dict) -> str:
        if "url" not in metadata:
            raise KeyError("Image metadata is missing 'url'")
        return os.path.join(self.image_dir, metadata["url"].lstrip("./"))

    def __call__(self, data: dict) -> dict:
        if "raw_lines" not in data or "meta_data" not in data:
            raise ValueError(
                "LoadCurrentAndMemoryVideoFrames requires raw_lines and meta_data"
            )
        frame_index = int(data["meta_data"]["frame_index"])
        history_indices = past_history_indices(frame_index, self.frames, self.stride)
        current_images = [None] * len(self.current_image_keys)
        missing_frames = self.frames - len(history_indices) if self.left_pad else 0
        history_images = [None] * (
            missing_frames * len(self.history_image_keys)
            + len(history_indices) * len(self.history_image_keys)
        )

        requests_by_video: dict[str, list[tuple[str, int, int]]] = {}

        def register(metadata: dict, destination: str, slot: int) -> None:
            media_type = metadata.get("type")
            media_path = self._absolute_path(metadata)
            if media_type == "image":
                image = _load_image(media_path)
                if destination == "current":
                    current_images[slot] = image
                else:
                    history_images[slot] = image
                return
            if media_type != "video":
                raise ValueError(f"Invalid image type: {media_type}")
            if "frame_idx" not in metadata:
                raise KeyError("Video metadata is missing 'frame_idx'")
            requests_by_video.setdefault(media_path, []).append(
                (destination, slot, int(metadata["frame_idx"]))
            )

        for slot, key in enumerate(self.current_image_keys):
            if key not in data:
                raise KeyError(f"Current row is missing image key: {key}")
            register(data[key], "current", slot)

        history_offset = missing_frames * len(self.history_image_keys)
        for position, history_index in enumerate(history_indices):
            history_row = orjson.loads(data["raw_lines"][history_index])
            for key_offset, key in enumerate(self.history_image_keys):
                if key not in history_row:
                    raise KeyError(
                        f"History row {history_index} is missing image key: {key}"
                    )
                slot = history_offset + position * len(self.history_image_keys) + key_offset
                register(history_row[key], "history", slot)

        for video_path, requests in requests_by_video.items():
            # Ascending indices let TorchCodec continue decoding within a GOP
            # instead of seeking backward from the current frame to history.
            requests = sorted(requests, key=lambda request: request[2])
            decoded = self.video_reader.read_frames(
                video_path,
                [frame_index for _, _, frame_index in requests],
            )
            for (destination, slot, _), image in zip(requests, decoded, strict=True):
                if destination == "current":
                    current_images[slot] = image
                else:
                    history_images[slot] = image

        if any(image is None for image in current_images):
            raise RuntimeError("Current-image decoding left an unfilled slot")
        expected_history_images = self.frames * len(self.history_image_keys)
        if self.left_pad and len(history_images) != expected_history_images:
            raise RuntimeError(
                "Fixed-layout history produced an unexpected number of slots: "
                f"expected {expected_history_images}, got {len(history_images)}"
            )
        data["images"] = current_images
        data["history_images"] = history_images
        return data
