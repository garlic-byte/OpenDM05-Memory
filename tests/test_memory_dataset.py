"""Unit tests for generic DM05 visual-memory sampling."""

import orjson
import pytest

from opendm.dataset.memory_dataset import (
    LoadCurrentAndMemoryVideoFrames,
    LoadMemoryImages,
    past_history_indices,
)


def test_history_indices_are_strictly_past_and_oldest_first():
    assert past_history_indices(80, frames=5, stride=16) == [0, 16, 32, 48, 64]
    assert past_history_indices(96, frames=5, stride=16) == [16, 32, 48, 64, 80]


def test_history_indices_do_not_cross_episode_start():
    assert past_history_indices(0, frames=5, stride=16) == []
    assert past_history_indices(16, frames=5, stride=16) == [0]
    assert past_history_indices(32, frames=5, stride=16) == [0, 16]


def test_invalid_history_settings_are_rejected():
    with pytest.raises(ValueError, match="positive"):
        past_history_indices(10, frames=0, stride=1)
    with pytest.raises(ValueError, match="inference capacity"):
        LoadMemoryImages(
            ["head", "wrist"],
            ".",
            frames=3,
            stride=1,
            max_history_images=5,
        )


def test_default_history_layout_keeps_only_available_frames():
    loader = LoadMemoryImages(
        ["images_1"],
        ".",
        frames=3,
        stride=1,
        max_history_images=3,
    )
    sample = {"raw_lines": [orjson.dumps({})], "meta_data": {"frame_index": 0}}
    assert loader(sample)["history_images"] == []


def test_fixed_history_layout_left_pads_missing_episode_frames():
    loader = LoadMemoryImages(
        ["images_1"],
        ".",
        frames=3,
        stride=1,
        max_history_images=3,
        left_pad=True,
    )

    def fake_image_loader(row):
        row["images"] = [row["frame"]]
        return row

    loader.image_loader = fake_image_loader
    rows = [
        orjson.dumps({"images_1": {}, "frame": frame_index})
        for frame_index in range(3)
    ]
    sample = {"raw_lines": rows, "meta_data": {"frame_index": 2}}
    assert loader(sample)["history_images"] == [None, 0, 1]


def test_fixed_history_layout_has_twenty_slots_at_episode_start():
    loader = LoadMemoryImages(
        ["images_1"],
        ".",
        frames=20,
        stride=25,
        max_history_images=20,
        left_pad=True,
    )
    sample = {"raw_lines": [orjson.dumps({})], "meta_data": {"frame_index": 0}}
    assert loader(sample)["history_images"] == [None] * 20


def test_torchcodec_loader_batches_indices_per_video_and_preserves_slots():
    class FakeVideoReader:
        def __init__(self):
            self.calls = []

        def read_frames(self, video_path, frame_indices):
            self.calls.append((video_path, frame_indices))
            return [f"{video_path}:{frame_index}" for frame_index in frame_indices]

    reader = FakeVideoReader()
    loader = LoadCurrentAndMemoryVideoFrames(
        current_image_keys=["head", "left", "right"],
        history_image_keys=["head"],
        image_dir="/dataset",
        frames=2,
        stride=1,
        max_history_images=2,
        left_pad=True,
        video_reader=reader,
    )
    rows = []
    for frame_index in range(3):
        rows.append(
            orjson.dumps(
                {
                    "head": {"type": "video", "url": "head.mp4", "frame_idx": frame_index},
                    "left": {"type": "video", "url": "left.mp4", "frame_idx": frame_index},
                    "right": {"type": "video", "url": "right.mp4", "frame_idx": frame_index},
                }
            )
        )
    sample = {
        **orjson.loads(rows[2]),
        "raw_lines": rows,
        "meta_data": {"frame_index": 2},
    }

    result = loader(sample)

    assert reader.calls == [
        ("/dataset/head.mp4", [0, 1, 2]),
        ("/dataset/left.mp4", [2]),
        ("/dataset/right.mp4", [2]),
    ]
    assert result["images"] == [
        "/dataset/head.mp4:2",
        "/dataset/left.mp4:2",
        "/dataset/right.mp4:2",
    ]
    assert result["history_images"] == [
        "/dataset/head.mp4:0",
        "/dataset/head.mp4:1",
    ]


def test_torchcodec_loader_keeps_episode_prefix_padding_without_decoding_it():
    class FakeVideoReader:
        def read_frames(self, video_path, frame_indices):
            return [frame_index for frame_index in frame_indices]

    loader = LoadCurrentAndMemoryVideoFrames(
        current_image_keys=["head"],
        history_image_keys=["head"],
        image_dir="/dataset",
        frames=3,
        stride=1,
        max_history_images=3,
        left_pad=True,
        video_reader=FakeVideoReader(),
    )
    row = {
        "head": {"type": "video", "url": "head.mp4", "frame_idx": 0},
    }
    sample = {
        **row,
        "raw_lines": [orjson.dumps(row)],
        "meta_data": {"frame_index": 0},
    }

    result = loader(sample)

    assert result["images"] == [0]
    assert result["history_images"] == [None, None, None]
