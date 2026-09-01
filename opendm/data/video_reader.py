"""Video-frame readers used by offline DM05 training pipelines."""

from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np
from PIL import Image


class TorchCodecVideoReader:
    """Read sparse RGB frames with worker-local cached TorchCodec decoders.

    Decoder objects are created lazily after DataLoader workers fork. Each
    process owns an independent LRU, avoiding cross-process FFmpeg state and
    bounding the number of simultaneously open videos.
    """

    def __init__(
        self,
        *,
        cache_size: int = 32,
        seek_mode: str = "exact",
        decoder_threads: int = 1,
    ):
        if cache_size <= 0:
            raise ValueError(f"video_decoder_cache_size must be positive, got {cache_size}")
        if decoder_threads <= 0:
            raise ValueError(
                f"video_decoder_threads must be positive, got {decoder_threads}"
            )
        if seek_mode not in {"exact", "approximate"}:
            raise ValueError(
                "video_seek_mode must be 'exact' or 'approximate', "
                f"got {seek_mode!r}"
            )
        self.cache_size = int(cache_size)
        self.seek_mode = seek_mode
        self.decoder_threads = int(decoder_threads)
        self._owner_pid = os.getpid()
        self._decoders: OrderedDict[str, object] = OrderedDict()

    def __getstate__(self) -> dict:
        """Drop native decoder state when a dataset is serialized."""
        state = self.__dict__.copy()
        state["_owner_pid"] = None
        state["_decoders"] = OrderedDict()
        return state

    def _reset_after_fork(self) -> None:
        current_pid = os.getpid()
        if self._owner_pid == current_pid:
            return
        self._decoders = OrderedDict()
        self._owner_pid = current_pid

    def _decoder(self, video_path: str):
        self._reset_after_fork()
        decoder = self._decoders.pop(video_path, None)
        if decoder is None:
            try:
                from torchcodec.decoders import VideoDecoder
            except (ImportError, RuntimeError) as exc:
                raise RuntimeError(
                    "video_backend='torchcodec' requires a TorchCodec build "
                    "compatible with the active PyTorch and FFmpeg libraries"
                ) from exc
            decoder = VideoDecoder(
                video_path,
                dimension_order="NHWC",
                num_ffmpeg_threads=self.decoder_threads,
                device="cpu",
                seek_mode=self.seek_mode,
            )
        self._decoders[video_path] = decoder
        while len(self._decoders) > self.cache_size:
            self._decoders.popitem(last=False)
        return decoder

    def read_frames(self, video_path: str, frame_indices: list[int]) -> list[Image.Image]:
        """Decode only the requested indices and return RGB PIL images."""
        if not frame_indices:
            return []
        if any(index < 0 for index in frame_indices):
            raise ValueError(f"Frame indices must be non-negative: {frame_indices}")
        decoder = self._decoder(video_path)
        frame_batch = decoder.get_frames_at(frame_indices)
        images = []
        for frame in frame_batch.data:
            array = frame.detach().cpu().contiguous().numpy()
            if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
                raise RuntimeError(
                    "TorchCodec returned an unexpected frame layout: "
                    f"shape={array.shape}, dtype={array.dtype}"
                )
            images.append(Image.fromarray(array, mode="RGB"))
        if len(images) != len(frame_indices):
            raise RuntimeError(
                f"Requested {len(frame_indices)} frames but decoded {len(images)}"
            )
        return images
