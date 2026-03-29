"""
Unit tests for YOLO Vision core modules.
Run with: pytest tests/ -v
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from app.core.detector import Detection, YOLODetector
from app.utils.helpers import (
    is_image, is_video, summarize_detections,
    build_output_path, encode_frame_to_bytes,
)


# ─── Detection dataclass ─────────────────────────────────────────────────────

class TestDetection:
    def _make(self, cls="cat", conf=0.92, bbox=(10, 20, 110, 120)):
        return Detection(class_id=0, class_label=cls, confidence=conf, bbox=bbox)

    def test_to_dict_keys(self):
        d = self._make()
        dd = d.to_dict()
        assert "class_label" in dd
        assert "confidence" in dd
        assert "bbox" in dd

    def test_bbox_dimensions(self):
        d = self._make(bbox=(0, 0, 50, 80))
        assert d.bbox_width == 50
        assert d.bbox_height == 80

    def test_confidence_rounded(self):
        d = self._make(conf=0.912345)
        assert d.to_dict()["confidence"] == round(0.912345, 4)


# ─── Helper utilities ─────────────────────────────────────────────────────────

class TestHelpers:
    def test_is_image(self):
        assert is_image("photo.jpg") is True
        assert is_image("photo.PNG") is True
        assert is_image("clip.mp4") is False

    def test_is_video(self):
        assert is_video("clip.mp4") is True
        assert is_video("clip.AVI") is True
        assert is_video("photo.jpeg") is False

    def test_summarize_empty(self):
        s = summarize_detections([])
        assert s["total_objects"] == 0
        assert s["avg_confidence"] == 0.0
        assert s["class_counts"] == {}

    def test_summarize_counts(self):
        dets = [
            Detection(0, "cat", 0.9, (0, 0, 10, 10)),
            Detection(0, "cat", 0.8, (0, 0, 10, 10)),
            Detection(1, "dog", 0.7, (0, 0, 10, 10)),
        ]
        s = summarize_detections(dets)
        assert s["total_objects"] == 3
        assert s["class_counts"]["cat"] == 2
        assert s["class_counts"]["dog"] == 1
        assert abs(s["avg_confidence"] - round((0.9 + 0.8 + 0.7) / 3, 4)) < 1e-6

    def test_build_output_path(self):
        p = build_output_path("/tmp/video.mp4", "/out", "_det")
        assert p.endswith("_det.mp4")
        assert "/out/" in p

    def test_encode_frame_bytes(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        b = encode_frame_to_bytes(frame)
        assert isinstance(b, bytes)
        assert len(b) > 0


# ─── Detector (mocked) ────────────────────────────────────────────────────────

class TestYOLODetector:
    def test_not_loaded_initially(self):
        d = YOLODetector("fake.pt")
        assert not d.is_loaded

    def test_get_color_consistent(self):
        d = YOLODetector("fake.pt")
        c1 = d._get_color(0)
        c2 = d._get_color(0)
        assert c1 == c2

    def test_draw_no_detections(self):
        d = YOLODetector("fake.pt")
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        out = d.draw_detections(frame, [])
        assert out.shape == frame.shape

    def test_load_model_bad_path(self):
        d = YOLODetector("nonexistent_model.pt")
        with pytest.raises(RuntimeError):
            d.load_model()

    def test_detect_image_not_loaded(self):
        d = YOLODetector("fake.pt")
        with pytest.raises(RuntimeError, match="not loaded"):
            d.detect_image("some_image.jpg")
