import pytest
import numpy as np
import cv2
from detect import (
    detect_hotspots, nms, CentroidTracker, compute_confidence, Detection,
    adaptive_threshold
)
from colorize import contrast_stretch


def make_synthetic_frame(shape=(480, 640), hotspot_pos=(320, 240), hotspot_size=50, hotspot_val=255, noise_level=10):
    frame = np.random.randint(0, noise_level, shape, dtype=np.uint8)
    y, x = hotspot_pos
    h, w = shape
    y1, y2 = max(0, y - hotspot_size//2), min(h, y + hotspot_size//2)
    x1, x2 = max(0, x - hotspot_size//2), min(w, x + hotspot_size//2)
    frame[y1:y2, x1:x2] = hotspot_val
    return frame


def test_adaptive_threshold():
    gray = np.zeros((100, 100), dtype=np.uint8)
    gray[20:80, 20:80] = 200
    thresh = adaptive_threshold(gray)
    assert thresh.shape == gray.shape
    assert thresh.dtype == np.uint8
    assert np.any(thresh > 0)


def test_contrast_stretch():
    frame = np.full((100, 100), 100, dtype=np.uint8)
    frame[20:80, 20:80] = 200
    stretched = contrast_stretch(frame)
    assert stretched.min() == 0
    assert stretched.max() == 255


def test_detect_hotspot_detected():
    frame = make_synthetic_frame(hotspot_val=255, noise_level=5)
    dets = detect_hotspots(frame, min_area=100, threshold_method='otsu')
    assert len(dets) >= 1
    det = dets[0]
    assert det.w > 0 and det.h > 0
    assert 0 <= det.confidence <= 1


def test_detect_no_hotspot_in_noise():
    frame = np.full((480, 640), 10, dtype=np.uint8)
    frame += np.random.randint(0, 5, (480, 640), dtype=np.uint8)
    dets = detect_hotspots(frame, min_area=5000, threshold_method='otsu')
    assert len(dets) == 0


def test_confidence_rises_with_intensity():
    frame_low = make_synthetic_frame(hotspot_val=100, noise_level=5)
    frame_high = make_synthetic_frame(hotspot_val=255, noise_level=5)
    dets_low = detect_hotspots(frame_low, min_area=100, threshold_method='otsu')
    dets_high = detect_hotspots(frame_high, min_area=100, threshold_method='otsu')
    assert len(dets_low) >= 1 and len(dets_high) >= 1
    assert dets_high[0].confidence > dets_low[0].confidence


def test_nms_merges_overlapping():
    d1 = Detection(100, 100, 50, 50, 0.9)
    d2 = Detection(110, 110, 50, 50, 0.8)
    d3 = Detection(300, 300, 50, 50, 0.7)
    kept = nms([d1, d2, d3], iou_threshold=0.3)
    assert len(kept) == 2
    assert kept[0].confidence == 0.9


def test_centroid_tracker_persists_identity():
    tracker = CentroidTracker(max_distance=30)
    d1 = Detection(100, 100, 20, 20, 0.8)
    d2 = Detection(105, 105, 20, 20, 0.8)
    d3 = Detection(200, 200, 20, 20, 0.8)

    out1 = tracker.update([d1])
    assert len(out1) == 1
    assert out1[0].track_id == 0
    assert out1[0].persistence == 1

    out2 = tracker.update([d2])
    assert len(out2) == 1
    assert out2[0].track_id == 0
    assert out2[0].persistence == 2

    out3 = tracker.update([d3])
    assert len(out3) == 2
    assert any(d.track_id == 0 for d in out3)
    assert any(d.track_id == 1 for d in out3)


def test_centroid_tracker_new_track_on_jump():
    tracker = CentroidTracker(max_distance=30)
    d1 = Detection(100, 100, 20, 20, 0.8)
    d2 = Detection(300, 300, 20, 20, 0.8)

    out1 = tracker.update([d1])
    out2 = tracker.update([d2])
    assert len(out2) == 2
    track_ids = {d.track_id for d in out2}
    assert len(track_ids) == 2


def test_compute_confidence_formula():
    gray = np.zeros((100, 100), dtype=np.uint8)
    gray[40:60, 40:60] = 200
    det = Detection(40, 40, 20, 20, 0.5, persistence=10)
    conf = compute_confidence(det, gray)
    peak = 200 / 255.0
    persistence_factor = min(10 / 30.0, 1.0)
    expected = 0.6 * peak + 0.4 * persistence_factor
    assert abs(conf - expected) < 0.01


def test_confidence_bounds():
    gray = np.zeros((100, 100), dtype=np.uint8)
    det = Detection(10, 10, 20, 20, 0.5, persistence=100)
    conf = compute_confidence(det, gray)
    assert 0 <= conf <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])