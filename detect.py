import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import time


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    confidence: float
    persistence: int = 1
    track_id: int = -1

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)

    @property
    def area(self) -> float:
        return self.w * self.h

    def iou(self, other: 'Detection') -> float:
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.w, other.x + other.w)
        y2 = min(self.y + self.h, other.y + other.h)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        inter = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


def adaptive_threshold(gray: np.ndarray, block_size: int = 31, C: float = 5.0) -> np.ndarray:
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, block_size, C)


def detect_hotspots(gray: np.ndarray, min_area: int = 100,
                    threshold_method: str = 'adaptive') -> List[Detection]:
    if threshold_method == 'adaptive':
        thresh = adaptive_threshold(gray)
    else:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_area = gray.shape[0] * gray.shape[1]
    max_area_ratio = 0.5

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > frame_area * max_area_ratio:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_val = cv2.mean(gray, mask=mask)[0]
        confidence = min(mean_val / 255.0, 1.0)
        detections.append(Detection(x, y, w, h, confidence))

    return detections


def nms(detections: List[Detection], iou_threshold: float = 0.3) -> List[Detection]:
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept = []

    for det in sorted_dets:
        keep = True
        for kept_det in kept:
            if det.iou(kept_det) > iou_threshold:
                keep = False
                break
        if keep:
            kept.append(det)

    return kept


class CentroidTracker:
    def __init__(self, max_distance: float = 50.0, max_disappeared: int = 10):
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self.next_id = 0
        self.tracks: dict[int, Detection] = {}
        self.disappeared: dict[int, int] = {}

    def update(self, detections: List[Detection]) -> List[Detection]:
        if not detections:
            for tid in list(self.disappeared.keys()):
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    del self.tracks[tid]
                    del self.disappeared[tid]
            return []

        if not self.tracks:
            for det in detections:
                det.track_id = self.next_id
                self.tracks[self.next_id] = det
                self.disappeared[self.next_id] = 0
                self.next_id += 1
            return detections

        track_centers = {tid: trk.center for tid, trk in self.tracks.items()}
        det_centers = [det.center for det in detections]

        used_tracks = set()
        used_dets = set()

        for det_idx, det_center in enumerate(det_centers):
            best_tid = None
            best_dist = float('inf')
            for tid, trk_center in track_centers.items():
                if tid in used_tracks:
                    continue
                dist = np.hypot(det_center[0] - trk_center[0], det_center[1] - trk_center[1])
                if dist < best_dist and dist <= self.max_distance:
                    best_dist = dist
                    best_tid = tid

            if best_tid is not None:
                self.tracks[best_tid] = detections[det_idx]
                self.tracks[best_tid].track_id = best_tid
                self.tracks[best_tid].persistence += 1
                self.disappeared[best_tid] = 0
                used_tracks.add(best_tid)
                used_dets.add(det_idx)

        for det_idx, det in enumerate(detections):
            if det_idx not in used_dets:
                det.track_id = self.next_id
                self.tracks[self.next_id] = det
                self.disappeared[self.next_id] = 0
                self.next_id += 1

        for tid in list(self.disappeared.keys()):
            if tid not in used_tracks:
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    del self.tracks[tid]
                    del self.disappeared[tid]

        return list(self.tracks.values())


def compute_confidence(detection: Detection, gray: np.ndarray) -> float:
    x, y, w, h = detection.x, detection.y, detection.w, detection.h
    roi = gray[y:y+h, x:x+w]
    if roi.size == 0:
        return detection.confidence

    peak = np.max(roi) / 255.0
    persistence_factor = min(detection.persistence / 30.0, 1.0)
    confidence = 0.6 * peak + 0.4 * persistence_factor
    return min(confidence, 1.0)