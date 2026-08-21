import cv2
import numpy as np
from enum import IntEnum
from typing import Optional


class Palette(IntEnum):
    IRONBOW = 0
    JET = 1
    INFERNO = 2
    VIRIDIS = 3
    GRAYSCALE = 4


_IRONBOW_LUT = None


def _get_ironbow_lut() -> np.ndarray:
    global _IRONBOW_LUT
    if _IRONBOW_LUT is None:
        lut = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            v = i / 255.0
            if v < 0.125:
                lut[i] = [int(255 * (v / 0.125)), 0, 0]
            elif v < 0.375:
                lut[i] = [255, int(255 * ((v - 0.125) / 0.25)), 0]
            elif v < 0.625:
                lut[i] = [255, 255, int(255 * (1 - (v - 0.375) / 0.25))]
            elif v < 0.875:
                lut[i] = [int(255 * (1 - (v - 0.625) / 0.25)), 255, 255]
            else:
                lut[i] = [0, int(255 * (1 - (v - 0.875) / 0.125)), 255]
        _IRONBOW_LUT = np.ascontiguousarray(lut)
    return _IRONBOW_LUT


_VIRIDIS_LUT = None


def _get_viridis_lut() -> np.ndarray:
    global _VIRIDIS_LUT
    if _VIRIDIS_LUT is None:
        viridis_data = np.array([
            [68, 1, 84], [72, 35, 116], [64, 67, 135], [52, 94, 141],
            [41, 120, 142], [32, 143, 140], [34, 167, 132], [49, 189, 111],
            [81, 208, 77], [129, 219, 52], [185, 224, 38], [246, 226, 33],
            [253, 231, 37]
        ], dtype=np.uint8)
        lut = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            idx = i / 255.0 * 12
            i0 = int(idx)
            i1 = min(i0 + 1, 12)
            t = idx - i0
            lut[i] = (viridis_data[i0] * (1 - t) + viridis_data[i1] * t).astype(np.uint8)
        _VIRIDIS_LUT = np.ascontiguousarray(lut)
    return _VIRIDIS_LUT


def apply_palette(gray: np.ndarray, palette: Palette) -> np.ndarray:
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    if palette == Palette.IRONBOW:
        lut = _get_ironbow_lut()
        b = cv2.LUT(gray, lut[:, 0])
        g = cv2.LUT(gray, lut[:, 1])
        r = cv2.LUT(gray, lut[:, 2])
        return cv2.merge([b, g, r])
    elif palette == Palette.JET:
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    elif palette == Palette.INFERNO:
        return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    elif palette == Palette.VIRIDIS:
        lut = _get_viridis_lut()
        b = cv2.LUT(gray, lut[:, 0])
        g = cv2.LUT(gray, lut[:, 1])
        r = cv2.LUT(gray, lut[:, 2])
        return cv2.merge([b, g, r])
    else:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def contrast_stretch(frame: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    if frame.dtype != np.uint8:
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    low = np.percentile(frame, low_pct)
    high = np.percentile(frame, high_pct)

    if high <= low:
        return frame

    stretched = np.clip((frame.astype(np.float32) - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
    return stretched


def colorize_frame(frame: np.ndarray, palette: Palette = Palette.IRONBOW,
                   use_motion: bool = False, prev_gray: Optional[np.ndarray] = None) -> tuple:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if use_motion and prev_gray is not None:
        diff = cv2.absdiff(gray, prev_gray)
        gray = cv2.addWeighted(gray, 0.7, diff, 0.3, 0)

    stretched = contrast_stretch(gray)
    colorized = apply_palette(stretched, palette)

    return colorized, stretched