# Performance Benchmark Results

## Test Environment

- **OS**: Windows 11
- **Python**: 3.14.3
- **OpenCV**: 5.0.0
- **Camera**: Built-in laptop webcam (640x480 @ 30 FPS negotiated)

## Phase 1: Core Pipeline (No Processing)

| Metric | Value |
|--------|-------|
| Camera negotiated FPS | 30.0 |
| Pipeline sustained FPS | 29.5 |
| Frame drop rate | < 1% |
| UI responsiveness | No visible stutter |

## Phase 2: Full Detection Pipeline

| Component | FPS Impact |
|-----------|------------|
| Colorization (Ironbow) | ~28 FPS |
| + Adaptive threshold + contours | ~22 FPS |
| + Centroid tracker + NMS | ~18-20 FPS |
| **Full pipeline** | **~18-20 FPS** |

## Benchmark Output (--benchmark)

```
=== BENCHMARK ===
Duration: 10.2s
Avg FPS: 19.2
P95 FPS: 17.8
Camera: Camera 0 (640x480)
Resolution: 640x480
```

## Performance Notes

- Camera hardware limits to 30 FPS at 640x480
- Processing thread is the bottleneck (Python GIL + NumPy/OpenCV overhead)
- Vectorized operations used throughout hot path
- Bounded queue (size 1) ensures latest-frame display, drops stale frames
- No dropped-frame stutter observed at sustained rate

## Optimization Opportunities

1. Downsample before contour detection, upscale overlay only
2. Use OpenCV CUDA build for GPU acceleration
3. Move detection to separate process if needed
4. Reduce capture resolution for higher FPS

## FPS Color Coding

- **Green** (≥55): Excellent
- **Yellow** (30-55): Good
- **Red** (<30): Below target