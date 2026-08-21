# Agent Prompt: Real-Time Thermal-Style Detection App (Webcam-Based)

Copy everything below this line into your coding agent (Claude Code, Cursor, etc.).
Work through the three phases in order. Do not start a phase until the previous
phase's exit criteria are met and reported.

---

## Role

You are a senior computer vision / desktop application engineer. Build a real-time
detection application that captures live video from a laptop's built-in camera,
applies a thermal-style false-color visualization, detects "hot spot" regions of
interest with a confidence score, and displays everything in a responsive UI —
all while sustaining the highest frame rate the hardware allows, with a 60 FPS
target and no visible stutter.

## Critical Hardware Clarification (read first)

A standard laptop webcam is an RGB sensor. It does **not** measure infrared
radiation or real temperature. Do not claim or imply true thermal/heat
measurement anywhere in the UI or code comments. Build the app around this
architecture instead:

- **Primary mode — "Simulated Thermal"**: colorize the RGB feed using a false-color
  palette (Ironbow, Rainbow, or Grayscale-Iron) driven by pixel luminance and
  frame-to-frame motion delta, and treat the brightest/most active regions as
  "hot spots." Label this mode clearly in the UI as *simulated*, not real
  temperature.
- **Secondary mode — "Real Thermal Camera"**: if a UVC-compatible thermal camera
  (e.g., Seek Thermal, FLIR One, Therm-App) is connected, it will enumerate as a
  normal video capture device. Build a `CameraSource` abstraction so the same
  pipeline can read frames from either the built-in webcam or an external thermal
  UVC device without changing downstream code.
  - **Important caveat to document, not just code**: most consumer thermal
    cameras stream an *already colorized* image over plain UVC — they do not
    expose raw per-pixel temperature through generic OpenCV capture. Getting
    actual radiometric (real-temperature) data almost always requires the
    vendor's own SDK/driver, not `cv2.VideoCapture`. So "Real Thermal Camera"
    mode should be labeled in the UI as "reading a pre-colorized thermal feed,"
    not "reading true temperature values," unless a vendor SDK is integrated
    later. Confidence scoring in this mode must not imply calibrated
    temperature accuracy it doesn't have.
- **Device enumeration limitation**: OpenCV alone does not expose human-readable
  camera names on most platforms — only an index. Do not assume device names
  are available by default. Use a best-effort approach: try a platform-specific
  helper if easily available (e.g., `pygrabber` on Windows, `AVFoundation` via
  `pyobjc` on macOS) and fall back to generic labels like "Camera 0 (1280x720)"
  using index + probed resolution when a name can't be retrieved. Do not let
  the app crash or hang if name lookup fails — it must degrade to index-based
  labels silently.

## Non-Negotiable Implementation Details (common failure points)

These are specific, easy-to-get-wrong mechanics. Follow them exactly:

1. **Qt threading**: PyQt6/PySide6 widgets are not thread-safe. Never update UI
   elements directly from the capture or processing thread. Use `QThread` +
   `pyqtSignal`/`Signal` to emit finished frames back to the main thread, and
   connect that signal to a slot that updates the video label. Directly calling
   `.setPixmap()` or similar from a worker thread will cause intermittent
   crashes or corrupted rendering that are hard to reproduce and debug later.
2. **Color channel order**: OpenCV reads/writes frames in BGR. Qt's `QImage`
   expects RGB. Convert with `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` before
   constructing a `QImage`, or colors will be visibly swapped (skin tones will
   look blue). Also call `np.ascontiguousarray()` on the frame before wrapping
   it in `QImage` and pass the correct `bytesPerLine` (`width * channels`) —
   skipping this produces sheared/garbled frames intermittently depending on
   memory layout.
3. **Bounded "latest frame" queue**: implement the capture→process handoff with
   a queue of size 1, using non-blocking writes that drop the old frame instead
   of blocking:
   ```python
   if not q.empty():
       try:
           q.get_nowait()
       except queue.Empty:
           pass
   q.put_nowait(frame)
   ```
   A plain blocking `queue.Queue(maxsize=1)` with a blocking `put()` will stall
   the capture thread once the processing thread falls behind, which is the
   opposite of the intended "always show latest frame, drop stale ones" design.
4. **GIL awareness**: Python threads do not give true CPU parallelism for
   Python-level code. OpenCV's own C++ functions release the GIL internally,
   which is why a multi-threaded pipeline still helps here — but any detection
   logic you write in pure Python/NumPy loops will still serialize. If profiling
   in Phase 2 shows the processing thread is the bottleneck, prefer vectorized
   NumPy/OpenCV calls over a `multiprocessing`-based redesign first; only move
   detection to a separate process if vectorization genuinely can't hit target
   FPS, since multiprocessing adds frame-serialization overhead of its own.
5. **Non-max suppression**: don't rely on `cv2.groupRectangles` — it requires
   near-duplicate rectangles and behaves unpredictably with differently-sized
   overlapping boxes. Implement a small IoU-based NMS function (sort by
   confidence, suppress boxes with IoU above a threshold against a
   higher-confidence box) so merging behaves predictably and is unit-testable.
6. **Temporal stability tracking**: "confidence rises with persistence" requires
   identifying the *same* blob across frames, not just counting blobs per
   frame. Implement a lightweight centroid-distance tracker: each detection
   gets matched to the nearest previous-frame detection within a distance
   threshold; if matched, carry forward and increment its persistence counter,
   otherwise start a new track at persistence 1. Without this, "temporal
   stability" has no defined meaning and the confidence score becomes
   arbitrary. Document the distance threshold and matching logic in a
   docstring.
7. **VideoWriter codec fallback**: `cv2.VideoWriter` can report success while
   silently producing a 0-byte or unplayable file if the requested fourcc codec
   isn't available on the platform. Try a small ordered list of fourccs (e.g.,
   `mp4v`, then `avc1`, then `XVID` with a `.avi` fallback extension) and verify
   with `writer.isOpened()` after each attempt before committing to one. Also
   ensure every frame written matches the exact width/height the writer was
   initialized with — a mismatch fails silently per-frame rather than raising.
8. **Camera switch race condition**: switching the selected camera device while
   the pipeline is running must stop the capture thread cleanly (release the
   `VideoCapture`, join the thread) before starting a new one. Do this behind a
   lock or a clear stop→confirm→start sequence in the UI so a rapid
   double-click on the device dropdown can't spawn two capture threads against
   the same or different devices simultaneously.
9. **`cv2.VideoCapture` FPS/resolution requests are not guarantees**: after
   calling `cap.set(cv2.CAP_PROP_FPS, 60)` and `cap.set(cv2.CAP_PROP_FRAME_...)`,
   always read back the actual values with `cap.get(...)` — many drivers accept
   the call but silently ignore the request. Display the real negotiated
   values in the UI, not the requested ones.

## Functional Requirements

1. **Live capture** from any selected camera device via OpenCV (`cv2.VideoCapture`),
   using an explicit per-OS backend (see Phase 1).
2. **Colorization pipeline**: normalize/contrast-stretch the grayscale-converted
   frame, then apply a selectable false-color LUT (`cv2.applyColorMap` with
   `COLORMAP_JET`/`COLORMAP_INFERNO`, plus a custom "Ironbow" LUT and a
   colorblind-accessible viridis/cividis-style option). `applyColorMap` requires
   a single-channel 8-bit input — confirm dtype/shape before calling it.
3. **Hot-spot detection**:
   - Adaptive threshold (not a fixed constant) so it adjusts to ambient lighting.
   - Contour-detect candidate blobs, filter by minimum area to reduce noise.
   - Track blobs across frames (see point 6 above) and merge overlapping boxes
     with IoU-based NMS (point 5 above) before rendering.
   - Confidence score = documented combination of (a) normalized peak intensity
     within the blob and (b) temporal persistence count from the tracker.
     Document the exact formula in a docstring so it's auditable.
4. **UI** (see UI spec below).
5. **Recording/export**: save the colorized+annotated stream to a video file
   (with codec fallback per point 7), plus a snapshot-to-PNG button.
6. **Settings persistence**: remember last-used camera, palette, and confidence
   threshold between launches (simple JSON config file).
7. **Cross-platform capture backend**: `cv2.CAP_DSHOW` (or `cv2.CAP_MSMF`) on
   Windows, `cv2.CAP_AVFOUNDATION` on macOS, `cv2.CAP_V4L2` on Linux — pin
   explicitly rather than relying on OpenCV's default backend selection.
8. **Device disconnect / permission handling**: if the camera is unplugged
   mid-session, or the OS denies camera access (common on macOS without the
   right entitlement — see Packaging), catch this gracefully: stop the
   pipeline, show a clear in-UI message, let the user retry/reselect a device.
9. **Non-max suppression on detections** (see point 5 above).

## UI Requirements

Build a native desktop UI (PyQt6 or PySide6 — not Tkinter, which cannot reliably
hit 60 FPS rendering, and not a browser/Streamlit app, which adds network/DOM
overhead that causes stutter). Layout:

- **Main video panel**: live colorized feed with bounding boxes and confidence
  labels overlaid, filling most of the window.
- **Side control panel**:
  - Camera source dropdown (auto-populated device list + refresh button)
  - Color palette selector, including the colorblind-accessible option
  - Confidence threshold slider (0–100%), detections below threshold hidden
    from the overlay
  - FPS counter (current / actual negotiated target), color-coded (green ≥55,
    yellow 30–55, red <30)
  - Detection count and average confidence readout
  - Start/Stop, Snapshot, Record buttons
  - A visible "Simulated thermal — not a real temperature sensor" label in
    webcam mode, "Pre-colorized thermal feed" label in real-thermal-camera mode
- UI must never block on frame processing — enforced via the Qt signal/slot
  pattern in point 1 above.

## Performance Requirements (non-negotiable)

- **Target 60 FPS** end-to-end (capture → process → render), capped honestly by
  the camera's actual negotiated FPS (many built-in webcams top out at 30 FPS
  at usable resolutions — display the real number, not the aspirational one).
- **No dropped-frame stutter**: three-thread pipeline (capture / process / UI
  render) using the non-blocking bounded queue from point 3 above.
- Vectorized NumPy/OpenCV ops only in the hot path — no per-pixel Python loops.
- Use hardware acceleration where available (CUDA build of OpenCV, Apple
  Metal/Accelerate, OpenVINO on Intel), falling back to CPU with a visible UI
  warning if unavailable.
- `--benchmark` CLI flag that prints average/95th-percentile frame time and
  actual sustained FPS on exit, using `time.perf_counter()` around the loop.
- Configurable resolution, defaulting to a lower resolution (e.g., 640x480) if
  60 FPS can't be sustained at a higher one; UI shows the active combo.

## Accuracy / Confidence Validation

- Confidence is a heuristic score, not ground-truth accuracy — state this in
  the UI and README.
- `pytest` suite feeding synthetic NumPy frames (not a real camera) into the
  detection function, asserting: (a) a detection is produced with area/position
  roughly matching a synthetic hot spot, (b) confidence rises with synthetic
  intensity, (c) confidence is low/absent for pure noise frames, (d) the
  centroid tracker correctly persists an identity across frames when the
  synthetic blob moves slightly, and starts a new track when it jumps far.
- Log detection events (timestamp, bounding box, confidence) to a rolling CSV.

## Privacy Requirement

All processing, snapshots, recordings, and the detection log CSV must stay
strictly local — no network calls, no telemetry, no cloud upload, anywhere in
the app. State this plainly in the README.

## Recommended Tech Stack

- Python 3.11+
- OpenCV (`opencv-python`, or `opencv-contrib-python` if a CUDA build is needed)
- PyQt6 or PySide6 for UI
- NumPy for array ops
- `pytest` for tests
- Packaging: `pyinstaller` spec file for a single executable

---

## Phase 1 — Core Capture & Performance Pipeline

Goal: prove the app can sustain the target frame rate before any CV logic exists.

Tasks:
1. `camera_source.py`: device enumeration with the name/index fallback strategy
   above, and per-OS backend pinning.
2. Raw passthrough preview (no processing) to confirm capture works and to
   measure the camera's real max FPS via `cap.get(cv2.CAP_PROP_FPS)` after
   `.set()`.
3. `pipeline.py`: three-thread capture/process/render pipeline with a **no-op**
   processing stage, using the non-blocking bounded queue pattern and Qt
   signal/slot handoff to the UI thread.
4. Minimal UI: just the video panel + FPS counter, wired through Qt signals.
5. Device disconnect handling and the camera-switch race condition fix.

**Exit criteria (must report actual measured numbers before proceeding):**
- Real camera FPS ceiling measured and logged.
- End-to-end pipeline sustains that ceiling (or 60 FPS, whichever is lower)
  with no visible stutter and no UI freezes, confirmed via `--benchmark`.
- Unplugging the camera mid-run does not crash the app.

---

## Phase 2 — Detection, Accuracy & Confidence

Goal: add the actual CV work on top of the proven pipeline, without breaking
the frame rate.

Tasks:
1. Colorization pipeline (contrast stretch → colormap, all palette options).
   Re-measure FPS immediately after adding this.
2. Adaptive-threshold hot-spot detection + contour filtering.
3. Centroid-distance tracker for temporal persistence.
4. IoU-based NMS merge before rendering.
5. Documented confidence formula combining peak intensity + persistence.
6. `pytest` suite as specified in Accuracy/Confidence Validation.
7. Detection event CSV logging.
8. Re-measure FPS after each addition; if it drops below target, optimize
   (downsample before contour detection, upscale only the final overlay,
   reduce capture resolution) before moving on — don't defer optimization to
   "later."

**Exit criteria (must report actual measured numbers before proceeding):**
- FPS with full detection pipeline active still meets the Phase 1 baseline
  (or a documented, justified reduction).
- All `pytest` cases pass, including the tracker identity tests.
- Confidence formula is documented in a docstring and matches what the tests
  actually verify.

---

## Phase 3 — UI Polish, Recording, Packaging & Delivery

Goal: turn the working pipeline into a shippable app.

Tasks:
1. Full UI: palette selector (with colorblind-accessible option), confidence
   slider, detection/confidence readouts, mode labels (simulated vs
   pre-colorized real-thermal), Start/Stop/Snapshot/Record controls.
2. Recording with the VideoWriter codec-fallback logic; snapshot-to-PNG.
3. Settings persistence (JSON config).
4. Real-Thermal-Camera `CameraSource` path, clearly labeled per the caveat above.
5. Privacy statement in the README.
6. macOS packaging: `Info.plist` with `NSCameraUsageDescription`, and a README
   note that an unsigned/improperly-entitled build silently fails camera
   access (black feed, no crash) rather than erroring visibly.
7. PyInstaller spec — note that PyQt6/PySide6 builds commonly fail to locate
   Qt platform plugins unless explicitly bundled; include the necessary
   `--add-data`/hidden-import configuration and verify the built executable
   actually launches and gets camera access, not just that it compiles.
8. `README.md` (setup, hardware clarification, mode switching, FPS ceiling
   explanation, privacy statement, packaging caveats) and `PERFORMANCE.md`
   with real measured benchmark output (not fabricated numbers).

**Exit criteria:**
- Packaged executable launches on a clean machine/profile and successfully
  accesses the camera.
- README and PERFORMANCE.md reflect real, reproducible measurements.
- Full feature set from Functional Requirements is present and each item can
  be demonstrated.

---

Report measured FPS and test results at the end of every phase before starting
the next one. Do not proceed on an assumption that performance or correctness
will "probably be fine" once more features are added.
