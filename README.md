# Thermal Imaging Simulation

<div align="center">

**A real-time thermal detection application using OpenCV and Python**

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow)

</div>

---

## 📋 Description

Thermal Imaging Simulation is a real-time thermal-style detection application that works with standard webcams and UVC thermal cameras. It provides simulated thermal visualization for standard RGB cameras by applying false-color palettes based on pixel luminance and motion detection, while also supporting actual thermal camera hardware for true thermal imaging.

The application combines computer vision techniques with an intuitive PyQt6 GUI to deliver live hot-spot detection, tracking, recording, and analysis capabilities—all running locally on your machine with no cloud connectivity.

---

## ✨ Features

- **🌡️ Simulated Thermal Mode**: Converts RGB webcam feeds into thermal-style visualization using false-color palettes (Ironbow, Jet, Inferno, Viridis, Grayscale) driven by pixel luminance and frame-to-frame motion delta

- **🔥 Real Thermal Camera Support**: Compatible with UVC thermal cameras (Seek Thermal, FLIR One, etc.) that stream pre-colorized thermal feeds

- **🎯 Live Detection & Tracking**: 
  - Adaptive threshold hot-spot detection with contour filtering
  - Centroid tracking for temporal persistence
  - IoU-based non-max suppression for accuracy

- **🎨 Multiple Color Palettes**: Ironbow, Jet, Inferno, Viridis (colorblind-accessible), Grayscale

- **📹 Recording & Snapshots**: Save annotated video with automatic codec fallback and PNG snapshots

- **⚙️ Settings Persistence**: Camera, palette, and confidence threshold settings saved between sessions

- **🔒 Privacy-First**: 100% local processing—no network calls, telemetry, or cloud uploads

---

## 📚 Table of Contents

- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Configuration](#configuration)
- [Detection System](#detection-system)
- [Performance](#performance)
- [Packaging](#packaging)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## 💻 Hardware Requirements

### Standard Webcam Mode
- Any USB webcam or built-in laptop camera
- **Note**: Standard webcams are RGB sensors, NOT thermal cameras. They do not measure infrared radiation. This app simulates thermal imaging using false-color visualization based on luminance and motion.

### Thermal Camera Mode
- **UVC-compatible thermal cameras**: Seek Thermal, FLIR One, etc.
- These cameras stream pre-colorized thermal images over standard UVC protocol
- **Limitation**: Raw per-pixel temperature data is not exposed through OpenCV (requires vendor SDKs)

---

## 📦 Software Requirements

- **Python**: 3.11 or higher
- **OpenCV**: 4.0+
- **PyQt6**: For GUI
- **NumPy**: For numerical operations
- **pytest**: For running tests (optional)

### Installation

```bash
pip install opencv-python PyQt6 numpy pytest
```

Or install directly from requirements:

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Run the Application

```bash
# Basic launch
python main.py

# With performance benchmarking (prints FPS stats on exit)
python main.py --benchmark

# Run test suite
python -m pytest test_detect.py -v
```

---

## 📖 Usage Guide

### Step-by-Step Operation

1. **Select Camera**: Choose your camera from the dropdown (auto-detected on startup)
2. **Choose Palette**: Pick a color palette that suits your preference or accessibility needs
3. **Adjust Threshold**: Use the confidence threshold slider to fine-tune detection sensitivity
4. **Enable Motion Delta** (optional): Toggle "Use motion delta" for enhanced motion-based colorization
5. **Start Stream**: Click the **Start** button to begin live feed
6. **Capture Snapshot**: Click **Snapshot** to save the current frame as PNG
7. **Record Video**: Click **Record** to start/stop video recording with annotations

### Output Files

- **Snapshots**: Saved as PNG in the application directory
- **Videos**: Saved with automatic codec selection and fallback
- **Logs**: Detection data exported to `detections.csv`

---

## ⚙️ Configuration

Settings are automatically saved to `config.json` in the application directory:

```json
{
  "camera_index": 0,
  "palette": 0,
  "confidence_threshold": 50,
  "use_motion": true,
  "min_area": 100,
  "show_detections": true
}
```

### Configuration Parameters

| Parameter | Description | Type | Range |
|-----------|-------------|------|-------|
| `camera_index` | Selected camera device | Integer | 0+ |
| `palette` | Color palette | Integer | 0=Ironbow, 1=Jet, 2=Inferno, 3=Viridis, 4=Grayscale |
| `confidence_threshold` | Detection confidence cutoff | Integer | 0-100 |
| `use_motion` | Enable motion-based colorization | Boolean | true/false |
| `min_area` | Minimum detection region area (pixels) | Integer | 1+ |
| `show_detections` | Display bounding boxes | Boolean | true/false |

---

## 🔍 Detection System

### Detection Log Format

Detections are logged to `detections.csv`:

```
Timestamp, FPS, Detection_Count, TrackID, X, Y, Width, Height, Confidence, Persistence
```

### Confidence Scoring

The confidence score combines intensity and temporal persistence:

```
confidence = 0.6 × (peak_intensity / 255) + 0.4 × min(persistence / 30, 1.0)
```

**Where:**
- `peak_intensity`: Maximum pixel value in the detection region of interest (ROI)
- `persistence`: Number of frames this detection has been continuously tracked

**Interpretation:**
- Higher intensity (brighter regions) = higher confidence
- Longer tracking history = higher confidence
- Score ranges from 0 to 100

---

## ⚡ Performance

For detailed benchmark results and performance analysis, see `PERFORMANCE.md`.

### Tips for Optimal Performance

- Lower the `min_area` parameter for detecting smaller objects
- Adjust `confidence_threshold` to reduce false positives/negatives
- Disable motion delta if not needed to save processing power
- Use lower resolution input for higher FPS on slower machines

---

## 📦 Packaging

### Build Standalone Executable

```bash
pip install pyinstaller
pyinstaller thermal_app.spec
```

The executable will be created in the `dist/ThermalDetectionApp/` directory.

### Platform-Specific Notes

#### macOS

- Requires `Info.plist` with `NSCameraUsageDescription` for camera access
- Grant camera permissions in System Preferences > Security & Privacy
- Unsigned builds may fail silently (black camera feed)

#### Windows

- Should work out-of-the-box with most webcams
- Thermal camera drivers must be installed separately

#### Linux

- May require `v4l2-ctl` for camera selection
- Some thermal cameras may need additional udev rules

---

## 🔒 Privacy

**Your privacy is our priority.**

- ✅ **100% Local Processing**: All video processing, detection, and recording happens on your machine
- ✅ **No Network Calls**: Zero internet connectivity—no telemetry, analytics, or cloud uploads
- ✅ **Local Storage Only**: Snapshots, recordings, and detection logs are stored exclusively on your disk
- ✅ **No Third-Party Libraries**: No dependencies that phone home

---

## 🐛 Troubleshooting

### Black Camera Feed
- **macOS**: Grant camera permissions in System Preferences
- **Windows/Linux**: Restart the application or unplug/replug the camera
- Check that the camera index in `config.json` is correct

### No Thermal Cameras Detected
- Verify USB connection and driver installation
- Check device manager/system settings to confirm camera is recognized
- Try resetting the camera or using a different USB port

### Low FPS or Performance Issues
- Close other applications
- Reduce video resolution (if supported by your camera)
- Disable motion delta feature
- Increase `confidence_threshold` to reduce processing load

### Recording Not Working
- Ensure disk space is available
- Check file permissions in the output directory
- Try using a different codec (automatic fallback should handle most cases)

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, fork the repository, and create pull requests.

---

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing issues for similar problems
- Review the `PERFORMANCE.md` for optimization tips

---

<div align="center">

Made with ❤️ by CoderOriz

</div>
