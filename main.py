import sys
import json
import csv
import argparse
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QSlider, QMessageBox, QCheckBox,
    QGroupBox, QFormLayout
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap

from camera_source import enumerate_cameras, CameraSource, CameraDevice
from pipeline import Pipeline
from colorize import colorize_frame, Palette, contrast_stretch
from detect import detect_hotspots, nms, CentroidTracker, compute_confidence, Detection


CONFIG_FILE = Path("config.json")
LOG_FILE = Path("detections.csv")


def load_config():
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except Exception:
            pass
    return {
        "camera_index": 0, "palette": 0, "confidence_threshold": 50,
        "use_motion": False, "min_area": 100, "show_detections": True
    }


def save_config(config):
    CONFIG_FILE.write_text(json.dumps(config))


def log_detection(detections: list[Detection], fps: float):
    if not detections:
        return
    row = [datetime.now().isoformat(), fps, len(detections)]
    for d in detections:
        row.extend([d.track_id, d.x, d.y, d.w, d.h, f"{d.confidence:.3f}", d.persistence])
    try:
        with LOG_FILE.open('a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(['timestamp', 'fps', 'count', 'track_id', 'x', 'y', 'w', 'h', 'confidence', 'persistence'])
            writer.writerow(row)
    except Exception:
        pass


class VideoWidget(QLabel):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: black;")
        self._frame = None
        self._detections: list[Detection] = []

    @pyqtSlot(object)
    def update_frame(self, data):
        frame, detections = data
        self._frame = frame
        self._detections = detections
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    def clear(self):
        self._frame = None
        self._detections = []
        self.setPixmap(QPixmap())


class MainWindow(QMainWindow):
    def __init__(self, benchmark: bool = False):
        super().__init__()
        self.benchmark = benchmark
        self.config = load_config()
        self.pipeline: Pipeline = None
        self.current_device: CameraDevice = None
        self._frame_times = []
        self._bench_start = time.perf_counter()

        self.prev_gray = None
        self.tracker = CentroidTracker()
        self.detection_count = 0
        self.avg_confidence = 0.0
        self.is_recording = False
        self.video_writer = None

        self.setWindowTitle("Thermal Detection App")
        self.resize(1200, 800)

        self.video = VideoWidget()
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setStyleSheet("font-size: 14px; font-weight: bold; color: green;")

        self.det_count_label = QLabel("Detections: 0")
        self.avg_conf_label = QLabel("Avg Confidence: 0%")

        self.mode_label = QLabel("Simulated thermal — not a real temperature sensor")
        self.mode_label.setStyleSheet("color: orange; font-weight: bold;")

        self.camera_combo = QComboBox()
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_cameras)

        self.palette_combo = QComboBox()
        for p in Palette:
            self.palette_combo.addItem(p.name, p)
        self.palette_combo.setCurrentIndex(self.config.get("palette", 0))
        self.palette_combo.currentIndexChanged.connect(self.on_palette_change)

        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(self.config.get("confidence_threshold", 50))
        self.conf_slider.valueChanged.connect(self.on_conf_change)
        self.conf_label = QLabel(f"Confidence: {self.conf_slider.value()}%")

        self.motion_check = QCheckBox("Use motion delta")
        self.motion_check.setChecked(self.config.get("use_motion", False))
        self.motion_check.toggled.connect(self.on_motion_change)

        self.min_area_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_area_slider.setRange(10, 2000)
        self.min_area_slider.setValue(self.config.get("min_area", 100))
        self.min_area_slider.valueChanged.connect(self.on_min_area_change)
        self.min_area_label = QLabel(f"Min Area: {self.min_area_slider.value()}")

        self.show_det_check = QCheckBox("Show detections")
        self.show_det_check.setChecked(self.config.get("show_detections", True))

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.toggle_pipeline)

        self.snapshot_btn = QPushButton("Snapshot")
        self.snapshot_btn.clicked.connect(self.take_snapshot)
        self.snapshot_btn.setEnabled(False)

        self.record_btn = QPushButton("Record")
        self.record_btn.clicked.connect(self.toggle_record)
        self.record_btn.setEnabled(False)

        left_panel = QVBoxLayout()
        left_panel.addWidget(self.video, 1)

        right_panel = QVBoxLayout()
        right_panel.setAlignment(Qt.AlignmentFlag.AlignTop)

        cam_group = QGroupBox("Camera")
        cam_layout = QFormLayout()
        cam_layout.addRow("Source:", self.camera_combo)
        cam_layout.addRow(self.refresh_btn)
        cam_group.setLayout(cam_layout)
        right_panel.addWidget(cam_group)

        proc_group = QGroupBox("Processing")
        proc_layout = QFormLayout()
        proc_layout.addRow("Palette:", self.palette_combo)
        proc_layout.addRow(self.conf_label, self.conf_slider)
        proc_layout.addRow(self.min_area_label, self.min_area_slider)
        proc_layout.addRow(self.motion_check)
        proc_layout.addRow(self.show_det_check)
        proc_group.setLayout(proc_layout)
        right_panel.addWidget(proc_group)

        stats_group = QGroupBox("Stats")
        stats_layout = QFormLayout()
        stats_layout.addRow(self.fps_label)
        stats_layout.addRow(self.det_count_label)
        stats_layout.addRow(self.avg_conf_label)
        stats_layout.addRow(self.mode_label)
        stats_group.setLayout(stats_layout)
        right_panel.addWidget(stats_group)

        ctrl_group = QGroupBox("Controls")
        ctrl_layout = QVBoxLayout()
        ctrl_layout.addWidget(self.start_btn)
        ctrl_layout.addWidget(self.snapshot_btn)
        ctrl_layout.addWidget(self.record_btn)
        ctrl_group.setLayout(ctrl_layout)
        right_panel.addWidget(ctrl_group)

        right_panel.addStretch()

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_panel, 3)
        main_layout.addLayout(right_panel, 1)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.refresh_cameras()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_stats)
        self.timer.start(500)

    def refresh_cameras(self):
        self.camera_combo.clear()
        devices = enumerate_cameras()
        for d in devices:
            self.camera_combo.addItem(f"{d.name} ({d.width}x{d.height} @ {d.fps:.0f}fps)", d)
        if devices:
            idx = min(self.config.get("camera_index", 0), len(devices) - 1)
            self.camera_combo.setCurrentIndex(idx)

    def on_palette_change(self, idx):
        self.config["palette"] = idx
        save_config(self.config)

    def on_conf_change(self, val):
        self.config["confidence_threshold"] = val
        self.conf_label.setText(f"Confidence: {val}%")
        save_config(self.config)

    def on_motion_change(self, checked):
        self.config["use_motion"] = checked
        save_config(self.config)

    def on_min_area_change(self, val):
        self.config["min_area"] = val
        self.min_area_label.setText(f"Min Area: {val}")
        save_config(self.config)

    def toggle_pipeline(self):
        if self.pipeline and self.pipeline._running:
            self.stop_pipeline()
        else:
            self.start_pipeline()

    def start_pipeline(self):
        device = self.camera_combo.currentData()
        if not device:
            QMessageBox.warning(self, "Error", "No camera selected")
            return

        self.current_device = device
        self.config["camera_index"] = self.camera_combo.currentIndex()
        save_config(self.config)

        source = CameraSource(device)
        self.pipeline = Pipeline(source, process_fn=self.process_frame)
        self.pipeline.signals.processed_ready.connect(self.video.update_frame)

        self.tracker = CentroidTracker()
        self.prev_gray = None

        if self.pipeline.start():
            self.start_btn.setText("Stop")
            self.snapshot_btn.setEnabled(True)
            self.record_btn.setEnabled(True)
        else:
            QMessageBox.warning(self, "Error", "Failed to start camera")

    def stop_pipeline(self):
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None
        self.video.clear()
        self.start_btn.setText("Start")
        self.snapshot_btn.setEnabled(False)
        self.record_btn.setEnabled(False)
        self.stop_recording()

    def process_frame(self, frame):
        palette = Palette(self.palette_combo.currentIndex())
        use_motion = self.motion_check.isChecked()
        min_area = self.min_area_slider.value()
        conf_thresh = self.conf_slider.value() / 100.0
        show_det = self.show_det_check.isChecked()

        colorized, gray = colorize_frame(frame, palette, use_motion, self.prev_gray)
        self.prev_gray = gray.copy()

        detections = detect_hotspots(gray, min_area=min_area)
        detections = nms(detections)
        detections = self.tracker.update(detections)

        for det in detections:
            det.confidence = compute_confidence(det, gray)

        detections = [d for d in detections if d.confidence >= conf_thresh]

        self.detection_count = len(detections)
        self.avg_confidence = np.mean([d.confidence for d in detections]) * 100 if detections else 0

        log_detection(detections, self.pipeline.get_fps() if self.pipeline else 0)

        if show_det and detections:
            for det in detections:
                cv2.rectangle(colorized, (det.x, det.y),
                              (det.x + det.w, det.y + det.h), (0, 255, 0), 2)
                label = f"ID:{det.track_id} {det.confidence:.0%}"
                cv2.putText(colorized, label, (det.x, det.y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if self.is_recording and self.video_writer:
            self.video_writer.write(colorized)

        return colorized, detections

    def take_snapshot(self):
        if self.video._frame is not None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"snapshot_{ts}.png", self.video._frame)

    def toggle_record(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        if self.video._frame is None:
            return
        h, w = self.video._frame.shape[:2]
        fourccs = [('mp4v', '.mp4'), ('avc1', '.mp4'), ('XVID', '.avi')]
        for fourcc, ext in fourccs:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"recording_{ts}{ext}"
            self.video_writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), 30, (w, h))
            if self.video_writer.isOpened():
                self.is_recording = True
                self.record_btn.setText("Stop Recording")
                return
        QMessageBox.warning(self, "Error", "Could not start recording")

    def stop_recording(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.is_recording = False
        self.record_btn.setText("Record")

    def update_stats(self):
        if self.pipeline:
            fps = self.pipeline.get_fps()
            self.fps_label.setText(f"FPS: {fps:.1f}")
            color = "green" if fps >= 55 else "yellow" if fps >= 30 else "red"
            self.fps_label.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {color};")

            if self.benchmark:
                self._frame_times.append(fps)

        self.det_count_label.setText(f"Detections: {self.detection_count}")
        self.avg_conf_label.setText(f"Avg Confidence: {self.avg_confidence:.0f}%")

    def closeEvent(self, event):
        self.stop_pipeline()
        if self.benchmark and self._frame_times:
            elapsed = time.perf_counter() - self._bench_start
            avg = sum(self._frame_times) / len(self._frame_times)
            p95 = sorted(self._frame_times)[int(len(self._frame_times) * 0.95)]
            print(f"\n=== BENCHMARK ===")
            print(f"Duration: {elapsed:.1f}s")
            print(f"Avg FPS: {avg:.1f}")
            print(f"P95 FPS: {p95:.1f}")
            print(f"Camera: {self.current_device.name if self.current_device else 'N/A'}")
            if self.current_device:
                print(f"Resolution: {self.current_device.width}x{self.current_device.height}")
        event.accept()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true", help="Print FPS stats on exit")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = MainWindow(benchmark=args.benchmark)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()