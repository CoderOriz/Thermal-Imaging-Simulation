import queue
import threading
import time
import numpy as np
from typing import Callable, Optional
from PyQt6.QtCore import QObject, pyqtSignal


class BoundedQueue:
    def __init__(self, maxsize: int = 1):
        self._queue = queue.Queue(maxsize=maxsize)

    def put(self, item):
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(item)

    def get(self):
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def empty(self):
        return self._queue.empty()


class FrameSignals(QObject):
    processed_ready = pyqtSignal(object)


class Pipeline:
    def __init__(self, camera_source, process_fn: Callable = None):
        self.camera_source = camera_source
        self.process_fn = process_fn or (lambda f: f)

        self.capture_queue = BoundedQueue(1)
        self.signals = FrameSignals()

        self._capture_thread = None
        self._process_thread = None
        self._running = False

        self._frame_count = 0
        self._fps_start = time.perf_counter()
        self._current_fps = 0.0

    def _capture_loop(self):
        while self._running:
            frame = self.camera_source.read()
            if frame is not None:
                self.capture_queue.put(frame)
            else:
                time.sleep(0.001)

    def _process_loop(self):
        while self._running:
            frame = self.capture_queue.get()
            if frame is not None:
                processed = self.process_fn(frame)
                self.signals.processed_ready.emit(processed)
                self._frame_count += 1
            else:
                time.sleep(0.001)

    def start(self):
        if not self.camera_source.start():
            return False
        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._capture_thread.start()
        self._process_thread.start()
        return True

    def stop(self):
        self._running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=1.0)
        if self._process_thread:
            self._process_thread.join(timeout=1.0)
        self.camera_source.stop()

    def get_fps(self) -> float:
        elapsed = time.perf_counter() - self._fps_start
        if elapsed >= 1.0:
            self._current_fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_start = time.perf_counter()
        return self._current_fps