import cv2
import platform
import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CameraDevice:
    index: int
    name: str
    width: int
    height: int
    fps: float
    backend: int
    is_thermal: bool = False


def get_backend() -> int:
    system = platform.system()
    if system == "Windows":
        return cv2.CAP_DSHOW
    elif system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    else:
        return cv2.CAP_V4L2


def probe_device(index: int, backend: int) -> Optional[CameraDevice]:
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    name = f"Camera {index} ({width}x{height})"

    try:
        if platform.system() == "Windows":
            import pygrabber.dshow_graph as dshow
            devices = dshow.get_device_names()
            if index < len(devices):
                name = devices[index]
    except Exception:
        pass

    cap.release()
    return CameraDevice(index, name, width, height, fps, backend)


def enumerate_cameras(max_index: int = 10) -> List[CameraDevice]:
    backend = get_backend()
    devices = []
    for i in range(max_index):
        dev = probe_device(i, backend)
        if dev:
            devices.append(dev)
        else:
            break
    return devices


def enumerate_thermal_cameras(max_index: int = 10) -> List[CameraDevice]:
    """Enumerate UVC thermal cameras (Seek Thermal, FLIR One, etc.).
    These appear as standard video devices but stream pre-colorized thermal feeds."""
    backend = get_backend()
    devices = []
    thermal_keywords = ['seek', 'flir', 'thermal', 'therm', 'lepton', 'boson']
    for i in range(max_index):
        dev = probe_device(i, backend)
        if dev:
            name_lower = dev.name.lower()
            if any(kw in name_lower for kw in thermal_keywords):
                dev.is_thermal = True
                dev.name += " (Pre-colorized thermal feed)"
            devices.append(dev)
        else:
            break
    return devices


class CameraSource:
    def __init__(self, device: CameraDevice):
        self.device = device
        self.cap: Optional[cv2.VideoCapture] = None
        self._running = False

    def start(self) -> bool:
        self.cap = cv2.VideoCapture(self.device.index, self.device.backend)
        if not self.cap.isOpened():
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.device.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.device.height)
        self.cap.set(cv2.CAP_PROP_FPS, 60)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.device.width = actual_w
        self.device.height = actual_h
        self.device.fps = actual_fps if actual_fps > 0 else 30.0

        self._running = True
        return True

    def read(self) -> Optional["np.ndarray"]:
        if not self.cap or not self._running:
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def stop(self):
        self._running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()