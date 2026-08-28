#!/usr/bin/env python3
"""Real-time traffic-light recognition with offline TTS guidance.

The program detects the traffic-light region with a COCO YOLO model, classifies
the active lamp using HSV color masks, stabilizes the result across multiple
frames, and speaks only when the stable signal state changes.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass
import importlib
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_DIR.parent
DEFAULT_YOLO_MODEL = REPOSITORY_ROOT / "src/models/YOLO/yolo11n.pt"

TRAFFIC_LIGHT_CLASS_ID = 9
SIGNAL_STATES = ("red", "yellow", "green", "unknown")

TTS_MESSAGES = {
    "red": "빨간불입니다. 정차하세요.",
    "yellow": "노란불입니다. 감속하고 정차를 준비하세요.",
    "green": "초록불입니다. 주변을 확인하고 출발하세요.",
}

STATE_COLORS = {
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "green": (0, 210, 0),
    "unknown": (180, 180, 180),
}

# Lazy-loaded runtime dependencies. This keeps --help usable on a machine that
# does not have the Jetson/OpenCV/YOLO environment installed.
cv2: Any = None
np: Any = None
YOLO: Any = None


@dataclass(frozen=True)
class SignalObservation:
    """One frame's selected traffic-light detection and color analysis."""

    box: tuple[int, int, int, int] | None
    confidence: float
    raw_state: str
    scores: dict[str, float]


class TemporalSignalFilter:
    """Require repeated frame votes before changing the public signal state."""

    def __init__(self, window_size: int, required_votes: int) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        if required_votes < 1 or required_votes > window_size:
            raise ValueError("required_votes must be between 1 and window_size")

        self.history: deque[str] = deque(maxlen=window_size)
        self.required_votes = required_votes
        self.stable_state = "unknown"

    def update(self, raw_state: str) -> tuple[str, bool, int]:
        if raw_state not in SIGNAL_STATES:
            raise ValueError(f"Unsupported signal state: {raw_state}")

        self.history.append(raw_state)
        candidate, votes = Counter(self.history).most_common(1)[0]
        previous_state = self.stable_state

        if votes >= self.required_votes:
            self.stable_state = candidate

        changed = self.stable_state != previous_state
        return self.stable_state, changed, votes

    def reset(self) -> None:
        self.history.clear()
        self.stable_state = "unknown"


class TTSWorker:
    """Speak announcements in a background thread without blocking video FPS."""

    def __init__(self, backend: str, rate: int, volume: int, muted: bool) -> None:
        self.backend = backend
        self.rate = rate
        self.volume = volume
        self.muted = muted
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=1)
        self._thread: threading.Thread | None = None
        self._executable: str | None = None
        self._pyttsx3: Any = None

        if backend == "none":
            return

        if backend == "espeak":
            self._executable = shutil.which("espeak-ng") or shutil.which("espeak")
            if self._executable is None:
                raise RuntimeError(
                    "espeak-ng was not found. Install it or use "
                    "--tts-backend pyttsx3/none."
                )
        elif backend == "pyttsx3":
            try:
                self._pyttsx3 = importlib.import_module("pyttsx3")
            except ModuleNotFoundError as error:
                raise RuntimeError(
                    "pyttsx3 is not installed. Install it or use "
                    "--tts-backend espeak/none."
                ) from error

        self._thread = threading.Thread(
            target=self._run,
            name="traffic-light-tts",
            daemon=True,
        )
        self._thread.start()

    def speak(self, message: str) -> bool:
        if self.backend == "none" or self.muted:
            return False

        # Keep only the most recent pending announcement. A stale traffic-light
        # message is less useful than the newest state.
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

        self._queue.put_nowait(message)
        return True

    def toggle_mute(self) -> bool:
        self.muted = not self.muted
        return self.muted

    def close(self) -> None:
        if self._thread is None:
            return

        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

        self._queue.put_nowait(None)
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        if self.backend == "pyttsx3":
            try:
                engine = self._pyttsx3.init()
                engine.setProperty("rate", self.rate)
                engine.setProperty("volume", min(max(self.volume / 200.0, 0.0), 1.0))
                self._select_korean_voice(engine)
            except Exception as error:
                print(f"[TTS error] pyttsx3 initialization failed: {error}", file=sys.stderr)
                return
        else:
            engine = None

        while True:
            message = self._queue.get()
            if message is None:
                return

            try:
                if self.backend == "espeak":
                    result = subprocess.run(
                        [
                            self._executable,
                            "-v",
                            "ko",
                            "-s",
                            str(self.rate),
                            "-a",
                            str(self.volume),
                            message,
                        ],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    if result.returncode != 0:
                        print(
                            f"[TTS error] espeak exited with {result.returncode}",
                            file=sys.stderr,
                        )
                else:
                    engine.say(message)
                    engine.runAndWait()
            except Exception as error:
                print(f"[TTS error] {error}", file=sys.stderr)

    @staticmethod
    def _select_korean_voice(engine: Any) -> None:
        """Select a Korean pyttsx3 voice when the system provides one."""
        for voice in engine.getProperty("voices"):
            searchable = " ".join(
                [
                    str(getattr(voice, "id", "")),
                    str(getattr(voice, "name", "")),
                    str(getattr(voice, "languages", "")),
                ]
            ).lower()
            if "korean" in searchable or "ko" in searchable:
                engine.setProperty("voice", voice.id)
                return


def import_runtime_dependencies() -> None:
    """Import the vision dependencies and report all missing packages."""
    global cv2, np, YOLO

    missing: list[str] = []
    try:
        cv2 = importlib.import_module("cv2")
    except ModuleNotFoundError:
        missing.append("Jetson OpenCV (cv2)")

    try:
        np = importlib.import_module("numpy")
    except ModuleNotFoundError:
        missing.append("numpy")

    try:
        YOLO = importlib.import_module("ultralytics").YOLO
    except ModuleNotFoundError:
        missing.append("ultralytics")

    if missing:
        packages = "\n  - ".join(missing)
        raise SystemExit(f"Missing runtime dependencies:\n  - {packages}")


def build_csi_pipeline(sensor_id: int, width: int, height: int, fps: int) -> str:
    """Build the GStreamer pipeline used by the existing course examples."""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"framerate={fps}/1 ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "queue leaky=downstream max-size-buffers=1 ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def open_video_source(args: argparse.Namespace) -> Any:
    if args.source == "csi":
        pipeline = build_csi_pipeline(
            args.sensor_id,
            args.width,
            args.height,
            args.camera_fps,
        )
        capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    elif args.source == "usb":
        capture = cv2.VideoCapture(args.camera_index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        capture.set(cv2.CAP_PROP_FPS, args.camera_fps)
    else:
        capture = cv2.VideoCapture(str(args.video_path))

    if not capture.isOpened():
        raise RuntimeError(f"카메라/영상을 열 수 없습니다: {args.source}")
    return capture


def make_color_masks(hsv: Any) -> dict[str, Any]:
    """Create masks for bright red, yellow, and green signal lamps."""
    red_low = cv2.inRange(hsv, np.array((0, 80, 100)), np.array((12, 255, 255)))
    red_high = cv2.inRange(
        hsv,
        np.array((165, 80, 100)),
        np.array((179, 255, 255)),
    )

    masks = {
        "red": red_low | red_high,
        "yellow": cv2.inRange(
            hsv,
            np.array((13, 80, 110)),
            np.array((38, 255, 255)),
        ),
        # A wide upper bound also covers cyan-looking green traffic lamps.
        "green": cv2.inRange(
            hsv,
            np.array((38, 60, 80)),
            np.array((100, 255, 255)),
        ),
    }

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for state, mask in masks.items():
        masks[state] = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return masks


def classify_signal_color(
    bgr_crop: Any,
    minimum_ratio: float,
    minimum_dominance: float,
) -> tuple[str, dict[str, float]]:
    """Classify an illuminated signal by HSV pixel ratio and dominance."""
    empty_scores = {"red": 0.0, "yellow": 0.0, "green": 0.0}
    if bgr_crop.size == 0:
        return "unknown", empty_scores

    blurred = cv2.GaussianBlur(bgr_crop, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    masks = make_color_masks(hsv)
    area = float(hsv.shape[0] * hsv.shape[1])
    scores = {
        state: float(cv2.countNonZero(mask)) / area
        for state, mask in masks.items()
    }

    best_state = max(scores, key=scores.get)
    best_score = scores[best_state]
    total_colored_ratio = sum(scores.values())
    dominance = best_score / max(total_colored_ratio, 1e-9)

    if best_score < minimum_ratio or dominance < minimum_dominance:
        return "unknown", scores
    return best_state, scores


def detect_signal(
    frame: Any,
    detector: Any,
    args: argparse.Namespace,
) -> SignalObservation:
    predict_options: dict[str, Any] = {
        "source": frame,
        "conf": args.confidence,
        "iou": args.iou,
        "classes": [TRAFFIC_LIGHT_CLASS_ID],
        "imgsz": args.image_size,
        "verbose": False,
    }
    if Path(args.yolo_model).suffix.lower() == ".pt":
        predict_options["device"] = args.device

    result = detector.predict(**predict_options)[0]
    empty_scores = {"red": 0.0, "yellow": 0.0, "green": 0.0}
    if result.boxes is None or len(result.boxes) == 0:
        return SignalObservation(None, 0.0, "unknown", empty_scores)

    height, width = frame.shape[:2]
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()

    candidates: list[tuple[tuple[int, int, int, int], float]] = []
    for box, confidence in zip(boxes, confidences):
        x1, y1, x2, y2 = box.tolist()
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        candidates.append(((x1, y1, x2, y2), float(confidence)))

    if args.signal_selection == "confidence":
        selected_box, selected_confidence = max(candidates, key=lambda item: item[1])
    else:
        selected_box, selected_confidence = max(
            candidates,
            key=lambda item: (
                (item[0][2] - item[0][0]) * (item[0][3] - item[0][1]),
                item[1],
            ),
        )

    x1, y1, x2, y2 = selected_box
    raw_state, scores = classify_signal_color(
        frame[y1:y2, x1:x2],
        minimum_ratio=args.minimum_color_ratio,
        minimum_dominance=args.minimum_color_dominance,
    )
    return SignalObservation(selected_box, selected_confidence, raw_state, scores)


def draw_interface(
    frame: Any,
    observation: SignalObservation,
    stable_state: str,
    votes: int,
    fps: float,
    muted: bool,
) -> Any:
    output = frame.copy()

    if observation.box is not None:
        x1, y1, x2, y2 = observation.box
        color = STATE_COLORS[observation.raw_state]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
        label = f"traffic light {observation.confidence:.2f} | {observation.raw_state}"
        cv2.rectangle(output, (x1, max(0, y1 - 31)), (x2, y1), color, -1)
        cv2.putText(
            output,
            label,
            (x1 + 5, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    score_text = "  ".join(
        f"{state[0].upper()}:{observation.scores[state] * 100:.1f}%"
        for state in ("red", "yellow", "green")
    )
    audio_state = "MUTED" if muted else "ON"
    lines = [
        "Q: quit | M: mute | R: repeat guidance",
        f"Raw: {observation.raw_state} | Stable: {stable_state} | Votes: {votes}",
        f"Color pixels: {score_text}",
        f"TTS: {audio_state} | FPS: {fps:.1f}",
    ]

    panel_width = min(output.shape[1] - 20, 780)
    panel_height = 25 + len(lines) * 31
    overlay = output.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_width, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.60, output, 0.40, 0, output)
    for index, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (22, 39 + index * 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    banner_text = {
        "red": "STOP",
        "yellow": "CAUTION",
        "green": "GO - CHECK SURROUNDINGS",
        "unknown": "NO RELIABLE SIGNAL",
    }[stable_state]
    banner_color = STATE_COLORS[stable_state]
    text_size = cv2.getTextSize(
        banner_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        3,
    )[0]
    banner_x = max(20, (output.shape[1] - text_size[0]) // 2)
    banner_y = output.shape[0] - 30
    cv2.putText(
        output,
        banner_text,
        (banner_x, banner_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        banner_color,
        3,
        cv2.LINE_AA,
    )
    return output


def validate_arguments(args: argparse.Namespace) -> None:
    model_path = Path(args.yolo_model).expanduser().resolve()
    if not model_path.is_file():
        raise SystemExit(f"YOLO model not found: {model_path}")
    args.yolo_model = str(model_path)

    if args.source == "video":
        if args.video_path is None:
            raise SystemExit("--video-path is required when --source video is used.")
        video_path = args.video_path.expanduser().resolve()
        if not video_path.is_file():
            raise SystemExit(f"Video file not found: {video_path}")
        args.video_path = video_path

    if args.stability_window < 1:
        raise SystemExit("--stability-window must be at least 1.")
    if args.stability_votes < 1:
        raise SystemExit("--stability-votes must be at least 1.")
    if args.stability_votes > args.stability_window:
        raise SystemExit("--stability-votes cannot exceed --stability-window.")
    if not 0.0 <= args.minimum_color_ratio <= 1.0:
        raise SystemExit("--minimum-color-ratio must be between 0 and 1.")
    if not 0.0 <= args.minimum_color_dominance <= 1.0:
        raise SystemExit("--minimum-color-dominance must be between 0 and 1.")
    if not 0 <= args.tts_volume <= 200:
        raise SystemExit("--tts-volume must be between 0 and 200.")
    if args.tts_rate < 1:
        raise SystemExit("--tts-rate must be at least 1.")


def run(args: argparse.Namespace) -> None:
    print(f"[YOLO] Loading: {args.yolo_model}")
    detector = YOLO(args.yolo_model)
    capture = open_video_source(args)
    try:
        tts = TTSWorker(
            backend=args.tts_backend,
            rate=args.tts_rate,
            volume=args.tts_volume,
            muted=args.mute,
        )
    except Exception:
        capture.release()
        raise
    signal_filter = TemporalSignalFilter(
        window_size=args.stability_window,
        required_votes=args.stability_votes,
    )

    displayed_fps = 0.0
    last_announced_state: str | None = None
    last_announcement_time = float("-inf")

    print("\nTraffic Light Voice Safety Assistant")
    print("Q: quit | M: mute/unmute | R: repeat guidance\n")

    try:
        while True:
            frame_start = time.perf_counter()
            success, frame = capture.read()

            if not success:
                if args.source == "video" and args.loop:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    signal_filter.reset()
                    last_announced_state = None
                    continue
                print("[Video] No more frames or frame read failed.", file=sys.stderr)
                break

            if args.flip is not None:
                frame = cv2.flip(frame, args.flip)

            observation = detect_signal(frame, detector, args)
            stable_state, changed, votes = signal_filter.update(observation.raw_state)

            if changed and stable_state in TTS_MESSAGES:
                now = time.monotonic()
                same_recent_state = (
                    stable_state == last_announced_state
                    and now - last_announcement_time < args.announcement_cooldown
                )
                if not same_recent_state:
                    message = TTS_MESSAGES[stable_state]
                    print(f"[Signal] {stable_state}: {message}")
                    if tts.speak(message):
                        last_announced_state = stable_state
                        last_announcement_time = now

            elapsed = max(time.perf_counter() - frame_start, 1e-9)
            current_fps = 1.0 / elapsed
            displayed_fps = (
                current_fps
                if displayed_fps == 0.0
                else 0.9 * displayed_fps + 0.1 * current_fps
            )

            output = draw_interface(
                frame,
                observation,
                stable_state,
                votes,
                displayed_fps,
                tts.muted,
            )
            cv2.imshow("Traffic Light Voice Safety Assistant", output)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("m"), ord("M")):
                muted = tts.toggle_mute()
                print(f"[TTS] {'Muted' if muted else 'Unmuted'}")
            if key in (ord("r"), ord("R")) and stable_state in TTS_MESSAGES:
                message = TTS_MESSAGES[stable_state]
                print(f"[Repeat] {message}")
                if tts.speak(message):
                    last_announced_state = stable_state
                    last_announcement_time = time.monotonic()
    finally:
        capture.release()
        cv2.destroyAllWindows()
        tts.close()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect traffic-light colors with YOLO/OpenCV and announce stable "
            "state changes through offline TTS."
        )
    )
    parser.add_argument(
        "--yolo-model",
        default=str(DEFAULT_YOLO_MODEL),
        help="Path to a COCO YOLO .pt or TensorRT .engine model.",
    )
    parser.add_argument(
        "--source",
        choices=("csi", "usb", "video"),
        default="csi",
        help="Camera/video source (default: csi).",
    )
    parser.add_argument("--sensor-id", type=int, default=0, help="CSI sensor ID.")
    parser.add_argument("--camera-index", type=int, default=0, help="USB camera index.")
    parser.add_argument("--video-path", type=Path, help="Path used with --source video.")
    parser.add_argument("--loop", action="store_true", help="Loop an input video.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument(
        "--flip",
        type=int,
        choices=(-1, 0, 1),
        default=None,
        help="OpenCV flip code: 0 vertical, 1 horizontal, -1 both.",
    )
    parser.add_argument("--device", default="cuda", help="YOLO .pt inference device.")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument(
        "--signal-selection",
        choices=("largest", "confidence"),
        default="largest",
        help="Select the largest or highest-confidence traffic light.",
    )
    parser.add_argument(
        "--minimum-color-ratio",
        type=float,
        default=0.01,
        help="Minimum active-color pixels inside the traffic-light box.",
    )
    parser.add_argument(
        "--minimum-color-dominance",
        type=float,
        default=0.55,
        help="Required winning color share among red/yellow/green pixels.",
    )
    parser.add_argument("--stability-window", type=int, default=7)
    parser.add_argument("--stability-votes", type=int, default=5)
    parser.add_argument(
        "--announcement-cooldown",
        type=float,
        default=5.0,
        help="Seconds before the same state may be announced again.",
    )
    parser.add_argument(
        "--tts-backend",
        choices=("espeak", "pyttsx3", "none"),
        default="espeak",
    )
    parser.add_argument("--tts-rate", type=int, default=155)
    parser.add_argument("--tts-volume", type=int, default=150)
    parser.add_argument("--mute", action="store_true", help="Start with TTS muted.")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    validate_arguments(args)
    import_runtime_dependencies()

    try:
        run(args)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except RuntimeError as error:
        raise SystemExit(f"Runtime error: {error}") from error


if __name__ == "__main__":
    main()
