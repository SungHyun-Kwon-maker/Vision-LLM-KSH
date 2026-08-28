#!/usr/bin/env python3
"""Educational Jetson vehicle assistant built from the course examples.

Inputs:
    recorded road video -> YOLO road objects -> HSV traffic-light state
    USB hand camera     -> MediaPipe hand gesture -> system volume
    microphone          -> Whisper.cpp -> Gemma 4 Vision -> Piper TTS

The program is a stationary classroom demonstration.  It must not be used to
control a vehicle or to decide whether real-world driving is safe.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter, deque
from dataclasses import dataclass, field, replace
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Sequence


PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_DIR.parent

DEFAULT_YOLO_MODEL = REPOSITORY_ROOT / "src/models/YOLO/yolo11n_int8.engine"
DEFAULT_GEMMA_MODEL = (
    REPOSITORY_ROOT
    / "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
)
DEFAULT_MMPROJ_MODEL = (
    REPOSITORY_ROOT
    / "src/models/Gemma4/mmproj-google_gemma-4-E2B-it-f16.gguf"
)
DEFAULT_HAND_MODEL = REPOSITORY_ROOT / "src/models/MediaPipe/hand_landmarker.task"
DEFAULT_WHISPER_CLI = REPOSITORY_ROOT / "whisper.cpp/build-cpu/bin/whisper-cli"
DEFAULT_WHISPER_MODEL = REPOSITORY_ROOT / "whisper.cpp/models/ggml-base.bin"
DEFAULT_PIPER_PYTHON = REPOSITORY_ROOT / ".piper_venv/bin/python"
DEFAULT_PIPER_MODEL = REPOSITORY_ROOT / "src/models/Piper/ko_KR-kss-medium.onnx"
DEFAULT_RUNTIME_OUTPUT = PROJECT_DIR / "runtime/vehicle_assistant"

TRAFFIC_LIGHT_CLASS_ID = 9
ROAD_CLASS_IDS = (0, 1, 2, 3, 5, 7, 9, 11)
ROAD_CLASS_NAMES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
}
ROAD_CLASS_KOREAN = {
    "person": "사람",
    "bicycle": "자전거",
    "car": "자동차",
    "motorcycle": "오토바이",
    "bus": "버스",
    "truck": "트럭",
    "traffic light": "신호등",
    "stop sign": "정지 표지판",
}

SIGNAL_STATES = ("red", "yellow", "green", "unknown")
SIGNAL_MESSAGES = {
    "red": "빨간불로 감지되었습니다. 정차 상태를 유지하세요.",
    "yellow": "노란불로 감지되었습니다. 주의가 필요합니다.",
    "green": "초록불로 감지되었습니다. 주변 교통 상황을 직접 확인하세요.",
}
STATE_COLORS = {
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "green": (0, 210, 0),
    "unknown": (180, 180, 180),
}

GESTURE_ACTIVATION_FRAMES = 6
GESTURE_RELEASE_FRAMES = 8
GESTURE_EMA_ALPHA = 0.25
GESTURE_VOLUME_STEP = 5
GESTURE_COMMAND_COOLDOWN = 0.20
PINCH_RATIO_MIN = 0.20
PINCH_RATIO_MAX = 1.50
MAX_CONTEXT_OBJECTS = 12

# Heavy runtime packages are loaded only after CLI validation.  This keeps
# --help and model-independent unit tests usable on a non-Jetson machine.
cv2: Any = None
np: Any = None
YOLO: Any = None
Llama: Any = None
Gemma4ChatHandler: Any = None
mp: Any = None
mp_python: Any = None
mp_vision: Any = None


@dataclass(frozen=True)
class SignalObservation:
    box: tuple[int, int, int, int] | None
    confidence: float
    raw_state: str
    scores: dict[str, float]


@dataclass(frozen=True)
class RoadScene:
    signal: SignalObservation
    objects: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class RoadSnapshot:
    sequence: int
    captured_at: float
    frame: Any
    scene: RoadScene
    stable_state: str
    stable_votes: int
    vision_data: dict[str, Any]
    inference_fps: float


@dataclass(frozen=True)
class GestureDecision:
    active: bool
    valid_pose: bool
    just_activated: bool
    just_deactivated: bool
    target_volume: int | None
    displayed_volume: int
    pinch_ratio: float | None


@dataclass(frozen=True)
class HandDisplaySnapshot:
    frame: Any | None
    active: bool
    valid_pose: bool
    volume: int
    status: str
    processing_fps: float


@dataclass
class InteractionState:
    busy: bool = False
    gemma_busy: bool = False
    status: str = "Ready - press V to ask"
    last_question: str = ""
    last_answer: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "busy": self.busy,
                "gemma_busy": self.gemma_busy,
                "status": self.status,
                "last_question": self.last_question,
                "last_answer": self.last_answer,
            }


class TemporalSignalFilter:
    """Require repeated YOLO/HSV samples before changing signal state."""

    def __init__(self, window_size: int = 7, required_votes: int = 5) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        if required_votes < 1 or required_votes > window_size:
            raise ValueError("required_votes must be between 1 and window_size")
        self.history: deque[str] = deque(maxlen=window_size)
        self.required_votes = required_votes
        self.stable_state = "unknown"

    def update(self, raw_state: str) -> tuple[str, bool, int]:
        if raw_state not in SIGNAL_STATES:
            raise ValueError(f"unsupported signal state: {raw_state}")
        self.history.append(raw_state)
        candidate, votes = Counter(self.history).most_common(1)[0]
        previous = self.stable_state
        if votes >= self.required_votes:
            self.stable_state = candidate
        return self.stable_state, self.stable_state != previous, votes

    def reset(self) -> None:
        self.history.clear()
        self.stable_state = "unknown"


class RateGate:
    """A deterministic wall-clock rate limiter for inference scheduling."""

    def __init__(self, rate_hz: float) -> None:
        if rate_hz <= 0:
            raise ValueError("rate_hz must be positive")
        self.interval = 1.0 / rate_hz
        self.next_due = float("-inf")

    def due(self, now: float) -> bool:
        if now < self.next_due:
            return False
        self.next_due = now + self.interval
        return True

    def reset(self) -> None:
        self.next_due = float("-inf")


def clamp_volume(value: float | int) -> int:
    return max(0, min(100, int(round(value))))


def volume_from_pinch_ratio(
    ratio: float,
    minimum: float = PINCH_RATIO_MIN,
    maximum: float = PINCH_RATIO_MAX,
    step: int = GESTURE_VOLUME_STEP,
) -> int:
    """Map a palm-normalized thumb/index distance to stepped volume."""

    if maximum <= minimum:
        raise ValueError("maximum pinch ratio must be greater than minimum")
    normalized = (ratio - minimum) / (maximum - minimum)
    raw_volume = max(0.0, min(1.0, normalized)) * 100.0
    return clamp_volume(round(raw_volume / step) * step)


class GestureVolumeFilter:
    """Debounce the control pose and smooth volume commands."""

    def __init__(
        self,
        initial_volume: int,
        activation_frames: int = GESTURE_ACTIVATION_FRAMES,
        release_frames: int = GESTURE_RELEASE_FRAMES,
        alpha: float = GESTURE_EMA_ALPHA,
        command_cooldown: float = GESTURE_COMMAND_COOLDOWN,
    ) -> None:
        if activation_frames < 1 or release_frames < 1:
            raise ValueError("gesture frame thresholds must be positive")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.activation_frames = activation_frames
        self.release_frames = release_frames
        self.alpha = alpha
        self.command_cooldown = command_cooldown
        self.active = False
        self.valid_frames = 0
        self.lost_frames = 0
        self.smoothed_volume: float | None = None
        self.last_command_volume = clamp_volume(initial_volume)
        self.last_command_time = float("-inf")

    def update(
        self,
        valid_pose: bool,
        pinch_ratio: float | None,
        now: float,
    ) -> GestureDecision:
        just_activated = False
        just_deactivated = False

        if valid_pose and pinch_ratio is not None:
            self.valid_frames += 1
            self.lost_frames = 0
            if not self.active and self.valid_frames >= self.activation_frames:
                self.active = True
                just_activated = True
                self.smoothed_volume = float(volume_from_pinch_ratio(pinch_ratio))
        else:
            self.valid_frames = 0
            self.lost_frames += 1
            if self.active and self.lost_frames >= self.release_frames:
                self.active = False
                self.smoothed_volume = None
                just_deactivated = True

        target: int | None = None
        if self.active and valid_pose and pinch_ratio is not None:
            raw_target = float(volume_from_pinch_ratio(pinch_ratio))
            if self.smoothed_volume is None:
                self.smoothed_volume = raw_target
            else:
                self.smoothed_volume = (
                    self.alpha * raw_target
                    + (1.0 - self.alpha) * self.smoothed_volume
                )
            quantized = clamp_volume(
                round(self.smoothed_volume / GESTURE_VOLUME_STEP)
                * GESTURE_VOLUME_STEP
            )
            enough_change = (
                abs(quantized - self.last_command_volume) >= GESTURE_VOLUME_STEP
            )
            cooldown_done = now - self.last_command_time >= self.command_cooldown
            if enough_change and cooldown_done:
                target = quantized
                self.last_command_volume = quantized
                self.last_command_time = now

        # Display the 5%-quantized value that was most recently issued, not
        # the intermediate EMA value.
        displayed = self.last_command_volume
        return GestureDecision(
            active=self.active,
            valid_pose=valid_pose,
            just_activated=just_activated,
            just_deactivated=just_deactivated,
            target_volume=target,
            displayed_volume=displayed,
            pinch_ratio=pinch_ratio,
        )


def point_distance(point1: Any, point2: Any) -> float:
    return math.sqrt(
        (float(point1.x) - float(point2.x)) ** 2
        + (float(point1.y) - float(point2.y)) ** 2
    )


def landmark_angle(point1: Any, point2: Any, point3: Any) -> float:
    """Return the angle at point2 using normalized MediaPipe coordinates."""

    vector1 = (
        float(point1.x) - float(point2.x),
        float(point1.y) - float(point2.y),
        float(point1.z) - float(point2.z),
    )
    vector2 = (
        float(point3.x) - float(point2.x),
        float(point3.y) - float(point2.y),
        float(point3.z) - float(point2.z),
    )
    magnitude1 = math.sqrt(sum(value * value for value in vector1))
    magnitude2 = math.sqrt(sum(value * value for value in vector2))
    if magnitude1 == 0.0 or magnitude2 == 0.0:
        return 0.0
    cosine = sum(a * b for a, b in zip(vector1, vector2)) / (
        magnitude1 * magnitude2
    )
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def hand_pose_measurements(hand: Sequence[Any]) -> tuple[bool, float | None]:
    """Check the dedicated pose and return its palm-normalized pinch ratio."""

    if len(hand) < 21:
        return False, None
    index_angle = landmark_angle(hand[5], hand[6], hand[7])
    folded_angles = (
        landmark_angle(hand[9], hand[10], hand[11]),
        landmark_angle(hand[13], hand[14], hand[15]),
        landmark_angle(hand[17], hand[18], hand[19]),
    )
    valid_pose = index_angle >= 150.0 and all(
        angle <= 135.0 for angle in folded_angles
    )
    palm_width = point_distance(hand[5], hand[17])
    if palm_width <= 1e-6:
        return False, None
    pinch_ratio = point_distance(hand[4], hand[8]) / palm_width
    return valid_pose, pinch_ratio


def horizontal_position(center_x: float, image_width: int) -> str:
    ratio = center_x / max(image_width, 1)
    if ratio < 1 / 3:
        return "왼쪽"
    if ratio > 2 / 3:
        return "오른쪽"
    return "정면"


def proximity_hint(area_ratio: float) -> str:
    """Return an image-size hint, not a physical distance measurement."""

    if area_ratio >= 0.25:
        return "화면에서 매우 크게 보임"
    if area_ratio >= 0.10:
        return "화면에서 크게 보임"
    if area_ratio >= 0.03:
        return "화면에서 중간 크기로 보임"
    return "화면에서 작게 보임"


def is_road_class(class_name: str) -> bool:
    return class_name in ROAD_CLASS_NAMES


def detection_to_context_object(
    class_id: int,
    class_name: str,
    confidence: float,
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    box_area = max(0, x2 - x1) * max(0, y2 - y1)
    area_ratio = box_area / max(image_width * image_height, 1)
    return {
        "class_id": class_id,
        "class": class_name,
        "class_ko": ROAD_CLASS_KOREAN.get(class_name, class_name),
        "confidence": round(confidence, 3),
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "horizontal_position": horizontal_position(center_x, image_width),
        "area_ratio": round(area_ratio, 4),
        "size_hint": proximity_hint(area_ratio),
    }


def make_color_masks(hsv: Any) -> dict[str, Any]:
    red_low = cv2.inRange(hsv, np.array([0, 80, 100]), np.array([12, 255, 255]))
    red_high = cv2.inRange(
        hsv,
        np.array([168, 80, 100]),
        np.array([179, 255, 255]),
    )
    red = cv2.bitwise_or(red_low, red_high)
    yellow = cv2.inRange(
        hsv,
        np.array([15, 70, 100]),
        np.array([38, 255, 255]),
    )
    # Korean traffic lights can appear blue-green in camera images.
    green = cv2.inRange(
        hsv,
        np.array([38, 55, 70]),
        np.array([105, 255, 255]),
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return {
        name: cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        for name, mask in {"red": red, "yellow": yellow, "green": green}.items()
    }


def classify_signal_color(
    roi: Any,
    minimum_ratio: float,
    minimum_dominance: float,
) -> tuple[str, dict[str, float]]:
    empty = {"red": 0.0, "yellow": 0.0, "green": 0.0}
    if roi is None or roi.size == 0:
        return "unknown", empty
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    masks = make_color_masks(hsv)
    area = float(max(hsv.shape[0] * hsv.shape[1], 1))
    scores = {
        state: float(cv2.countNonZero(mask)) / area
        for state, mask in masks.items()
    }
    best_state = max(scores, key=scores.get)
    best_score = scores[best_state]
    dominance = best_score / max(sum(scores.values()), 1e-9)
    if best_score < minimum_ratio or dominance < minimum_dominance:
        return "unknown", scores
    return best_state, scores


def detect_road_scene(
    frame: Any,
    detector: Any,
    args: argparse.Namespace,
) -> RoadScene:
    """Run one YOLO sample and select the topmost traffic light."""

    options: dict[str, Any] = {
        "source": frame,
        "conf": args.confidence,
        "iou": args.iou,
        "classes": list(ROAD_CLASS_IDS),
        "imgsz": args.image_size,
        "verbose": False,
    }
    if Path(args.yolo_model).suffix.lower() == ".pt":
        options["device"] = args.device
    result = detector.predict(**options)[0]

    empty_scores = {"red": 0.0, "yellow": 0.0, "green": 0.0}
    if result.boxes is None or len(result.boxes) == 0:
        return RoadScene(
            SignalObservation(None, 0.0, "unknown", empty_scores),
            tuple(),
        )

    height, width = frame.shape[:2]
    objects: list[dict[str, Any]] = []
    signal_candidates: list[tuple[tuple[int, int, int, int], float]] = []
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        class_name = result.names[class_id]
        if not is_road_class(class_name):
            continue
        confidence = float(box.conf[0].item())
        raw_x1, raw_y1, raw_x2, raw_y2 = box.xyxy[0].cpu().tolist()
        x1 = max(0, min(int(raw_x1), width - 1))
        y1 = max(0, min(int(raw_y1), height - 1))
        x2 = max(x1 + 1, min(int(raw_x2), width))
        y2 = max(y1 + 1, min(int(raw_y2), height))
        bbox = (x1, y1, x2, y2)

        if class_id == TRAFFIC_LIGHT_CLASS_ID:
            signal_candidates.append((bbox, confidence))
        if confidence >= args.scene_confidence or class_id == TRAFFIC_LIGHT_CLASS_ID:
            objects.append(
                detection_to_context_object(
                    class_id,
                    class_name,
                    confidence,
                    bbox,
                    width,
                    height,
                )
            )

    objects.sort(key=lambda item: item["area_ratio"], reverse=True)
    if not signal_candidates:
        return RoadScene(
            SignalObservation(None, 0.0, "unknown", empty_scores),
            tuple(objects),
        )

    if args.signal_selection == "confidence":
        selected_box, signal_confidence = max(
            signal_candidates,
            key=lambda item: item[1],
        )
    elif args.signal_selection == "largest":
        selected_box, signal_confidence = max(
            signal_candidates,
            key=lambda item: (
                (item[0][2] - item[0][0]) * (item[0][3] - item[0][1]),
                item[1],
            ),
        )
    else:
        selected_box, signal_confidence = min(
            signal_candidates,
            key=lambda item: item[0][1],
        )

    x1, y1, x2, y2 = selected_box
    raw_state, scores = classify_signal_color(
        frame[y1:y2, x1:x2],
        args.minimum_color_ratio,
        args.minimum_color_dominance,
    )
    return RoadScene(
        SignalObservation(selected_box, signal_confidence, raw_state, scores),
        tuple(objects),
    )


def build_vision_context(
    scene: RoadScene,
    stable_state: str,
    frame: Any,
    captured_at: float,
) -> dict[str, Any]:
    height, width = frame.shape[:2]
    selected_signal: dict[str, Any] | None = None
    if scene.signal.box is not None:
        x1, y1, x2, y2 = scene.signal.box
        selected_signal = {
            "class": "traffic light",
            "confidence": round(scene.signal.confidence, 3),
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "raw_color_state": scene.signal.raw_state,
            "stable_color_state": stable_state,
            "color_pixel_ratios": {
                key: round(value, 4) for key, value in scene.signal.scores.items()
            },
        }
    return {
        "timestamp": captured_at,
        "image_width": width,
        "image_height": height,
        "source": "recorded_road_video",
        "position_reference": "카메라 이미지 기준",
        "size_method": "BBox 화면 점유율이며 실제 거리 측정이 아님",
        "objects": [dict(obj) for obj in scene.objects],
        "selected_traffic_light": selected_signal,
    }


def detections_to_text(vision_data: dict[str, Any]) -> str:
    objects = vision_data.get("objects", [])
    sentences: list[str] = []
    if not objects:
        sentences.append(
            "현재 YOLO 탐지 결과에는 도로 객체가 없습니다. 탐지 실패 가능성은 있습니다."
        )
    else:
        counts = Counter(obj["class_ko"] for obj in objects)
        count_text = ", ".join(
            f"{class_name} {count}개" for class_name, count in counts.items()
        )
        sentences.append(f"탐지 개수 요약: {count_text}.")
        for index, obj in enumerate(objects[:MAX_CONTEXT_OBJECTS], start=1):
            sentences.append(
                f"{index}번 객체는 {obj['class_ko']}({obj['class']})이고 "
                f"confidence는 {obj['confidence']:.2f}이며, 화면의 "
                f"{obj['horizontal_position']}에 있고 {obj['size_hint']}입니다."
            )
        omitted = len(objects) - MAX_CONTEXT_OBJECTS
        if omitted > 0:
            sentences.append(f"나머지 {omitted}개 객체의 상세 위치는 생략했습니다.")

    signal = vision_data.get("selected_traffic_light")
    if signal is None:
        sentences.append("안정적으로 선택된 신호등은 없습니다.")
    else:
        sentences.append(
            f"선택된 신호등의 단일 샘플 색은 {signal['raw_color_state']}이고, "
            f"다중 샘플 다수결 규칙으로 확정한 상태는 "
            f"{signal['stable_color_state']}입니다."
        )
    sentences.append(
        "화면 크기 정보는 실제 거리나 충돌 가능성을 의미하지 않습니다."
    )
    return "\n".join(sentences)


def frame_to_image_data(frame: Any) -> str:
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        raise RuntimeError("도로 프레임 이미지 인코딩에 실패했습니다.")
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")


def compact_vision_context(vision_data: dict[str, Any]) -> dict[str, Any]:
    """Limit detailed objects so one image and the prompt fit n_ctx=2048."""

    compact = dict(vision_data)
    objects = vision_data.get("objects", [])
    compact["objects"] = [dict(obj) for obj in objects[:MAX_CONTEXT_OBJECTS]]
    compact["omitted_object_count"] = max(0, len(objects) - MAX_CONTEXT_OBJECTS)
    return compact


def build_gemma_messages(
    question: str,
    image_data: str,
    vision_data: dict[str, Any],
    recent_turns: Sequence[tuple[str, str]] = (),
) -> list[dict[str, Any]]:
    """Build a prompt containing exactly one current road frame."""

    history_text = (
        "\n".join(
            f"사용자: {user}\nAI: {answer}" for user, answer in recent_turns
        )
        if recent_turns
        else "없음"
    )
    system_prompt = f"""
너는 정차 상태 교육용 차량비서 AI다.
YOLO 도로 객체 Context, 규칙 기반 신호등 안정 상태, 현재 도로 이미지 한 장을 참고해 질문에 답하라.

안전 규칙:
- 신호등 안전 상태는 Gemma의 이미지 추측보다 stable_color_state를 우선한다.
- red에서는 정차 상태 유지를 안내한다.
- yellow에서는 주의가 필요하다고 안내한다.
- green에서도 출발 가능 또는 안전하다고 단정하지 말고 주변 직접 확인을 안내한다.
- unknown이면 신호를 확실히 판단할 수 없다고 말한다.
- 탐지되지 않은 객체가 실제로 없다고 단정하지 않는다.
- BBox 크기로 실제 거리, 속도, 충돌 가능성을 추측하지 않는다.
- 실제 차량 제어 명령을 만들지 않는다.
- 한국어 두 문장 이내로 답한다.

최근 대화:
{history_text}
""".strip()
    user_text = f"""
사용자 음성 질문: {question}

Vision Natural Language Context:
{detections_to_text(vision_data)}

Vision Structured Context(JSON):
{json.dumps(compact_vision_context(vision_data), ensure_ascii=False, indent=2)}
""".strip()
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_data}},
            ],
        },
    ]


def build_volume_get_command(
    backend: str,
    alsa_card: str = "default",
    alsa_control: str = "Master",
) -> list[str]:
    if backend == "wpctl":
        return ["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"]
    if backend == "pactl":
        return ["pactl", "get-sink-volume", "@DEFAULT_SINK@"]
    if backend == "amixer":
        return ["amixer", "-D", alsa_card, "get", alsa_control]
    raise ValueError(f"unsupported volume backend: {backend}")


def build_volume_set_command(
    backend: str,
    volume: int,
    alsa_card: str = "default",
    alsa_control: str = "Master",
) -> list[str]:
    value = clamp_volume(volume)
    if backend == "wpctl":
        return [
            "wpctl",
            "set-volume",
            "-l",
            "1.0",
            "@DEFAULT_AUDIO_SINK@",
            f"{value / 100:.2f}",
        ]
    if backend == "pactl":
        return ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{value}%"]
    if backend == "amixer":
        return ["amixer", "-D", alsa_card, "sset", alsa_control, f"{value}%"]
    raise ValueError(f"unsupported volume backend: {backend}")


def parse_volume_output(backend: str, output: str) -> int:
    if backend == "wpctl":
        match = re.search(r"Volume:\s*([0-9]*\.?[0-9]+)", output)
        if match:
            return clamp_volume(float(match.group(1)) * 100)
    else:
        match = re.search(r"\[?([0-9]{1,3})%\]?", output)
        if match:
            return clamp_volume(int(match.group(1)))
    raise RuntimeError(f"{backend} 볼륨 출력을 해석할 수 없습니다: {output.strip()}")


class VolumeController:
    """Control only a whitelisted system volume command."""

    def __init__(
        self,
        backend: str,
        alsa_card: str,
        alsa_control: str,
        runner: Callable[..., Any] = subprocess.run,
    ) -> None:
        self.alsa_card = alsa_card
        self.alsa_control = alsa_control
        self.runner = runner
        self.backend = self._detect_backend(backend)

    def _run(self, command: list[str]) -> Any:
        return self.runner(
            command,
            text=True,
            capture_output=True,
            check=True,
        )

    def _detect_backend(self, requested: str) -> str:
        candidates = ("wpctl", "pactl", "amixer") if requested == "auto" else (requested,)
        errors: list[str] = []
        for candidate in candidates:
            executable = build_volume_get_command(
                candidate,
                self.alsa_card,
                self.alsa_control,
            )[0]
            if shutil.which(executable) is None:
                errors.append(f"{candidate}: 실행 파일 없음")
                continue
            try:
                result = self._run(
                    build_volume_get_command(
                        candidate,
                        self.alsa_card,
                        self.alsa_control,
                    )
                )
                parse_volume_output(candidate, result.stdout)
                return candidate
            except (OSError, subprocess.SubprocessError, RuntimeError) as error:
                errors.append(f"{candidate}: {error}")
        raise RuntimeError(
            "사용 가능한 시스템 볼륨 백엔드가 없습니다. " + "; ".join(errors)
        )

    def get_volume(self) -> int:
        result = self._run(
            build_volume_get_command(
                self.backend,
                self.alsa_card,
                self.alsa_control,
            )
        )
        return parse_volume_output(self.backend, result.stdout)

    def set_volume(self, volume: int) -> int:
        value = clamp_volume(volume)
        self._run(
            build_volume_set_command(
                self.backend,
                value,
                self.alsa_card,
                self.alsa_control,
            )
        )
        return value


class SystemAudioPlayer:
    """Play Piper WAV through the same system route controlled by gestures."""

    def __init__(self, volume_backend: str, speaker_device: str) -> None:
        self.volume_backend = volume_backend
        self.speaker_device = speaker_device
        executable = {
            "wpctl": "pw-play",
            "pactl": "paplay",
            "amixer": "aplay",
        }[volume_backend]
        if shutil.which(executable) is None:
            raise RuntimeError(f"오디오 재생 명령을 찾을 수 없습니다: {executable}")
        self.executable = executable

    def play(self, audio_file: Path) -> None:
        if self.volume_backend == "amixer":
            command = [self.executable, "-D", self.speaker_device, str(audio_file)]
        else:
            command = [self.executable, str(audio_file)]
        subprocess.run(command, check=True)


class TTSWorker:
    def __init__(
        self,
        backend: str,
        piper_python: Path,
        piper_model: Path,
        output_file: Path,
        player: SystemAudioPlayer | None,
        espeak_rate: int,
        muted: bool,
    ) -> None:
        self.backend = backend
        self.piper_python = piper_python
        self.piper_model = piper_model
        self.output_file = output_file
        self.player = player
        self.espeak_rate = espeak_rate
        self.muted = muted
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=1)
        self._thread: threading.Thread | None = None
        self._espeak = shutil.which("espeak-ng") or shutil.which("espeak")
        if backend != "none":
            self._thread = threading.Thread(
                target=self._run,
                name="vehicle-tts",
                daemon=True,
            )
            self._thread.start()

    def speak(self, text: str) -> bool:
        if self.backend == "none" or self.muted:
            return False
        self._discard_pending()
        self._queue.put_nowait(text)
        return True

    def wait_until_idle(self) -> None:
        if self._thread is not None:
            self._queue.join()

    def toggle_mute(self) -> bool:
        self.muted = not self.muted
        return self.muted

    def _discard_pending(self) -> None:
        try:
            while True:
                self._queue.get_nowait()
                self._queue.task_done()
        except queue.Empty:
            return

    def _run(self) -> None:
        while True:
            text = self._queue.get()
            try:
                if text is None:
                    return
                if self.backend == "piper":
                    subprocess.run(
                        [
                            str(self.piper_python),
                            "-m",
                            "piper",
                            "-m",
                            str(self.piper_model),
                            "-f",
                            str(self.output_file),
                            "--",
                            text,
                        ],
                        check=True,
                    )
                    if self.player is None:
                        raise RuntimeError("Piper 오디오 플레이어가 없습니다.")
                    self.player.play(self.output_file)
                else:
                    if self._espeak is None:
                        raise RuntimeError("espeak-ng/espeak를 찾을 수 없습니다.")
                    subprocess.run(
                        [
                            self._espeak,
                            "-v",
                            "ko",
                            "-s",
                            str(self.espeak_rate),
                            text,
                        ],
                        check=True,
                    )
            except Exception as error:
                print(f"[TTS error] {error}", file=sys.stderr)
            finally:
                self._queue.task_done()

    def close(self) -> None:
        if self._thread is None:
            return
        self._discard_pending()
        self._queue.put_nowait(None)
        self._thread.join(timeout=3.0)


class SpeechToText:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        args.audio_input.parent.mkdir(parents=True, exist_ok=True)

    def listen(self) -> str:
        if self.args.stt_backend == "keyboard":
            return input("질문을 입력하세요: ").strip()
        command: list[str] = []
        if not self.args.no_pasuspender:
            command.extend(["pasuspender", "--"])
        command.extend(
            [
                "arecord",
                "-D",
                self.args.microphone_device,
                "-f",
                "S16_LE",
                "-r",
                "16000",
                "-c",
                "1",
                "-d",
                str(self.args.record_seconds),
                str(self.args.audio_input),
            ]
        )
        print(f"[STT] {self.args.record_seconds}초 동안 말씀하세요.")
        subprocess.run(command, check=True)
        result = subprocess.run(
            [
                str(self.args.whisper_cli),
                "-m",
                str(self.args.whisper_model),
                "-f",
                str(self.args.audio_input),
                "-l",
                "ko",
                "--no-timestamps",
            ],
            text=True,
            capture_output=True,
            check=True,
        )
        text = result.stdout.strip()
        if not text:
            raise RuntimeError("Whisper.cpp가 빈 문장을 반환했습니다.")
        return text


class GemmaRoadAssistant:
    def __init__(self, args: argparse.Namespace) -> None:
        print(f"[Gemma] Loading model: {args.gemma_model}")
        handler = Gemma4ChatHandler(clip_model_path=str(args.mmproj_model))
        self.llm = Llama(
            model_path=str(args.gemma_model),
            chat_handler=handler,
            n_gpu_layers=args.gemma_gpu_layers,
            n_ctx=args.context_window,
            n_batch=32,
            n_ubatch=32,
            verbose=False,
        )
        self.max_tokens = args.max_tokens
        self.recent_turns: deque[tuple[str, str]] = deque(maxlen=2)

    def answer(
        self,
        question: str,
        frame: Any,
        vision_data: dict[str, Any],
    ) -> str:
        messages = build_gemma_messages(
            question,
            frame_to_image_data(frame),
            vision_data,
            tuple(self.recent_turns),
        )
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=0.2,
        )
        answer = response["choices"][0]["message"]["content"].strip()
        if not answer:
            raise RuntimeError("Gemma가 빈 응답을 반환했습니다.")
        self.recent_turns.append((question, answer))
        return answer


class RoadVisionWorker:
    """Process only the most recent queued road frame at the YOLO target rate."""

    def __init__(
        self,
        detector: Any,
        args: argparse.Namespace,
        gpu_lock: threading.Lock,
        interaction: InteractionState,
        tts: TTSWorker,
    ) -> None:
        self.detector = detector
        self.args = args
        self.gpu_lock = gpu_lock
        self.interaction = interaction
        self.tts = tts
        self.signal_filter = TemporalSignalFilter(
            args.stability_window,
            args.stability_votes,
        )
        self._queue: queue.Queue[tuple[int, float, Any] | None] = queue.Queue(
            maxsize=1
        )
        self._lock = threading.Lock()
        self._latest: RoadSnapshot | None = None
        self._reset_requested = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="road-yolo",
            daemon=True,
        )
        self._last_announced_state: str | None = None
        self._last_announcement_time = float("-inf")
        self._pending_announcement: str | None = None
        self._previous_inference_time: float | None = None
        self._displayed_fps = 0.0

    def start(self) -> None:
        self._thread.start()

    def submit(self, sequence: int, captured_at: float, frame: Any) -> None:
        try:
            self._queue.get_nowait()
            self._queue.task_done()
        except queue.Empty:
            pass
        self._queue.put_nowait((sequence, captured_at, frame.copy()))

    def latest(self) -> RoadSnapshot | None:
        with self._lock:
            return self._latest

    def request_reset(self) -> None:
        self._reset_requested.set()
        with self._lock:
            self._latest = None

    def _perform_reset(self) -> None:
        self.signal_filter.reset()
        self._last_announced_state = None
        self._pending_announcement = None
        with self._lock:
            self._latest = None
        self._reset_requested.clear()

    def _announce_if_ready(self, stable_state: str) -> None:
        if self._pending_announcement != stable_state:
            return
        if self.interaction.snapshot()["busy"]:
            return
        now = time.monotonic()
        same_recent = (
            stable_state == self._last_announced_state
            and now - self._last_announcement_time < self.args.announcement_cooldown
        )
        if not same_recent:
            message = SIGNAL_MESSAGES[stable_state]
            print(f"[Signal event] {stable_state}: {message}")
            if self.tts.speak(message):
                self._last_announced_state = stable_state
                self._last_announcement_time = now
        self._pending_announcement = None

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                if self._reset_requested.is_set():
                    self._perform_reset()
                sequence, captured_at, frame = item
                started = time.perf_counter()
                with self.gpu_lock:
                    scene = detect_road_scene(frame, self.detector, self.args)
                # If the video looped while this old end-of-file frame was
                # being inferred, discard it instead of leaking state across
                # the loop boundary.
                if self._reset_requested.is_set():
                    self._perform_reset()
                    continue
                stable_state, changed, votes = self.signal_filter.update(
                    scene.signal.raw_state
                )
                if changed:
                    self._pending_announcement = (
                        stable_state if stable_state in SIGNAL_MESSAGES else None
                    )
                self._announce_if_ready(stable_state)

                finished = time.perf_counter()
                if self._previous_inference_time is None:
                    instantaneous = 1.0 / max(finished - started, 1e-9)
                else:
                    instantaneous = 1.0 / max(
                        finished - self._previous_inference_time,
                        1e-9,
                    )
                self._previous_inference_time = finished
                self._displayed_fps = (
                    instantaneous
                    if self._displayed_fps == 0.0
                    else 0.9 * self._displayed_fps + 0.1 * instantaneous
                )
                vision_data = build_vision_context(
                    scene,
                    stable_state,
                    frame,
                    captured_at,
                )
                snapshot = RoadSnapshot(
                    sequence=sequence,
                    captured_at=captured_at,
                    frame=frame,
                    scene=scene,
                    stable_state=stable_state,
                    stable_votes=votes,
                    vision_data=vision_data,
                    inference_fps=self._displayed_fps,
                )
                with self._lock:
                    self._latest = snapshot
            except Exception as error:
                print(f"[YOLO worker error] {error}", file=sys.stderr)
            finally:
                self._queue.task_done()

    def close(self) -> None:
        try:
            self._queue.get_nowait()
            self._queue.task_done()
        except queue.Empty:
            pass
        self._queue.put_nowait(None)
        self._thread.join(timeout=5.0)


class HandVolumeWorker:
    """Own the USB camera and run MediaPipe at the configured rate."""

    def __init__(
        self,
        args: argparse.Namespace,
        volume_controller: VolumeController,
    ) -> None:
        self.args = args
        self.volume_controller = volume_controller
        initial_volume = volume_controller.get_volume()
        self.gesture_filter = GestureVolumeFilter(initial_volume)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = HandDisplaySnapshot(
            frame=None,
            active=False,
            valid_pose=False,
            volume=initial_volume,
            status="Hand camera starting...",
            processing_fps=0.0,
        )
        self._thread = threading.Thread(
            target=self._run,
            name="hand-volume",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def latest(self) -> HandDisplaySnapshot:
        with self._lock:
            return self._latest

    def _publish(self, snapshot: HandDisplaySnapshot) -> None:
        with self._lock:
            self._latest = snapshot

    def _create_hand_detector(self) -> Any:
        base_options = mp_python.BaseOptions(
            model_asset_path=str(self.args.hand_model)
        )
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=self.args.hand_confidence,
            min_hand_presence_confidence=self.args.hand_confidence,
            min_tracking_confidence=self.args.hand_confidence,
        )
        return mp_vision.HandLandmarker.create_from_options(options)

    def _open_camera(self) -> Any:
        capture = cv2.VideoCapture(self.args.hand_camera_index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.hand_camera_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.hand_camera_height)
        return capture

    def _annotate(
        self,
        frame: Any,
        hand: Sequence[Any] | None,
        decision: GestureDecision,
        fps: float,
        status: str,
    ) -> Any:
        output = frame.copy()
        height, width = output.shape[:2]
        if hand is not None:
            points = [(int(point.x * width), int(point.y * height)) for point in hand]
            for connection in mp_vision.HandLandmarksConnections.HAND_CONNECTIONS:
                cv2.line(
                    output,
                    points[connection.start],
                    points[connection.end],
                    (0, 220, 0),
                    2,
                )
            for index, point in enumerate(points):
                color = (0, 0, 255) if index in (4, 8) else (255, 120, 0)
                cv2.circle(output, point, 6 if index in (4, 8) else 3, color, -1)
            cv2.line(output, points[4], points[8], (0, 255, 255), 3)

        bar_x1, bar_y1 = 25, max(80, height - 55)
        bar_x2 = min(width - 25, bar_x1 + 300)
        bar_y2 = bar_y1 + 24
        fill_x = bar_x1 + int(
            (bar_x2 - bar_x1) * decision.displayed_volume / 100
        )
        cv2.rectangle(output, (bar_x1, bar_y1), (bar_x2, bar_y2), (180, 180, 180), 2)
        cv2.rectangle(output, (bar_x1, bar_y1), (fill_x, bar_y2), (0, 220, 255), -1)
        mode = "ACTIVE" if decision.active else "LOCKED"
        lines = [
            f"Gesture volume: {decision.displayed_volume}% | {mode}",
            f"Pose: {'VALID' if decision.valid_pose else 'WAIT'} | FPS: {fps:.1f}",
            status,
        ]
        overlay = output.copy()
        cv2.rectangle(overlay, (0, 0), (width, 72), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.60, output, 0.40, 0, output)
        for index, line in enumerate(lines):
            cv2.putText(
                output,
                line,
                (15, 22 + index * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return output

    def _run(self) -> None:
        detector: Any = None
        capture: Any = None
        displayed_fps = 0.0
        previous_time: float | None = None
        try:
            detector = self._create_hand_detector()
            while not self._stop.is_set():
                if capture is None or not capture.isOpened():
                    if capture is not None:
                        capture.release()
                    capture = self._open_camera()
                    if not capture.isOpened():
                        self._publish(
                            HandDisplaySnapshot(
                                frame=None,
                                active=False,
                                valid_pose=False,
                                volume=self.gesture_filter.last_command_volume,
                                status="Hand camera unavailable - retrying",
                                processing_fps=0.0,
                            )
                        )
                        self._stop.wait(2.0)
                        continue

                loop_started = time.perf_counter()
                success, frame = capture.read()
                if not success:
                    capture.release()
                    capture = None
                    continue
                if self.args.mirror_hand_camera:
                    frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detector.detect(mp_image)
                selected_hand: Sequence[Any] | None = None
                valid_pose = False
                ratio: float | None = None
                handedness_status = "No control hand"

                for hand, categories in zip(
                    result.hand_landmarks,
                    result.handedness,
                ):
                    if not categories:
                        continue
                    category = categories[0]
                    label = category.category_name
                    # MediaPipe handedness assumes a mirrored selfie image.
                    # Swap only when the user explicitly disables mirroring.
                    if not self.args.mirror_hand_camera:
                        label = "Left" if label == "Right" else "Right"
                    score = float(category.score)
                    if (
                        label.lower() == self.args.control_hand
                        and score >= self.args.hand_confidence
                    ):
                        selected_hand = hand
                        valid_pose, ratio = hand_pose_measurements(hand)
                        handedness_status = f"{label} hand {score:.2f}"
                        break

                previous_volume = self.gesture_filter.last_command_volume
                decision = self.gesture_filter.update(
                    valid_pose,
                    ratio,
                    time.monotonic(),
                )
                status = handedness_status
                if decision.target_volume is not None:
                    try:
                        applied = self.volume_controller.set_volume(
                            decision.target_volume
                        )
                        status = f"System volume set to {applied}%"
                    except Exception as error:
                        # The filter commits optimistically. Roll back so the
                        # same gesture can retry after a transient audio error.
                        self.gesture_filter.last_command_volume = previous_volume
                        self.gesture_filter.last_command_time = float("-inf")
                        decision = replace(
                            decision,
                            target_volume=None,
                            displayed_volume=previous_volume,
                        )
                        status = f"Volume error: {error}"
                        print(f"[Volume error] {error}", file=sys.stderr)

                now = time.perf_counter()
                if previous_time is not None:
                    current_fps = 1.0 / max(now - previous_time, 1e-9)
                    displayed_fps = (
                        current_fps
                        if displayed_fps == 0.0
                        else 0.9 * displayed_fps + 0.1 * current_fps
                    )
                previous_time = now
                annotated = self._annotate(
                    frame,
                    selected_hand,
                    decision,
                    displayed_fps,
                    status,
                )
                self._publish(
                    HandDisplaySnapshot(
                        frame=annotated,
                        active=decision.active,
                        valid_pose=decision.valid_pose,
                        volume=decision.displayed_volume,
                        status=status,
                        processing_fps=displayed_fps,
                    )
                )
                elapsed = time.perf_counter() - loop_started
                self._stop.wait(max(0.0, 1.0 / self.args.hand_fps - elapsed))
        except Exception as error:
            print(f"[MediaPipe worker error] {error}", file=sys.stderr)
            self._publish(
                HandDisplaySnapshot(
                    frame=None,
                    active=False,
                    valid_pose=False,
                    volume=self.gesture_filter.last_command_volume,
                    status=f"MediaPipe failed: {error}",
                    processing_fps=0.0,
                )
            )
        finally:
            if capture is not None:
                capture.release()
            if detector is not None:
                detector.close()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)


def save_interaction_snapshot(snapshot: RoadSnapshot, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "road_vision_data.json"
    json_temp = output_dir / "road_vision_data_tmp.json"
    image_path = output_dir / "road_frame.jpg"
    image_temp = output_dir / "road_frame_tmp.jpg"
    with json_temp.open("w", encoding="utf-8") as file:
        json.dump(snapshot.vision_data, file, ensure_ascii=False, indent=2)
    os.replace(json_temp, json_path)
    if not cv2.imwrite(str(image_temp), snapshot.frame):
        raise RuntimeError("도로 스냅샷 저장에 실패했습니다.")
    os.replace(image_temp, image_path)


def start_interaction_worker(
    snapshot: RoadSnapshot,
    interaction: InteractionState,
    stt: SpeechToText,
    assistant: GemmaRoadAssistant,
    tts: TTSWorker,
    gpu_lock: threading.Lock,
    output_dir: Path,
) -> None:
    with interaction.lock:
        if interaction.busy:
            return
        interaction.busy = True
        interaction.status = "Saving current road frame..."

    def worker() -> None:
        try:
            save_interaction_snapshot(snapshot, output_dir)
            tts.wait_until_idle()
            with interaction.lock:
                interaction.status = "Waiting for microphone..."
            question = stt.listen()
            if not question:
                raise RuntimeError("음성 질문이 비어 있습니다.")
            print(f"[User/STT] {question}")
            with interaction.lock:
                interaction.last_question = question
                interaction.status = "Gemma Vision is answering..."
                interaction.gemma_busy = True
            with gpu_lock:
                answer = assistant.answer(
                    question,
                    snapshot.frame,
                    snapshot.vision_data,
                )
            with interaction.lock:
                interaction.gemma_busy = False
                interaction.last_answer = answer
                interaction.status = "Speaking Gemma answer..."
            print(f"[Gemma] {answer}")
            if tts.speak(answer):
                tts.wait_until_idle()
        except Exception as error:
            print(f"[Interaction error] {error}", file=sys.stderr)
            with interaction.lock:
                interaction.status = "Interaction failed - see terminal"
        finally:
            with interaction.lock:
                interaction.busy = False
                interaction.gemma_busy = False
                if not interaction.status.startswith("Interaction failed"):
                    interaction.status = "Ready - press V to ask"

    threading.Thread(target=worker, name="stt-gemma", daemon=True).start()


def draw_road_interface(
    frame: Any,
    snapshot: RoadSnapshot | None,
    interaction_status: str,
    tts_muted: bool,
    volume: int | None,
) -> Any:
    output = frame.copy()
    if snapshot is None:
        lines = [
            "Waiting for first YOLO sample...",
            f"Interaction: {interaction_status}",
        ]
        stable_state = "unknown"
    else:
        scene = snapshot.scene
        for obj in scene.objects:
            bbox = obj["bbox"]
            x1, y1, x2, y2 = (
                bbox["x1"],
                bbox["y1"],
                bbox["x2"],
                bbox["y2"],
            )
            selected = scene.signal.box == (x1, y1, x2, y2)
            color = STATE_COLORS[scene.signal.raw_state] if selected else (180, 180, 180)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 3 if selected else 1)
            cv2.putText(
                output,
                f"{obj['class']} {obj['confidence']:.2f} | {obj['horizontal_position']}",
                (x1, max(18, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )
        stable_state = snapshot.stable_state
        lines = [
            "V: ask Gemma | R: repeat signal | M: AI mute | Q: quit",
            (
                f"Signal raw: {scene.signal.raw_state} | stable: "
                f"{stable_state} | votes: {snapshot.stable_votes}"
            ),
            f"Road objects: {len(scene.objects)} | YOLO: {snapshot.inference_fps:.1f} FPS",
            (
                f"System volume: {volume if volume is not None else 'disabled'} | "
                f"TTS: {'MUTED' if tts_muted else 'ON'}"
            ),
            f"Interaction: {interaction_status}",
        ]

    panel_height = 20 + 27 * len(lines)
    overlay = output.copy()
    cv2.rectangle(
        overlay,
        (10, 10),
        (min(output.shape[1] - 10, 900), panel_height),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.60, output, 0.40, 0, output)
    for index, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (22, 37 + index * 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    banner = {
        "red": "RED - KEEP STOPPED",
        "yellow": "YELLOW - CAUTION",
        "green": "GREEN - CHECK SURROUNDINGS",
        "unknown": "NO RELIABLE SIGNAL",
    }[stable_state]
    cv2.putText(
        output,
        banner,
        (20, output.shape[0] - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        STATE_COLORS[stable_state],
        3,
        cv2.LINE_AA,
    )
    return output


def import_runtime_dependencies(
    load_gemma: bool,
    load_mediapipe: bool,
) -> None:
    global cv2, np, YOLO, Llama, Gemma4ChatHandler, mp, mp_python, mp_vision
    try:
        import cv2 as cv2_module
        import numpy as numpy_module
        from ultralytics import YOLO as yolo_class
    except ImportError as error:
        raise RuntimeError(
            "OpenCV, NumPy, ultralytics 설치를 확인하세요."
        ) from error
    cv2 = cv2_module
    np = numpy_module
    YOLO = yolo_class

    if load_gemma:
        try:
            from llama_cpp import Llama as llama_class
            from llama_cpp.llama_chat_format import Gemma4ChatHandler as handler_class
        except ImportError as error:
            raise RuntimeError(
                "Gemma4ChatHandler를 지원하는 llama-cpp-python이 필요합니다."
            ) from error
        Llama = llama_class
        Gemma4ChatHandler = handler_class

    if load_mediapipe:
        try:
            import mediapipe as mediapipe_module
            from mediapipe.tasks import python as mediapipe_python
            from mediapipe.tasks.python import vision as mediapipe_vision
        except ImportError as error:
            raise RuntimeError("MediaPipe 설치를 확인하세요.") from error
        mp = mediapipe_module
        mp_python = mediapipe_python
        mp_vision = mediapipe_vision


def require_file(path: Path | str, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise SystemExit(f"{label} not found: {resolved}")
    return resolved


def validate_arguments(args: argparse.Namespace) -> None:
    args.road_video = require_file(args.road_video, "Road video")
    args.yolo_model = str(require_file(args.yolo_model, "YOLO model"))
    if not args.disable_volume_control:
        args.hand_model = require_file(args.hand_model, "MediaPipe hand model")
    if not args.disable_interaction:
        args.gemma_model = require_file(args.gemma_model, "Gemma model")
        args.mmproj_model = require_file(args.mmproj_model, "Gemma mmproj model")
        if args.stt_backend == "whisper":
            args.whisper_cli = require_file(args.whisper_cli, "Whisper CLI")
            args.whisper_model = require_file(args.whisper_model, "Whisper model")
            if shutil.which("arecord") is None:
                raise SystemExit("arecord not found. Install alsa-utils.")
            if not args.no_pasuspender and shutil.which("pasuspender") is None:
                raise SystemExit(
                    "pasuspender not found. Install pulseaudio-utils or use --no-pasuspender."
                )
    if args.tts_backend == "piper":
        args.piper_python = require_file(args.piper_python, "Piper Python")
        args.piper_model = require_file(args.piper_model, "Piper model")
        config_path = Path(str(args.piper_model) + ".json")
        if not config_path.is_file():
            raise SystemExit(f"Piper model config not found: {config_path}")
    elif args.tts_backend == "espeak":
        if shutil.which("espeak-ng") is None and shutil.which("espeak") is None:
            raise SystemExit("espeak-ng/espeak not found.")

    if args.road_fps <= 0 or args.hand_fps <= 0 or args.display_fps <= 0:
        raise SystemExit("Road, hand, and display FPS must be positive.")
    if args.stability_window < 1 or args.stability_votes < 1:
        raise SystemExit("Stability window and votes must be at least 1.")
    if args.stability_votes > args.stability_window:
        raise SystemExit("--stability-votes cannot exceed --stability-window.")
    if args.image_size < 32:
        raise SystemExit("--image-size must be at least 32.")
    if args.announcement_cooldown < 0:
        raise SystemExit("--announcement-cooldown cannot be negative.")
    if args.hand_camera_width < 1 or args.hand_camera_height < 1:
        raise SystemExit("Hand camera width and height must be positive.")
    for name in (
        "confidence",
        "scene_confidence",
        "iou",
        "hand_confidence",
        "minimum_color_ratio",
        "minimum_color_dominance",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be between 0 and 1.")
    if args.context_window < 512:
        raise SystemExit("--context-window must be at least 512.")
    if args.record_seconds < 1:
        raise SystemExit("--record-seconds must be at least 1.")
    args.audio_input = Path(args.audio_input).expanduser().resolve()
    args.audio_output = Path(args.audio_output).expanduser().resolve()
    args.runtime_output = Path(args.runtime_output).expanduser().resolve()


def run(args: argparse.Namespace) -> None:
    volume_controller: VolumeController | None = None
    audio_player: SystemAudioPlayer | None = None
    volume_backend: str | None = None
    if not args.disable_volume_control:
        print("[Audio] Detecting system volume backend...")
        volume_controller = VolumeController(
            args.volume_backend,
            args.alsa_card,
            args.alsa_control,
        )
        volume_backend = volume_controller.backend
        print(
            f"[Audio] Volume backend: {volume_backend}, "
            f"current volume: {volume_controller.get_volume()}%"
        )
    elif args.tts_backend == "piper":
        # Piper still needs a default system route even when gesture control is off.
        temporary_controller = VolumeController(
            args.volume_backend,
            args.alsa_card,
            args.alsa_control,
        )
        volume_backend = temporary_controller.backend

    if args.tts_backend == "piper":
        if volume_backend is None:
            raise RuntimeError("Piper 재생용 시스템 오디오 백엔드가 없습니다.")
        audio_player = SystemAudioPlayer(volume_backend, args.speaker_device)

    tts = TTSWorker(
        args.tts_backend,
        args.piper_python,
        args.piper_model,
        args.audio_output,
        audio_player,
        args.tts_rate,
        args.mute,
    )
    interaction = InteractionState(
        status=(
            "Interaction disabled"
            if args.disable_interaction
            else "Ready - press V to ask"
        )
    )
    gpu_lock = threading.Lock()

    assistant = None if args.disable_interaction else GemmaRoadAssistant(args)
    stt = None if args.disable_interaction else SpeechToText(args)

    print(f"[YOLO] Loading model: {args.yolo_model}")
    detector = YOLO(args.yolo_model)
    road_worker = RoadVisionWorker(
        detector,
        args,
        gpu_lock,
        interaction,
        tts,
    )
    road_worker.start()

    hand_worker: HandVolumeWorker | None = None
    if not args.disable_volume_control:
        assert volume_controller is not None
        hand_worker = HandVolumeWorker(args, volume_controller)
        hand_worker.start()

    road_capture = cv2.VideoCapture(str(args.road_video))
    if not road_capture.isOpened():
        if hand_worker is not None:
            hand_worker.close()
        road_worker.close()
        tts.close()
        raise RuntimeError(f"도로 영상을 열 수 없습니다: {args.road_video}")

    source_fps = road_capture.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = args.display_fps
    playback_fps = min(source_fps, args.display_fps)
    playback_interval = 1.0 / playback_fps
    road_gate = RateGate(args.road_fps)
    sequence = 0

    print("\nEducational Vehicle Assistant")
    print(f"Road video: {args.road_video}")
    print(
        f"Display {playback_fps:.1f} FPS | YOLO {args.road_fps:.1f} FPS | "
        f"MediaPipe {args.hand_fps:.1f} FPS"
    )
    print("V: ask Gemma | R: repeat signal | M: AI mute | Q: quit")
    print("Stationary classroom demo only; no vehicle control or driving safety decision.\n")

    try:
        while True:
            loop_started = time.perf_counter()
            success, frame = road_capture.read()
            if not success:
                if args.loop:
                    road_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    road_worker.request_reset()
                    road_gate.reset()
                    continue
                break

            captured_at = time.time()
            sequence += 1
            now = time.monotonic()
            if road_gate.due(now):
                road_worker.submit(sequence, captured_at, frame)

            road_snapshot = road_worker.latest()
            interaction_snapshot = interaction.snapshot()
            hand_snapshot = hand_worker.latest() if hand_worker is not None else None
            volume = hand_snapshot.volume if hand_snapshot is not None else None
            road_output = draw_road_interface(
                frame,
                road_snapshot,
                interaction_snapshot["status"],
                tts.muted,
                volume,
            )
            cv2.imshow("Vehicle Assistant - Road", road_output)
            if hand_snapshot is not None and hand_snapshot.frame is not None:
                cv2.imshow("Vehicle Assistant - Gesture Volume", hand_snapshot.frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("m"), ord("M")):
                muted = tts.toggle_mute()
                print(f"[TTS] {'Muted' if muted else 'Unmuted'}")
            if (
                key in (ord("r"), ord("R"))
                and road_snapshot is not None
                and road_snapshot.stable_state in SIGNAL_MESSAGES
            ):
                message = SIGNAL_MESSAGES[road_snapshot.stable_state]
                print(f"[Repeat] {message}")
                tts.speak(message)
            if key in (ord("v"), ord("V")) and not args.disable_interaction:
                if interaction_snapshot["busy"]:
                    print("[Interaction] A request is already running.")
                elif road_snapshot is None:
                    print("[Interaction] 아직 YOLO 도로 스냅샷이 없습니다.")
                else:
                    assert stt is not None and assistant is not None
                    start_interaction_worker(
                        road_snapshot,
                        interaction,
                        stt,
                        assistant,
                        tts,
                        gpu_lock,
                        args.runtime_output,
                    )

            elapsed = time.perf_counter() - loop_started
            time.sleep(max(0.0, playback_interval - elapsed))
    finally:
        road_capture.release()
        cv2.destroyAllWindows()
        if hand_worker is not None:
            hand_worker.close()
        road_worker.close()
        tts.close()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recorded-road YOLO/Gemma assistant plus MediaPipe system-volume "
            "control for a stationary Jetson demo."
        )
    )
    parser.add_argument("--road-video", type=Path, required=True)
    parser.add_argument("--no-loop", action="store_false", dest="loop")
    parser.set_defaults(loop=True)
    parser.add_argument("--road-fps", type=float, default=10.0)
    parser.add_argument("--hand-fps", type=float, default=15.0)
    parser.add_argument("--display-fps", type=float, default=30.0)

    parser.add_argument("--yolo-model", default=str(DEFAULT_YOLO_MODEL))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confidence", type=float, default=0.015)
    parser.add_argument("--scene-confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument(
        "--signal-selection",
        choices=("topmost", "largest", "confidence"),
        default="topmost",
    )
    parser.add_argument("--minimum-color-ratio", type=float, default=0.01)
    parser.add_argument("--minimum-color-dominance", type=float, default=0.55)
    parser.add_argument("--stability-window", type=int, default=7)
    parser.add_argument("--stability-votes", type=int, default=5)
    parser.add_argument("--announcement-cooldown", type=float, default=5.0)

    parser.add_argument("--disable-volume-control", action="store_true")
    parser.add_argument("--hand-model", type=Path, default=DEFAULT_HAND_MODEL)
    parser.add_argument("--hand-camera-index", type=int, default=0)
    parser.add_argument("--hand-camera-width", type=int, default=640)
    parser.add_argument("--hand-camera-height", type=int, default=480)
    parser.add_argument("--hand-confidence", type=float, default=0.70)
    parser.add_argument("--control-hand", choices=("left", "right"), default="right")
    parser.add_argument(
        "--no-mirror-hand-camera",
        action="store_false",
        dest="mirror_hand_camera",
    )
    parser.set_defaults(mirror_hand_camera=True)
    parser.add_argument(
        "--volume-backend",
        choices=("auto", "wpctl", "pactl", "amixer"),
        default="auto",
    )
    parser.add_argument("--alsa-card", default="default")
    parser.add_argument("--alsa-control", default="Master")

    parser.add_argument("--disable-interaction", action="store_true")
    parser.add_argument("--gemma-model", type=Path, default=DEFAULT_GEMMA_MODEL)
    parser.add_argument("--mmproj-model", type=Path, default=DEFAULT_MMPROJ_MODEL)
    parser.add_argument("--gemma-gpu-layers", type=int, default=-1)
    parser.add_argument("--context-window", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument(
        "--stt-backend",
        choices=("whisper", "keyboard"),
        default="whisper",
    )
    parser.add_argument("--whisper-cli", type=Path, default=DEFAULT_WHISPER_CLI)
    parser.add_argument("--whisper-model", type=Path, default=DEFAULT_WHISPER_MODEL)
    parser.add_argument("--microphone-device", default="plughw:3,0")
    parser.add_argument("--record-seconds", type=int, default=5)
    parser.add_argument("--no-pasuspender", action="store_true")

    parser.add_argument(
        "--tts-backend",
        choices=("piper", "espeak", "none"),
        default="piper",
    )
    parser.add_argument("--piper-python", type=Path, default=DEFAULT_PIPER_PYTHON)
    parser.add_argument("--piper-model", type=Path, default=DEFAULT_PIPER_MODEL)
    parser.add_argument("--speaker-device", default="plughw:2,0")
    parser.add_argument("--tts-rate", type=int, default=155)
    parser.add_argument("--mute", action="store_true")
    parser.add_argument(
        "--runtime-output",
        type=Path,
        default=DEFAULT_RUNTIME_OUTPUT,
    )
    parser.add_argument(
        "--audio-input",
        type=Path,
        default=DEFAULT_RUNTIME_OUTPUT / "input.wav",
    )
    parser.add_argument(
        "--audio-output",
        type=Path,
        default=DEFAULT_RUNTIME_OUTPUT / "response.wav",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    validate_arguments(args)
    try:
        import_runtime_dependencies(
            load_gemma=not args.disable_interaction,
            load_mediapipe=not args.disable_volume_control,
        )
        run(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except RuntimeError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
