#!/usr/bin/env python3
"""Traffic-light Vision-LLM audiovisual assistant for NVIDIA Jetson.

Pipeline:
    camera -> YOLO traffic-light ROI -> HSV event detection -> Piper TTS
                                          |
              microphone -> Whisper.cpp -> Gemma 4 Vision -> Piper TTS

Safety-critical stop/go state is determined by deterministic vision rules.
Gemma receives the selected ROI, structured detection context, and the user's
speech question to generate a short Korean explanation.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter, deque
from dataclasses import dataclass, field
import importlib
import json
import os
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

DEFAULT_YOLO_MODEL = REPOSITORY_ROOT / "src/models/YOLO/yolo11n_int8.engine"
DEFAULT_GEMMA_MODEL = (
    REPOSITORY_ROOT
    / "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
)
DEFAULT_MMPROJ_MODEL = (
    REPOSITORY_ROOT
    / "src/models/Gemma4/mmproj-google_gemma-4-E2B-it-f16.gguf"
)
DEFAULT_WHISPER_CLI = REPOSITORY_ROOT / "whisper.cpp/build-cpu/bin/whisper-cli"
DEFAULT_WHISPER_MODEL = REPOSITORY_ROOT / "whisper.cpp/models/ggml-base.bin"
DEFAULT_PIPER_PYTHON = REPOSITORY_ROOT / ".piper_venv/bin/python"
DEFAULT_PIPER_MODEL = REPOSITORY_ROOT / "src/models/Piper/ko_KR-kss-medium.onnx"
DEFAULT_AUDIO_INPUT = REPOSITORY_ROOT / "src/audio/input.wav"
DEFAULT_AUDIO_OUTPUT = REPOSITORY_ROOT / "src/audio/response.wav"
DEFAULT_RUNTIME_OUTPUT = PROJECT_DIR / "runtime"

TRAFFIC_LIGHT_CLASS_ID = 9
SIGNAL_STATES = ("red", "yellow", "green", "unknown")

SAFETY_MESSAGES = {
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

# Lazy-loaded after argument validation so --help and pure logic tests do not
# require the Jetson AI environment.
cv2: Any = None
np: Any = None
YOLO: Any = None
Llama: Any = None
Gemma4ChatHandler: Any = None


@dataclass(frozen=True)
class SignalObservation:
    """One frame's selected traffic-light detection and color analysis."""

    box: tuple[int, int, int, int] | None
    confidence: float
    raw_state: str
    scores: dict[str, float]


@dataclass(frozen=True)
class SceneObservation:
    """All YOLO objects plus the traffic light selected for safety analysis."""

    signal: SignalObservation
    objects: tuple[dict[str, Any], ...]


@dataclass
class InteractionState:
    """Thread-safe status shared by the camera and STT/Gemma worker."""

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
    """Require repeated frame votes before changing the safety state."""

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
        previous = self.stable_state

        if votes >= self.required_votes:
            self.stable_state = candidate

        return self.stable_state, self.stable_state != previous, votes

    def reset(self) -> None:
        self.history.clear()
        self.stable_state = "unknown"


class TTSWorker:
    """Serialize Piper/espeak announcements without blocking the camera loop."""

    def __init__(
        self,
        backend: str,
        piper_python: Path,
        piper_model: Path,
        output_file: Path,
        speaker_device: str,
        rate: int,
        volume: int,
        muted: bool,
    ) -> None:
        self.backend = backend
        self.piper_python = piper_python
        self.piper_model = piper_model
        self.output_file = output_file
        self.speaker_device = speaker_device
        self.rate = rate
        self.volume = volume
        self.muted = muted
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=1)
        self._thread: threading.Thread | None = None
        self._espeak_executable: str | None = None

        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        if backend == "none":
            return
        if backend == "espeak":
            self._espeak_executable = (
                shutil.which("espeak-ng") or shutil.which("espeak")
            )

        self._thread = threading.Thread(
            target=self._run,
            name="piper-tts",
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

    def close(self) -> None:
        if self._thread is None:
            return
        self._discard_pending()
        self._queue.put_nowait(None)
        self._thread.join(timeout=3.0)

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
                    self._speak_with_piper(text)
                else:
                    self._speak_with_espeak(text)
            except Exception as error:
                print(f"[TTS error] {error}", file=sys.stderr)
            finally:
                self._queue.task_done()

    def _speak_with_piper(self, text: str) -> None:
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
        subprocess.run(
            [
                "aplay",
                "-D",
                self.speaker_device,
                str(self.output_file),
            ],
            check=True,
        )

    def _speak_with_espeak(self, text: str) -> None:
        subprocess.run(
            [
                self._espeak_executable,
                "-v",
                "ko",
                "-s",
                str(self.rate),
                "-a",
                str(self.volume),
                text,
            ],
            check=True,
        )


class SpeechToText:
    """Record with arecord and transcribe with the course Whisper.cpp CLI."""

    def __init__(
        self,
        backend: str,
        whisper_cli: Path,
        whisper_model: Path,
        audio_file: Path,
        microphone_device: str,
        record_seconds: int,
        use_pasuspender: bool,
    ) -> None:
        self.backend = backend
        self.whisper_cli = whisper_cli
        self.whisper_model = whisper_model
        self.audio_file = audio_file
        self.microphone_device = microphone_device
        self.record_seconds = record_seconds
        self.use_pasuspender = use_pasuspender
        self.audio_file.parent.mkdir(parents=True, exist_ok=True)

    def listen(self) -> str:
        if self.backend == "keyboard":
            return input("질문을 입력하세요: ").strip()

        command: list[str] = []
        if self.use_pasuspender:
            command.extend(["pasuspender", "--"])
        command.extend(
            [
                "arecord",
                "-D",
                self.microphone_device,
                "-f",
                "S16_LE",
                "-r",
                "16000",
                "-c",
                "1",
                "-d",
                str(self.record_seconds),
                str(self.audio_file),
            ]
        )

        print(f"[STT] {self.record_seconds}초 동안 말씀하세요.")
        subprocess.run(command, check=True)

        result = subprocess.run(
            [
                str(self.whisper_cli),
                "-m",
                str(self.whisper_model),
                "-f",
                str(self.audio_file),
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


class GemmaVisionAssistant:
    """Answer a speech question using ROI image and structured YOLO context."""

    def __init__(
        self,
        model_path: Path,
        mmproj_path: Path,
        context_window: int,
        gpu_layers: int,
        max_tokens: int,
    ) -> None:
        print(f"[Gemma] Loading model: {model_path}")
        chat_handler = Gemma4ChatHandler(clip_model_path=str(mmproj_path))
        self.llm = Llama(
            model_path=str(model_path),
            chat_handler=chat_handler,
            n_gpu_layers=gpu_layers,
            n_ctx=context_window,
            n_batch=32,
            n_ubatch=32,
            verbose=False,
        )
        self.max_tokens = max_tokens
        self.recent_turns: deque[tuple[str, str]] = deque(maxlen=2)

    def answer(
        self,
        question: str,
        roi_image: Any,
        vision_data: dict[str, Any],
    ) -> str:
        success, buffer = cv2.imencode(".jpg", roi_image)
        if not success:
            raise RuntimeError("신호등 ROI 이미지 인코딩에 실패했습니다.")

        image_base64 = base64.b64encode(buffer).decode("utf-8")
        image_data = "data:image/jpeg;base64," + image_base64
        vision_json = json.dumps(vision_data, ensure_ascii=False, indent=2)
        vision_text = detections_to_text(vision_data)

        if self.recent_turns:
            history_text = "\n".join(
                f"사용자: {user}\nAI: {assistant}"
                for user, assistant in self.recent_turns
            )
        else:
            history_text = "없음"

        system_prompt = f"""
너는 차량용 신호등 안전 보조 AI다.
YOLO가 선택한 신호등 ROI 이미지, 구조화된 탐지 결과, 규칙 기반 안정 상태를 참고해 사용자의 질문에 답하라.

안전 규칙:
- stable_state가 red이면 반드시 정차를 안내하고 출발해도 된다고 말하지 마라.
- stable_state가 yellow이면 감속하고 정차 준비를 안내하라.
- stable_state가 green이면 주변과 보행자를 확인한 뒤 출발하도록 안내하라.
- stable_state가 unknown이면 확실히 판별할 수 없다고 답하라.
- 규칙 기반 stable_state를 안전 제어의 기준으로 사용하라.
- 이미지나 탐지 정보에 없는 사실을 추측하지 마라.
- 한국어 두 문장 이내로 간결하게 답하라.

최근 대화(Sliding Window Memory):
{history_text}
""".strip()

        user_text = f"""
사용자 음성 질문: {question}

Vision Natural Language Context:
{vision_text}

Vision Structured Context(JSON):
{vision_json}
""".strip()

        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data},
                        },
                    ],
                },
            ],
            max_tokens=self.max_tokens,
            temperature=0.2,
        )
        answer = response["choices"][0]["message"]["content"].strip()
        if not answer:
            raise RuntimeError("Gemma가 빈 응답을 반환했습니다.")

        self.recent_turns.append((question, answer))
        return answer


def import_runtime_dependencies(load_gemma: bool) -> None:
    """Import vision and optional Gemma packages after CLI validation."""
    global cv2, np, YOLO, Llama, Gemma4ChatHandler

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

    if load_gemma:
        try:
            llama_cpp = importlib.import_module("llama_cpp")
            chat_format = importlib.import_module("llama_cpp.llama_chat_format")
            Llama = llama_cpp.Llama
            Gemma4ChatHandler = chat_format.Gemma4ChatHandler
        except (ModuleNotFoundError, AttributeError):
            missing.append("llama-cpp-python with Gemma4ChatHandler")

    if missing:
        packages = "\n  - ".join(missing)
        raise SystemExit(f"Missing runtime dependencies:\n  - {packages}")


def build_csi_pipeline(sensor_id: int, width: int, height: int, fps: int) -> str:
    """Use the same Jetson GStreamer pipeline as the course YOLO examples."""
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
    """Create HSV masks for bright red, yellow, and green signal lamps."""
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
        # Korean traffic lights may look cyan; include the blue-green hue area.
        "green": cv2.inRange(
            hsv,
            np.array((38, 60, 80)),
            np.array((110, 255, 255)),
        ),
    }

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return {
        state: cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        for state, mask in masks.items()
    }


def classify_signal_color(
    bgr_crop: Any,
    minimum_ratio: float,
    minimum_dominance: float,
) -> tuple[str, dict[str, float]]:
    """Classify the active lamp using HSV ratio and winning-color dominance."""
    empty = {"red": 0.0, "yellow": 0.0, "green": 0.0}
    if bgr_crop.size == 0:
        return "unknown", empty

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
    dominance = best_score / max(sum(scores.values()), 1e-9)
    if best_score < minimum_ratio or dominance < minimum_dominance:
        return "unknown", scores
    return best_state, scores


def detect_scene(
    frame: Any,
    detector: Any,
    args: argparse.Namespace,
) -> SceneObservation:
    """Run YOLO once, retain the full scene, and select one traffic light."""
    options: dict[str, Any] = {
        "source": frame,
        "conf": args.confidence,
        "iou": args.iou,
        "classes": None,
        "imgsz": args.image_size,
        "verbose": False,
    }
    if Path(args.yolo_model).suffix.lower() == ".pt":
        options["device"] = args.device

    result = detector.predict(**options)[0]
    empty = {"red": 0.0, "yellow": 0.0, "green": 0.0}
    if result.boxes is None or len(result.boxes) == 0:
        return SceneObservation(
            SignalObservation(None, 0.0, "unknown", empty),
            tuple(),
        )

    height, width = frame.shape[:2]
    candidates: list[tuple[tuple[int, int, int, int], float]] = []
    objects: list[dict[str, Any]] = []
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
        x1 = max(0, min(int(x1), width - 1))
        y1 = max(0, min(int(y1), height - 1))
        x2 = max(x1 + 1, min(int(x2), width))
        y2 = max(y1 + 1, min(int(y2), height))
        bbox = (x1, y1, x2, y2)

        if class_id == TRAFFIC_LIGHT_CLASS_ID:
            candidates.append((bbox, confidence))

        # Traffic lights use a deliberately low threshold because they are
        # small. Other scene objects follow the 0.25 threshold used in the new
        # cam-json-llm.py answer code to avoid noisy Gemma context.
        if confidence >= args.scene_confidence or class_id == TRAFFIC_LIGHT_CLASS_ID:
            objects.append(
                {
                    "class_id": class_id,
                    "class": result.names[class_id],
                    "confidence": round(confidence, 3),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                }
            )

    if not candidates:
        return SceneObservation(
            SignalObservation(None, 0.0, "unknown", empty),
            tuple(objects),
        )

    if args.signal_selection == "topmost":
        selected_box, confidence = min(candidates, key=lambda item: item[0][1])
    elif args.signal_selection == "confidence":
        selected_box, confidence = max(candidates, key=lambda item: item[1])
    else:
        selected_box, confidence = max(
            candidates,
            key=lambda item: (
                (item[0][2] - item[0][0]) * (item[0][3] - item[0][1]),
                item[1],
            ),
        )

    x1, y1, x2, y2 = selected_box
    raw_state, scores = classify_signal_color(
        frame[y1:y2, x1:x2],
        args.minimum_color_ratio,
        args.minimum_color_dominance,
    )
    return SceneObservation(
        SignalObservation(selected_box, confidence, raw_state, scores),
        tuple(objects),
    )


def build_vision_context(
    scene: SceneObservation,
    stable_state: str,
    frame: Any,
) -> dict[str, Any]:
    """Convert all YOLO results and the signal event to section-06 JSON."""
    height, width = frame.shape[:2]
    observation = scene.signal
    objects = [
        {
            "class_id": obj["class_id"],
            "class": obj["class"],
            "confidence": obj["confidence"],
            "bbox": dict(obj["bbox"]),
        }
        for obj in scene.objects
    ]

    selected_signal: dict[str, Any] | None = None
    if observation.box is not None:
        x1, y1, x2, y2 = observation.box
        selected_signal = {
            "class": "traffic light",
            "confidence": round(observation.confidence, 3),
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "raw_color_state": observation.raw_state,
            "stable_color_state": stable_state,
            "color_pixel_ratios": {
                key: round(value, 4)
                for key, value in observation.scores.items()
            },
        }

    return {
        "timestamp": time.time(),
        "image_width": width,
        "image_height": height,
        "objects": objects,
        "selected_traffic_light": selected_signal,
    }


def detections_to_text(vision_data: dict[str, Any]) -> str:
    """Convert full-frame YOLO JSON to the answer code's sentence template."""
    objects = vision_data.get("objects", [])
    sentences: list[str] = []
    if not objects:
        sentences.append("현재 탐지된 객체가 없습니다.")
    else:
        for index, obj in enumerate(objects, start=1):
            bbox = obj["bbox"]
            sentences.append(
                f"{index}번 객체는 {obj['class']}이며, "
                f"confidence는 {obj['confidence']:.2f}이고, "
                f"bbox는 ({bbox['x1']}, {bbox['y1']}, "
                f"{bbox['x2']}, {bbox['y2']})입니다."
            )

    signal = vision_data.get("selected_traffic_light")
    if signal is None:
        sentences.append("안전 이벤트 대상으로 선택된 신호등은 없습니다.")
    else:
        sentences.append(
            f"선택된 신호등의 HSV 단일 프레임 판정은 "
            f"{signal['raw_color_state']}이고, 다중 프레임 규칙 기반 "
            f"안정 상태는 {signal['stable_color_state']}입니다."
        )

    return "\n".join(sentences)


def save_vision_snapshot(
    vision_data: dict[str, Any],
    roi_image: Any,
    output_dir: Path,
) -> None:
    """Atomically save JSON and ROI image as taught in section 06."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "vision_data.json"
    json_temp_path = output_dir / "vision_data_tmp.json"
    image_path = output_dir / "traffic_light_roi.jpg"
    image_temp_path = output_dir / "traffic_light_roi_tmp.jpg"

    with json_temp_path.open("w", encoding="utf-8") as file:
        json.dump(vision_data, file, ensure_ascii=False, indent=2)
    os.replace(json_temp_path, json_path)

    if not cv2.imwrite(str(image_temp_path), roi_image):
        raise RuntimeError("ROI 이미지 저장에 실패했습니다.")
    os.replace(image_temp_path, image_path)


def start_interaction_worker(
    state: InteractionState,
    stt: SpeechToText,
    assistant: GemmaVisionAssistant,
    tts: TTSWorker,
    roi_image: Any,
    vision_data: dict[str, Any],
    gpu_lock: threading.Lock,
) -> None:
    """Run microphone -> Whisper -> Gemma Vision -> Piper off the UI loop."""
    with state.lock:
        if state.busy:
            return
        state.busy = True
        state.status = "Waiting for microphone..."

    def worker() -> None:
        try:
            # Do not let a previous safety announcement leak into the microphone.
            tts.wait_until_idle()
            question = stt.listen()
            if not question:
                raise RuntimeError("음성 질문이 비어 있습니다.")

            print(f"[User/STT] {question}")
            with state.lock:
                state.last_question = question
                state.status = "Gemma Vision is answering..."
                state.gemma_busy = True

            # YOLO and Gemma share Jetson GPU memory. Serialize the two inference
            # calls and let the camera loop skip YOLO while Gemma is active.
            with gpu_lock:
                answer = assistant.answer(question, roi_image, vision_data)

            with state.lock:
                state.gemma_busy = False
                state.last_answer = answer
                state.status = "Speaking Gemma answer..."

            print(f"[Gemma] {answer}")
            if tts.speak(answer):
                tts.wait_until_idle()
        except Exception as error:
            print(f"[Interaction error] {error}", file=sys.stderr)
            with state.lock:
                state.status = "Interaction failed - see terminal"
        finally:
            with state.lock:
                state.busy = False
                state.gemma_busy = False
                if not state.status.startswith("Interaction failed"):
                    state.status = "Ready - press V to ask"

    threading.Thread(target=worker, name="stt-gemma", daemon=True).start()


def draw_interface(
    frame: Any,
    scene: SceneObservation,
    stable_state: str,
    votes: int,
    fps: float,
    muted: bool,
    interaction_status: str,
    interaction_enabled: bool,
) -> Any:
    output = frame.copy()
    observation = scene.signal

    # Show the same full-frame object information used for JSON/Gemma context.
    for obj in scene.objects:
        bbox = obj["bbox"]
        object_box = (bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"])
        if observation.box is not None and object_box == observation.box:
            continue
        x1, y1, x2, y2 = object_box
        cv2.rectangle(output, (x1, y1), (x2, y2), (160, 160, 160), 1)
        cv2.putText(
            output,
            f"{obj['class']} {obj['confidence']:.2f}",
            (x1, max(18, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

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
    voice_key = "V: ask Gemma | " if interaction_enabled else ""
    lines = [
        f"{voice_key}Q: quit | M: mute | R: repeat",
        f"Raw: {observation.raw_state} | Stable: {stable_state} | Votes: {votes}",
        f"Scene objects in Context: {len(scene.objects)}",
        f"Color pixels: {score_text}",
        f"Interaction: {interaction_status}",
        f"TTS: {'MUTED' if muted else 'ON'} | FPS: {fps:.1f}",
    ]

    panel_width = min(output.shape[1] - 20, 850)
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
            0.66,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    banner = {
        "red": "STOP",
        "yellow": "CAUTION",
        "green": "GO - CHECK SURROUNDINGS",
        "unknown": "NO RELIABLE SIGNAL",
    }[stable_state]
    text_size = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
    cv2.putText(
        output,
        banner,
        (max(20, (output.shape[1] - text_size[0]) // 2), output.shape[0] - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        STATE_COLORS[stable_state],
        3,
        cv2.LINE_AA,
    )
    return output


def validate_arguments(args: argparse.Namespace) -> None:
    """Resolve required files before importing large AI dependencies."""

    def require_file(value: Path | str, label: str) -> Path:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"{label} not found: {path}")
        return path

    args.yolo_model = str(require_file(args.yolo_model, "YOLO model"))

    if args.source == "video":
        if args.video_path is None:
            raise SystemExit("--video-path is required with --source video.")
        args.video_path = require_file(args.video_path, "Video file")

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
                    "pasuspender not found. Install pulseaudio-utils or use "
                    "--no-pasuspender."
                )

    if args.tts_backend == "piper":
        args.piper_python = require_file(args.piper_python, "Piper Python")
        args.piper_model = require_file(args.piper_model, "Piper model")
        piper_config = Path(str(args.piper_model) + ".json")
        if not piper_config.is_file():
            raise SystemExit(f"Piper model config not found: {piper_config}")
        if shutil.which("aplay") is None:
            raise SystemExit("aplay not found. Install alsa-utils.")
    elif args.tts_backend == "espeak":
        if shutil.which("espeak-ng") is None and shutil.which("espeak") is None:
            raise SystemExit("espeak-ng/espeak not found.")

    if args.stability_window < 1 or args.stability_votes < 1:
        raise SystemExit("Stability window and votes must be at least 1.")
    if args.stability_votes > args.stability_window:
        raise SystemExit("--stability-votes cannot exceed --stability-window.")
    if not 0.0 <= args.confidence <= 1.0:
        raise SystemExit("--confidence must be between 0 and 1.")
    if not 0.0 <= args.scene_confidence <= 1.0:
        raise SystemExit("--scene-confidence must be between 0 and 1.")
    if not 0.0 <= args.minimum_color_ratio <= 1.0:
        raise SystemExit("--minimum-color-ratio must be between 0 and 1.")
    if not 0.0 <= args.minimum_color_dominance <= 1.0:
        raise SystemExit("--minimum-color-dominance must be between 0 and 1.")
    if args.record_seconds < 1:
        raise SystemExit("--record-seconds must be at least 1.")
    if not 0 <= args.tts_volume <= 200:
        raise SystemExit("--tts-volume must be between 0 and 200.")

    args.audio_input = Path(args.audio_input).expanduser().resolve()
    args.audio_output = Path(args.audio_output).expanduser().resolve()
    args.runtime_output = Path(args.runtime_output).expanduser().resolve()


def run(args: argparse.Namespace) -> None:
    print(f"[YOLO] Loading model: {args.yolo_model}")
    detector = YOLO(args.yolo_model)

    if args.disable_interaction:
        assistant = None
        stt = None
    else:
        assistant = GemmaVisionAssistant(
            args.gemma_model,
            args.mmproj_model,
            args.context_window,
            args.gemma_gpu_layers,
            args.max_tokens,
        )
        stt = SpeechToText(
            args.stt_backend,
            args.whisper_cli,
            args.whisper_model,
            args.audio_input,
            args.microphone_device,
            args.record_seconds,
            not args.no_pasuspender,
        )

    capture = open_video_source(args)
    try:
        tts = TTSWorker(
            args.tts_backend,
            args.piper_python,
            args.piper_model,
            args.audio_output,
            args.speaker_device,
            args.tts_rate,
            args.tts_volume,
            args.mute,
        )
    except Exception:
        capture.release()
        raise

    signal_filter = TemporalSignalFilter(
        args.stability_window,
        args.stability_votes,
    )
    interaction = InteractionState(
        status=(
            "Interaction disabled"
            if args.disable_interaction
            else "Ready - press V to ask"
        )
    )
    gpu_lock = threading.Lock()

    stable_state = "unknown"
    stable_votes = 0
    last_scene = SceneObservation(
        SignalObservation(
            None,
            0.0,
            "unknown",
            {"red": 0.0, "yellow": 0.0, "green": 0.0},
        ),
        tuple(),
    )
    displayed_fps = 0.0
    last_announced_state: str | None = None
    last_announcement_time = float("-inf")

    print("\nTraffic Light Vision-LLM Audiovisual Assistant")
    print("Q: quit | M: mute | R: repeat safety guidance")
    if not args.disable_interaction:
        print("V: record a question and ask Gemma Vision")
    print()

    try:
        while True:
            frame_start = time.perf_counter()
            success, frame = capture.read()
            if not success:
                if args.source == "video" and args.loop:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    signal_filter.reset()
                    stable_state = "unknown"
                    last_announced_state = None
                    continue
                print("[Video] No more frames or frame read failed.", file=sys.stderr)
                break

            if args.flip is not None:
                frame = cv2.flip(frame, args.flip)

            interaction_snapshot = interaction.snapshot()
            if not interaction_snapshot["gemma_busy"]:
                with gpu_lock:
                    scene = detect_scene(frame, detector, args)
                last_scene = scene
                observation = scene.signal
                stable_state, changed, stable_votes = signal_filter.update(
                    observation.raw_state
                )
            else:
                # Continue displaying camera frames while the shared GPU runs Gemma.
                scene = last_scene
                observation = scene.signal
                changed = False

            if (
                changed
                and stable_state in SAFETY_MESSAGES
                and not interaction_snapshot["busy"]
            ):
                now = time.monotonic()
                same_recent_state = (
                    stable_state == last_announced_state
                    and now - last_announcement_time < args.announcement_cooldown
                )
                if not same_recent_state:
                    message = SAFETY_MESSAGES[stable_state]
                    print(f"[Safety event] {stable_state}: {message}")
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
                scene,
                stable_state,
                stable_votes,
                displayed_fps,
                tts.muted,
                interaction_snapshot["status"],
                not args.disable_interaction,
            )
            cv2.imshow("Traffic Light Vision-LLM Assistant", output)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("m"), ord("M")):
                muted = tts.toggle_mute()
                print(f"[TTS] {'Muted' if muted else 'Unmuted'}")
            if key in (ord("r"), ord("R")) and stable_state in SAFETY_MESSAGES:
                message = SAFETY_MESSAGES[stable_state]
                print(f"[Repeat] {message}")
                if tts.speak(message):
                    last_announced_state = stable_state
                    last_announcement_time = time.monotonic()

            if key in (ord("v"), ord("V")) and not args.disable_interaction:
                if interaction_snapshot["busy"]:
                    print("[Interaction] A request is already running.")
                    continue
                if observation.box is None:
                    message = "현재 카메라에서 신호등을 찾을 수 없습니다."
                    print(f"[Interaction] {message}")
                    tts.speak(message)
                    continue

                x1, y1, x2, y2 = observation.box
                roi_image = frame[y1:y2, x1:x2].copy()
                if roi_image.size == 0:
                    print("[Interaction] Empty traffic-light ROI.", file=sys.stderr)
                    continue

                vision_data = build_vision_context(
                    scene,
                    stable_state,
                    frame,
                )
                try:
                    save_vision_snapshot(
                        vision_data,
                        roi_image,
                        args.runtime_output,
                    )
                except RuntimeError as error:
                    print(f"[Snapshot error] {error}", file=sys.stderr)
                    continue

                start_interaction_worker(
                    interaction,
                    stt,
                    assistant,
                    tts,
                    roi_image,
                    vision_data,
                    gpu_lock,
                )
    finally:
        capture.release()
        cv2.destroyAllWindows()
        tts.close()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "YOLO traffic-light events + Whisper.cpp STT + Gemma 4 Vision + "
            "Piper TTS for NVIDIA Jetson."
        )
    )
    parser.add_argument("--yolo-model", default=str(DEFAULT_YOLO_MODEL))
    parser.add_argument("--gemma-model", type=Path, default=DEFAULT_GEMMA_MODEL)
    parser.add_argument("--mmproj-model", type=Path, default=DEFAULT_MMPROJ_MODEL)
    parser.add_argument(
        "--source",
        choices=("csi", "usb", "video"),
        default="csi",
    )
    parser.add_argument("--sensor-id", type=int, default=0)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--video-path", type=Path)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--flip", type=int, choices=(-1, 0, 1), default=None)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confidence", type=float, default=0.015)
    parser.add_argument(
        "--scene-confidence",
        type=float,
        default=0.25,
        help="Minimum confidence for non-traffic-light objects in Gemma Context.",
    )
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

    parser.add_argument(
        "--disable-interaction",
        action="store_true",
        help="Run only YOLO/HSV event detection and safety TTS.",
    )
    parser.add_argument(
        "--stt-backend",
        choices=("whisper", "keyboard"),
        default="whisper",
    )
    parser.add_argument("--whisper-cli", type=Path, default=DEFAULT_WHISPER_CLI)
    parser.add_argument("--whisper-model", type=Path, default=DEFAULT_WHISPER_MODEL)
    parser.add_argument("--microphone-device", default="plughw:3,0")
    parser.add_argument("--record-seconds", type=int, default=5)
    parser.add_argument("--audio-input", type=Path, default=DEFAULT_AUDIO_INPUT)
    parser.add_argument("--no-pasuspender", action="store_true")

    parser.add_argument("--gemma-gpu-layers", type=int, default=-1)
    parser.add_argument("--context-window", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=120)

    parser.add_argument(
        "--tts-backend",
        choices=("piper", "espeak", "none"),
        default="piper",
    )
    parser.add_argument("--piper-python", type=Path, default=DEFAULT_PIPER_PYTHON)
    parser.add_argument("--piper-model", type=Path, default=DEFAULT_PIPER_MODEL)
    parser.add_argument("--audio-output", type=Path, default=DEFAULT_AUDIO_OUTPUT)
    parser.add_argument("--speaker-device", default="plughw:2,0")
    parser.add_argument("--tts-rate", type=int, default=155)
    parser.add_argument("--tts-volume", type=int, default=150)
    parser.add_argument("--mute", action="store_true")
    parser.add_argument("--runtime-output", type=Path, default=DEFAULT_RUNTIME_OUTPUT)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    validate_arguments(args)
    import_runtime_dependencies(load_gemma=not args.disable_interaction)
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except RuntimeError as error:
        raise SystemExit(f"Runtime error: {error}") from error


if __name__ == "__main__":
    main()
