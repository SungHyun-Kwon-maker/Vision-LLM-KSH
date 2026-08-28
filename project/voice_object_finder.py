#!/usr/bin/env python3
"""Voice-controlled object finder for Jetson, YOLO, and local Gemma.

Press V in the OpenCV window, speak a request, and the program will:
1. transcribe the microphone input,
2. ask local Gemma to select a COCO class and optional color,
3. run YOLO only for that class, and
4. highlight matching objects in the camera frame.

The runtime dependencies are imported after argument parsing so ``--help`` works
even on a development machine without the Jetson packages installed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import importlib
import json
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_DIR.parent

DEFAULT_YOLO_MODEL = REPOSITORY_ROOT / "src/models/YOLO/yolo11n.pt"
DEFAULT_GEMMA_MODEL = (
    REPOSITORY_ROOT
    / "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
)

# Ultralytics COCO class order. TensorRT engines created from yolo11n use the
# same indices, so this list works before either a .pt model or an engine has
# performed its first inference.
COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)

SUPPORTED_COLORS = (
    "red", "orange", "yellow", "green", "blue", "purple", "pink", "black",
    "white", "gray", "brown",
)

KOREAN_TARGET_ALIASES = {
    "신호등": "traffic light",
    "소화전": "fire hydrant",
    "정지 표지판": "stop sign",
    "주차 미터기": "parking meter",
    "야구 방망이": "baseball bat",
    "야구 글러브": "baseball glove",
    "테니스 라켓": "tennis racket",
    "와인잔": "wine glass",
    "화분": "potted plant",
    "식탁": "dining table",
    "휴대전화": "cell phone",
    "휴대폰": "cell phone",
    "핸드폰": "cell phone",
    "스마트폰": "cell phone",
    "전자레인지": "microwave",
    "냉장고": "refrigerator",
    "곰인형": "teddy bear",
    "드라이기": "hair drier",
    "오토바이": "motorcycle",
    "자전거": "bicycle",
    "비행기": "airplane",
    "자동차": "car",
    "승용차": "car",
    "버스": "bus",
    "기차": "train",
    "트럭": "truck",
    "배": "boat",
    "벤치": "bench",
    "새": "bird",
    "고양이": "cat",
    "강아지": "dog",
    "말": "horse",
    "양": "sheep",
    "소": "cow",
    "코끼리": "elephant",
    "곰": "bear",
    "얼룩말": "zebra",
    "기린": "giraffe",
    "백팩": "backpack",
    "가방": "backpack",
    "우산": "umbrella",
    "넥타이": "tie",
    "캐리어": "suitcase",
    "스키": "skis",
    "스노보드": "snowboard",
    "공": "sports ball",
    "연": "kite",
    "스케이트보드": "skateboard",
    "서핑보드": "surfboard",
    "물병": "bottle",
    "병": "bottle",
    "컵": "cup",
    "포크": "fork",
    "나이프": "knife",
    "칼": "knife",
    "숟가락": "spoon",
    "그릇": "bowl",
    "바나나": "banana",
    "사과": "apple",
    "샌드위치": "sandwich",
    "오렌지": "orange",
    "브로콜리": "broccoli",
    "당근": "carrot",
    "핫도그": "hot dog",
    "피자": "pizza",
    "도넛": "donut",
    "케이크": "cake",
    "의자": "chair",
    "소파": "couch",
    "침대": "bed",
    "변기": "toilet",
    "텔레비전": "tv",
    "티비": "tv",
    "노트북": "laptop",
    "마우스": "mouse",
    "리모컨": "remote",
    "키보드": "keyboard",
    "오븐": "oven",
    "토스터": "toaster",
    "싱크대": "sink",
    "책": "book",
    "시계": "clock",
    "꽃병": "vase",
    "가위": "scissors",
    "칫솔": "toothbrush",
    "사람": "person",
}

KOREAN_COLOR_ALIASES = {
    "빨간": "red",
    "빨강": "red",
    "붉은": "red",
    "주황": "orange",
    "노란": "yellow",
    "노랑": "yellow",
    "초록": "green",
    "녹색": "green",
    "파란": "blue",
    "파랑": "blue",
    "보라": "purple",
    "분홍": "pink",
    "핑크": "pink",
    "검은": "black",
    "검정": "black",
    "하얀": "white",
    "흰색": "white",
    "회색": "gray",
    "갈색": "brown",
}

# Assigned after command-line parsing. Lazy imports allow --help and py_compile
# to work on machines that do not have the Jetson runtime packages.
cv2: Any = None
np: Any = None
YOLO: Any = None
Llama: Any = None


@dataclass(frozen=True)
class SearchIntent:
    target_class: str
    color: str | None
    raw_response: str


@dataclass
class SharedState:
    busy: bool = False
    target_class: str | None = None
    target_id: int | None = None
    color: str | None = None
    status: str = "Press V and speak"
    last_command: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "busy": self.busy,
                "target_class": self.target_class,
                "target_id": self.target_id,
                "color": self.color,
                "status": self.status,
                "last_command": self.last_command,
            }

    def clear_target(self) -> None:
        with self.lock:
            self.target_class = None
            self.target_id = None
            self.color = None
            self.status = "Target cleared - press V"


def import_runtime_dependencies() -> None:
    """Import large third-party packages and report all missing packages."""
    global cv2, np, YOLO, Llama

    missing: list[str] = []

    try:
        cv2 = importlib.import_module("cv2")
    except ModuleNotFoundError:
        missing.append("opencv-python / Jetson OpenCV (cv2)")

    try:
        np = importlib.import_module("numpy")
    except ModuleNotFoundError:
        missing.append("numpy")

    try:
        YOLO = importlib.import_module("ultralytics").YOLO
    except ModuleNotFoundError:
        missing.append("ultralytics")

    try:
        Llama = importlib.import_module("llama_cpp").Llama
    except ModuleNotFoundError:
        missing.append("llama-cpp-python")

    if missing:
        package_list = "\n  - ".join(missing)
        raise SystemExit(f"Missing runtime dependencies:\n  - {package_list}")


class SpeechToText:
    """Capture microphone audio and transcribe it with a selected backend."""

    def __init__(
        self,
        backend: str,
        language: str,
        microphone_index: int | None,
        timeout: float,
        phrase_time_limit: float,
        whisper_model_name: str,
        whisper_device: str,
    ) -> None:
        self.backend = backend
        self.language = language
        self.microphone_index = microphone_index
        self.timeout = timeout
        self.phrase_time_limit = phrase_time_limit
        self.whisper_device = whisper_device
        self.sr: Any = None
        self.recognizer: Any = None
        self.whisper_model: Any = None

        if backend == "keyboard":
            return

        try:
            self.sr = importlib.import_module("speech_recognition")
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "SpeechRecognition is required for microphone input."
            ) from error

        self.recognizer = self.sr.Recognizer()
        self.recognizer.pause_threshold = 0.6

        if backend == "whisper":
            try:
                whisper = importlib.import_module("whisper")
            except ModuleNotFoundError as error:
                raise RuntimeError(
                    "openai-whisper is required when --stt whisper is used."
                ) from error

            print(
                f"[STT] Loading Whisper '{whisper_model_name}' "
                f"on {whisper_device}..."
            )
            self.whisper_model = whisper.load_model(
                whisper_model_name,
                device=whisper_device,
            )

    def listen(self) -> str:
        if self.backend == "keyboard":
            return input("찾고 싶은 물체를 입력하세요: ").strip()

        try:
            with self.sr.Microphone(device_index=self.microphone_index) as source:
                print("[STT] 주변 소음 보정 중...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                print("[STT] 말씀하세요.")
                audio = self.recognizer.listen(
                    source,
                    timeout=self.timeout,
                    phrase_time_limit=self.phrase_time_limit,
                )
        except self.sr.WaitTimeoutError as error:
            raise RuntimeError("제한 시간 안에 음성이 감지되지 않았습니다.") from error

        if self.backend == "google":
            try:
                text = self.recognizer.recognize_google(
                    audio,
                    language=self.language,
                )
            except self.sr.UnknownValueError as error:
                raise RuntimeError("음성을 이해하지 못했습니다.") from error
            except self.sr.RequestError as error:
                raise RuntimeError(f"Google STT 요청 실패: {error}") from error
            return text.strip()

        raw_audio = audio.get_raw_data(convert_rate=16000, convert_width=2)
        waveform = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32)
        waveform /= 32768.0

        whisper_language = self.language.split("-")[0].lower()
        result = self.whisper_model.transcribe(
            waveform,
            language=whisper_language,
            fp16=self.whisper_device == "cuda",
            temperature=0.0,
        )
        return result["text"].strip()


class GemmaIntentParser:
    """Use local Gemma to turn natural language into a detection target."""

    def __init__(self, model_path: Path, gpu_layers: int, context_window: int) -> None:
        print(f"[Gemma] Loading model: {model_path}")
        self.llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=gpu_layers,
            n_ctx=context_window,
            n_batch=32,
            n_ubatch=32,
            verbose=False,
        )

    def parse(self, command: str, previous_target: str | None) -> SearchIntent:
        class_list = ", ".join(COCO_CLASSES)
        color_list = ", ".join(SUPPORTED_COLORS)
        previous = previous_target or "none"

        prompt = f"""
너는 음성 기반 객체 탐색 시스템의 명령 해석기다.
사용자의 명령에서 찾을 물체와 선택적인 색상을 추출하라.

규칙:
1. target_class는 반드시 아래 COCO 클래스 중 정확히 하나를 영어로 선택한다.
2. 색상이 없으면 color는 null이다.
3. 색상이 있으면 color는 아래 영어 색상 중 하나다.
4. 설명이나 Markdown 없이 JSON 객체 한 개만 출력한다.
5. 직접 일치하는 클래스가 없으면 의미상 가장 가까운 클래스를 고른다.

COCO classes: {class_list}
Allowed colors: {color_list}
Previous target: {previous}

출력 형식:
{{"target_class": "bottle", "color": null}}

사용자 명령: {command}
""".strip()

        raw_response = ""
        try:
            response = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.0,
            )
            raw_response = response["choices"][0]["message"]["content"].strip()
            intent = self._intent_from_json(raw_response)
            if intent is not None:
                return intent
            print("[Gemma] JSON validation failed; using keyword fallback.")
        except Exception as error:
            print(f"[Gemma] Inference failed; using keyword fallback: {error}")

        fallback = self._fallback_intent(command, previous_target)
        if fallback is None:
            raise RuntimeError(
                "명령에서 COCO 객체 클래스를 결정하지 못했습니다. "
                "예: '물병을 찾아줘'"
            )

        return SearchIntent(
            target_class=fallback[0],
            color=fallback[1],
            raw_response=raw_response or "keyword fallback",
        )

    @staticmethod
    def _intent_from_json(raw_response: str) -> SearchIntent | None:
        match = re.search(r"\{.*?\}", raw_response, flags=re.DOTALL)
        if match is None:
            return None

        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

        target = str(data.get("target_class", "")).strip().lower()
        target = target.replace("_", " ")
        target = {
            "cellphone": "cell phone",
            "mobile phone": "cell phone",
            "sofa": "couch",
            "plant": "potted plant",
            "television": "tv",
        }.get(target, target)

        if target not in COCO_CLASSES:
            return None

        color_value = data.get("color")
        color = None if color_value is None else str(color_value).strip().lower()
        color = {
            "빨간색": "red",
            "빨강": "red",
            "주황색": "orange",
            "노란색": "yellow",
            "노랑": "yellow",
            "초록색": "green",
            "녹색": "green",
            "파란색": "blue",
            "파랑": "blue",
            "보라색": "purple",
            "분홍색": "pink",
            "검은색": "black",
            "검정": "black",
            "흰색": "white",
            "하얀색": "white",
            "회색": "gray",
            "갈색": "brown",
        }.get(color, color)
        if color in ("", "none", "null", "없음"):
            color = None
        if color is not None and color not in SUPPORTED_COLORS:
            color = None

        return SearchIntent(target, color, raw_response)

    @staticmethod
    def _fallback_intent(
        command: str,
        previous_target: str | None,
    ) -> tuple[str, str | None] | None:
        normalized = command.lower()
        target: str | None = None

        # Longer aliases take priority, e.g. 휴대전화 before 전화-like terms.
        for korean, coco_class in sorted(
            KOREAN_TARGET_ALIASES.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            if korean in normalized:
                target = coco_class
                break

        if target is None:
            for coco_class in sorted(COCO_CLASSES, key=len, reverse=True):
                if coco_class in normalized:
                    target = coco_class
                    break

        if target is None and previous_target is not None:
            if any(word in normalized for word in ("그거", "그것", "이전", "아까")):
                target = previous_target

        if target is None:
            return None

        color: str | None = None
        for korean, english in KOREAN_COLOR_ALIASES.items():
            if korean in normalized:
                color = english
                break
        if color is None:
            for english in SUPPORTED_COLORS:
                if english in normalized:
                    color = english
                    break

        return target, color


def build_csi_pipeline(sensor_id: int, width: int, height: int, fps: int) -> str:
    """Return the GStreamer pipeline used in the course Jetson examples."""
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
        raise RuntimeError(f"카메라/영상 소스를 열 수 없습니다: {args.source}")
    return capture


def color_ratio(bgr_crop: Any, color: str) -> float:
    """Return the ratio of pixels that fall in a coarse HSV color range."""
    if bgr_crop.size == 0:
        return 0.0

    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    ranges: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]] | list[Any]] = {
        "red": [((0, 80, 55), (10, 255, 255)), ((170, 80, 55), (179, 255, 255))],
        "orange": ((10, 90, 70), (22, 255, 255)),
        "yellow": ((22, 80, 70), (35, 255, 255)),
        "green": ((35, 55, 45), (85, 255, 255)),
        "blue": ((85, 70, 45), (130, 255, 255)),
        "purple": ((130, 55, 45), (155, 255, 255)),
        "pink": ((155, 45, 70), (170, 255, 255)),
        "black": ((0, 0, 0), (179, 255, 55)),
        "white": ((0, 0, 190), (179, 50, 255)),
        "gray": ((0, 0, 55), (179, 55, 210)),
        "brown": ((5, 55, 20), (22, 255, 180)),
    }

    selected = ranges[color]
    if color == "red":
        mask = cv2.inRange(hsv, np.array(selected[0][0]), np.array(selected[0][1]))
        mask |= cv2.inRange(hsv, np.array(selected[1][0]), np.array(selected[1][1]))
    else:
        lower, upper = selected
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

    return float(cv2.countNonZero(mask)) / float(mask.size)


def draw_status_panel(frame: Any, lines: list[str]) -> None:
    """Draw ASCII status text; cv2.putText does not render Korean reliably."""
    panel_width = min(frame.shape[1] - 20, 650)
    panel_height = 25 + 31 * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_width, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (22, 39 + index * 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def process_detections(
    frame: Any,
    detector: Any,
    target_id: int,
    target_class: str,
    target_color: str | None,
    args: argparse.Namespace,
) -> tuple[Any, int]:
    predict_options: dict[str, Any] = {
        "source": frame,
        "conf": args.confidence,
        "iou": args.iou,
        "verbose": False,
        "classes": [target_id],
        "imgsz": args.image_size,
    }
    if Path(args.yolo_model).suffix.lower() == ".pt":
        predict_options["device"] = args.yolo_device

    result = detector.predict(**predict_options)[0]
    output = frame.copy()
    found_count = 0

    if result.boxes is None:
        return output, found_count

    height, width = frame.shape[:2]
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()

    for box, confidence in zip(boxes, confidences):
        x1, y1, x2, y2 = box.tolist()
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))

        ratio = 1.0
        if target_color is not None:
            ratio = color_ratio(frame[y1:y2, x1:x2], target_color)
            if ratio < args.minimum_color_ratio:
                continue

        found_count += 1
        label = f"{target_class} {float(confidence):.2f}"
        if target_color is not None:
            label += f" | {target_color} {ratio:.0%}"

        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.rectangle(output, (x1, max(0, y1 - 31)), (x2, y1), (0, 170, 0), -1)
        cv2.putText(
            output,
            label,
            (x1 + 5, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output, found_count


def start_command_worker(
    state: SharedState,
    speech_to_text: SpeechToText,
    intent_parser: GemmaIntentParser,
) -> None:
    """Start one non-blocking STT/Gemma command task."""
    with state.lock:
        if state.busy:
            return
        state.busy = True
        state.status = "Listening..."
        previous_target = state.target_class

    def worker() -> None:
        try:
            command = speech_to_text.listen()
            if not command:
                raise RuntimeError("빈 명령이 입력되었습니다.")

            print(f"[Command] {command}")
            with state.lock:
                state.status = "Gemma is parsing..."
                state.last_command = command

            intent = intent_parser.parse(command, previous_target)
            target_id = COCO_CLASSES.index(intent.target_class)
            print(f"[Gemma] {intent.raw_response}")
            print(
                f"[Target] class={intent.target_class}, "
                f"color={intent.color or 'any'}"
            )

            with state.lock:
                state.target_class = intent.target_class
                state.target_id = target_id
                state.color = intent.color
                state.status = "Searching..."
        except Exception as error:
            print(f"[Command error] {error}", file=sys.stderr)
            with state.lock:
                state.status = "Command failed - see terminal"
        finally:
            with state.lock:
                state.busy = False

    threading.Thread(target=worker, name="voice-command", daemon=True).start()


def run(args: argparse.Namespace) -> None:
    yolo_model_path = Path(args.yolo_model).expanduser().resolve()
    gemma_model_path = Path(args.gemma_model).expanduser().resolve()

    if not yolo_model_path.is_file():
        raise SystemExit(f"YOLO model not found: {yolo_model_path}")
    if not gemma_model_path.is_file():
        raise SystemExit(f"Gemma model not found: {gemma_model_path}")
    if args.source == "video" and not args.video_path:
        raise SystemExit("--video-path is required when --source video is used.")

    print(f"[YOLO] Loading model: {yolo_model_path}")
    detector = YOLO(str(yolo_model_path))
    intent_parser = GemmaIntentParser(
        gemma_model_path,
        gpu_layers=args.gemma_gpu_layers,
        context_window=args.context_window,
    )
    speech_to_text = SpeechToText(
        backend=args.stt,
        language=args.language,
        microphone_index=args.microphone_index,
        timeout=args.listen_timeout,
        phrase_time_limit=args.phrase_time_limit,
        whisper_model_name=args.whisper_model,
        whisper_device=args.whisper_device,
    )
    capture = open_video_source(args)
    state = SharedState()

    print("\nVoice-Controlled Visual Object Finder")
    print("V: voice/text command | C: clear target | Q: quit\n")

    displayed_fps = 0.0

    try:
        while True:
            frame_start = time.perf_counter()
            success, frame = capture.read()
            if not success:
                print("[Camera] Failed to read a frame.", file=sys.stderr)
                break

            if args.flip is not None:
                frame = cv2.flip(frame, args.flip)

            snapshot = state.snapshot()
            output = frame.copy()
            found_count = 0

            # Pause GPU YOLO calls while Whisper/Gemma is processing a command.
            if snapshot["target_id"] is not None and not snapshot["busy"]:
                output, found_count = process_detections(
                    frame,
                    detector,
                    snapshot["target_id"],
                    snapshot["target_class"],
                    snapshot["color"],
                    args,
                )

            elapsed = max(time.perf_counter() - frame_start, 1e-9)
            current_fps = 1.0 / elapsed
            displayed_fps = (
                current_fps
                if displayed_fps == 0.0
                else 0.9 * displayed_fps + 0.1 * current_fps
            )

            target_text = "none"
            if snapshot["target_class"] is not None:
                color = snapshot["color"] or "any color"
                target_text = f"{color} {snapshot['target_class']}"

            if snapshot["target_id"] is None:
                result_text = "Waiting for command"
            elif snapshot["busy"]:
                result_text = "Command processing"
            elif found_count > 0:
                result_text = f"FOUND ({found_count})"
            else:
                result_text = "Not found"

            draw_status_panel(
                output,
                [
                    "V: speak | C: clear | Q: quit",
                    f"Status: {snapshot['status']}",
                    f"Target: {target_text}",
                    f"Result: {result_text} | FPS: {displayed_fps:.1f}",
                ],
            )

            cv2.imshow("Voice-Controlled Visual Object Finder", output)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("v"), ord("V")):
                start_command_worker(state, speech_to_text, intent_parser)
            if key in (ord("c"), ord("C")):
                state.clear_target()
    finally:
        capture.release()
        cv2.destroyAllWindows()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Use speech, local Gemma, and YOLO to find requested objects in a "
            "Jetson/USB camera feed."
        )
    )
    parser.add_argument(
        "--yolo-model",
        default=str(DEFAULT_YOLO_MODEL),
        help="Path to a COCO YOLO .pt or TensorRT .engine model.",
    )
    parser.add_argument(
        "--gemma-model",
        default=str(DEFAULT_GEMMA_MODEL),
        help="Path to the local Gemma GGUF model.",
    )
    parser.add_argument(
        "--source",
        choices=("csi", "usb", "video"),
        default="csi",
        help="Input video source (default: csi).",
    )
    parser.add_argument("--sensor-id", type=int, default=0, help="CSI sensor ID.")
    parser.add_argument("--camera-index", type=int, default=0, help="USB camera index.")
    parser.add_argument("--video-path", type=Path, help="Input path for --source video.")
    parser.add_argument("--width", type=int, default=1280, help="Camera width.")
    parser.add_argument("--height", type=int, default=720, help="Camera height.")
    parser.add_argument("--camera-fps", type=int, default=30, help="Requested camera FPS.")
    parser.add_argument(
        "--flip",
        type=int,
        choices=(-1, 0, 1),
        default=None,
        help="OpenCV flip code: 0 vertical, 1 horizontal, -1 both.",
    )
    parser.add_argument(
        "--stt",
        choices=("google", "whisper", "keyboard"),
        default="google",
        help="Speech-to-text backend (default: google).",
    )
    parser.add_argument("--language", default="ko-KR", help="STT language code.")
    parser.add_argument("--microphone-index", type=int, help="PyAudio microphone index.")
    parser.add_argument("--listen-timeout", type=float, default=5.0)
    parser.add_argument("--phrase-time-limit", type=float, default=6.0)
    parser.add_argument("--whisper-model", default="tiny")
    parser.add_argument(
        "--whisper-device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    parser.add_argument("--yolo-device", default="cuda")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument(
        "--minimum-color-ratio",
        type=float,
        default=0.08,
        help="Minimum matching HSV pixel ratio inside a bounding box.",
    )
    parser.add_argument("--gemma-gpu-layers", type=int, default=-1)
    parser.add_argument("--context-window", type=int, default=2048)
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    import_runtime_dependencies()

    try:
        run(args)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except RuntimeError as error:
        raise SystemExit(f"Runtime error: {error}") from error


if __name__ == "__main__":
    main()
