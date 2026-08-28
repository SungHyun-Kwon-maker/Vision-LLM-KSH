#!/usr/bin/env python3
"""YOLO + Gemma 4 + STT/TTS educational visual assistance prototype.

Pipeline:
    camera -> YOLO -> Dictionary/JSON -> Natural Language Context
       microphone -> Whisper.cpp -> question + frame + context -> Gemma 4
                                                     Gemma answer -> Piper TTS

This implementation intentionally follows the examples in
06-Vision_LLM_Multimodal_Systems.ipynb.  Position and proximity labels are
computed with simple image-coordinate rules so that the result is easy to
inspect and explain.

This is an educational prototype, not a navigation or safety device.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
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
DEFAULT_RUNTIME_DIR = PROJECT_DIR / "runtime/visual_assistant"

CONTEXT_WINDOW = 2048
MAX_TOKENS = 180
MAX_CONTEXT_OBJECTS = 12
DEFAULT_SCENE_QUESTION = "현재 내 앞에 무엇이 있고 각각 어디에 있는지 알려줘."

# 자주 탐지되는 COCO 클래스만 한국어 이름을 함께 제공합니다. 매핑에 없는
# 클래스는 YOLO의 영문 클래스명을 그대로 사용합니다.
COCO_KOREAN = {
    "person": "사람",
    "bicycle": "자전거",
    "car": "자동차",
    "motorcycle": "오토바이",
    "bus": "버스",
    "truck": "트럭",
    "traffic light": "신호등",
    "stop sign": "정지 표지판",
    "bench": "벤치",
    "bird": "새",
    "cat": "고양이",
    "dog": "개",
    "backpack": "가방",
    "umbrella": "우산",
    "handbag": "손가방",
    "suitcase": "여행가방",
    "bottle": "물병",
    "cup": "컵",
    "chair": "의자",
    "couch": "소파",
    "bed": "침대",
    "dining table": "식탁",
    "toilet": "변기",
    "tv": "텔레비전",
    "laptop": "노트북",
    "mouse": "마우스",
    "keyboard": "키보드",
    "cell phone": "휴대전화",
    "book": "책",
    "clock": "시계",
    "potted plant": "화분",
}

# Jetson 전용 패키지는 main()에서 불러옵니다. 덕분에 모델이 없는 PC에서도
# --help와 아래의 좌표/JSON 순수 함수 테스트를 실행할 수 있습니다.
cv2: Any = None
YOLO: Any = None
Llama: Any = None
Gemma4ChatHandler: Any = None


def horizontal_position(center_x: float, image_width: int) -> str:
    """Return a camera-image horizontal region for an x coordinate."""

    ratio = center_x / max(image_width, 1)
    if ratio < 1 / 3:
        return "왼쪽"
    if ratio > 2 / 3:
        return "오른쪽"
    return "정면"


def vertical_position(center_y: float, image_height: int) -> str:
    """Return a camera-image vertical region for a y coordinate."""

    ratio = center_y / max(image_height, 1)
    if ratio < 1 / 3:
        return "위쪽"
    if ratio > 2 / 3:
        return "아래쪽"
    return "중앙 높이"


def estimate_proximity(area_ratio: float) -> str:
    """Estimate proximity only from bounding-box occupancy in the image."""

    if area_ratio >= 0.25:
        return "매우 가까워 보입니다"
    if area_ratio >= 0.10:
        return "가까워 보입니다"
    if area_ratio >= 0.03:
        return "중간 거리로 보입니다"
    return "멀리 있어 보입니다"


def detection_to_object(
    class_name: str,
    confidence: float,
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    """Convert one YOLO detection into the project's structured output."""

    raw_x1, raw_y1, raw_x2, raw_y2 = bbox
    x1 = max(0, min(image_width, int(raw_x1)))
    y1 = max(0, min(image_height, int(raw_y1)))
    x2 = max(x1, min(image_width, int(raw_x2)))
    y2 = max(y1, min(image_height, int(raw_y2)))

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    box_area = max(0, x2 - x1) * max(0, y2 - y1)
    image_area = max(image_width * image_height, 1)
    area_ratio = box_area / image_area

    return {
        "class": class_name,
        "class_ko": COCO_KOREAN.get(class_name, class_name),
        "confidence": round(confidence, 3),
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "position": {
            "horizontal": horizontal_position(center_x, image_width),
            "vertical": vertical_position(center_y, image_height),
        },
        "area_ratio": round(area_ratio, 4),
        "proximity_hint": estimate_proximity(area_ratio),
    }


def result_to_vision_dict(
    result: Any,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    """Convert an Ultralytics result into Dictionary/JSON-ready data."""

    objects: list[dict[str, Any]] = []

    # 06번 노트북의 `for box in result.boxes` 구조를 그대로 사용합니다.
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
        class_name = result.names[class_id]

        objects.append(
            detection_to_object(
                class_name,
                confidence,
                (x1, y1, x2, y2),
                image_width,
                image_height,
            )
        )

    # 화면에서 크게 보이는 객체부터 설명하면 짧은 답변에서도 중요한 객체가
    # 먼저 전달됩니다. 실제 물리적 거리를 정렬하는 것은 아닙니다.
    objects.sort(key=lambda obj: obj["area_ratio"], reverse=True)

    return {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "image_width": image_width,
        "image_height": image_height,
        "position_reference": "카메라 이미지 기준",
        "distance_method": "BBox 화면 점유율 기반 상대적 추정",
        "objects": objects,
    }


def detections_to_text(vision_data: dict[str, Any]) -> str:
    """Convert structured detections to the notebook-style natural language."""

    objects = vision_data["objects"]
    if len(objects) == 0:
        return "현재 YOLO 탐지 결과에는 객체가 없습니다. 탐지 실패 가능성은 있습니다."

    counts = Counter(obj["class_ko"] for obj in objects)
    count_text = ", ".join(f"{name} {count}개" for name, count in counts.items())
    sentences = [f"탐지 개수 요약: {count_text}."]

    # Gemma의 2,048 context 안에 이미지와 질문도 함께 들어가므로 큰 객체
    # 12개까지만 상세 문장으로 변환합니다. 전체 탐지 결과는 JSON에 남습니다.
    context_objects = objects[:MAX_CONTEXT_OBJECTS]
    for index, obj in enumerate(context_objects, start=1):
        position = obj["position"]
        sentences.append(
            f"{index}번 객체는 {obj['class_ko']}({obj['class']})이며, "
            f"confidence는 {obj['confidence']:.2f}입니다. "
            f"카메라 화면의 {position['horizontal']}, {position['vertical']}에 있고 "
            f"{obj['proximity_hint']}."
        )

    omitted_count = len(objects) - len(context_objects)
    if omitted_count > 0:
        sentences.append(
            f"Context 길이를 줄이기 위해 나머지 {omitted_count}개 객체의 상세 위치는 생략했습니다."
        )

    sentences.append(
        "거리 표현은 실제 깊이가 아니라 BBox 화면 점유율만 사용한 상대적 추정입니다."
    )
    return "\n".join(sentences)


def save_json_atomically(data: dict[str, Any], output_path: Path) -> None:
    """Write JSON through a temporary file, as demonstrated in section 6."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.stem + "_temp.json")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    os.replace(temp_path, output_path)


def save_frame_atomically(frame: Any, output_path: Path) -> None:
    """Save the current frame through a temporary JPG file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.stem + "_temp.jpg")
    success = cv2.imwrite(str(temp_path), frame)
    if not success:
        raise RuntimeError("현재 카메라 프레임 저장에 실패했습니다.")
    os.replace(temp_path, output_path)


def frame_to_image_data(frame: Any) -> str:
    """Encode an OpenCV frame as the Gemma image_url data URI."""

    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        raise RuntimeError("이미지 인코딩에 실패했습니다.")
    image_base64 = base64.b64encode(buffer).decode("utf-8")
    return "data:image/jpeg;base64," + image_base64


def speech_to_text(args: argparse.Namespace) -> str:
    """Record speech with arecord and transcribe it with Whisper.cpp."""

    args.audio_input.parent.mkdir(parents=True, exist_ok=True)
    record_command = [
        "arecord",
        "-D",
        args.microphone_device,
        "-f",
        "S16_LE",
        "-r",
        "16000",
        "-c",
        "1",
        "-d",
        str(args.record_seconds),
        str(args.audio_input),
    ]
    if shutil.which("pasuspender"):
        record_command = ["pasuspender", "--"] + record_command

    print(f"\n[STT] {args.record_seconds}초 동안 말씀해 주세요.")
    subprocess.run(record_command, check=True)

    result = subprocess.run(
        [
            str(args.whisper_cli),
            "-m",
            str(args.whisper_model),
            "-f",
            str(args.audio_input),
            "-l",
            "ko",
            "--no-timestamps",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def text_to_speech(text: str, args: argparse.Namespace) -> None:
    """Synthesize Korean speech with Piper and play it with aplay."""

    if args.mute:
        return

    args.audio_output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(args.piper_python),
            "-m",
            "piper",
            "-m",
            str(args.piper_model),
            "-f",
            str(args.audio_output),
            "--",
            text,
        ],
        check=True,
    )
    subprocess.run(
        ["aplay", "-D", args.speaker_device, str(args.audio_output)],
        check=True,
    )


def ask_gemma(
    llm: Any,
    question: str,
    vision_text: str,
    frame: Any,
) -> str:
    """Send question + structured context + current image to Gemma 4 Vision."""

    image_data = frame_to_image_data(frame)

    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": """
Instruction:
당신은 카메라 장면을 짧게 설명하는 교육용 시각 보조 AI입니다.
YOLO 탐지 Context를 우선 근거로 사용하고 이미지는 보조 근거로만 사용하여
사용자의 질문에 답하십시오. 객체의 종류와 카메라 화면 기준 방향을 먼저 말하십시오.

Constraint:
탐지 결과나 이미지에서 명확히 확인되지 않는 객체와 상황은 추측하지 마십시오.
객체가 탐지되지 않았다고 해서 실제로 없다고 단정하지 마십시오.
가까움/멀리 표현은 BBox 크기 기반의 상대적 추정이며 실제 거리나 미터로 말하지 마십시오.
안전하다, 길을 건너도 된다, 진행해도 된다는 판단을 내리지 마십시오.
위험 또는 이동 관련 질문에는 카메라 정보의 한계를 알리고 직접 확인하도록 안내하십시오.

Output Format:
자연스러운 한국어 두 문장 이내로 답하십시오.
""",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"사용자 질문:\n{question}\n\n"
                            f"YOLO Context:\n{vision_text}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data},
                    },
                ],
            },
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.2,
    )
    return response["choices"][0]["message"]["content"].strip()


def load_jetson_packages() -> None:
    """Import heavy packages only when the actual application starts."""

    global cv2, YOLO, Llama, Gemma4ChatHandler

    try:
        import cv2 as cv2_module
        from ultralytics import YOLO as yolo_class
        from llama_cpp import Llama as llama_class
        from llama_cpp.llama_chat_format import Gemma4ChatHandler as handler_class
    except ImportError as error:
        raise RuntimeError(
            "필수 Python 패키지가 없습니다. OpenCV, ultralytics, "
            "llama-cpp-python 설치를 확인하세요."
        ) from error

    cv2 = cv2_module
    YOLO = yolo_class
    Llama = llama_class
    Gemma4ChatHandler = handler_class


def create_llm(args: argparse.Namespace) -> Any:
    """Create Gemma exactly with the notebook's Gemma4ChatHandler pattern."""

    print("\n[Gemma] 모델을 불러오는 중입니다...")
    chat_handler = Gemma4ChatHandler(clip_model_path=str(args.mmproj_model))
    return Llama(
        model_path=str(args.gemma_model),
        chat_handler=chat_handler,
        n_gpu_layers=-1,
        n_ctx=CONTEXT_WINDOW,
        n_batch=32,
        n_ubatch=32,
        verbose=False,
    )


def csi_pipeline(sensor_id: int) -> str:
    """Return the Jetson CSI GStreamer pipeline used in the Colab material."""

    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        "video/x-raw(memory:NVMM), "
        "width=1280, height=720, framerate=30/1 ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "queue leaky=downstream max-size-buffers=1 ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def open_camera(source: str, sensor_id: int) -> Any:
    """Open a CSI camera, USB camera index, or video file."""

    if source.lower() == "csi":
        return cv2.VideoCapture(csi_pipeline(sensor_id), cv2.CAP_GSTREAMER)
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def check_runtime_files(args: argparse.Namespace) -> None:
    """Fail early with a readable list of model files that are missing."""

    required = {
        "YOLO 모델": args.yolo_model,
        "Gemma 모델": args.gemma_model,
        "Gemma mmproj": args.mmproj_model,
    }
    missing = [f"- {label}: {path}" for label, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("필수 모델 파일을 찾을 수 없습니다.\n" + "\n".join(missing))


def check_voice_files(args: argparse.Namespace, needs_stt: bool) -> None:
    """Validate speech files only when the corresponding feature is used."""

    required: dict[str, Path] = {}
    if needs_stt:
        required.update(
            {
                "Whisper 실행 파일": args.whisper_cli,
                "Whisper 모델": args.whisper_model,
            }
        )
    if not args.mute:
        required.update(
            {
                "Piper Python": args.piper_python,
                "Piper 모델": args.piper_model,
            }
        )
    missing = [f"- {label}: {path}" for label, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("음성 기능 파일을 찾을 수 없습니다.\n" + "\n".join(missing))


def process_question(
    question: str,
    frame: Any,
    vision_data: dict[str, Any],
    llm: Any,
    args: argparse.Namespace,
) -> None:
    """Save evidence, ask Gemma, print the answer, and speak it."""

    if not question.strip():
        print("[STT] 음성을 인식하지 못했습니다. 다시 시도해 주세요.")
        return

    vision_text = detections_to_text(vision_data)
    save_json_atomically(vision_data, args.json_output)
    save_frame_atomically(frame, args.frame_output)

    print(f"\n[질문]\n{question}")
    print(f"\n[Vision Context]\n{vision_text}")
    answer = ask_gemma(llm, question, vision_text, frame)
    print(f"\n[Gemma]\n{answer}")
    text_to_speech(answer, args)


def draw_controls(frame: Any, muted: bool) -> None:
    """Draw ASCII controls because cv2.putText does not render Korean."""

    mute_state = "ON" if muted else "OFF"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (20, 20, 20), -1)
    cv2.putText(
        frame,
        f"V: voice  S: scene  M: mute({mute_state})  Q: quit",
        (15, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO/Gemma 4 기반 교육용 생활환경 음성 안내기"
    )
    parser.add_argument(
        "--source",
        default="csi",
        help="csi, USB 카메라 번호(예: 0), 또는 동영상 파일 경로",
    )
    parser.add_argument("--sensor-id", type=int, default=0)
    parser.add_argument("--flip", type=int, choices=(-1, 0, 1), default=None)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--yolo-model", type=Path, default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--gemma-model", type=Path, default=DEFAULT_GEMMA_MODEL)
    parser.add_argument("--mmproj-model", type=Path, default=DEFAULT_MMPROJ_MODEL)
    parser.add_argument("--whisper-cli", type=Path, default=DEFAULT_WHISPER_CLI)
    parser.add_argument("--whisper-model", type=Path, default=DEFAULT_WHISPER_MODEL)
    parser.add_argument("--piper-python", type=Path, default=DEFAULT_PIPER_PYTHON)
    parser.add_argument("--piper-model", type=Path, default=DEFAULT_PIPER_MODEL)
    parser.add_argument("--microphone-device", default="plughw:3,0")
    parser.add_argument("--speaker-device", default="plughw:2,0")
    parser.add_argument("--record-seconds", type=int, default=5)
    parser.add_argument("--mute", action="store_true", help="Piper 음성 출력을 끕니다.")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_RUNTIME_DIR / "vision_data.json",
    )
    parser.add_argument(
        "--frame-output",
        type=Path,
        default=DEFAULT_RUNTIME_DIR / "current_frame.jpg",
    )
    parser.add_argument(
        "--audio-input",
        type=Path,
        default=DEFAULT_RUNTIME_DIR / "input.wav",
    )
    parser.add_argument(
        "--audio-output",
        type=Path,
        default=DEFAULT_RUNTIME_DIR / "response.wav",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not 0.0 <= args.conf <= 1.0 or not 0.0 <= args.iou <= 1.0:
        print("--conf와 --iou는 0에서 1 사이여야 합니다.", file=sys.stderr)
        return 2
    if args.record_seconds < 1:
        print("--record-seconds는 1 이상이어야 합니다.", file=sys.stderr)
        return 2

    try:
        check_runtime_files(args)
        load_jetson_packages()
    except (FileNotFoundError, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 1

    # 1. YOLO와 카메라 준비: 06번 노트북과 동일한 시작점입니다.
    yolo = YOLO(str(args.yolo_model))
    cap = open_camera(args.source, args.sensor_id)
    if not cap.isOpened():
        print(f"카메라를 열 수 없습니다: {args.source}", file=sys.stderr)
        return 1

    llm = None  # 첫 질문 때 로드하여 카메라 시작 시간을 줄입니다.

    print("\n=== Educational Visual Assistant ===")
    print("V : 음성으로 질문")
    print("S : 기본 장면 설명")
    print("M : TTS 음소거 전환")
    print("Q : 종료")
    print("주의: 실제 거리·안전·이동 가능 여부를 판단하는 장치가 아닙니다.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("카메라 프레임을 읽을 수 없습니다.", file=sys.stderr)
                break

            if args.flip is not None:
                frame = cv2.flip(frame, args.flip)

            height, width = frame.shape[:2]

            # 2. 현재 프레임 YOLO 탐지
            results = yolo.predict(
                source=frame,
                conf=args.conf,
                iou=args.iou,
                verbose=False,
            )
            result = results[0]

            # 3. YOLO 결과를 Dictionary 형태로 변환
            vision_data = result_to_vision_dict(result, width, height)

            output_frame = result.plot()
            draw_controls(output_frame, args.mute)
            cv2.imshow("YOLO + Gemma Visual Assistant", output_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("m"):
                args.mute = not args.mute
                print(f"[TTS] {'음소거' if args.mute else '음소거 해제'}")
                continue
            if key not in (ord("v"), ord("s")):
                continue

            try:
                check_voice_files(args, needs_stt=(key == ord("v")))
                if key == ord("v"):
                    question = speech_to_text(args)
                else:
                    question = DEFAULT_SCENE_QUESTION

                # 4. 첫 질의 시 Gemma Vision 모델 준비
                if llm is None:
                    llm = create_llm(args)

                # 5. JSON→자연어, Image+Text→Gemma, Gemma→TTS
                process_question(question, frame.copy(), vision_data, llm, args)
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                print(f"[질의 처리 오류] {error}", file=sys.stderr)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
