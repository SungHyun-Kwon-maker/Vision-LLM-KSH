# YOLO–Gemma 생활환경 음성 안내기

## 1. 프로젝트 개요

Jetson 카메라로 주변 객체를 탐지하고, 사용자가 음성으로 질문하면 현재
프레임과 YOLO 탐지 정보를 함께 본 Gemma 4가 한국어로 답하는 교육용 시각
보조 프로토타입입니다.

예시 질문:

- “내 앞에 무엇이 있어?”
- “의자가 어느 쪽에 있어?”
- “사람이 몇 명 보여?”
- “가장 크게 보이는 물체를 알려줘.”

이 프로그램은 실제 보행·길 안내·충돌 방지 장치가 아닙니다. 단안 카메라와
일반 객체 탐지 모델은 물체를 놓칠 수 있고, 실제 거리와 안전 여부를 판단할
수 없습니다.

## 2. 전체 파이프라인

```text
카메라 → YOLO 객체 탐지 → Dictionary/JSON → 자연어 Vision Context ─┐
                                                                     │
마이크 → arecord → Whisper.cpp STT → 사용자 질문 ────────────────────┤
                                                                     ▼
                                      현재 프레임 + Context + 질문 → Gemma 4
                                                                     │
                                                                     ▼
                                                       한국어 답변 → Piper TTS
```

## 3. Colab 수업 코드와의 대응

프로그램의 핵심 코드는 `06-Vision_LLM_Multimodal_Systems.ipynb`의 흐름을
유지했습니다.

| Colab 내용 | 이 프로젝트의 함수/코드 |
|---|---|
| `YOLO(...).predict()` | `main()`의 실시간 YOLO 탐지 |
| `for box in result.boxes` | `result_to_vision_dict()` |
| Dictionary → JSON | `save_json_atomically()` |
| `detections_to_text()` | 같은 이름의 `detections_to_text()` |
| OpenCV 이미지 → Base64 | `frame_to_image_data()` |
| `Gemma4ChatHandler` + Image/Text | `create_llm()`, `ask_gemma()` |
| `arecord` + Whisper.cpp | `speech_to_text()` |
| Piper + `aplay` | `text_to_speech()` |
| CSI GStreamer 문자열 | `csi_pipeline()` |

수업 코드에서 확장한 부분은 다음 세 가지뿐입니다.

1. BBox 중심 좌표를 이용해 화면을 `왼쪽/정면/오른쪽`으로 구분합니다.
2. BBox 면적 비율로 `가까움/중간/멀리` 힌트를 만듭니다.
3. YOLO Context와 전체 카메라 프레임을 Gemma에 동시에 전달합니다.

## 4. 위치와 거리 힌트 계산

### 화면 방향

BBox 중심의 x 좌표를 영상 너비로 나누고 화면을 3등분합니다.

```text
0%             33%             67%            100%
┌───────────────┬───────────────┬───────────────┐
│     왼쪽      │      정면     │     오른쪽    │
└───────────────┴───────────────┴───────────────┘
```

y 좌표도 `위쪽/중앙 높이/아래쪽`으로 3등분합니다. 모든 방향은 사람의
절대 방향이 아니라 **카메라에 보이는 이미지 기준**입니다.

### 상대적 거리 힌트

```python
area_ratio = bbox_area / image_area
```

| BBox 점유율 | 출력 |
|---:|---|
| 25% 이상 | 매우 가까워 보임 |
| 10% 이상 | 가까워 보임 |
| 3% 이상 | 중간 거리로 보임 |
| 3% 미만 | 멀리 있어 보임 |

같은 실제 거리에 있어도 물체의 원래 크기에 따라 결과가 달라집니다. 따라서
이 값은 실제 거리 측정값이 아니며, 프로그램도 미터 단위 거리를 말하지 않게
제한했습니다.

## 5. 필요한 파일

노트북과 같은 기본 경로를 사용합니다.

```text
src/models/YOLO/yolo11n_int8.engine

src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf
src/models/Gemma4/mmproj-google_gemma-4-E2B-it-f16.gguf

whisper.cpp/build-cpu/bin/whisper-cli
whisper.cpp/models/ggml-base.bin

.piper_venv/bin/python
src/models/Piper/ko_KR-kss-medium.onnx
src/models/Piper/ko_KR-kss-medium.onnx.json
```

모델과 실행 파일은 용량 때문에 Git에서 제외될 수 있습니다. 현재 저장소에
파일이 없으면 06번 노트북의 설치 절차로 준비해야 합니다.

## 6. 실행

저장소 최상위 폴더에서 Jetson CSI 카메라를 사용할 때:

```bash
python3 project/visual_assistant.py --source csi
```

USB 카메라가 0번일 때:

```bash
python3 project/visual_assistant.py --source 0
```

카메라 영상이 거꾸로 보일 때는 다음 옵션을 추가합니다.

```bash
python3 project/visual_assistant.py --source csi --flip 0
```

오디오 없이 화면과 텍스트 답변부터 시험하려면:

```bash
python3 project/visual_assistant.py --source csi --mute
```

장치 번호가 다르면 `arecord -l`, `aplay -l`로 확인한 후 지정합니다.

```bash
python3 project/visual_assistant.py \
    --source csi \
    --microphone-device plughw:3,0 \
    --speaker-device plughw:2,0
```

## 7. 조작법

| 키 | 기능 |
|---|---|
| `V` | 5초간 녹음하고 음성 질문에 답하기 |
| `S` | “현재 내 앞에 무엇이 있고 어디에 있는지” 바로 질문하기 |
| `M` | TTS 음소거/해제 |
| `Q` | 종료 |

학습하기 쉬운 단일 루프 구조를 사용했기 때문에 STT, Gemma, TTS가 실행되는
동안 카메라 화면은 잠시 멈춥니다. 기능을 이해한 뒤 작업 스레드로 분리하는
것을 다음 확장 과제로 권장합니다.

## 8. Structured Output 예시

질문할 때 다음 파일이 원자적으로 저장됩니다.

```text
project/runtime/visual_assistant/
├── vision_data.json
├── current_frame.jpg
├── input.wav
└── response.wav
```

`vision_data.json` 예시:

```json
{
    "image_width": 1280,
    "image_height": 720,
    "position_reference": "카메라 이미지 기준",
    "distance_method": "BBox 화면 점유율 기반 상대적 추정",
    "objects": [
        {
            "class": "chair",
            "class_ko": "의자",
            "confidence": 0.91,
            "bbox": {"x1": 820, "y1": 260, "x2": 1190, "y2": 700},
            "position": {
                "horizontal": "오른쪽",
                "vertical": "아래쪽"
            },
            "area_ratio": 0.1767,
            "proximity_hint": "가까워 보입니다"
        }
    ]
}
```

## 9. 안전 제한

- YOLO가 탐지하지 못한 물체가 실제로 없다고 단정하지 않습니다.
- Gemma는 카메라에 없는 객체를 추측하지 않도록 제한합니다.
- BBox 기반 가까움은 실제 깊이나 충돌 거리로 사용하지 않습니다.
- “안전하다”, “건너도 된다”, “진행해도 된다”는 판단을 출력하지 않습니다.
- 화면 방향은 카메라 설치 방향과 좌우 반전 설정에 영향을 받습니다.

향후 실제 보조 연구로 확장하려면 깊이 카메라, 카메라 자세 보정, 전용 장애물
데이터셋, 다중 프레임 추적, 진동 피드백과 별도의 규칙 기반 안전 계층이
필요합니다.

## 10. 순수 로직 테스트

Jetson 모델 없이도 화면 구역, BBox 점유율, JSON 자연어 변환을 검사할 수
있습니다.

```bash
python3 -m unittest project.test_visual_assistant -v
```
