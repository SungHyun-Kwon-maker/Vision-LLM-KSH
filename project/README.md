# Traffic Light Vision-LLM Audiovisual Assistant

## 1. 프로젝트 개요

Jetson 카메라에서 신호등을 탐지하고 자동 안전 음성 안내를 제공하며, 사용자가 현재 상황에 대해 음성으로 질문하면 신호등 ROI 이미지를 보는 로컬 Gemma가 한국어 음성으로 답하는 Vision-LLM 시청각 멀티모달 시스템입니다.

이 프로젝트는 `06-Vision_LLM_Multimodal_Systems.ipynb`의 Final Project 요구사항을 다음과 같이 모두 포함합니다.

| 최종 과제 요구사항 | 프로젝트 구현 |
|---|---|
| YOLO 객체 탐지 | COCO `traffic light` 클래스 탐지 |
| 특정 이벤트 감지 | 안정된 빨강·노랑·초록 상태 변화 |
| Structured Output | BBox·Confidence·색상 비율을 JSON으로 생성 |
| Vision-to-Text | JSON 탐지 결과를 Gemma용 자연어 Context로 변환 |
| Image + Text 멀티모달 | 신호등 ROI 이미지와 음성 질문을 Gemma 4에 전달 |
| 사용자 음성 입력 | `arecord` + Whisper.cpp 한국어 STT |
| 로컬 LLM | GGUF Gemma 4 + `Gemma4ChatHandler` + `mmproj` |
| 음성 출력 | Piper 한국어 TTS + `aplay` |
| LLM Memory | 최근 2개 질의응답 Sliding Window |

## 2. 사용 시나리오

### 자동 안전 이벤트

```text
카메라에서 빨간 신호 감지
        ↓
최근 7프레임 중 5프레임 이상 빨강
        ↓
Piper: “빨간불입니다. 정차하세요.”
```

신호가 계속 빨간색이어도 음성을 매 프레임 반복하지 않습니다. 안정 상태가 바뀌거나 신호를 잃었다가 쿨다운 이후 다시 같은 상태가 확인될 때만 안내합니다.

### 사용자 음성 질의응답

```text
사용자가 V 키를 누름
        ↓
“지금 출발해도 돼?”라고 질문
        ↓
arecord로 5초 녹음
        ↓
Whisper.cpp 한국어 STT
        ↓
질문 + YOLO JSON + 신호등 ROI 이미지
        ↓
Gemma 4 Vision
        ↓
Piper: “현재 빨간불이므로 정차해서 기다려야 합니다.”
```

## 3. 전체 시스템 구조

```text
                              ┌─→ HSV 색상 분석
Jetson CSI/USB 카메라 → YOLO ─┤   └→ 다중 프레임 안정화
                              │          └→ 안전 이벤트 → Piper TTS
                              │
                              └─→ 신호등 ROI 이미지 ───────────────┐
                                                                  │
마이크 → arecord → Whisper.cpp STT → 사용자 질문 ─────────────────┤
                                                                  ▼
                  YOLO Structured JSON + Natural Language → Gemma 4 Vision
                                                                  │
                                                                  ▼
                                                         Piper → aplay
```

YOLO와 Gemma가 같은 Jetson GPU를 동시에 사용하지 않도록 Gemma 추론 중에는 YOLO 추론을 잠시 멈추고 카메라 화면만 계속 표시합니다. STT와 Gemma는 별도 작업 스레드에서 실행되어 OpenCV 키 입력과 화면 처리가 중단되지 않습니다.

## 4. 안전 설계

`정차`와 `출발` 판단을 Gemma의 자유로운 응답에 맡기지 않습니다.

- HSV 색상 분석과 최근 프레임 다수결이 `stable_state`를 결정합니다.
- 자동 TTS는 규칙 기반 `stable_state`만 사용합니다.
- Gemma 프롬프트에도 이 상태를 안전 기준으로 사용하도록 전달합니다.
- `red`에서는 출발을 권하지 않고, `unknown`에서는 판별할 수 없다고 답하도록 제한합니다.
- `green`에서도 주변과 보행자를 확인하도록 안내합니다.

이 코드는 교육용 프로토타입이며 실제 차량 제어에는 사용하지 않습니다.

## 5. 강의 00~06과의 연결

| 강의 | 프로젝트 적용 내용 |
|---|---|
| 00 Initial Setup | Jetson 프로젝트 폴더와 실행 환경 |
| 01 Linux and Python | 함수·클래스·자료구조·파일 입출력·subprocess |
| 02 Computer Vision | OpenCV, HSV 색공간, 마스크, Morphology, Overlay |
| 03 DL and GPU | CUDA GPU 추론과 처리 속도/FPS |
| 04 Object Detection | YOLO, COCO 클래스 9, TensorRT INT8 엔진, BBox |
| 05 LLM and Gemma | 로컬 GGUF Gemma, Prompt, Context, Sliding Window Memory |
| 06 Multimodal Systems | JSON/Natural Language Context, ROI 이미지, Whisper.cpp, Piper |

## 6. 프로젝트 파일

```text
project/
├── .gitignore
├── README.md
└── traffic_light_multimodal.py
```

프로그램을 실행하고 `V`로 질문하면 다음 런타임 파일이 원자적으로 저장됩니다.

```text
project/runtime/
├── vision_data.json
└── traffic_light_roi.jpg
```

`runtime/`은 실행 중 생성되는 데이터이므로 `project/.gitignore`에서 제외합니다.

`vision_data.json` 예시:

```json
{
  "image_width": 1280,
  "image_height": 720,
  "objects": [
    {
      "class": "traffic light",
      "confidence": 0.91,
      "bbox": {"x1": 100, "y1": 50, "x2": 160, "y2": 190},
      "raw_color_state": "red",
      "stable_color_state": "red",
      "color_pixel_ratios": {
        "red": 0.12,
        "yellow": 0.0,
        "green": 0.0
      }
    }
  ]
}
```

## 7. 필요한 파일 배치

06번 강의와 동일한 기본 경로를 사용합니다.

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

모델 경로는 `.gitignore` 대상이므로 Git에 표시되지 않습니다. 실행하는 Jetson에 실제 파일이 있어야 합니다.

## 8. Whisper.cpp 설치

06번 강의의 CPU 빌드 기준입니다. YOLO와 Gemma가 GPU를 사용하므로 STT는 CPU 버전으로 시작하는 것이 메모리 관리에 유리합니다.

```bash
git clone https://github.com/ggml-org/whisper.cpp.git
sudo apt install -y cmake alsa-utils pulseaudio-utils

cd whisper.cpp
cmake -B build-cpu
cmake --build build-cpu --target whisper-cli -j4 --config Release
sh ./models/download-ggml-model.sh base
cd ..
```

생성 파일 확인:

```bash
ls -lh whisper.cpp/build-cpu/bin/whisper-cli
ls -lh whisper.cpp/models/ggml-base.bin
```

마이크 장치 목록 확인:

```bash
arecord -l
```

기본 마이크 장치는 강의와 같은 `plughw:3,0`입니다. 장치 번호가 다르면 실행 시 `--microphone-device`를 변경합니다.

## 9. Piper 설치

06번 강의와 동일하게 별도 가상환경을 사용합니다.

```bash
python3 -m venv .piper_venv
source .piper_venv/bin/activate
pip install piper-tts

mkdir -p src/models/Piper
python3 -m piper.download_voices \
    ko_KR-kss-medium \
    --data-dir src/models/Piper

deactivate
```

스피커 장치 목록 확인:

```bash
aplay -l
```

기본 스피커 장치는 강의와 같은 `plughw:2,0`입니다.

Piper 단독 시험:

```bash
mkdir -p src/audio

.piper_venv/bin/python -m piper \
    -m src/models/Piper/ko_KR-kss-medium.onnx \
    -f src/audio/response.wav \
    -- \
    "빨간불입니다. 정차하세요."

aplay -D plughw:2,0 src/audio/response.wav
```

## 10. Gemma 환경

06번의 이미지 입력 방식은 `Gemma4ChatHandler`가 필요합니다. CUDA 기반 `llama-cpp-python`은 05번 강의와 동일하게 설치합니다.

```bash
export PATH=/usr/local/cuda/bin:$PATH
CUDACXX=/usr/local/cuda/bin/nvcc \
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc" \
pip install --no-cache-dir llama-cpp-python
```

## 11. 전체 시스템 실행

저장소 최상위 폴더에서 실행합니다.

```bash
python3 project/traffic_light_multimodal.py \
    --source csi
```

기본 설정:

```text
YOLO       src/models/YOLO/yolo11n_int8.engine
Gemma      src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf
mmproj     src/models/Gemma4/mmproj-google_gemma-4-E2B-it-f16.gguf
Whisper    whisper.cpp/build-cpu/bin/whisper-cli
Piper      .piper_venv/bin/python
마이크     plughw:3,0
스피커     plughw:2,0
```

카메라 화면이 상하로 뒤집혔다면:

```bash
python3 project/traffic_light_multimodal.py \
    --source csi \
    --flip 0
```

마이크와 스피커 번호가 다르다면:

```bash
python3 project/traffic_light_multimodal.py \
    --source csi \
    --microphone-device plughw:1,0 \
    --speaker-device plughw:0,0
```

PulseAudio를 사용하지 않는 환경에서 `pasuspender` 오류가 나면:

```bash
python3 project/traffic_light_multimodal.py \
    --source csi \
    --no-pasuspender
```

## 12. 단계별 점검 방법

전체 모델을 한꺼번에 실행하기 전에 단계별로 확인하는 것이 좋습니다.

### 1단계: Vision만 점검

Gemma·Whisper·TTS 없이 YOLO와 색상 판별만 실행합니다.

```bash
python3 project/traffic_light_multimodal.py \
    --source csi \
    --disable-interaction \
    --tts-backend none
```

### 2단계: Vision + 자동 Piper TTS

```bash
python3 project/traffic_light_multimodal.py \
    --source csi \
    --disable-interaction \
    --tts-backend piper
```

### 3단계: 마이크 대신 키보드 + Gemma Vision

STT 문제를 분리하고 터미널에 질문을 입력합니다.

```bash
python3 project/traffic_light_multimodal.py \
    --source csi \
    --stt-backend keyboard
```

### 4단계: 전체 시청각 시스템

```bash
python3 project/traffic_light_multimodal.py \
    --source csi \
    --stt-backend whisper \
    --tts-backend piper
```

### 강의 동영상 입력

```bash
python3 project/traffic_light_multimodal.py \
    --source video \
    --video-path src/videos/section4_project_traffic.mp4 \
    --loop
```

현재 저장소에는 해당 동영상이 없으므로 실제 파일을 전달한 경로에 배치해야 합니다.

## 13. 조작 방법

OpenCV 영상 창에서 사용합니다.

| 키 | 기능 |
|---|---|
| `V` | 5초간 음성 녹음 → Whisper → Gemma Vision → Piper 응답 |
| `R` | 현재 안정 신호의 안전 문장 다시 안내 |
| `M` | 음성 출력 음소거/해제 |
| `Q` | 프로그램 종료 |

OpenCV 창에 포커스가 있어야 키 입력이 전달됩니다. `--stt-backend keyboard`에서는 `V`를 누른 후 터미널에 질문을 입력합니다.

## 14. 신호 판정 로직

### YOLO ROI 선택

06번 신호등 ROI 예제와 같이 기본 신뢰도는 `0.015`, 클래스는 COCO 번호 9번, 선택 기준은 화면에서 가장 위쪽에 있는 신호등입니다.

```text
confidence       = 0.015
classes          = [9]
signal_selection = topmost
```

다른 환경에서는 다음 선택 기준도 사용할 수 있습니다.

```bash
--signal-selection largest
--signal-selection confidence
```

### HSV 판정

선택된 신호등 ROI에서 빨강·노랑·초록 픽셀 비율을 계산합니다. 가장 높은 색상이 최소 픽셀 비율과 우세 비율을 모두 통과해야 유효합니다.

```text
minimum_color_ratio     = 0.01
minimum_color_dominance = 0.55
```

### 다중 프레임 안정화

기본적으로 최근 7프레임 중 같은 판정이 5번 이상일 때만 상태를 변경합니다.

```text
stability_window = 7
stability_votes  = 5
```

## 15. 추천 발표 시연

1. 프로젝트 구조에서 네 가지 모달리티(Camera Image, JSON/Text, User Audio, TTS Audio)를 설명합니다.
2. 빨간 신호를 보여주고 `정차하세요` 자동 이벤트 음성을 시연합니다.
3. 빨간불이 유지돼도 음성이 반복되지 않는 다중 프레임 안정화와 쿨다운을 설명합니다.
4. `V`를 누르고 `지금 출발해도 돼?`라고 질문합니다.
5. Whisper의 STT 결과와 Gemma 답변을 터미널에서 보여줍니다.
6. `project/runtime/vision_data.json`과 ROI 이미지가 Gemma Context로 전달됐음을 보여줍니다.
7. 초록불로 바꾼 후 같은 질문을 반복해 응답 차이를 시연합니다.
8. 안전 판정은 규칙 기반이고 Gemma는 설명을 담당한다는 점을 강조합니다.

## 16. 문제 해결

### 신호등이 탐지되지 않는 경우

신호등이 너무 작으면 입력 크기를 높이거나 선택 기준을 조정합니다.

```bash
--image-size 960 --confidence 0.01
```

### 색상이 계속 unknown인 경우

조명에 맞게 임계값을 완화합니다.

```bash
--minimum-color-ratio 0.005 \
--minimum-color-dominance 0.45
```

### CUDA 메모리가 부족한 경우

1. Whisper.cpp CPU 빌드를 사용합니다.
2. TensorRT INT8 YOLO 엔진을 사용합니다.
3. Gemma 실행 중 YOLO가 일시 중지되는 기본 동작을 유지합니다.
4. 그래도 부족하면 Gemma 일부 또는 전체 레이어를 CPU로 이동합니다.

```bash
--gemma-gpu-layers 20
```

### TTS 없이 먼저 실행하는 경우

```bash
--tts-backend none
```

## 17. 한계와 확장 방향

- 실제 신호등의 크기·역광·LED 깜빡임에 따라 HSV 임계값 조정이 필요합니다.
- 여러 방향의 신호등이 보이면 기본적으로 가장 위쪽 신호등 하나만 선택합니다.
- 향후 보행자·횡단보도·차량 탐지를 JSON Context에 추가할 수 있습니다.
- 음성 질문이 아닌 특정 손동작을 이벤트로 추가할 수 있습니다.
- 프로세스를 YOLO, Gemma, Audio 세 개로 분리하고 JSON/이미지 파일 또는 IPC로 연결하면 장애 격리와 실시간성이 향상됩니다.
