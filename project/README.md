# Traffic Light Voice Safety Assistant

## 1. 프로젝트 개요

Jetson 카메라 영상에서 YOLO로 신호등을 탐지하고, OpenCV 색상 분석으로 현재 신호를 판단한 뒤 신호가 바뀔 때 음성으로 안내하는 교통안전 보조 시스템입니다.

| 인식 상태 | 화면 표시 | TTS 안내 |
|---|---|---|
| 빨간불 | `STOP` | `빨간불입니다. 정차하세요.` |
| 노란불 | `CAUTION` | `노란불입니다. 감속하고 정차를 준비하세요.` |
| 초록불 | `GO - CHECK SURROUNDINGS` | `초록불입니다. 주변을 확인하고 출발하세요.` |
| 미확인 | `NO RELIABLE SIGNAL` | 안내하지 않음 |

단일 프레임의 오검출로 잘못 안내하지 않도록 최근 여러 프레임의 판정 결과가 일정 횟수 이상 일치할 때만 신호 변경을 확정합니다. TTS도 매 프레임 반복하지 않고 안정된 신호 상태가 바뀔 때만 실행합니다.

## 2. 시스템 구성

```text
Jetson CSI/USB 카메라 또는 동영상
                │
                ▼
       YOLO 신호등 객체 탐지
                │
                ▼
    Bounding Box 내부 HSV 색상 분석
      red / yellow / green / unknown
                │
                ▼
       최근 N프레임 다수결 필터
                │
                ▼
      화면 Overlay + 비동기 TTS 안내
```

## 3. 강의 내용과의 연결

| 프로젝트 기능 | 활용한 강의 내용 및 파일 |
|---|---|
| Jetson CSI 카메라 | `yolo11n_test.py`, `tensorrt_object_detection.py` |
| YOLO 신호등 탐지 | `04_DL-Object-Detection.ipynb` |
| HSV 색공간 및 마스크 | `02_Computer-Vision.ipynb`, `object_detection.py` |
| GPU/TensorRT 가속 | `03_DL-and-GPU.ipynb`, `04_DL-Object-Detection.ipynb` |
| 음성 안내 | 오프라인 `espeak-ng` 또는 `pyttsx3` |

이 버전은 안전 판단을 규칙 기반으로 처리하며 Gemma를 사용하지 않습니다. `빨간불이면 정차` 같은 안전 규칙을 LLM의 자유로운 응답에 맡기지 않기 위해서입니다. 향후 Gemma는 사용자의 질문을 이해하거나 판정 이유를 자연어로 설명하는 보조 기능으로 추가할 수 있습니다.

## 4. 파일 구성

```text
project/
├── README.md
└── traffic_light_tts.py
```

## 5. 주요 구현 기능

- COCO YOLO의 `traffic light` 클래스만 선택적으로 탐지
- `.pt` 모델과 TensorRT `.engine` 모델 지원
- 여러 신호등이 보이면 가장 큰 신호등 또는 최고 신뢰도 신호등 선택
- 선택된 Bounding Box 안에서 빨강·노랑·초록 HSV 픽셀 비율 계산
- 최소 색상 비율과 우세 비율을 통과하지 못하면 `unknown` 처리
- 최근 7프레임 중 5프레임 이상 일치해야 안정 상태 변경
- 신호 변경 시에만 TTS 실행
- 같은 신호를 재검출했을 때 반복 안내를 막는 쿨다운
- TTS를 별도 스레드에서 실행하여 영상 FPS 저하 방지
- CSI 카메라, USB 카메라, 동영상 파일 지원
- FPS, YOLO 신뢰도, 원시/안정 상태, 색상 비율 Overlay

## 6. 준비 사항

### YOLO 모델

기본 경로는 기존 강의 코드와 같습니다.

```text
src/models/YOLO/yolo11n.pt
```

TensorRT 엔진도 사용할 수 있습니다.

```text
src/models/YOLO/yolo11n_fp16.engine
src/models/YOLO/yolo11n_int8.engine
```

저장소의 `.gitignore`가 `src/models/*`를 제외하므로 모델 파일은 Git에 표시되지 않습니다. 실행하는 Jetson에 모델이 실제로 존재해야 합니다.

### Python 및 TTS 패키지

YOLO와 OpenCV는 강의 실습 환경에 설치된 버전을 우선 사용합니다. Jetson CSI 카메라는 GStreamer가 활성화된 OpenCV가 필요하므로 일반 `opencv-python`으로 기존 Jetson OpenCV를 덮어쓰지 않는 것이 좋습니다.

오프라인 TTS로 `espeak-ng`를 사용하는 방법이 가장 간단합니다.

```bash
sudo apt update
sudo apt install espeak-ng
```

TTS만 먼저 시험합니다.

```bash
espeak-ng -v ko "빨간불입니다. 정차하세요."
```

대안으로 `pyttsx3`를 사용할 수 있습니다.

```bash
pip install pyttsx3
```

`pyttsx3`의 한국어 음성 품질과 지원 여부는 Jetson에 설치된 시스템 음성에 따라 달라집니다.

## 7. 실행 방법

모든 명령은 저장소 최상위 폴더에서 실행합니다.

### Jetson CSI 카메라

```bash
python3 project/traffic_light_tts.py \
    --source csi \
    --tts-backend espeak
```

카메라 화면이 상하로 뒤집혀 있으면 다음과 같이 실행합니다.

```bash
python3 project/traffic_light_tts.py \
    --source csi \
    --flip 0
```

### USB 카메라

```bash
python3 project/traffic_light_tts.py \
    --source usb \
    --camera-index 0
```

### 강의 신호등 동영상

```bash
python3 project/traffic_light_tts.py \
    --source video \
    --video-path src/videos/section4_project_traffic.mp4 \
    --loop
```

현재 저장소에는 해당 동영상이 없으므로 강의에서 제공받은 파일을 위 경로에 넣거나 실제 경로를 `--video-path`로 전달해야 합니다.

### TensorRT 가속

```bash
python3 project/traffic_light_tts.py \
    --source csi \
    --yolo-model src/models/YOLO/yolo11n_fp16.engine
```

### 음성 없이 영상 기능만 점검

```bash
python3 project/traffic_light_tts.py \
    --source csi \
    --tts-backend none
```

전체 옵션은 다음 명령으로 확인합니다.

```bash
python3 project/traffic_light_tts.py --help
```

## 8. 조작 방법

OpenCV 영상 창에서 다음 키를 사용합니다.

| 키 | 동작 |
|---|---|
| `Q` | 프로그램 종료 |
| `M` | TTS 음소거/해제 |
| `R` | 현재 안정 상태의 안내 문장 다시 말하기 |

## 9. 판정 과정

### 1단계: 신호등 탐지

YOLO 추론 시 COCO 클래스 번호 9번인 `traffic light`만 탐지합니다. 기본값은 화면에서 가장 크게 보이는 신호등을 안내 대상으로 선택합니다.

### 2단계: 색상 분석

선택된 Bounding Box를 HSV로 변환하고 빨강·노랑·초록 마스크의 픽셀 비율을 계산합니다. 가장 높은 색상이 다음 조건을 모두 만족해야 유효한 판정으로 사용됩니다.

- Bounding Box 전체에서 해당 색상이 차지하는 최소 비율
- 세 색상 픽셀 중 해당 색상이 차지하는 최소 우세 비율

기본 임계값은 다음과 같습니다.

```text
minimum_color_ratio     = 0.01
minimum_color_dominance = 0.55
```

### 3단계: 시간 안정화

기본적으로 최근 7프레임 중 같은 상태가 5번 이상 나타나야 안정 상태로 확정합니다.

```text
stability_window = 7
stability_votes  = 5
```

카메라가 많이 흔들리거나 오검출이 잦으면 `--stability-window 10 --stability-votes 7`처럼 높일 수 있습니다. 반응이 너무 느리면 값을 낮춥니다.

## 10. 작은 신호등이 잘 탐지되지 않을 때

신호등은 화면에서 작게 보이는 경우가 많습니다. 다음 순서로 조정합니다.

1. 신호등이 카메라에서 충분히 크게 보이도록 거리를 조절합니다.
2. YOLO 신뢰도 기준을 `0.20` 정도로 낮춥니다.
3. 입력 크기를 `960`으로 높입니다.

```bash
python3 project/traffic_light_tts.py \
    --source csi \
    --confidence 0.20 \
    --image-size 960
```

색상은 인식되지만 상태가 계속 `unknown`이면 다음과 같이 색상 임계값을 완화합니다.

```bash
python3 project/traffic_light_tts.py \
    --source csi \
    --minimum-color-ratio 0.005 \
    --minimum-color-dominance 0.45
```

## 11. 추천 시연 순서

1. 신호등 영상 또는 모형 신호등을 카메라에 보여줍니다.
2. 화면에서 YOLO Bounding Box와 색상 픽셀 비율을 설명합니다.
3. 빨간불이 5프레임 이상 유지되면 `정차하세요` 음성이 한 번만 나오는 것을 보여줍니다.
4. 계속 빨간불이어도 음성이 반복되지 않는 점을 설명합니다.
5. 초록불로 바꾸고 안정화 이후 출발 안내가 나오는 것을 보여줍니다.
6. `M`과 `R` 키로 음소거와 재안내 기능을 시연합니다.

## 12. 한계와 확장 방향

- 실제 신호등의 크기, 조명, 역광에 따라 HSV 임계값 조정이 필요합니다.
- 여러 방향의 신호등이 동시에 보이면 현재는 가장 큰 신호등 하나만 선택합니다.
- 실제 차량 제어에 연결하지 않는 교육용 프로토타입입니다.
- 보행자와 횡단보도 탐지를 추가하면 보행 안전 안내 시스템으로 확장할 수 있습니다.
- 마이크 질문과 Gemma를 추가하면 `지금 출발해도 돼?` 같은 질의응답을 제공할 수 있습니다. 안전 상태 판정은 계속 규칙 기반으로 유지하는 것이 좋습니다.

