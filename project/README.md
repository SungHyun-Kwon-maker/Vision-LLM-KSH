# Voice-Controlled Visual Object Finder

## 1. 프로젝트 개요

사용자가 마이크에 자연어로 찾고 싶은 물체를 말하면, 로컬 Gemma가 명령에서 탐색 대상과 색상 조건을 추출하고 YOLO가 Jetson 카메라 영상에서 해당 물체를 실시간으로 찾아 강조하는 시청각 멀티모달 프로젝트입니다.

예시 명령:

- `물병을 찾아줘.`
- `카메라에서 빨간 컵을 찾아줘.`
- `휴대전화를 찾아줘.`
- `의자가 어디 있는지 알려줘.`

이 프로젝트는 기존 강의 자료의 다음 코드를 통합합니다.

| 기능 | 기반 자료 |
|---|---|
| Jetson CSI 카메라 입력 | `yolo11n_test.py`, `tensorrt_object_detection.py` |
| YOLO/TensorRT 객체 탐지 | `04_DL-Object-Detection.ipynb` |
| HSV 색상 조건 검사 | `object_detection.py`, `02_Computer-Vision.ipynb` |
| 로컬 Gemma 자연어 처리 | `05-LLM_and_Gemma.ipynb` |
| 마이크 음성 입력 | SpeechRecognition 또는 로컬 Whisper |

## 2. 시스템 구조

```text
사용자 음성
   │
   ▼
음성 인식(STT)
   │  "빨간 컵을 찾아줘"
   ▼
로컬 Gemma 명령 해석
   │  {"target_class": "cup", "color": "red"}
   ▼
YOLO/TensorRT 실시간 객체 탐지
   │
   ▼
OpenCV 색상 조건 검사 및 Bounding Box 표시
```

Gemma가 카메라 영상을 직접 보는 구조는 아닙니다. Gemma는 자연어 명령을 YOLO가 이해할 수 있는 COCO 클래스와 선택적인 색상 조건으로 변환하고, 실제 영상 인식은 YOLO가 담당합니다. 이 역할 분리가 실시간 성능과 시스템 설명 측면에서 중요합니다.

## 3. 구현 범위

현재 구현된 기능:

- 한국어/영어 음성 명령 입력
- Google 음성 인식 또는 로컬 Whisper STT 선택
- 로컬 GGUF Gemma를 이용한 자연어 의도 분석
- YOLO 기본 COCO 80개 클래스 중 탐색 대상 선택
- 빨강, 주황, 노랑, 초록, 파랑, 보라, 분홍, 검정, 흰색, 회색, 갈색 조건 처리
- Jetson CSI 카메라, USB 카메라, 동영상 파일 입력
- YOLO `.pt` 및 TensorRT `.engine` 모델 지원
- 탐지 결과, 신뢰도, FPS 화면 표시
- 이전 탐색 대상을 Gemma 프롬프트에 전달하는 간단한 문맥 유지
- LLM 응답이 올바른 JSON이 아닐 때 기본 한국어 키워드 매핑으로 복구

기본 YOLO 모델이 학습한 COCO 80개 클래스만 찾을 수 있습니다. `person`, `bottle`, `cup`, `chair`, `backpack`, `cell phone`, `laptop` 등은 가능하지만 개인 지갑이나 특정 제품처럼 COCO에 없는 물체는 커스텀 모델이 필요합니다.

## 4. 파일 구성

```text
project/
├── README.md
└── voice_object_finder.py
```

## 5. 필요한 모델

강의 노트북과 동일한 기본 경로를 사용합니다.

```text
src/models/YOLO/yolo11n.pt
src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf
```

TensorRT를 사용할 때는 실행 옵션으로 엔진 경로를 전달합니다.

```text
src/models/YOLO/yolo11n_fp16.engine
src/models/YOLO/yolo11n_int8.engine
```

이 저장소의 `.gitignore`가 `src/models/*`를 제외하므로 모델은 Git에 나타나지 않습니다. Jetson의 위 경로에 모델 파일이 실제로 존재하는지 먼저 확인해야 합니다.

## 6. 패키지 준비

YOLO, OpenCV, CUDA PyTorch, `llama-cpp-python`은 강의 실습 환경에 설치된 버전을 우선 사용합니다. 특히 Jetson CSI 카메라는 GStreamer가 활성화된 OpenCV가 필요하므로 일반 `opencv-python` 패키지로 기존 Jetson OpenCV를 덮어쓰지 않는 것이 좋습니다.

Google STT를 사용하는 가장 간단한 시연 구성:

```bash
sudo apt install portaudio19-dev python3-pyaudio
pip install SpeechRecognition PyAudio
```

Google STT는 인터넷 연결이 필요합니다. 인터넷 없이 실행하려면 Whisper 계열 STT를 설치합니다.

```bash
pip install SpeechRecognition PyAudio openai-whisper
```

로컬 Gemma를 위한 CUDA 기반 `llama-cpp-python` 설치는 강의 자료와 동일합니다.

```bash
export PATH=/usr/local/cuda/bin:$PATH
CUDACXX=/usr/local/cuda/bin/nvcc CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc" \
pip install --no-cache-dir llama-cpp-python
```

## 7. 실행 방법

모든 명령은 저장소 최상위 폴더에서 실행합니다.

### CSI 카메라 + Google STT

```bash
python3 project/voice_object_finder.py \
    --source csi \
    --stt google
```

### CSI 카메라 + 로컬 Whisper STT

Jetson 메모리를 고려해 기본적으로 작은 `tiny` 모델과 CPU 추론을 사용합니다.

```bash
python3 project/voice_object_finder.py \
    --source csi \
    --stt whisper \
    --whisper-model tiny \
    --whisper-device cpu
```

### TensorRT YOLO 엔진 사용

```bash
python3 project/voice_object_finder.py \
    --source csi \
    --stt google \
    --yolo-model src/models/YOLO/yolo11n_fp16.engine
```

### USB 카메라 사용

```bash
python3 project/voice_object_finder.py \
    --source usb \
    --camera-index 0 \
    --stt google
```

### 마이크 없이 기능 점검

터미널에 자연어 명령을 직접 입력할 수 있습니다.

```bash
python3 project/voice_object_finder.py \
    --source usb \
    --stt keyboard
```

전체 실행 옵션은 다음 명령으로 확인합니다.

```bash
python3 project/voice_object_finder.py --help
```

## 8. 조작 방법

OpenCV 카메라 창에서 다음 키를 사용합니다.

| 키 | 동작 |
|---|---|
| `V` | 마이크 녹음 시작 또는 키보드 명령 입력 |
| `C` | 현재 탐색 대상 초기화 |
| `Q` | 프로그램 종료 |

카메라가 뒤집혀 보이면 `--flip 0`(상하), `--flip 1`(좌우), `--flip -1`(상하좌우) 옵션을 추가합니다.

## 9. 추천 시연 순서

1. `V`를 누르고 `물병을 찾아줘`라고 말합니다.
2. 터미널에서 STT 문장과 Gemma가 만든 JSON을 확인합니다.
3. 물병을 카메라에 보여주고 Bounding Box와 `FOUND` 표시를 확인합니다.
4. 다시 `V`를 누르고 `빨간 컵을 찾아줘`라고 말합니다.
5. 빨간 컵과 다른 색 컵을 함께 보여주며 색상 필터의 차이를 설명합니다.
6. 화면에 없는 물체를 요청해 `Not found` 상태도 시연합니다.

## 10. 평가 항목 예시

- 서로 다른 5개 COCO 물체에 대한 음성 명령 성공률
- STT부터 탐색 대상 설정까지 걸린 시간
- YOLO 탐지 FPS
- 물체가 없을 때의 예외 처리
- 자연어 표현이 달라도 동일한 클래스로 변환되는지 확인

## 11. 한계와 확장 방향

- 색상 판별은 Bounding Box 내부 HSV 픽셀 비율을 사용하므로 조명과 배경의 영향을 받습니다.
- `사람 옆의 가방` 같은 공간 관계는 아직 구현하지 않았습니다.
- 항상 듣는 방식이 아니라 `V` 키를 눌러 녹음을 시작하므로 원치 않는 음성 입력을 방지할 수 있습니다.
- 향후 TTS를 추가하면 `물병을 찾았습니다`라는 결과를 음성으로 안내할 수 있습니다.
- 객체 추적을 추가하면 물체가 움직여도 같은 대상을 안정적으로 유지할 수 있습니다.
- Summary Memory를 추가하면 `아까 찾은 것`, `그거 말고 다른 것` 같은 후속 명령을 더 정확히 처리할 수 있습니다.

