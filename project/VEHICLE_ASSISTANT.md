# YOLO–Gemma–MediaPipe 차량비서 AI

## 1. 프로젝트 개요

Jetson에서 녹화 도로 영상과 USB 웹캠을 동시에 사용하는 정차 상태 교육용
프로젝트입니다.

- 녹화 도로 영상: YOLO 도로 객체 탐지와 신호등 색 안정화
- USB 웹캠: MediaPipe 손동작을 이용한 시스템 전체 볼륨 조절
- 마이크: Whisper.cpp 한국어 STT
- Gemma 4: 현재 도로 장면에 관한 음성 질문 응답
- 스피커: Piper 한국어 TTS

실제 차량 제어, 실시간 주행 보조, 충돌 회피 또는 출발 가능 판단에는 사용할
수 없습니다.

## 2. 시스템 구조

```text
녹화 도로 영상(최대 30 FPS)
        │
        ├─ 최신 프레임 큐(maxsize=1) → YOLO(10 FPS) → 도로 객체 JSON
        │                                  │
        │                                  └→ 신호등 ROI → HSV → 7회 중 5표
        │                                                        │
        │                                                        └→ 규칙 기반 TTS
        │
        └─ 음성 질문 시 최신 YOLO 시점 전체 프레임 1장 ───────────────┐
                                                                      │
마이크 → arecord → Whisper.cpp → 질문 ─────────────────────────────────┤
                                                                      ▼
                                     JSON + 이미지 1장 + 질문 → Gemma 4
                                                                      │
                                                                      ▼
                                                          Piper → 시스템 출력

USB 웹캠 → MediaPipe(15 FPS) → 전용 손모양 0.4초 → 엄지/검지 거리
                                                        │
                                                        ▼
                                            시스템 볼륨 0~100%
```

YOLO와 Gemma는 Jetson GPU 잠금을 공유합니다. Gemma가 답하는 동안 YOLO는
기다리지만, 입력 큐에는 항상 최신 도로 프레임 한 장만 남기므로 오래된 프레임이
누적되지 않습니다. MediaPipe는 별도 작업자에서 계속 동작합니다.

## 3. 프레임 정책

| 기능 | 기본 처리율 | 이유 |
|---|---:|---|
| 도로 영상 표시 | 원본과 30 FPS 중 작은 값 | 자연스러운 데모 화면 |
| YOLO | 10 FPS | Jetson 부하와 신호 반응성의 균형 |
| 신호 안정화 | 7회 중 5회 | 약 0.5~0.7초 동안 반복 확인 |
| MediaPipe | 15 FPS | 손동작 반응성과 CPU 부하의 균형 |
| Gemma | 질문당 최신 프레임 1장 | 2,048 context와 Jetson 메모리 절약 |

Gemma에는 손 카메라 이미지나 연속 도로 프레임을 보내지 않습니다. 움직임이나
실제 속도도 추측하지 않습니다.

## 4. Colab 코드와의 대응

핵심 코드는 `04_DL-Object-Detection.ipynb`와
`06-Vision_LLM_Multimodal_Systems.ipynb`의 순서를 유지했습니다.

| 수업 내용 | 통합 프로그램 |
|---|---|
| `YOLO(...).predict()` | `detect_road_scene()` |
| `for box in result.boxes` | 도로 클래스 Dictionary 변환 |
| Dictionary → JSON → 자연어 | `build_vision_context()`, `detections_to_text()` |
| OpenCV 이미지 → Base64 | `frame_to_image_data()` |
| `Gemma4ChatHandler` | `GemmaRoadAssistant` |
| `arecord` + Whisper.cpp | `SpeechToText` |
| Piper + WAV 재생 | `TTSWorker`, `SystemAudioPlayer` |
| MediaPipe HandLandmarker | `HandVolumeWorker` |

수업 코드에 추가된 부분은 작업자 스레드, 최신 프레임 큐, 시스템 볼륨 백엔드,
제스처 안정화입니다.

## 5. 도로 탐지와 신호등 판정

추가 학습 없이 COCO 클래스 중 다음 객체만 사용합니다.

```text
person, bicycle, car, motorcycle,
bus, truck, traffic light, stop sign
```

신호등은 화면 위쪽의 탐지 결과를 기본 선택하고, 해당 ROI에서 HSV 색상 비율을
계산합니다. 단일 판정을 바로 사용하지 않고 최근 7회의 결과 중 같은 상태가
5회 이상일 때만 안정 상태를 바꿉니다.

안내 문구도 Gemma가 아닌 규칙으로 결정합니다.

- 빨강: 정차 상태 유지
- 노랑: 주의 필요
- 초록: 주변 교통 상황 직접 확인
- 알 수 없음: 자동 음성 없음

초록 상태에서도 “출발해도 된다”거나 “안전하다”고 출력하지 않습니다.

## 6. 볼륨 제스처

기본 제어 손은 오른손입니다. 미러링된 웹캠 화면에서 다음 순서로 동작합니다.

1. 검지는 펴고 중지·약지·소지는 접습니다.
2. 이 자세를 6프레임, 약 0.4초 유지하면 `ACTIVE`가 됩니다.
3. 엄지 끝과 검지 끝의 거리를 벌리거나 좁힙니다.
4. 자세를 8프레임 이상 잃으면 다시 `LOCKED`가 됩니다.

카메라와 손 사이 거리가 변해도 비교적 일정하도록 다음 값을 사용합니다.

```python
pinch_ratio = distance(thumb_tip_4, index_tip_8) \
              / distance(index_mcp_5, pinky_mcp_17)
```

`0.20`은 0%, `1.50`은 100%로 선형 변환합니다. 결과에는 EMA 0.25,
5% 단계, 최소 변화폭 5%, 명령 쿨다운 200ms를 적용합니다. 볼륨 변경을
말로 반복 안내하지 않고 웹캠 창의 막대로 표시합니다.

손 카메라 프레임은 파일에 저장하지 않습니다.

## 7. 시스템 볼륨과 TTS 출력

`--volume-backend auto`는 실제 연결 가능한 백엔드를 다음 순서로 검사합니다.

1. PipeWire: `wpctl` + `pw-play`
2. PulseAudio: `pactl` + `paplay`
3. ALSA: `amixer` + `aplay`

Piper WAV도 선택된 시스템 출력으로 재생하므로 손동작 볼륨이 AI 음성에도
적용됩니다. Gemma는 운영체제 명령을 만들지 않으며, Python 코드에 정의된
볼륨 명령만 인자 목록으로 실행합니다.

Jetson 오디오 확인:

```bash
wpctl status
wpctl get-volume @DEFAULT_AUDIO_SINK@

pactl info
pactl get-sink-volume @DEFAULT_SINK@

amixer -D default scontrols
aplay -l
```

ALSA를 직접 사용해야 한다면 카드와 mixer 이름을 지정합니다.

```bash
--volume-backend amixer \
--alsa-card hw:2 \
--alsa-control Master \
--speaker-device plughw:2,0
```

## 8. 필요한 파일

```text
src/models/YOLO/yolo11n_int8.engine
src/models/MediaPipe/hand_landmarker.task

src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf
src/models/Gemma4/mmproj-google_gemma-4-E2B-it-f16.gguf

whisper.cpp/build-cpu/bin/whisper-cli
whisper.cpp/models/ggml-base.bin

.piper_venv/bin/python
src/models/Piper/ko_KR-kss-medium.onnx
src/models/Piper/ko_KR-kss-medium.onnx.json
```

현재 저장소에는 용량이 큰 YOLO TensorRT 엔진과 Gemma 모델이 포함되어 있지
않을 수 있습니다. 실행하는 Jetson에 위 경로로 배치해야 합니다.

녹화 도로 영상은 별도 경로에 준비합니다. 영상에는 COCO YOLO가 탐지할 수 있는
차량·사람·신호등이 보이는 구간이 포함되어야 합니다.

## 9. 실행

저장소 최상위 폴더에서:

```bash
python3 project/vehicle_assistant.py \
    --road-video src/videos/road_demo.mp4 \
    --hand-camera-index 0
```

기본 처리율을 명시하려면:

```bash
python3 project/vehicle_assistant.py \
    --road-video src/videos/road_demo.mp4 \
    --road-fps 10 \
    --hand-fps 15 \
    --display-fps 30 \
    --volume-backend auto
```

마이크 없이 키보드 질문으로 먼저 확인:

```bash
python3 project/vehicle_assistant.py \
    --road-video src/videos/road_demo.mp4 \
    --stt-backend keyboard
```

단계별 진단을 위해 Gemma와 손동작을 모두 끄고 YOLO만 실행:

```bash
python3 project/vehicle_assistant.py \
    --road-video src/videos/road_demo.mp4 \
    --disable-interaction \
    --disable-volume-control \
    --tts-backend none
```

도로 영상은 기본 반복됩니다. 한 번 재생 후 종료하려면 `--no-loop`를 추가합니다.

## 10. 조작키

| 키 | 기능 |
|---|---|
| `V` | Whisper 녹음 후 최신 도로 프레임 한 장으로 Gemma 질문 |
| `R` | 현재 규칙 기반 신호 상태 다시 안내 |
| `M` | AI 안내 TTS만 음소거/해제 |
| `Q` | 프로그램 종료 |

`M`은 시스템 전체 볼륨을 바꾸지 않습니다. 시스템 전체 볼륨은 MediaPipe
손동작으로만 조절합니다.

## 11. 런타임 출력

음성 질문을 시작한 시점의 도로 정보만 저장됩니다.

```text
project/runtime/vehicle_assistant/
├── road_vision_data.json
├── road_frame.jpg
├── input.wav
└── response.wav
```

Gemma에 전달한 프레임과 JSON은 같은 YOLO 처리 시점의 데이터입니다.

## 12. 테스트

Jetson 모델 없이도 신호 필터, 프레임 스케줄, 제스처, Gemma 이미지 개수와
오디오 명령을 검사할 수 있습니다.

```bash
python3 -m unittest project.test_vehicle_assistant -v
python3 -m py_compile \
    project/vehicle_assistant.py \
    project/test_vehicle_assistant.py
```

Jetson 수동 확인 항목:

- YOLO 표시가 약 10 FPS로 갱신되고 오래된 프레임이 누적되지 않는지
- 같은 신호 색이 5회 확인되기 전에는 안정 상태가 바뀌지 않는지
- 전용 자세를 약 0.4초 유지해야만 볼륨 모드가 활성화되는지
- 활성화 후 볼륨이 5% 단위로 바뀌는지
- 손을 내리면 마지막 볼륨이 유지되는지
- 손동작 볼륨이 음악·영상·Piper 출력에 함께 적용되는지
- `V` 질문 한 번에 Gemma가 최신 도로 프레임 한 장만 받는지

## 13. 문제 해결

오른손이 선택되지 않으면 다음을 순서대로 시험합니다.

```bash
--control-hand left
--no-mirror-hand-camera
```

Jetson 부하가 높으면:

```bash
--road-fps 5 --hand-fps 10
```

신호등이 너무 작아 탐지되지 않으면 `--confidence`를 조금 낮출 수 있지만,
오탐도 함께 늘어납니다. 실제 테스트에서는 `0.01~0.03` 범위에서 녹화 영상에
맞춰 조정합니다.

시스템 볼륨 자동 탐지가 실패하면 `wpctl`, `pactl`, `amixer` 중 실제로
동작하는 명령을 먼저 터미널에서 확인하고 `--volume-backend`로 고정합니다.
