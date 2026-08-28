import json
from pathlib import Path

import cv2
import numpy as np
from llama_cpp import Llama
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
YOLO_MODEL_PATH = BASE_DIR / "src/models/YOLO/yolo11n_int8.engine"
GEMMA_MODEL_PATH = (
    BASE_DIR / "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
)
JSON_PATH = BASE_DIR / "src/output/vision_data.json"

CONTEXT_WINDOW = 2048
MAX_TOKENS = 150


def update_mask(image, h_upper1=10, h_lower2=170):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 100, 50], dtype=np.uint8)
    upper1 = np.array([h_upper1, 255, 255], dtype=np.uint8)
    lower2 = np.array([h_lower2, 100, 50], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv_image, lower1, upper1)
    mask2 = cv2.inRange(hsv_image, lower2, upper2)
    return cv2.bitwise_or(mask1, mask2)


def detections_to_text(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        vision_data = json.load(file)

    objects = vision_data.get("objects", [])

    if not objects:
        return "현재 탐지된 객체가 없습니다."

    sentences = []
    for index, obj in enumerate(objects, start=1):
        sentence = (
            f"{index}번 객체는 {obj['class']}이며, "
            f"confidence는 {obj['confidence']:.2f}입니다."
        )
        sentences.append(sentence)

    return "\n".join(sentences)


def create_mask_panel(masks):
    tiles = []
    for mask in masks:
        tile = cv2.resize(mask, (256, 144), interpolation=cv2.INTER_NEAREST)
        tiles.append(tile)
    return np.hstack(tiles)


def create_camera_pipeline(sensor_id=0):
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


def create_vision_dict(result, width, height):
    objects = []

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
        objects.append(
            {
                "class": result.names[class_id],
                "confidence": round(confidence, 3),
                "bbox": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                },
            }
        )

    return {
        "image_width": width,
        "image_height": height,
        "objects": objects,
    }


def create_llm():
    return Llama(
        model_path=str(GEMMA_MODEL_PATH),
        n_gpu_layers=-1,
        n_ctx=CONTEXT_WINDOW,
        n_batch=32,
        n_ubatch=32,
        verbose=False,
    )


def describe_detections(llm, vision_text):
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "주어진 객체 탐지 정보를 바탕으로 현재 상황을 설명하시오. "
                    "탐지 결과에 없는 객체를 추측하지 마시오. "
                    "한국어 두 문장 이내로 답하시오."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{vision_text}",
            },
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.7,
    )
    return response["choices"][0]["message"]["content"]


def main():
    if not YOLO_MODEL_PATH.is_file():
        raise FileNotFoundError(f"YOLO 모델을 찾을 수 없습니다: {YOLO_MODEL_PATH}")

    yolo = YOLO(str(YOLO_MODEL_PATH))
    llm = None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cap = cv2.VideoCapture(create_camera_pipeline(), cv2.CAP_GSTREAMER)

    try:
        if not cap.isOpened():
            raise RuntimeError("카메라를 열 수 없습니다.")

        print("q : 종료")
        print("l : 현재 YOLO 탐지 결과를 Gemma에게 전달")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("카메라 프레임을 읽을 수 없습니다.")
                break

            height, width = frame.shape[:2]
            results = yolo.predict(
                source=frame,
                conf=0.5,
                iou=0.5,
                verbose=False,
            )
            result = results[0]

            blurred_frame = cv2.GaussianBlur(frame, (7, 7), 0)
            mask = update_mask(frame)
            blurred_mask = update_mask(blurred_frame)
            eroded_mask = cv2.erode(blurred_mask, kernel, iterations=2)
            opened_mask = cv2.morphologyEx(blurred_mask, cv2.MORPH_OPEN, kernel)
            double_eroded_mask = cv2.erode(eroded_mask, kernel, iterations=2)
            mask_panel = create_mask_panel(
                [mask, blurred_mask, eroded_mask, opened_mask, double_eroded_mask]
            )

            output_frame = result.plot()
            cv2.imshow("YOLO + Gemma", output_frame)
            cv2.imshow("Red Mask Processing", mask_panel)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key != ord("l"):
                continue

            vision_dict = create_vision_dict(result, width, height)
            JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(JSON_PATH, "w", encoding="utf-8") as file:
                json.dump(vision_dict, file, ensure_ascii=False, indent=4)

            print("\n[JSON 파일 저장 완료]")
            vision_text = detections_to_text(JSON_PATH)
            print("\n[Vision Context]")
            print(vision_text)

            if llm is None:
                if not GEMMA_MODEL_PATH.is_file():
                    raise FileNotFoundError(
                        f"Gemma 모델을 찾을 수 없습니다: {GEMMA_MODEL_PATH}"
                    )
                llm = create_llm()

            answer = describe_detections(llm, vision_text)
            print("\n[Gemma]")
            print(answer)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
