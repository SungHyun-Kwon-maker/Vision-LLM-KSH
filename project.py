import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from ultralytics import YOLO
from llama_cpp import Llama


YOLO_MODEL_PATH = "src/models/YOLO/yolo11n_int8.engine"
GEMMA_MODEL_PATH = "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
JSON_PATH = "src/output/vision_data.json"

CONTEXT_WINDOW = 2048
MAX_TOKENS = 150

def update_mask(h_upper1=10, h_lower2=170):
    lower1 = np.array([0, 100, 50])
    upper1 = np.array([h_upper1, 255, 255])
    lower2 = np.array([h_lower2, 100, 50])
    upper2 = np.array([180, 255, 255])

    # 이미지를 HSV 색공간으로 변환
    result_hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)

    # Mask 생성
    mask1 = cv2.inRange(result_hsv, lower1, upper1)
    mask2 = cv2.inRange(result_hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

def detections_to_text(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        vision_data = json.load(file)
        
    objects = vision_data["objects"]

    if len(objects) == 0:
        return "현재 탐지된 객체가 없습니다."

    sentences = []

    for index, obj in enumerate(objects, start=1):
        sentence = f"{index}번 객체는 {obj['class']}이며, confidence는 {obj['confidence']:.2f}입니다."

        sentences.append(sentence)

    return "\n".join(sentences)


yolo = YOLO(YOLO_MODEL_PATH)

llm = Llama(
    model_path=GEMMA_MODEL_PATH,
    n_gpu_layers=-1,
    n_ctx=CONTEXT_WINDOW,
    n_batch=32,
    n_ubatch=32,
    verbose=False,
)

pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), "
    "width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 !100 "
    "appsink drop=true max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)


if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()


print("q : 종료")
print("l : 현재 YOLO 탐지 결과를 Gemma에게 전달")


while True:
    ret, frame = cap.read()
    key = cv2.waitKey(1) & 0xFF

    if not ret:
        break
    if key == ord("q"):
        break

    height, width = frame.shape[:2]


    # 1. 현재 프레임 YOLO 탐지
    results = yolo.predict(
        source=frame,
        conf=0.5,
        iou=0.5,
        verbose=False,
    )

    result = results[0]

    result_blur = cv2.GaussianBlur(result, (7,7), 0)

    result_blur_hsv = cv2.cvtColor(result_blur, cv2.COLOR_RGB2HSV)

    mask_hsv1 = cv2.inRange(result_blur_hsv, red_lower1, red_upper1)
    mask_hsv2 = cv2.inRange(result_blur_hsv, red_lower2, red_upper2)
    mask_blur_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)

    red_lower1 = np.array([0, 100, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 100, 50])
    red_upper2 = np.array([180, 255, 255])

    result_hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)

    mask_hsv1 = cv2.inRange(result_hsv, red_lower1, red_upper1)
    mask_hsv2 = cv2.inRange(result_hsv, red_lower2, red_upper2)
    mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)

    binmask_crop = mask_blur_hsv
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binmask_crop_erode = cv2.erode(binmask_crop, None, iterations=2)
    binmask_crop_opened = cv2.morphologyEx(binmask_crop, cv2.MORPH_OPEN, kernel)
    binmask_crop_double_erode = cv2.erode(binmask_crop_erode, None, iterations=2)

    plt.figure(figsize=(16,8))

    plt.subplot(1,5,1), plt.imshow(mask_hsv, cmap="gray"), plt.title("Original")
    plt.subplot(1,5,2), plt.imshow(binmask_crop, cmap="gray"), plt.title("Blured")

for ax in plt.gcf().axes:
    ax.axis("off")

    # 2. YOLO 결과 화면 출력
    output_frame = result.plot()
    cv2.imshow("YOLO + Gemma", output_frame)


    # 3. YOLO 탐지 결과를 Dictionary 형태로 변환
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
                }
            }
        )

    vision_dict = {
        "image_width": width,
        "image_height": height,
        "objects": objects,
    }



    # 4. Dictionary → JSON 파일 저장
    if key == ord("l"):
        with open(JSON_PATH, "w", encoding="utf-8") as file:
            json.dump(vision_dict, file, ensure_ascii=False, indent=4)
        
        print("\n[JSON 파일 저장 완료]")

        # 5. JSON 파일 → Natural Language 변환
        vision_text = detections_to_text(JSON_PATH)
        print("\n[Vision Context]")
        print(vision_text)

        # 6. 탐지 결과를 Gemma에 Context로 전달
        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": """
                                Instruction:
                                주어진 객체 탐지 정보를 바탕으로 현재 상황을 설명하시오.

                                Constraint:
                                탐지 결과에 없는 객체를 추측하지 마시오.

                                Output Format:
                                한국어 두 문장 이내.
                               """
                },
                {
                    "role": "user",
                    "content": f"""
                                Context:
                                {vision_text}
                               """
                },
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.7,
        )

        answer = response["choices"][0]["message"]["content"]

        print("\n[Gemma]")
        print(answer)


cap.release()
cv2.destroyAllWindows()