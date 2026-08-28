import cv2
import os

from ultralytics import YOLO


YOLO_MODEL_PATH = "src/models/YOLO/yolo11n_int8.engine"
IMAGE_PATH = "src/images/city.png"
OUTPUT_IMAGE_PATH = "src/output/vision_output.jpg"
TEMP_IMAGE_PATH = "src/output/vision_output_temp.jpg"

DISPLAY_WIDTH = 600
DISPLAY_HEIGHT = 600


yolo = YOLO(YOLO_MODEL_PATH)

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise RuntimeError("이미지를 불러올 수 없습니다.")


results = yolo.predict(
    source=image,
    conf=0.015,
    iou=0.5,
    classes=[9],
    verbose=False,
)

result = results[0]


if len(result.boxes) == 0:
    raise RuntimeError("신호등이 탐지되지 않았습니다.")


selected_box = min(result.boxes, key=lambda box: float(box.xyxy[0][1].item()))
confidence = float(selected_box.conf[0].item())
x1, y1, x2, y2 = (selected_box.xyxy[0].cpu().tolist())

x1 = int(x1)
y1 = int(y1)
x2 = int(x2)
y2 = int(y2)


height, width = image.shape[:2]

x1 = max(0, x1)
y1 = max(0, y1)
x2 = min(width, x2)
y2 = min(height, y2)

traffic_light_image = image[y1:y2, x1:x2].copy()

if traffic_light_image.size == 0:
    raise RuntimeError("BBox 이미지를 만들 수 없습니다.")


display_image = cv2.resize(traffic_light_image, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_NEAREST)

cv2.namedWindow("Traffic Light", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Traffic Light", DISPLAY_WIDTH, DISPLAY_HEIGHT)
cv2.imshow("Traffic Light", display_image)


while True:
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()


success = cv2.imwrite(TEMP_IMAGE_PATH, traffic_light_image)

if not success:
    raise RuntimeError("이미지 저장에 실패했습니다.")


os.replace(TEMP_IMAGE_PATH, OUTPUT_IMAGE_PATH)

print(f"\n이미지 저장 완료: {OUTPUT_IMAGE_PATH}")