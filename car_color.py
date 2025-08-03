from ultralytics import YOLO
import cv2
import numpy as np

# تحميل النموذج
model = YOLO("C:/Users/super/Desktop/yolo/yolov8n.pt")

# تحميل الفيديو
video_path = "yourpath"
cap = cv2.VideoCapture(video_path)

class_names = model.names  # أسماء الكلاسات

# دالة لتحديد لون السيارة بناءً على المتوسط
def get_car_color(avg_color):
    blue, green, red = avg_color

    if all(c > 200 for c in [red, green, blue]):
        return "White"
    elif all(c < 50 for c in [red, green, blue]):
        return "Black"
    elif red > blue and red > green:
        return "Red"
    elif blue > red and blue > green:
        return "Blue"
    elif green > red and green > blue:
        return "Green"
    elif abs(red - blue) < 20 and red > 100:
        return "Gray"
    else:
        return "Unknown"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, show=False, conf=0.5)
    boxes = results[0].boxes

    for box in boxes:
        cls_id = int(box.cls[0])
        class_name = class_names[cls_id]

        if class_name == "car":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            avg_color = crop.mean(axis=0).mean(axis=0)  # BGR
            color_name = get_car_color(avg_color)

            # طباعة الموقع واللون
            print(f"🚗 Car at ({x1},{y1}) → Color: {color_name}")

            # رسم المربع واللون
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{color_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

    # عرض الفيديو
    cv2.imshow("YOLO Car Detection", frame)

cap.release()
cv2.destroyAllWindows()
