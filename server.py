from flask import Flask, Response, render_template, request, jsonify
from picamera2 import Picamera2
import time
import cv2
from adafruit_servokit import ServoKit
import torch
import uuid
import numpy as np
import sys
sys.path.append('/home/pi/Desktop/scripts/app/sort')  # Додаємо шлях до репозиторію SORT
from sort import Sort  # Імпортуємо SORT з локального репозиторію

app = Flask(__name__)

# Ініціалізація камери
picam2 = Picamera2()

# Ініціалізація панелі керування для сервоприводів
kit = ServoKit(channels=16)

# Запуск камери
picam2.start()

# Поточні кути сервоприводів
servo_pan = 90
servo_tilt = 90

# Крок повороту
STEP = 5

# Межі кутів
MIN_ANGLE = 0
MAX_ANGLE = 180

# Завантаження моделі YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Змінна для зберігання об'єктів
detected_objects = {}

# Ініціалізація трекера SORT
tracker = Sort()

# Змінна для стеження
tracking_object_id = None


def generate():
    global detected_objects, tracking_object_id
    while True:
        frame = picam2.capture_array()
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        # Детекція
        results = model(frame)
        boxes = []
        for *box, conf, cls in results.xyxy[0]:
            if int(cls) == 0:  # person
                boxes.append([int(box[0]), int(box[1]), int(box[2]), int(box[3]), conf.item()])

        # Якщо є детекції, оновлюємо трекер
        if boxes:
            tracked_objects = tracker.update(np.array(boxes))

            current_objects = {}
            for obj in tracked_objects:
                x1, y1, x2, y2, track_id = obj
                track_id = int(track_id)
                object_name = f"Person ({track_id})"

                if track_id not in detected_objects:
                    detected_objects[track_id] = {
                        'name': object_name,
                        'first_seen': time.time(),
                        'box': [x1, y1, x2, y2],
                        'last_seen': time.time()
                    }

                detected_objects[track_id]['box'] = [x1, y1, x2, y2]
                detected_objects[track_id]['last_seen'] = time.time()
                current_objects[track_id] = detected_objects[track_id]

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            detected_objects = current_objects

            # 🎯 Автоматичне стеження
            if tracking_object_id is not None:
                if tracking_object_id in detected_objects:
                    track_object_with_servos(tracking_object_id, frame)
                else:
                    print(f"[WARN] Об'єкт з ID {tracking_object_id} більше не знайдено. Скасовуємо стеження.")
                    tracking_object_id = None

        else:
            # Якщо взагалі немає детекцій — теж скидаємо
            if tracking_object_id is not None:
                print(f"[INFO] Об'єктів немає в кадрі. Стеження зупинено.")
                tracking_object_id = None

        # Генерація кадру MJPEG
        _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        time.sleep(0.01)


@app.route('/video')
def video():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/objects')
def objects():
    return jsonify([{'name': data['name']} for name, data in detected_objects.items()])


@app.route('/move', methods=['POST'])
def move_servo():
    global servo_pan, servo_tilt

    direction = request.form.get('direction')
    if direction == 'left' and servo_pan < MAX_ANGLE:
        servo_pan += STEP
    elif direction == 'right' and servo_pan > MIN_ANGLE:
        servo_pan -= STEP
    elif direction == 'up' and servo_tilt < MAX_ANGLE:
        servo_tilt += STEP
    elif direction == 'down' and servo_tilt > MIN_ANGLE:
        servo_tilt -= STEP

    kit.servo[0].angle = servo_pan
    kit.servo[1].angle = servo_tilt

    print(f"Панорамування: {servo_pan}, Нахил: {servo_tilt}")
    return 'OK'


@app.route('/track', methods=['POST'])
def track_object():
    global tracking_object_id

    object_index = int(request.form.get('object_id'))  # index зі списку
    keys = list(detected_objects.keys())
    if 0 <= object_index < len(keys):
        tracking_object_id = keys[object_index]
        print(f"[INFO] Стеження активоване для ID: {tracking_object_id}")
        return 'Стеження активовано!'
    return 'Невірний індекс об\'єкта', 400


@app.route('/stop_tracking', methods=['POST'])
def stop_tracking():
    global tracking_object_id
    tracking_object_id = None
    print("[INFO] Стеження зупинено.")
    return 'Стеження зупинено!'


def track_object_with_servos(object_id, frame):
    """Пропорційне наведення серво на центр об'єкта з урахуванням перевернутої камери"""
    global servo_pan, servo_tilt, kit

    if object_id in detected_objects:
        obj = detected_objects[object_id]
        x1, y1, x2, y2 = obj['box']
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        frame_height, frame_width, _ = frame.shape
        delta_x = center_x - frame_width // 2
        delta_y = center_y - frame_height // 2

        # ⚠ Змінено: обернене управління, бо зображення перевернуто
        sensitivity = 0.05
        correction_x = delta_x * sensitivity      # без мінуса
        correction_y = -delta_y * sensitivity     # тут інвертуємо

        servo_pan += int(correction_x)
        servo_tilt += int(correction_y)

        servo_pan = max(MIN_ANGLE, min(servo_pan, MAX_ANGLE))
        servo_tilt = max(MIN_ANGLE, min(servo_tilt, MAX_ANGLE))

        kit.servo[0].angle = servo_pan
        kit.servo[1].angle = servo_tilt

        print(f"🎯 Пан: {servo_pan}, Нахил: {servo_tilt} | ΔX: {delta_x}, ΔY: {delta_y}")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)

