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
from ultralytics import YOLO  # Імпортуємо YOLOv8

app = Flask(__name__)

# Ініціалізація камери
picam2 = Picamera2()

# Ініціалізація панелі керування для сервоприводів
kit = ServoKit(channels=16)

# Запуск камери
picam2.start()

# Поточні кути сервоприводів
servo_pan = 90  # Змінено на 90 для точного центрування
servo_tilt = 90  # Змінено на 90 для точного центрування

# Крок повороту
STEP = 5

# Межі кутів
MIN_ANGLE = 0
MAX_ANGLE = 180

# Завантаження моделі YOLOv8
model = YOLO('yolov8n.pt')  # Використовуємо найшвидшу модель 'yolov8n.pt'

# Змінна для зберігання об'єктів
detected_objects = {}

# Ініціалізація трекера SORT
tracker = Sort()

# Змінна для стеження
tracking_object_id = None

# Змінні для PID контролера
prev_error_x = 0
prev_error_y = 0
integral_x = 0
integral_y = 0
last_move_time = 0

# Add this function definition at the top of your script
def draw_crosshair(frame):
    height, width, _ = frame.shape
    center_x = width // 2
    center_y = height // 2
    color = (0, 255, 0)  # Green color
    thickness = 1
    length = 10  # Length of the crosshair lines
    
    # Draw the horizontal line
    cv2.line(frame, (center_x - length, center_y), (center_x + length, center_y), color, thickness)
    # Draw the vertical line
    cv2.line(frame, (center_x, center_y - length), (center_x, center_y + length), color, thickness)
    
    return frame

# Update your 'generate' function to use 'draw_crosshair' after processing the frame.
def generate():
    global detected_objects, tracking_object_id
    
    while True:

        frame = picam2.capture_array()
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        results = model.predict(frame, verbose=False, stream=False)[0]
        boxes = []
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == 0:  # Class 0 is for people
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append([x1, y1, x2, y2, conf])
        
        if boxes:
            tracked_objects = tracker.update(np.array(boxes))
            detected_objects = {}
            
            for obj in tracked_objects:
                x1, y1, x2, y2, track_id = map(int, obj)
                detected_objects[track_id] = {
                    'box': [x1, y1, x2, y2],
                    'name': f'Person ({track_id})'
                }
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            if tracking_object_id is not None:
                for obj in tracked_objects:
                    x1, y1, x2, y2, track_id = obj
                    if int(track_id) == tracking_object_id:
                        track_object_with_servos((x1, y1, x2, y2), frame)
                        break
        else:
            if tracking_object_id is not None:
                print("[INFO] Об'єктів немає в кадрі. Стеження зупинено.")
                tracking_object_id = None
        
        # Call the draw_crosshair function to draw the crosshair on the frame
        frame = draw_crosshair(frame)
        
        _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

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

    if direction == 'left' and servo_pan > MIN_ANGLE:
        servo_pan -= STEP
    elif direction == 'right' and servo_pan < MAX_ANGLE:
        servo_pan += STEP
    elif direction == 'up' and servo_tilt > MIN_ANGLE:
        servo_tilt -= STEP
    elif direction == 'down' and servo_tilt < MAX_ANGLE:
        servo_tilt += STEP

    kit.servo[0].angle = servo_tilt
    kit.servo[1].angle = servo_pan
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



def track_object_with_servos(box, frame):
    global servo_pan, servo_tilt, prev_error_x, integral_x, prev_error_y, integral_y
    
    # Отримуємо координати прямокутника об'єкта
    x1, y1, x2, y2 = map(int, box)
    
    # Розміри кадру
    frame_height, frame_width = frame.shape[:2]
    
    # Центр кадру
    center_screen_x = frame_width // 2
    center_screen_y = frame_height // 2
    
    # Центр об'єкта
    center_obj_x = x1 + (x2 - x1) // 2
    center_obj_y = y1 + (y2 - y1) // 2
    
    # Візуалізація
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.circle(frame, (center_screen_x, center_screen_y), 5, (0, 0, 255), -1)
    cv2.circle(frame, (center_obj_x, center_obj_y), 5, (255, 0, 0), -1)
    
    # Виведення координат X та Y
    cv2.putText(frame, f"X: {center_obj_x}, Y: {center_obj_y}", 
                (20, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                (0, 255, 0), 2, cv2.LINE_AA)

    # Малюємо координатну площину (ось X і Y)
    cv2.line(frame, (center_obj_x, 0), (center_obj_x, frame_height), (0, 255, 0), 2)
    cv2.line(frame, (0, center_obj_y), (frame_width, center_obj_y), (0, 255, 0), 2)
    
    # Обчислення похибки по осі X
    error_x = center_screen_x - center_obj_x  # Перевертаємо похибку для правильного напрямку
    
    # Параметри PID-регулятора для стабілізації по X
    kp_x = 0.01  # Збільшили пропорційний коефіцієнт для X (для більш швидкої реакції)
    ki_x = 0.0001  # Інтегральний коефіцієнт для X
    kd_x = 0.01  # Диференційний коефіцієнт для X
    
    # Оновлення інтегральної складової для X
    integral_x += error_x
    max_integral_x = 50
    integral_x = max(-max_integral_x, min(integral_x, max_integral_x))
    
    # Обчислення диференційної складової для X
    derivative_x = error_x - prev_error_x
    
    # Обчислення коригуючого сигналу для сервоприводу по X
    pan_adjustment = kp_x * error_x + ki_x * integral_x + kd_x * derivative_x
    
    # Обмеження максимального кроку руху по X
    max_step_x = 0.5  # Збільшили максимальний крок для більш швидкого руху
    pan_adjustment = max(-max_step_x, min(pan_adjustment, max_step_x))
    
    # Збереження поточної похибки для наступної ітерації
    prev_error_x = error_x
    
    # Визначення мертвої зони для руху по осі X
    dead_zone_x = 10  # Зменшили мертву зону для більш чутливого реагування
    if abs(error_x) > dead_zone_x:
        # Оновлення положення сервоприводу по осі X
        servo_pan = max(MIN_ANGLE, min(MAX_ANGLE, servo_pan + pan_adjustment))
        kit.servo[1].angle = servo_pan
    
    # Обчислення похибки по осі Y
    error_y = center_obj_y - center_screen_y  # Змінили знак на правильний напрямок
    
    # Параметри PID-регулятора для стабілізації по Y
    kp_y = 0.01  # Збільшили пропорційний коефіцієнт для Y (для більш швидкої реакції)
    ki_y = 0.0001  # Інтегральний коефіцієнт для Y
    kd_y = 0.01  # Диференційний коефіцієнт для Y
    
    # Оновлення інтегральної складової для Y
    integral_y += error_y
    max_integral_y = 50
    integral_y = max(-max_integral_y, min(integral_y, max_integral_y))
    
    # Обчислення диференційної складової для Y
    derivative_y = error_y - prev_error_y
    
    # Обчислення коригуючого сигналу для сервоприводу по Y
    tilt_adjustment = kp_y * error_y + ki_y * integral_y + kd_y * derivative_y
    
    # Оновлення попередньої похибки для Y
    prev_error_y = error_y
    
    # Визначення мертвої зони для руху по осі Y
    dead_zone_y = 20  # Зменшили мертву зону для більш чутливого реагування
    if abs(error_y) > dead_zone_y:
        servo_tilt = max(MIN_ANGLE, min(MAX_ANGLE, servo_tilt + tilt_adjustment))
        kit.servo[0].angle = servo_tilt






if __name__ == '__main__':
    try:
        # Встановлюємо початкові положення сервоприводів
        kit.servo[0].angle = servo_tilt
        kit.servo[1].angle = servo_pan
        print(f"[INFO] Початкові положення: PAN={servo_pan}, TILT={servo_tilt}")
        
        print("[INFO] Запуск сервера...")
        app.run(host='0.0.0.0', port=5000, threaded=True)
    except Exception as e:
        print(f"[ERROR] Помилка при запуску сервера: {e}")

