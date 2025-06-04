from flask import Flask, Response, render_template, request, jsonify
from picamera2 import Picamera2
import time
import cv2
from adafruit_servokit import ServoKit
import torch
import uuid
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'app/sort'))
from sort import Sort
from ultralytics import YOLO  

import os
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), 'app/templates'),
    static_folder=os.path.join(os.path.dirname(__file__), 'app/static')
)


picam2 = Picamera2()

kit = ServoKit(channels=16)

picam2.start()

# Поточні кути сервоприводів
servo_pan = 90  
servo_tilt = 90  

# Крок повороту
STEP = 5

# Межі кутів
MIN_ANGLE = 0
MAX_ANGLE = 180

model_path = os.path.join('models', 'yolov8n.pt')

# Перевірка, чи файл існує.
if not os.path.exists(model_path):
    print('[INFO] Завантаження YOLOv8n.pt у папку models...')
    os.makedirs('models', exist_ok=True)
    import urllib.request
    yolov8_url = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt'
    urllib.request.urlretrieve(yolov8_url, model_path)
    print('[INFO] Завантаження завершено.')


model = YOLO('models/yolov8n.pt')  

detected_objects = {}

tracker = Sort()

tracking_object_id = None

prev_error_x = 0
prev_error_y = 0
integral_x = 0
integral_y = 0
last_move_time = 0




def draw_crosshair(frame):
    height, width, _ = frame.shape
    center_x = width // 2
    center_y = height // 2
    color = (0, 255, 0)  
    thickness = 1
    length = 10  
    
    cv2.line(frame, (center_x - length, center_y), (center_x + length, center_y), color, thickness)
    cv2.line(frame, (center_x, center_y - length), (center_x, center_y + length), color, thickness)
    
    return frame

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
            if cls_id == 0:  
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
    object_index = int(request.form.get('object_id'))  
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
    
    x1, y1, x2, y2 = map(int, box)
    frame_height, frame_width = frame.shape[:2]
    center_screen_x = frame_width // 2
    center_screen_y = frame_height // 2
    center_obj_x = x1 + (x2 - x1) // 2
    center_obj_y = y1 + (y2 - y1) // 2

    # Візуалізація
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.circle(frame, (center_screen_x, center_screen_y), 5, (0, 0, 255), -1)
    cv2.circle(frame, (center_obj_x, center_obj_y), 5, (255, 0, 0), -1)
    cv2.putText(frame, f"X: {center_obj_x}, Y: {center_obj_y}", 
                (20, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(frame, (center_obj_x, 0), (center_obj_x, frame_height), (0, 255, 0), 2)
    cv2.line(frame, (0, center_obj_y), (frame_width, center_obj_y), (0, 255, 0), 2)

    # PID параметри
    kp_x, ki_x, kd_x = 0.04, 0.0002, 0.02
    kp_y, ki_y, kd_y = 0.04, 0.0002, 0.02
    max_step_x = 3.0
    max_step_y = 3.0
    max_integral = 80
    dead_zone_x = 8
    dead_zone_y = 12

    # Похибка по X
    error_x = center_screen_x - center_obj_x
    integral_x += error_x
    integral_x = max(-max_integral, min(integral_x, max_integral))
    derivative_x = error_x - prev_error_x
    pan_adjustment = kp_x * error_x + ki_x * integral_x + kd_x * derivative_x
    pan_adjustment = max(-max_step_x, min(pan_adjustment, max_step_x))
    prev_error_x = error_x

    if abs(error_x) > dead_zone_x:
        servo_pan = max(MIN_ANGLE, min(MAX_ANGLE, servo_pan + pan_adjustment))
        kit.servo[1].angle = servo_pan

    # Похибка по Y
    error_y = center_obj_y - center_screen_y
    integral_y += error_y
    integral_y = max(-max_integral, min(integral_y, max_integral))
    derivative_y = error_y - prev_error_y
    tilt_adjustment = kp_y * error_y + ki_y * integral_y + kd_y * derivative_y
    tilt_adjustment = max(-max_step_y, min(tilt_adjustment, max_step_y))
    prev_error_y = error_y

    if abs(error_y) > dead_zone_y:
        servo_tilt = max(MIN_ANGLE, min(MAX_ANGLE, servo_tilt + tilt_adjustment))
        kit.servo[0].angle = servo_tilt







if __name__ == '__main__':
    try:
        kit.servo[0].angle = servo_tilt
        kit.servo[1].angle = servo_pan
        print(f"[INFO] Початкові положення: PAN={servo_pan}, TILT={servo_tilt}")
        
        print("[INFO] Запуск сервера...")
        app.run(host='0.0.0.0', port=5000, threaded=True)
    except Exception as e:
        print(f"[ERROR] Помилка при запуску сервера: {e}")

