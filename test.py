import time
from adafruit_servokit import ServoKit

# Ініціалізація сервоконтролера (16 каналів)
kit = ServoKit(channels=16)

# Початкові значення (середина)
servo_pan = 90   # X (вліво/вправо)
servo_tilt = 90  # Y (вгору/вниз)

# Встановлення початкових позицій
kit.servo[0].angle = servo_pan
kit.servo[1].angle = servo_tilt

time.sleep(1)

def move_left():
    global servo_pan
    servo_pan -= 10  # змінено на зменшення
    servo_pan = max(0, servo_pan)
    kit.servo[0].angle = servo_pan
    print(f"← Left to {servo_pan}°")

def move_right():
    global servo_pan
    servo_pan += 10  # змінено на збільшення
    servo_pan = min(180, servo_pan)
    kit.servo[0].angle = servo_pan
    print(f"→ Right to {servo_pan}°")

def move_up():
    global servo_tilt
    servo_tilt -= 10  # зменшення, якщо "менше" — це вгору (під камеру)
    servo_tilt = max(0, servo_tilt)
    kit.servo[1].angle = servo_tilt
    print(f"↑ Up to {servo_tilt}°")

def move_down():
    global servo_tilt
    servo_tilt += 10  # збільшення — вниз
    servo_tilt = min(180, servo_tilt)
    kit.servo[1].angle = servo_tilt
    print(f"↓ Down to {servo_tilt}°")

# --- Головний цикл ---
print("Перевірка напрямків сервоприводів.")
print("Натисни клавішу:")
print("[w] Вгору   [s] Вниз")
print("[a] Вліво   [d] Вправо")
print("[q] Вийти")

try:
    while True:
        key = input("→ ")
        if key == 'w':
            move_up()
        elif key == 's':
            move_down()
        elif key == 'a':
            move_left()
        elif key == 'd':
            move_right()
        elif key == 'q':
            print("Вихід...")
            break
        else:
            print("Невідома команда.")
        time.sleep(0.3)

except KeyboardInterrupt:
    print("\nЗупинено вручну.")

