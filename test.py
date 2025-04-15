import os

# Вказуємо шляхи до файлів
cfg_path = 'yolov3.cfg'
weights_path = 'yolov3.weights'

# Перевіряємо наявність файлів
if os.path.exists(cfg_path):
    print(f"Файл {cfg_path} знайдений.")
else:
    print(f"Файл {cfg_path} не знайдений.")

if os.path.exists(weights_path):
    print(f"Файл {weights_path} знайдений.")
else:
    print(f"Файл {weights_path} не знайдений.")

