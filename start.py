import os
import sys
import subprocess
from pathlib import Path
import sys

START_INFO = "[INFO]"
START_ERROR = "[ERROR]"

project_dir = Path(__file__).resolve().parent
venv_dir = project_dir / "myenv"
python_exec = venv_dir / "bin" / "python3"
requirements_file = project_dir / "requirements.txt"

def create_virtualenv():
    print(f"{START_INFO} Створюємо віртуальне середовище з підтримкою системних пакетів...")
    subprocess.run(["python3", "-m", "venv", "--system-site-packages", str(venv_dir)], check=True)

def install_requirements():
    if not requirements_file.exists():
        print(f"{START_ERROR} Файл requirements.txt не знайдено!")
        return
    print(f"{START_INFO} Встановлюю залежності...")
    subprocess.run([str(python_exec), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([str(python_exec), "-m", "pip", "install", "-r", str(requirements_file)], check=True)

def is_libcamera_available():
    result = subprocess.run(["which", "libcamera-hello"], capture_output=True)
    return result.returncode == 0


def clone_sort_if_needed():
    sort_dir = project_dir / "app" / "sort"
    if not sort_dir.exists():
        print(f"{START_INFO} Клоную SORT у {sort_dir}...")
        subprocess.run(["git", "clone", "https://github.com/abewley/sort.git", str(sort_dir)], check=True)
    else:
        print(f"{START_INFO} SORT вже існує в {sort_dir}")

# === ГОЛОВНА ЛОГІКА ===
if not venv_dir.exists():
    create_virtualenv()

if not python_exec.exists():
    print(f"{START_ERROR} Віртуальне середовище створено некоректно.")
    sys.exit(1)

install_requirements()
clone_sort_if_needed()


def is_picamera2_available():
    result = subprocess.run([str(python_exec), "-c", "from picamera2 import Picamera2"], capture_output=True)
    return result.returncode == 0

def run_app():
    print(f"\n{START_INFO} Запускаємо додаток...")
    subprocess.run([str(python_exec), "run.py"])

# === ГОЛОВНА ЛОГІКА ===
if not venv_dir.exists():
    create_virtualenv()

if not python_exec.exists():
    print(f"{START_ERROR} Віртуальне середовище створено некоректно.")
    sys.exit(1)

install_requirements()

if not is_libcamera_available():
    print(f"\n{START_ERROR} Не встановлено libcamera (не Python-пакет).")
    print("➡ Щоб встановити його, виконай:")
    print("   sudo apt install libcamera-dev libcamera-apps")
    sys.exit(1)

if not is_picamera2_available():
    print(f"\n{START_ERROR} Не встановлено підтримку libcamera через Python (picamera2).")
    print("➡ Щоб встановити, виконай:")
    print("   sudo apt install -y python3-picamera2")
    sys.exit(1)

run_app()

