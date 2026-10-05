import subprocess
import sys
import time
import cv2
import numpy as np

import config

def adb(*args: str, timeout: float = 15.0) -> bytes:
    """Запустить adb. Возвращает stdout как bytes. При ошибке — sys.exit"""
    cmd = ["adb", *args]
    try:
        res = subprocess.run(cmd, capture_output=True,
                             timeout=timeout, check=False)
    except FileNotFoundError:
        sys.exit("adb не найден в PATH.")
    except subprocess.TimeoutExpired:
        sys.exit(f"adb {' '.join(args)}: таймаут {timeout}s")
    if res.returncode != 0:
        err = res.stderr.decode("utf-8", "replace").strip()
        sys.exit(f"adb {' '.join(args)} упал: {err}")
    return res.stdout


def adb_str(*args: str, timeout: float = 15.0) -> str:
    return adb(*args, timeout=timeout).decode("utf-8", "replace")


def check_device() -> None:
    out = adb_str("devices")
    lines = [l for l in out.splitlines()[1:] if l.strip()]
    if not lines:
        sys.exit("Нет подключённых устройств.")
    states = [l.split("\t")[1] for l in lines if "\t" in l]
    if "device" not in states:
        sys.exit(f"Устройство не готово: {states}.")


def launch_app(package: str) -> None:
    adb("shell", "monkey", "-p", package,
        "-c", "android.intent.category.LAUNCHER", "1")


def screen_size() -> tuple[int, int]:
    out = adb_str("shell", "wm", "size")
    for line in out.splitlines():
        if "size:" in line:
            w, h = line.split(":")[-1].strip().split("x")
            return int(w), int(h)
    return 1080, 2340

def _screen_is_on() -> bool:
    """Проверка активности экрана"""
    out = adb_str("shell", "dumpsys", "power", timeout=5.0)
    return "mWakefulness=Awake" in out


def is_locked() -> bool:
    """Показан ли локскрин"""
    out = adb_str("shell", "dumpsys", "window", timeout=5.0)
    return ("mDreamingLockscreen=true" in out
            or "mShowingLockscreen=true" in out)


def _wait_for(predicate, timeout: float, poll: float) -> bool:
    """
    Ждать, пока predicate() вернёт True.

    Возвращает True при успехе,
    False при таймауте. Каждая итерация — adb-вызов
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll)
    return False

def wake_screen() -> None:
    """Разбудить экран, если он погашен"""
    if _screen_is_on():
        return
    adb("shell", "input", "keyevent", "KEYCODE_WAKEUP")
    _wait_for(_screen_is_on,
              timeout=config.SCREEN_WAKE_TIMEOUT,
              poll=config.POLL_INTERVAL)


def try_unlock() -> bool:
    """Попытка снять локскрин"""
    if not is_locked():
        return True

    # Попытка 1: dismiss-keyguard
    try:
        adb("shell", "wm", "dismiss-keyguard")
    except SystemExit:
        pass  # на части прошивок команды нет

    if _wait_for(lambda: not is_locked(),
                 timeout=config.UNLOCK_TIMEOUT,
                 poll=config.POLL_INTERVAL):
        return True

    # Попытка 2: свайп вверх (swipe-to-unlock)
    dev_w, dev_h = screen_size()
    x = dev_w // 2
    adb("shell", "input", "swipe",
        str(x), str(int(dev_h * 0.8)),
        str(x), str(int(dev_h * 0.3)), "300")

    return _wait_for(lambda: not is_locked(),
                     timeout=config.UNLOCK_TIMEOUT,
                     poll=config.POLL_INTERVAL)

def screencap_to_numpy() -> np.ndarray:
    """Скриншот"""
    raw = adb("exec-out", "screencap", "-p", timeout=20.0)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit("Не удалось декодировать screencap.")
    return img


def tap(x: int, y: int) -> None:
    """Нажатие"""
    adb("shell", "input", "tap", str(x), str(y))