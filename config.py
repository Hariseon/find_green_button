import numpy as np

CANNY_LOW = 50
CANNY_HIGH = 150
BLUR_KERNEL = (5, 5)
CLOSE_KERNEL = (7, 7)  # склейка разорванных границ 
CLOSE_ITER = 2

MIN_AREA_RATIO = 0.0005  # шум
MAX_AREA_RATIO = 0.50  # фон

TOP_MARGIN_RATIO = 0.05  # статус-бар
FULLSCREEN_RATIO = 0.95  # системные кнопки

SIM_ASPECT_TOL = 0.30  # чтобы считать похожими
SIM_AREA_REL_TOL = 0.30  # макс относительная разница площади
MIN_CLUSTER_SIZE = 3  # мин элементов чтобы кластер считался списком

GREEN_HSV_LOW = np.array([35, 40, 40], dtype=np.uint8)
GREEN_HSV_HIGH = np.array([90, 255, 255], dtype=np.uint8)
GREEN_RATIO_THRESHOLD = 0.30  # порог доли, при которой элемент считается зеленым

TEXT_MIN_COMPONENTS = 3  # элементов, чтобы считать словом
TEXT_MAX_LARGEST_RATIO = 0.30  # процент для заливки

RING_PAD_MIN = 6  # минимум пикселей
RING_PAD_DIV = 6  # pad = min(w,h) // 6
GREEN_CONTRAST_MIN = 0.15  # порог определения зеленого фона
SCREEN_GREEN_DOMINATED = 0.50  # порог размера экрана в % для определения что это фон

GREEN_MORPH_KERNEL = (5, 5)
GREEN_MORPH_ITER = 2
ORPHAN_OVERLAP_THRESHOLD = 0.50  # 50%+ пересечения = принадлежит Canny-элементу
ORPHAN_MIN_AREA_RATIO = 0.001  # порог игнорирования

MIN_SCORE = 0.25  # порог для считания кнопкой
AMBIGUITY_GAP = 0.10  # разница для неоднозначности

SCREEN_WAKE_TIMEOUT = 2.0  # включение экрана
UNLOCK_TIMEOUT = 2.0  # снятие локскрина
POLL_INTERVAL = 0.2  # пауза между итерациями