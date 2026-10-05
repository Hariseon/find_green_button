import cv2
import numpy as np

import config
from models import Element

def build_raw_green_mask(img: np.ndarray) -> np.ndarray:
    """
    RAW mask.
    Каждый элемент остается отдельным блобом
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, config.GREEN_HSV_LOW, config.GREEN_HSV_HIGH)


def build_morph_green_mask(img: np.ndarray, raw_mask: np.ndarray | None = None) -> np.ndarray:
    """
    MORPH. Используется в orphan green.
    """
    mask = raw_mask if raw_mask is not None else build_raw_green_mask(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       config.GREEN_MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,
                            iterations=config.GREEN_MORPH_ITER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def is_green_dominated(img: np.ndarray, mask: np.ndarray | None = None,
                       threshold: float = config.SCREEN_GREEN_DOMINATED) -> tuple[bool, float]:
    """
    Доля зелёного на экране по RAW mask.
    """
    if mask is None:
        mask = build_raw_green_mask(img)
    if mask.size == 0:
        return False, 0.0
    ratio = float(np.count_nonzero(mask)) / mask.size
    return ratio > threshold, ratio

def segment_elements(img: np.ndarray) -> list[Element]:
    """
    Найти все визуально выделенные элементы через Canny.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, config.BLUR_KERNEL, 0)
    edges = cv2.Canny(blurred, config.CANNY_LOW, config.CANNY_HIGH)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       config.CLOSE_KERNEL)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel,
                              iterations=config.CLOSE_ITER)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    img_h, img_w = img.shape[:2]
    screen_area = float(img_h * img_w)
    elements: list[Element] = []

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 10 or h < 10:
            continue
        bbox_area = float(w * h)
        area_ratio = bbox_area / screen_area
        if not (config.MIN_AREA_RATIO <= area_ratio <= config.MAX_AREA_RATIO):
            continue
        cnt_area = float(cv2.contourArea(cnt))
        fill = cnt_area / bbox_area if bbox_area > 0 else 0.0
        elements.append(Element(
            idx=i, x=x, y=y, w=w, h=h,
            area=bbox_area, area_ratio=area_ratio,
            aspect=w / h, fill=fill,
        ))

    return elements


def _reject_reason(e: Element, img_w: int, img_h: int) -> str:
    if e.y < img_h * config.TOP_MARGIN_RATIO:
        return "top_margin"
    if (e.w > img_w * config.FULLSCREEN_RATIO
            or e.h > img_h * config.FULLSCREEN_RATIO):
        return "fullscreen"
    return ""


def apply_hard_filters(elements: list[Element], img_w: int, img_h: int) -> None:
    """Отсеять статус-бар и элементы во весь экран."""
    for e in elements:
        reason = _reject_reason(e, img_w, img_h)
        if reason and not e.rejected:
            e.rejected = reason

def _green_structure(roi_mask: np.ndarray) -> tuple[int, float]:
    """
    Связные компоненты зелёной маски внутри bbox. 
    Возвращает число значимых и площадь самого крупного блоба
    """
    if roi_mask.size == 0:
        return 0, 0.0

    n, _, stats, _ = cv2.connectedComponentsWithStats(
        roi_mask, connectivity=8
    )
    if n <= 1:
        return 0, 0.0

    areas = stats[1:, cv2.CC_STAT_AREA]
    bbox_area = float(roi_mask.shape[0] * roi_mask.shape[1])
    if bbox_area == 0:
        return 0, 0.0

    largest_ratio = float(areas.max()) / bbox_area
    significant = int((areas > bbox_area * 0.02).sum())
    return significant, largest_ratio


def _ring_green_ratio(e: Element, mask: np.ndarray,
                      img_h: int, img_w: int) -> float:
    """
    Доля зелёного около границы объекта
    """
    pad = max(config.RING_PAD_MIN,
              min(e.w, e.h) // config.RING_PAD_DIV)

    x0 = max(0, e.x - pad)
    y0 = max(0, e.y - pad)
    x1 = min(img_w, e.x + e.w + pad)
    y1 = min(img_h, e.y + e.h + pad)

    ring = mask[y0:y1, x0:x1].copy()
    if ring.size == 0:
        return 0.0

    ix0 = e.x - x0
    iy0 = e.y - y0
    ring[iy0:iy0 + e.h, ix0:ix0 + e.w] = 0

    if ring.size == 0:
        return 0.0
    return float(np.count_nonzero(ring)) / ring.size


def check_green(img: np.ndarray, elements: list[Element],
                mask: np.ndarray | None = None) -> None:
    """
    Проверка цвета - на RAW mask.
    Для orphans is_green по определению True.
    """
    if mask is None:
        mask = build_raw_green_mask(img)
    img_h, img_w = mask.shape

    for e in elements:
        roi = mask[e.y:e.y + e.h, e.x:e.x + e.w]
        if roi.size == 0:
            continue
        e.green_ratio = float(np.count_nonzero(roi)) / roi.size

        if e.is_orphan:
            e.is_green = True
        else:
            e.is_green = e.green_ratio >= config.GREEN_RATIO_THRESHOLD

        if e.is_green:
            e.green_n_components, e.green_largest_ratio = _green_structure(roi)
            e.green_ring_ratio = _ring_green_ratio(e, mask, img_h, img_w)
            e.green_contrast = e.green_ratio - e.green_ring_ratio

            # Фильтр "зелёный текст"
            if (e.green_n_components >= config.TEXT_MIN_COMPONENTS
                    and e.green_largest_ratio < config.TEXT_MAX_LARGEST_RATIO):
                if not e.rejected:
                    e.rejected = "green_text"

            # Фильтр "зелёное — это фон"
            if e.green_contrast < config.GREEN_CONTRAST_MIN:
                if not e.rejected:
                    e.rejected = "green_is_background"