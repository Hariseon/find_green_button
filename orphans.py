"""
Ищем зелёные регионы из HSV-маски, которые не покрыты ни одним Canny-элементом. 
Это объекты - потенциально светло-зелёная кнопка на светлом фоне.
"""

import cv2
import numpy as np

import config
from models import Element, GreenRegion


def find_green_regions(mask: np.ndarray) -> list[GreenRegion]:
    """
    Найти контуры в зелёной маске (MORPH).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    regions: list[GreenRegion] = []
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 5 or h < 5:
            continue
        area = float(cv2.contourArea(cnt))
        if area <= 0:
            continue
        regions.append(GreenRegion(idx=i, x=x, y=y, w=w, h=h,
                                   contour_area=area))
    return regions


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """
    Доля пересечения bbox_a внутри bbox_b.
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / (aw * ah) if aw * ah > 0 else 0.0


def classify_green_regions(green_regions: list[GreenRegion],
                           elements: list[Element]) -> list[GreenRegion]:
    """
    Оставить элементы, что не попали в Canny
    """
    orphans: list[GreenRegion] = []
    for gr in green_regions:
        best_ratio = 0.0
        best_idx = -1
        for e in elements:
            ratio = _bbox_overlap(gr.bbox, e.bbox)
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = e.idx
        gr.overlap = best_ratio
        if best_ratio >= config.ORPHAN_OVERLAP_THRESHOLD:
            gr.owner_idx = best_idx
        else:
            gr.owner_idx = -1
            orphans.append(gr)
    return orphans


def orphans_to_elements(orphans: list[GreenRegion], img_w: int, img_h: int,
                        idx_offset: int = 1000) -> list[Element]:
    """
    Конвертировать найденное в Element.
    """
    screen_area = float(img_w * img_h)
    result: list[Element] = []
    for i, gr in enumerate(orphans):
        bbox_area = float(gr.w * gr.h)
        if bbox_area <= 0:
            continue
        result.append(Element(
            idx=idx_offset + i,
            x=gr.x, y=gr.y, w=gr.w, h=gr.h,
            area=bbox_area,
            area_ratio=bbox_area / screen_area,
            aspect=gr.w / gr.h,
            fill=gr.contour_area / bbox_area,
            is_orphan=True,
        ))
    return result