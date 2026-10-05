import cv2
import numpy as np

import config
from models import Element


def save_debug(img: np.ndarray, elements: list[Element], orphans: list[Element],
               raw_mask: np.ndarray, morph_mask: np.ndarray) -> None:
    """
    Сохранить отладочные изображения вместе с цветовыми индикациями:
        debug_edges.png             — Canny edges
        debug_green_mask_raw.png    — RAW зелёная маска
        debug_green_mask_morph.png  — MORPH зелёная маска
        debug_elements.png          — все bbox с метками
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, config.BLUR_KERNEL, 0),
                      config.CANNY_LOW, config.CANNY_HIGH)
    cv2.imwrite("debug_edges.png", edges)

    cv2.imwrite("debug_green_mask_raw.png", raw_mask)
    cv2.imwrite("debug_green_mask_morph.png", morph_mask)

    vis = img.copy()

    for e in elements:
        x, y, w, h = e.bbox
        if e.rejected:
            color = (100, 100, 100)      # серый — отсеян
            label = f"x{e.idx}"
        elif e.is_list:
            color = (180, 180, 180)      # светло-серый — список
            label = f"L{e.idx}"
        elif e.is_green:
            color = (0, 0, 255)          # красный — кандидат
            label = f"G{e.idx} s={e.score:.2f}"
        else:
            color = (255, 100, 0)        # синий — одиночка, не зелёный
            label = f"{e.idx}"
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cv2.putText(vis, label, (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    for e in orphans:
        x, y, w, h = e.bbox
        if e.rejected:
            color = (0, 128, 0)          # тёмно-зелёный — отсеян
            label = f"o{e.idx} x"
        else:
            color = (0, 255, 0)          # ярко-зелёный — кандидат
            label = f"O{e.idx} s={e.score:.2f}"
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cv2.putText(vis, label, (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.imwrite("debug_elements.png", vis)
    print("Сохранение картинок - debug_edges.png, debug_green_mask_raw.png, "
          "debug_green_mask_morph.png, debug_elements.png")