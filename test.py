import argparse
import sys
import time

from adb import check_device, launch_app, is_locked, try_unlock, screencap_to_numpy, tap, wake_screen
from cv_vision import (build_raw_green_mask, build_morph_green_mask, is_green_dominated, segment_elements,
    apply_hard_filters, check_green)
from orphans import find_green_regions, classify_green_regions, orphans_to_elements 
from analysis import cluster_elements, rank_candidates, decide
from debug import save_debug
import config

def process_frame(img):
    """
    Пайплайн кадра.

    Возвращает dict:
        decision       — "tap" / "ambiguous" / "not_found" / "green_dominated"
        target         — Element или None
        elements       — Canny-элементы
        orphans        — orphan-элементы
        candidates     — отсортированные кандидаты
        raw_mask       — RAW зелёная маска
        morph_mask     — MORPH зелёная маска
        green_ratio    — доля зелёного на экране
    """
    img_h, img_w = img.shape[:2]

    raw_mask = build_raw_green_mask(img)
    morph_mask = build_morph_green_mask(img, raw_mask=raw_mask)

    is_dom, green_ratio = is_green_dominated(img, mask=raw_mask)
    if is_dom:
        return {
            "decision": "green_dominated",
            "target": None,
            "elements": [], "orphans": [], "candidates": [],
            "raw_mask": raw_mask, "morph_mask": morph_mask,
            "green_ratio": green_ratio,
        }

    elements = segment_elements(img)
    apply_hard_filters(elements, img_w, img_h)
    check_green(img, elements, mask=raw_mask)
    cluster_elements(elements)

    green_regions = find_green_regions(morph_mask)
    orphans_raw = classify_green_regions(green_regions, elements)
    orphans = orphans_to_elements(orphans_raw, img_w, img_h)
    orphans = [e for e in orphans
               if e.area_ratio >= config.ORPHAN_MIN_AREA_RATIO]
    apply_hard_filters(orphans, img_w, img_h)
    check_green(img, orphans, mask=raw_mask)

    # Кандидаты
    canny_candidates = [e for e in elements
                        if e.is_green and not e.is_list and not e.rejected]
    orphan_candidates = [e for e in orphans if not e.rejected]
    candidates = canny_candidates + orphan_candidates
    candidates = rank_candidates(candidates, img_w, img_h)

    decision, target = decide(candidates)

    return {
        "decision": decision,
        "target": target,
        "elements": elements,
        "orphans": orphans,
        "candidates": candidates,
        "raw_mask": raw_mask,
        "morph_mask": morph_mask,
        "green_ratio": green_ratio,
    }

def finalize(img, result, args) -> int:
    """
    Финальный вывод результата и действие.
    """
    decision = result["decision"]
    target = result["target"]

    if args.debug:
        save_debug(img, result["elements"], result["orphans"],
                   result["raw_mask"], result["morph_mask"])

    if decision == "green_dominated":
        ratio = result["green_ratio"]
        print(f"Экран залит зелёным ({ratio * 100:.0f}%).")
        print("Зелёная кнопка не найдена.")
        return 1

    if decision == "not_found":
        print("Зелёная кнопка не найдена.")
        return 1

    if decision == "ambiguous":
        print("Неоднозначно: несколько кандидатов с близким скором.")
        return 3

    assert target is not None
    cx, cy = target.center
    print(f"Кнопка найдена: bbox={target.bbox} "
          f"score={target.score:.3f}")
    tap(cx, cy)
    return 0

def main() -> int:
    p = argparse.ArgumentParser(
        description="Найти и нажать зелёную кнопку в Android-приложении."
    )
    p.add_argument("package",
                   help="package name приложения, напр. com.example.app")
    p.add_argument("--timeout", type=float, default=10.0,
                   help="общий таймаут поиска, сек (по умолчанию 10)")
    p.add_argument("--debug", action="store_true",
                   help="сохранить скриншоты, маски, bbox")
    args = p.parse_args()

    check_device()

    print(f"Запуск {args.package}...")
    launch_app(args.package)

    wake_screen()

    if is_locked():
        if not try_unlock():
            print("Не удалось снять локскрин.")
            return 3

    deadline = time.monotonic() + args.timeout
    last_img = None
    last_result = None

    while time.monotonic() < deadline:
        img = screencap_to_numpy()
        result = process_frame(img)

        if result["decision"] == "tap":
            return finalize(img, result, args)

        last_img = img
        last_result = result

        if time.monotonic() + config.POLL_INTERVAL < deadline:
            time.sleep(config.POLL_INTERVAL)

    if last_result is not None:
        return finalize(last_img, last_result, args)
    return 1


if __name__ == "__main__":
    sys.exit(main())