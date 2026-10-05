"""
Анализ кандидатов: кластеризация, скоринг, решение.

Функции:
    cluster_elements  — Union-Find, пометить "списки"
    score_element     — скор одного элемента
    rank_candidates   — отсортировать по скору
    decide            — tap / ambiguous / not_found
"""

import config
from models import Element

def _similar(a: Element, b: Element) -> bool:
    """Похожи ли элементы по форме и размеру, без цвета."""
    if abs(a.aspect - b.aspect) > config.SIM_ASPECT_TOL:
        return False
    max_area = max(a.area, b.area)
    if max_area > 0:
        if abs(a.area - b.area) / max_area > config.SIM_AREA_REL_TOL:
            return False
    return True

def cluster_elements(elements: list[Element]) -> None:
    """
    Объединить похожие элементы в кластеры через Union-Find.
    Заполняет: cluster_id, cluster_size, is_list.
    """
    n = len(elements)
    if n == 0:
        return
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if _similar(elements[i], elements[j]):
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    for cid, members in enumerate(groups.values()):
        size = len(members)
        is_list = size >= config.MIN_CLUSTER_SIZE
        for i in members:
            elements[i].cluster_id = cid
            elements[i].cluster_size = size
            elements[i].is_list = is_list

def score_element(e: Element, img_w: int, img_h: int,
                  max_area: float) -> float:
    """
    Оценка объекта соотвествию значению "кнопка"

    Признаки:
        - площадь (не должно быть маленьким элементом)
        - вытянутость (кнопки более вытянутые в сторону прямоугольников и овалов)
        - Заливка (насколько много зеленого)
        - позиция Y (типично - ниже)
        - позиция X (типично - правее)
        - контраст с окружением (кнопка всегда заметна)
    """
    score = 0.0

    if max_area > 0:
        score += 0.3 * (e.area / max_area)

    if 2.0 <= e.aspect <= 6.0:
        score += 0.25
    elif e.aspect < 1.5:
        score -= 0.15

    if e.fill > 0.75:
        score += 0.2
    elif e.fill < 0.55:
        score -= 0.15

    y_center = (e.y + e.h / 2) / img_h
    if 0.6 <= y_center <= 0.95:
        score += 0.15

    x_center = (e.x + e.w / 2) / img_w
    if x_center < 0.25:
        score -= 0.15

    if e.green_contrast > 0.4:
        score += 0.1

    return max(0.0, score)


def rank_candidates(candidates: list[Element],
                    img_w: int, img_h: int) -> list[Element]:
    """
    Сортировка по убыванию score 
    """
    if not candidates:
        return []
    max_area = max(e.area for e in candidates)
    for e in candidates:
        e.score = score_element(e, img_w, img_h, max_area)
    candidates.sort(key=lambda e: e.score, reverse=True)
    return candidates

def decide(candidates: list[Element]) -> tuple[str, Element | None]:
    """
    Принятие решения по элементам на скрине
    """
    if not candidates:
        return "not_found", None

    top = candidates[0]
    if top.score < config.MIN_SCORE:
        return "not_found", None

    if len(candidates) >= 2:
        if (top.score - candidates[1].score) < config.AMBIGUITY_GAP:
            return "ambiguous", None

    return "tap", top
