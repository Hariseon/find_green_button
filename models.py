from dataclasses import dataclass


@dataclass
class Element:
    """
        1. segment_elements: idx, bbox, area, aspect, fill
        2. apply_hard_filters: rejected
        3. check_green: green_ratio, green_*, is_green, rejected
        4. cluster_elements: cluster_id, cluster_size, is_list
        5. rank_candidates: score
    """
    idx: int
    x: int
    y: int
    w: int
    h: int
    area: float
    area_ratio: float
    aspect: float
    fill: float = 0.0

    # Зелёный цвет
    green_ratio: float = 0.0
    green_ring_ratio: float = 0.0
    green_contrast: float = 0.0
    green_n_components: int = 0
    green_largest_ratio: float = 0.0
    is_green: bool = False

    # Orphan
    is_orphan: bool = False

    # Кластеризация
    cluster_id: int = -1
    cluster_size: int = 0
    is_list: bool = False

    # Решение
    score: float = 0.0
    rejected: str = ""

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2


@dataclass
class GreenRegion:
    """
    Зелёный регион из HSV-маски (MORPH)
    """
    idx: int
    x: int
    y: int
    w: int
    h: int
    contour_area: float
    owner_idx: int = -1  # idx Canny-элемента, если регион ему принадлежит
    overlap: float = 0.0  # доля пересечения с owner

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h