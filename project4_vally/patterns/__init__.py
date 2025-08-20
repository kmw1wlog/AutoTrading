# patterns/__init__.py
from .triangle import detect_triangle
from .wedge import detect_wedge
from .flag import detect_flag
from .head_and_shoulders import detect_head_and_shoulders
from .double_top import detect_double_top
from .triple_top import detect_triple_top
from .cup_with_handle import detect_cup_with_handle
from .quasimodo import detect_quasimodo
from .wolf_wave import detect_wolf_wave

__all__ = [
    "detect_triangle",
    "detect_wedge",
    "detect_flag",
    "detect_head_and_shoulders",
    "detect_double_top",
    "detect_triple_top",
    "detect_cup_with_handle",
    "detect_quasimodo",
    "detect_wolf_wave",
]
