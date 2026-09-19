"""系统矩阵连乘、光线追迹与物像求解。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .elements import IDENTITY, Matrix, Ray, inverse, matmul, matvec

# 钉死的浮点容差
DET_TOL = 1e-12  # 行列式与 1 的允许偏差
AFOCAL_TOL = 1e-9  # |C| 小于该值视为无焦
INFINITE_CONJUGATE_TOL = 1e-12  # |C*s + D| 小于该值视为像在无穷远
ROUND_TRIP_TOL = 1e-9  # 正向再逆向还原入射光线的容差


@dataclass(frozen=True)
class Imaging:
    """物像共轭解。image_distance / magnification 为 None 表示像在无穷远。"""

    object_distance: float
    image_distance: Optional[float]
    magnification: Optional[float]
    image_at_infinity: bool


def compose(matrices: Sequence[Matrix]) -> Matrix:
    """按光线前进方向把各元件矩阵依次乘起来：M = Mn ... M2 M1。"""
    result: Matrix = IDENTITY
    for m in matrices:
        result = matmul(m, result)
    return result


def trace_ray(system: Matrix, ray: Ray) -> Ray:
    """整段矩阵作用于入射光线，得到出射 (y, u)。"""
    return matvec(system, ray)


def effective_focal_length(system: Matrix) -> Optional[float]:
    """空气中有效焦距 EFL = -1 / C；|C| 在容差内视为无焦，返回 None。"""
    (_, _), (c, _) = system
    if abs(c) <= AFOCAL_TOL:
        return None
    return -1.0 / c


def solve_imaging(system: Matrix, object_distance: float) -> Imaging:
    """物在第一面左侧、物距 s 取正。

    物方面到像方面的总矩阵 B' = A*s + B + s'*(C*s + D)，共轭条件
    B' = 0 解出像距 s'；横向放大率 m = A + s'*C。
    单透镜时退化为 1/s + 1/s' = 1/f。
    """
    (a, b), (c, d) = system
    denom = c * object_distance + d
    if abs(denom) <= INFINITE_CONJUGATE_TOL:
        return Imaging(object_distance, None, None, True)
    image_distance = -(a * object_distance + b) / denom
    magnification = a + image_distance * c
    return Imaging(object_distance, image_distance, magnification, False)


def round_trip(matrices: Sequence[Matrix], ray: Ray) -> Ray:
    """正向追迹一遍，再按相反次序用各段矩阵的逆作用回去。"""
    forward = ray
    for m in matrices:
        forward = matvec(m, forward)
    back = forward
    for m in reversed(matrices):
        back = matvec(inverse(m), back)
    return back
