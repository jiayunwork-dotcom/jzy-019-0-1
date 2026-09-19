"""近轴元件的 2x2 光线传递矩阵。

光线全程用列向量 [y, u] 表示：y 为高度，u 为近轴角（角度本身，
不换算成光学方向余弦）。矩阵按行存储为嵌套元组 ((a, b), (c, d))。
"""
from __future__ import annotations

from typing import Any, Mapping

Matrix = tuple[tuple[float, float], tuple[float, float]]
Ray = tuple[float, float]

IDENTITY: Matrix = ((1.0, 0.0), (0.0, 1.0))


def space(length: float) -> Matrix:
    """空气中自由空间传播距离 L。"""
    return ((1.0, float(length)), (0.0, 1.0))


def thin_lens(focal_length: float) -> Matrix:
    """薄透镜，焦距 f（f 非零由入参检查保证）。"""
    return ((1.0, 0.0), (-1.0 / focal_length, 1.0))


def refraction(radius: float, n1: float, n2: float) -> Matrix:
    """球面折射：曲率半径 R，面两侧折射率 n1 -> n2。

    R 的符号与光路前进方向一致：凸向入射一侧取正。
    n1 == n2 时退化为单位矩阵（出射角等于入射角）。
    """
    return ((1.0, 0.0), ((n1 - n2) / (n2 * radius), n1 / n2))


def element_matrix(spec: Mapping[str, Any]) -> Matrix:
    """由已校验的元件规格生成矩阵。未知种类在入参检查阶段已被拦截。"""
    kind = spec["type"]
    if kind == "space":
        return space(spec["length"])
    if kind == "thin_lens":
        return thin_lens(spec["focal_length"])
    if kind == "refraction":
        return refraction(spec["radius"], spec["n1"], spec["n2"])
    raise ValueError(f"unknown element type: {kind!r}")


def matmul(p: Matrix, q: Matrix) -> Matrix:
    """矩阵乘积 p @ q（q 先作用于光线）。"""
    (a, b), (c, d) = p
    (e, f), (g, h) = q
    return ((a * e + b * g, a * f + b * h), (c * e + d * g, c * f + d * h))


def matvec(m: Matrix, ray: Ray) -> Ray:
    (a, b), (c, d) = m
    y, u = ray
    return (a * y + b * u, c * y + d * u)


def determinant(m: Matrix) -> float:
    (a, b), (c, d) = m
    return a * d - b * c


def inverse(m: Matrix) -> Matrix:
    (a, b), (c, d) = m
    det = determinant(m)
    return ((d / det, -b / det), (-c / det, a / det))
