"""入参检查：未知元件、负间隔、零焦距、缺项、非有限数、非正折射率
都在追迹前以带类型的错误退回，并说明哪一项不合规。"""
from __future__ import annotations

import math
from typing import Any


class OpticsError(Exception):
    """带类型的错误基类，由 FastAPI 统一转成 JSON 响应。"""

    status_code: int = 400
    error_type: str = "optics_error"

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        index: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.field = field
        self.index = index

    def payload(self) -> dict[str, Any]:
        error: dict[str, Any] = {"type": self.error_type, "message": self.message}
        if self.field is not None:
            error["field"] = self.field
        if self.index is not None:
            error["element_index"] = self.index
        return {"error": error}


class MissingField(OpticsError):
    error_type = "missing_field"


class InvalidValue(OpticsError):
    error_type = "invalid_value"


class UnknownElementType(OpticsError):
    error_type = "unknown_element_type"


class UnknownSystem(OpticsError):
    error_type = "unknown_system"
    status_code = 404


class DuplicateSystem(OpticsError):
    error_type = "duplicate_system"
    status_code = 409


class InvalidJson(OpticsError):
    error_type = "invalid_json"


KNOWN_ELEMENT_TYPES = ("space", "thin_lens", "refraction")


def _finite_number(value: Any, field: str, index: int | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidValue(f"{field} 必须是数值", field=field, index=index)
    number = float(value)
    if not math.isfinite(number):
        raise InvalidValue(f"{field} 必须是有限数", field=field, index=index)
    return number


def _required(spec: dict, key: str, index: int | None) -> Any:
    if key not in spec:
        raise MissingField(f"缺少字段 {key}", field=key, index=index)
    return spec[key]


def validate_element(raw: Any, index: int) -> dict:
    """校验单个元件并返回规范化规格，不合规时抛出带类型的错误。"""
    if not isinstance(raw, dict):
        raise InvalidValue("元件必须是对象", index=index)
    kind = _required(raw, "type", index)
    if kind not in KNOWN_ELEMENT_TYPES:
        raise UnknownElementType(f"未知元件种类: {kind!r}", field="type", index=index)

    if kind == "space":
        length = _finite_number(_required(raw, "length", index), "length", index)
        if length < 0:
            raise InvalidValue(
                "间隔不能为负：负间隔不等于正向传播", field="length", index=index
            )
        return {"type": "space", "length": length}

    if kind == "thin_lens":
        focal = _finite_number(_required(raw, "focal_length", index), "focal_length", index)
        if focal == 0.0:
            raise InvalidValue("焦距不得为零", field="focal_length", index=index)
        return {"type": "thin_lens", "focal_length": focal}

    radius = _finite_number(_required(raw, "radius", index), "radius", index)
    if radius == 0.0:
        raise InvalidValue("曲率半径不得为零", field="radius", index=index)
    n1 = _finite_number(_required(raw, "n1", index), "n1", index)
    n2 = _finite_number(_required(raw, "n2", index), "n2", index)
    if n1 <= 0:
        raise InvalidValue("折射率必须为正", field="n1", index=index)
    if n2 <= 0:
        raise InvalidValue("折射率必须为正", field="n2", index=index)
    return {"type": "refraction", "radius": radius, "n1": n1, "n2": n2}


def validate_elements(raw: Any) -> list[dict]:
    if raw is None:
        raise MissingField("缺少字段 elements", field="elements")
    if not isinstance(raw, list) or not raw:
        raise InvalidValue("elements 必须是非空数组", field="elements")
    return [validate_element(item, i) for i, item in enumerate(raw)]


def validate_name(raw: Any) -> str:
    if raw is None:
        raise MissingField("缺少字段 name", field="name")
    if not isinstance(raw, str) or not raw.strip():
        raise InvalidValue("name 必须是非空字符串", field="name")
    return raw


def validate_ray(raw: Any) -> tuple[float, float]:
    if raw is None:
        raise MissingField("缺少字段 ray", field="ray")
    if not isinstance(raw, dict):
        raise InvalidValue("ray 必须是对象", field="ray")
    y = _finite_number(_required(raw, "y", None), "ray.y")
    u = _finite_number(_required(raw, "u", None), "ray.u")
    return (y, u)


def validate_object_distance(raw: Any) -> float | None:
    """物距可选；给定时必须为正。"""
    if raw is None:
        return None
    distance = _finite_number(raw, "object_distance")
    if distance <= 0:
        raise InvalidValue("物距必须为正", field="object_distance")
    return distance
