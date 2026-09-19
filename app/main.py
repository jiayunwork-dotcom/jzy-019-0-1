"""近轴光路追迹与成像求解 HTTP 服务（FastAPI 入口）。"""
from __future__ import annotations

import json
import os
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import elements as elem
from . import system as sysmod
from .storage import SystemStore
from .validation import (
    InvalidJson,
    InvalidValue,
    MissingField,
    OpticsError,
    UnknownSystem,
    validate_elements,
    validate_name,
    validate_object_distance,
    validate_ray,
)

# 内置示范档：共焦望远镜，两透镜间距等于焦距之和
DEMO_NAME = "confocal_telescope"
DEMO_F1 = 100.0
DEMO_F2 = 50.0
DEMO_ELEMENTS: list[dict] = [
    {"type": "thin_lens", "focal_length": DEMO_F1},
    {"type": "space", "length": DEMO_F1 + DEMO_F2},
    {"type": "thin_lens", "focal_length": DEMO_F2},
]


def create_app(db_path: str | None = None) -> FastAPI:
    store = SystemStore(db_path or os.environ.get("OPTICS_DB", "systems.db"))
    store.ensure(DEMO_NAME, DEMO_ELEMENTS)

    app = FastAPI(title="paraxial-optics", summary="近轴几何光学追迹与成像求解")

    @app.exception_handler(OpticsError)
    async def _optics_error_handler(_request: Request, exc: OpticsError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.payload())

    def resolve_elements(body: dict) -> tuple[str | None, list[dict]]:
        """按名取档或使用当次请求里的元件清单（只用一次，不落库）。"""
        has_name = body.get("name") is not None
        has_elements = body.get("elements") is not None
        if has_name and has_elements:
            raise InvalidValue("name 与 elements 只能给一个", field="name")
        if not has_name and not has_elements:
            raise MissingField("缺少字段 name 或 elements", field="name")
        if has_name:
            name = validate_name(body["name"])
            stored = store.get(name)
            if stored is None:
                raise UnknownSystem(f"未登记的光路档: {name!r}", field="name")
            return name, stored
        return None, validate_elements(body["elements"])

    @app.get("/")
    async def index() -> dict:
        return {
            "service": "paraxial-optics",
            "endpoints": ["POST /systems", "GET /systems", "POST /trace"],
            "demo_system": DEMO_NAME,
        }

    @app.post("/systems", status_code=201)
    async def register_system(request: Request) -> dict:
        body = await _json_body(request)
        name = validate_name(body.get("name"))
        elements = validate_elements(body.get("elements"))
        store.register(name, elements)
        return {"name": name, "elements": elements}

    @app.get("/systems")
    async def list_systems() -> dict:
        return {"systems": store.list()}

    @app.post("/trace")
    async def trace(request: Request) -> dict:
        body = await _json_body(request)
        name, elements = resolve_elements(body)
        ray = validate_ray(body.get("ray"))
        object_distance = validate_object_distance(body.get("object_distance"))

        matrices = [elem.element_matrix(spec) for spec in elements]
        system = sysmod.compose(matrices)
        out_y, out_u = sysmod.trace_ray(system, ray)
        (a, b), (c, d) = system
        efl = sysmod.effective_focal_length(system)

        result: dict[str, Any] = {
            "system": {
                "A": a,
                "B": b,
                "C": c,
                "D": d,
                "determinant": elem.determinant(system),
            },
            "afocal": efl is None,
            "effective_focal_length": efl,
            "input_ray": {"y": ray[0], "u": ray[1]},
            "output_ray": {"y": out_y, "u": out_u},
            "imaging": None,
        }
        if name is not None:
            result["name"] = name
        if object_distance is not None:
            imaging = sysmod.solve_imaging(system, object_distance)
            result["imaging"] = {
                "object_distance": imaging.object_distance,
                "image_distance": imaging.image_distance,
                "magnification": imaging.magnification,
                "image_at_infinity": imaging.image_at_infinity,
            }
        return result

    return app


async def _json_body(request: Request) -> dict:
    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise InvalidJson("请求体不是合法 JSON") from None
    if not isinstance(body, dict):
        raise InvalidValue("请求体必须是 JSON 对象")
    return body


app = create_app()
