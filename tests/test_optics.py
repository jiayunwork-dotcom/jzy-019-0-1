"""钉死需求里列出的全部行为：行列式、高斯公式、共焦望远镜、
往返还原、各类带类型拒绝、并行追迹隔离。"""
import asyncio
import math

import httpx
import pytest
from fastapi.testclient import TestClient

from app import system as sysmod
from app.elements import determinant, element_matrix
from app.main import DEMO_ELEMENTS, DEMO_F1, DEMO_F2, DEMO_NAME, create_app


def make_client() -> TestClient:
    return TestClient(create_app(":memory:"))


# ---------- 矩阵与行列式 ----------

def test_air_system_determinant_is_one():
    """空气间隔夹薄透镜：每一段矩阵与整段乘积的行列式都是 1。"""
    elements = [
        {"type": "space", "length": 30.0},
        {"type": "thin_lens", "focal_length": 50.0},
        {"type": "space", "length": 20.0},
        {"type": "thin_lens", "focal_length": -40.0},
        {"type": "space", "length": 10.0},
    ]
    matrices = [element_matrix(e) for e in elements]
    for m in matrices:
        assert abs(determinant(m) - 1.0) <= sysmod.DET_TOL
    system = sysmod.compose(matrices)
    assert abs(determinant(system) - 1.0) <= sysmod.DET_TOL


def test_air_glass_air_determinant_returns_to_one():
    """空气进玻璃再出到空气，整段行列式回到 1。"""
    elements = [
        {"type": "refraction", "radius": 50.0, "n1": 1.0, "n2": 1.5},
        {"type": "space", "length": 10.0},
        {"type": "refraction", "radius": -50.0, "n1": 1.5, "n2": 1.0},
    ]
    system = sysmod.compose([element_matrix(e) for e in elements])
    assert abs(determinant(system) - 1.0) <= sysmod.DET_TOL


def test_refraction_with_equal_indices_is_identity():
    """两侧折射率相等时球面退化成什么也不做。"""
    m = element_matrix({"type": "refraction", "radius": 30.0, "n1": 1.5, "n2": 1.5})
    y, u = sysmod.trace_ray(m, (2.0, 0.1))
    assert y == pytest.approx(2.0)
    assert u == pytest.approx(0.1)


# ---------- 物像求解 ----------

def test_single_lens_recovers_gaussian_formula():
    """单个薄透镜从矩阵还原高斯公式：1/s + 1/s' = 1/f。"""
    client = make_client()
    f, s = 50.0, 120.0
    resp = client.post(
        "/trace",
        json={
            "elements": [{"type": "thin_lens", "focal_length": f}],
            "ray": {"y": 1.0, "u": 0.0},
            "object_distance": s,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    image_distance = data["imaging"]["image_distance"]
    assert 1.0 / s + 1.0 / image_distance == pytest.approx(1.0 / f, rel=1e-12)
    assert data["imaging"]["magnification"] == pytest.approx(
        -image_distance / s, rel=1e-12
    )
    assert data["effective_focal_length"] == pytest.approx(f, rel=1e-12)


def test_confocal_telescope_is_afocal_with_negative_magnification():
    """内置共焦望远镜：C 接近 0，放大率为负且逼近 -f2/f1。"""
    client = make_client()
    resp = client.post(
        "/trace",
        json={
            "name": DEMO_NAME,
            "ray": {"y": 2.0, "u": 0.01},
            "object_distance": 300.0,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert abs(data["system"]["C"]) <= sysmod.AFOCAL_TOL
    assert data["afocal"] is True
    assert data["effective_focal_length"] is None
    magnification = data["imaging"]["magnification"]
    assert magnification == pytest.approx(-DEMO_F2 / DEMO_F1, rel=1e-9)
    assert magnification < 0


def test_defocus_from_confocal_restores_power():
    """把间距拉开或缩短，焦度重新出现，不再报无焦。"""
    client = make_client()
    for spacing in (DEMO_F1 + DEMO_F2 - 10.0, DEMO_F1 + DEMO_F2 + 10.0):
        resp = client.post(
            "/trace",
            json={
                "elements": [
                    {"type": "thin_lens", "focal_length": DEMO_F1},
                    {"type": "space", "length": spacing},
                    {"type": "thin_lens", "focal_length": DEMO_F2},
                ],
                "ray": {"y": 1.0, "u": 0.0},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["afocal"] is False
        assert abs(data["system"]["C"]) > sysmod.AFOCAL_TOL
        assert data["effective_focal_length"] is not None
        assert math.isfinite(data["effective_focal_length"])


def test_changing_one_focal_length_changes_efl_and_magnification():
    """只改某一块透镜的焦距，有效焦距与放大率都跟着变。"""
    client = make_client()
    base = [
        {"type": "thin_lens", "focal_length": 100.0},
        {"type": "space", "length": 120.0},
        {"type": "thin_lens", "focal_length": 50.0},
    ]
    changed = [base[0], base[1], {"type": "thin_lens", "focal_length": 60.0}]
    results = []
    for elements in (base, changed):
        resp = client.post(
            "/trace",
            json={
                "elements": elements,
                "ray": {"y": 1.0, "u": 0.0},
                "object_distance": 200.0,
            },
        )
        assert resp.status_code == 200
        results.append(resp.json())
    efl_a, efl_b = (r["effective_focal_length"] for r in results)
    mag_a, mag_b = (r["imaging"]["magnification"] for r in results)
    assert not math.isclose(efl_a, efl_b, rel_tol=1e-9)
    assert not math.isclose(mag_a, mag_b, rel_tol=1e-9)


# ---------- 往返关系 ----------

def test_round_trip_recovers_input_ray():
    """正向追一遍再逆向作用回去，入射 y 与 u 在容差内还原。"""
    elements = [
        {"type": "space", "length": 25.0},
        {"type": "refraction", "radius": 50.0, "n1": 1.0, "n2": 1.5},
        {"type": "space", "length": 10.0},
        {"type": "refraction", "radius": -60.0, "n1": 1.5, "n2": 1.0},
        {"type": "thin_lens", "focal_length": 80.0},
        {"type": "space", "length": 40.0},
    ]
    matrices = [element_matrix(e) for e in elements]
    ray_in = (3.0, -0.02)
    back = sysmod.round_trip(matrices, ray_in)
    assert back[0] == pytest.approx(ray_in[0], abs=sysmod.ROUND_TRIP_TOL)
    assert back[1] == pytest.approx(ray_in[1], abs=sysmod.ROUND_TRIP_TOL)


# ---------- 带类型的拒绝 ----------

def test_zero_focal_length_rejected():
    client = make_client()
    resp = client.post(
        "/systems",
        json={"name": "bad", "elements": [{"type": "thin_lens", "focal_length": 0}]},
    )
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert error["type"] == "invalid_value"
    assert error["field"] == "focal_length"
    assert error["element_index"] == 0


def test_unknown_element_rejected():
    client = make_client()
    resp = client.post(
        "/systems",
        json={"name": "x", "elements": [{"type": "prism", "angle": 5}]},
    )
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert error["type"] == "unknown_element_type"
    assert error["element_index"] == 0


def test_unknown_system_rejected():
    client = make_client()
    resp = client.post(
        "/trace", json={"name": "does-not-exist", "ray": {"y": 1.0, "u": 0.0}}
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "unknown_system"


def test_negative_spacing_rejected():
    client = make_client()
    resp = client.post(
        "/trace",
        json={
            "elements": [{"type": "space", "length": -5.0}],
            "ray": {"y": 1.0, "u": 0.0},
        },
    )
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert error["type"] == "invalid_value"
    assert error["field"] == "length"


def test_missing_field_rejected():
    client = make_client()
    resp = client.post(
        "/systems",
        json={"name": "incomplete", "elements": [{"type": "space"}]},
    )
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert error["type"] == "missing_field"
    assert error["field"] == "length"


def test_non_finite_rejected():
    client = make_client()
    resp = client.post(
        "/trace",
        json={
            "elements": [{"type": "thin_lens", "focal_length": 10.0}],
            "ray": {"y": float("nan"), "u": 0.0},
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_value"


def test_non_positive_index_rejected():
    client = make_client()
    resp = client.post(
        "/systems",
        json={
            "name": "bad-n",
            "elements": [
                {"type": "refraction", "radius": 50.0, "n1": 1.0, "n2": -1.5}
            ],
        },
    )
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert error["type"] == "invalid_value"
    assert error["field"] == "n2"


# ---------- 档的登记、列出与一次性清单 ----------

def test_register_and_list_shows_full_element_specs():
    client = make_client()
    elements = [
        {"type": "refraction", "radius": 40.0, "n1": 1.0, "n2": 1.6},
        {"type": "space", "length": 8.0},
        {"type": "refraction", "radius": -40.0, "n1": 1.6, "n2": 1.0},
    ]
    resp = client.post("/systems", json={"name": "singlet", "elements": elements})
    assert resp.status_code == 201

    listing = client.get("/systems").json()["systems"]
    names = {s["name"] for s in listing}
    assert {"singlet", DEMO_NAME} <= names
    singlet = next(s for s in listing if s["name"] == "singlet")
    assert singlet["elements"] == elements
    demo = next(s for s in listing if s["name"] == DEMO_NAME)
    assert demo["elements"] == DEMO_ELEMENTS


def test_inline_elements_are_used_once_not_stored():
    client = make_client()
    resp = client.post(
        "/trace",
        json={
            "elements": [{"type": "thin_lens", "focal_length": 25.0}],
            "ray": {"y": 1.0, "u": 0.0},
        },
    )
    assert resp.status_code == 200
    names = {s["name"] for s in client.get("/systems").json()["systems"]}
    assert names == {DEMO_NAME}


# ---------- 并行追迹隔离 ----------

def test_parallel_traces_do_not_contaminate_each_other():
    app = create_app(":memory:")
    client = TestClient(app)
    assert client.post(
        "/systems",
        json={"name": "sys-a", "elements": [{"type": "thin_lens", "focal_length": 10.0}]},
    ).status_code == 201
    assert client.post(
        "/systems",
        json={"name": "sys-b", "elements": [{"type": "thin_lens", "focal_length": 40.0}]},
    ).status_code == 201

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as async_client:

            async def trace_many(name, y, u, s):
                return await asyncio.gather(
                    *(
                        async_client.post(
                            "/trace",
                            json={
                                "name": name,
                                "ray": {"y": y, "u": u},
                                "object_distance": s,
                            },
                        )
                        for _ in range(25)
                    )
                )

            return await asyncio.gather(
                trace_many("sys-a", 1.0, 0.0, 30.0),
                trace_many("sys-b", 2.0, 0.0, 60.0),
            )

    responses_a, responses_b = asyncio.run(run())

    for resp in responses_a:
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "sys-a"
        assert data["effective_focal_length"] == pytest.approx(10.0)
        assert data["output_ray"]["y"] == pytest.approx(1.0)
        assert data["output_ray"]["u"] == pytest.approx(-0.1)
        assert data["imaging"]["image_distance"] == pytest.approx(15.0)
        assert data["imaging"]["magnification"] == pytest.approx(-0.5)

    for resp in responses_b:
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "sys-b"
        assert data["effective_focal_length"] == pytest.approx(40.0)
        assert data["output_ray"]["y"] == pytest.approx(2.0)
        assert data["output_ray"]["u"] == pytest.approx(-0.05)
        assert data["imaging"]["image_distance"] == pytest.approx(120.0)
        assert data["imaging"]["magnification"] == pytest.approx(-2.0)
