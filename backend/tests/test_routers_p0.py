"""P0 GET 接口：demo-routes / teas / knowledge / flavor-profile。

断言结构与字段类型，不 hardcode seed 具体文本（数据在变）。
LLM 全程 disabled（conftest autouse），这些查询类接口本就不走 LLM。
"""

from tests.conftest import TEA_ID


def test_demo_routes(client):
    resp = client.get("/api/demo-routes")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["meta"]["fallback"] is False
    routes = body["data"]
    assert isinstance(routes, list)
    assert len(routes) >= 2
    for r in routes:
        for k in ("id", "tea_id", "market", "target_language",
                  "audience_reference", "asset_type", "enabled"):
            assert k in r, f"缺字段 {k}"
        assert isinstance(r["enabled"], bool)
    # 主路径两条都在且 enabled
    route_ids = [r["id"] for r in routes]
    assert "szz_domestic_poster" in route_ids
    assert "szz_western_coffee_poster" in route_ids
    assert all(r["enabled"] for r in routes if r["id"] in {
        "szz_domestic_poster", "szz_western_coffee_poster",
    })


def test_list_teas(client):
    resp = client.get("/api/teas")
    assert resp.status_code == 200
    teas = resp.json()["data"]
    assert isinstance(teas, list) and teas
    t = teas[0]
    for k in ("id", "name", "category", "origin", "brand", "demo_available"):
        assert k in t
    assert isinstance(t["demo_available"], bool)


def test_get_knowledge(client):
    resp = client.get(f"/api/teas/{TEA_ID}/knowledge")
    assert resp.status_code == 200
    d = resp.json()["data"]
    for k in ("tea", "origin", "process", "story", "evidence"):
        assert k in d
    assert isinstance(d["process"]["steps"], list) and d["process"]["steps"]
    assert isinstance(d["evidence"], list) and d["evidence"]
    ev = d["evidence"][0]
    for k in ("id", "source_type", "source", "confidence"):
        assert k in ev
    assert ev["confidence"] in ("high", "medium", "low")


def test_get_flavor_profile(client):
    resp = client.get(f"/api/teas/{TEA_ID}/flavor-profile")
    assert resp.status_code == 200
    d = resp.json()["data"]
    assert "dimensions" in d
    assert isinstance(d["dimensions"], list) and d["dimensions"]
    for dim in d["dimensions"]:
        assert "key" in dim and "label_zh" in dim and "label_en" in dim
        assert isinstance(dim["intensity"], int)
        assert 0 <= dim["intensity"] <= 10
        assert isinstance(dim["evidence_ids"], list)


def test_get_component_flavor(client):
    """成分追溯：返回该茶成分→口感映射，每条带证据溯源。"""
    resp = client.get(f"/api/teas/{TEA_ID}/component-flavor")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["meta"]["fallback"] is False
    d = body["data"]
    assert d["tea_id"] == TEA_ID
    assert isinstance(d["links"], list) and d["links"]
    lk = d["links"][0]
    for k in (
        "component",
        "component_category",
        "flavor_key",
        "flavor_label",
        "flavor_dimension",
        "mechanism",
        "relationship",
        "evidence",
        "confidence",
        "notes",
    ):
        assert k in lk
    # flavor_dimension 对齐该茶 dimension（flavor_key 非空时）
    if lk["flavor_key"]:
        assert lk["flavor_dimension"] is not None
        assert isinstance(lk["flavor_dimension"]["intensity"], int)
    # 证据溯源：evidence 非空，每条挂 id/source/confidence
    assert isinstance(lk["evidence"], list) and lk["evidence"]
    for ev in lk["evidence"]:
        assert ev["confidence"] in ("high", "medium", "low")


def test_component_flavor_not_found(client):
    resp = client.get("/api/teas/nonexistent_tea/component-flavor")
    assert resp.status_code == 200  # 业务错误不走 HTTP 4xx
    body = resp.json()
    assert body["success"] is False
    assert body["error"]["code"] == "TEA_NOT_FOUND"


def test_tea_not_found(client):
    resp = client.get("/api/teas/nonexistent_tea/knowledge")
    assert resp.status_code == 200  # 业务错误不走 HTTP 4xx
    body = resp.json()
    assert body["success"] is False
    assert body["error"]["code"] == "TEA_NOT_FOUND"

    resp2 = client.get("/api/teas/nonexistent_tea/flavor-profile")
    assert resp2.json()["error"]["code"] == "TEA_NOT_FOUND"


def test_root_and_health(client):
    """根路径与健康检查（非 /api/* 业务接口，但宜冒烟。"""
    assert client.get("/").status_code == 200
    assert client.get("/health").json()["status"] == "ok"
