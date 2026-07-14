"""image_service 生图服务契约测试（直调 service，monkeypatch 网络层）。

覆盖：
- 未启用 → (None, "disabled")
- 成功 → ({"url","model","size"}, "ok") + 写一条缓存
- 缓存命中（≤29 天）→ 不触网、不新增缓存
- 缓存过期（>29 天）→ 触网重生
- 各类失败（timeout/network/gateway/parse）→ (None, status)

用 monkeypatch 替换 OpenAI.images.generate 返假 ImagesResponse，不真调智谱。
"""

import httpx
import pytest
from datetime import datetime, timedelta, timezone
from openai import APIConnectionError, APIStatusError, APITimeoutError
from openai.types.images_response import ImagesResponse

from app.config import Settings
from app.llm_schemas import ImageResult
from app.services import image_service, output_store
from tests.conftest import _patch_get_settings

ENABLED_SETTINGS = Settings(
    image_api_key="fake-image-key",
    image_base_url="https://open.bigmodel.cn/api/paas/v4",
    image_model="cogview-4",
    image_size="1024x1024",
)
DISABLED_SETTINGS = Settings(image_api_key="", image_base_url="")


def _fake_images_response(url: str = "https://example.com/img.png") -> ImagesResponse:
    """构造一个最小可用的假 ImagesResponse。"""
    return ImagesResponse(
        id="fake",
        created=1710000000,
        model="cogview-4",
        data=[{"url": url}],
    )


# ---------------------------------------------------------------------------
# 未启用
# ---------------------------------------------------------------------------


def test_generate_image_disabled(monkeypatch):
    _patch_get_settings(monkeypatch, DISABLED_SETTINGS)
    # 若被触网就抛错
    monkeypatch.setattr(
        image_service, "_client",
        lambda: (_ for _ in ()).throw(AssertionError("未启用不应触网")),
    )
    result, status = image_service.generate_image(prompt="test")
    assert result is None
    assert status == "disabled"


# ---------------------------------------------------------------------------
# 成功
# ---------------------------------------------------------------------------


def test_generate_image_success(monkeypatch):
    _patch_get_settings(monkeypatch, ENABLED_SETTINGS)
    calls = []

    class _FakeImages:
        def generate(self, **kw):
            calls.append(kw)
            return _fake_images_response("https://example.com/ok.png")

    monkeypatch.setattr(image_service, "_client", lambda: type("C", (), {"images": _FakeImages()})())

    result, status = image_service.generate_image(prompt="赛珍珠铁观音海报")
    assert status == "ok"
    assert result is not None
    assert result["url"] == "https://example.com/ok.png"
    assert result["model"] == "cogview-4"
    assert result["size"] == "1024x1024"
    assert result["style"] == image_service.DEFAULT_STYLE, "不传 style 应回显默认 fresh"
    assert result["scene"] == image_service.DEFAULT_SCENE, "不传 scene 应回显默认 closeup"
    # 调用参数含 model/prompt/n/size/quality + extra_body(watermark)
    sent = calls[0]
    assert sent["model"] == "cogview-4"
    assert sent["n"] == 1
    assert sent["quality"] == "hd"
    assert sent["extra_body"] == {"watermark_enabled": False}
    # prompt 被富化：原"赛珍珠铁观音海报"+ 默认 closeup 镜头片段 + fresh 风格片段 + 技术后缀
    assert "赛珍珠铁观音海报" in sent["prompt"]
    # 商务信号词已清除（实测会把出图拽向商务老气风）
    assert "Professional commercial product photography" not in sent["prompt"]
    assert "elegant composition" not in sent["prompt"]
    # 默认 closeup 镜头片段 + fresh 风格片段 + 画质 + 负面词仍在
    assert "close-up product shot" in sent["prompt"]
    assert "morning daylight" in sent["prompt"]
    assert "photorealistic" in sent["prompt"]
    assert "No text, no watermark" in sent["prompt"]
    # 写了一条缓存
    assert output_store.count_rows() == 1


def test_generate_image_landscape_scene(monkeypatch):
    """显式传 scene=landscape → 富化含产地广角镜头片段。"""
    _patch_get_settings(monkeypatch, ENABLED_SETTINGS)
    calls = []

    class _FakeImages:
        def generate(self, **kw):
            calls.append(kw)
            return _fake_images_response("https://example.com/land.png")

    monkeypatch.setattr(image_service, "_client", lambda: type("C", (), {"images": _FakeImages()})())
    result, status = image_service.generate_image(prompt="岩茶海报", scene="landscape")
    assert status == "ok"
    assert result["scene"] == "landscape"
    sent = calls[0]["prompt"]
    # landscape 片段关键词（广角 / 山林 / 上下三分构图）
    assert "wide establishing shot" in sent
    assert "mountain and forest scenery" in sent


def test_generate_image_product_scene(monkeypatch):
    """显式传 scene=product → 富化含商品罐图镜头片段。"""
    _patch_get_settings(monkeypatch, ENABLED_SETTINGS)
    calls = []

    class _FakeImages:
        def generate(self, **kw):
            calls.append(kw)
            return _fake_images_response("https://example.com/prod.png")

    monkeypatch.setattr(image_service, "_client", lambda: type("C", (), {"images": _FakeImages()})())
    result, status = image_service.generate_image(prompt="金骏眉海报", scene="product")
    assert status == "ok"
    assert result["scene"] == "product"
    sent = calls[0]["prompt"]
    assert "tea canister as the main subject" in sent


def test_generate_image_unknown_scene_falls_back(monkeypatch):
    """未知 scene → 回退默认 closeup，不抛、不白屏。"""
    _patch_get_settings(monkeypatch, ENABLED_SETTINGS)
    calls = []

    class _FakeImages:
        def generate(self, **kw):
            calls.append(kw)
            return _fake_images_response("https://example.com/sfb.png")

    monkeypatch.setattr(image_service, "_client", lambda: type("C", (), {"images": _FakeImages()})())
    result, status = image_service.generate_image(prompt="海报", scene="nonexistent_scene")
    assert status == "ok"
    assert result["scene"] == image_service.DEFAULT_SCENE
    assert "close-up product shot" in calls[0]["prompt"]


def test_generate_image_business_style(monkeypatch):
    """显式传 style=business → 富化含商务片段，business 信号词进 prompt。"""
    _patch_get_settings(monkeypatch, ENABLED_SETTINGS)
    calls = []

    class _FakeImages:
        def generate(self, **kw):
            calls.append(kw)
            return _fake_images_response("https://example.com/biz.png")

    monkeypatch.setattr(image_service, "_client", lambda: type("C", (), {"images": _FakeImages()})())
    result, status = image_service.generate_image(prompt="铁观音海报", style="business")
    assert status == "ok"
    assert result["style"] == "business"
    sent = calls[0]["prompt"]
    # business 片段关键词（低光照 / 深色奢华背景）
    assert "low-key studio lighting" in sent
    assert "dark charcoal" in sent
    # 商务美学信号词仍不出现（这些是禁词，business 片段也不含）
    assert "Professional commercial product photography" not in sent
    assert "elegant composition" not in sent


def test_generate_image_unknown_style_falls_back(monkeypatch):
    """未知 style → 回退默认 fresh，不抛、不白屏。"""
    _patch_get_settings(monkeypatch, ENABLED_SETTINGS)
    calls = []

    class _FakeImages:
        def generate(self, **kw):
            calls.append(kw)
            return _fake_images_response("https://example.com/fb.png")

    monkeypatch.setattr(image_service, "_client", lambda: type("C", (), {"images": _FakeImages()})())
    result, status = image_service.generate_image(prompt="海报", style="nonexistent_style")
    assert status == "ok"
    assert result["style"] == image_service.DEFAULT_STYLE
    assert "morning daylight" in calls[0]["prompt"], "未知 style 应用默认 fresh 片段"


def test_generate_image_style_in_cache_key(monkeypatch):
    """同 prompt+size、不同 style → 不命中彼此缓存（style 进了哈希键）。"""
    _patch_get_settings(monkeypatch, ENABLED_SETTINGS)

    class _FakeImages:
        def generate(self, **kw):
            return _fake_images_response("https://example.com/" + kw["prompt"][:1] + ".png")

    monkeypatch.setattr(image_service, "_client", lambda: type("C", (), {"images": _FakeImages()})())
    image_service.generate_image(prompt="同款茶", style="fresh")
    # 第二次换 business，即使 prompt 相同也不应命中 fresh 的缓存
    r2, s2 = image_service.generate_image(prompt="同款茶", style="business")
    assert s2 == "ok"
    assert r2["style"] == "business", "换 style 必须重新生图（缓存键含 style）"


def test_generate_image_scene_in_cache_key(monkeypatch):
    """同 prompt+size+style、不同 scene → 不命中彼此缓存（scene 进了哈希键）。"""
    _patch_get_settings(monkeypatch, ENABLED_SETTINGS)

    class _FakeImages:
        def generate(self, **kw):
            return _fake_images_response("https://example.com/" + kw["prompt"][:1] + ".png")

    monkeypatch.setattr(image_service, "_client", lambda: type("C", (), {"images": _FakeImages()})())
    image_service.generate_image(prompt="同款茶", style="fresh", scene="closeup")
    # 第二次换 landscape，prompt/style 相同也不应命中 closeup 的缓存
    r2, s2 = image_service.generate_image(prompt="同款茶", style="fresh", scene="landscape")
    assert s2 == "ok"
    assert r2["scene"] == "landscape", "换 scene 必须重新生图（缓存键含 scene）"


def test_enrich_prompt_deterministic():
    """富化是纯函数、确定性：同输入两次结果一致；空 prompt 原样返回。"""
    a = image_service._enrich_prompt("茶海报", "fresh", "closeup")
    b = image_service._enrich_prompt("茶海报", "fresh", "closeup")
    assert a == b, "同 prompt + style + scene 富化结果应一致"
    # 含默认 closeup 镜头 + fresh 风格 + 中性画质 + 负面词，不含商务信号词
    assert "close-up product shot" in a
    assert "morning daylight" in a
    assert "photorealistic" in a
    assert "No text, no watermark" in a
    assert "Professional commercial product photography" not in a
    assert "elegant composition" not in a
    # business 风格片段含商务光照信号，但不含已禁的美学词
    biz = image_service._enrich_prompt("茶海报", "business", "closeup")
    assert "low-key studio lighting" in biz
    assert "elegant composition" not in biz
    # landscape / product 镜头片段关键词
    assert "wide establishing shot" in image_service._enrich_prompt("茶海报", "fresh", "landscape")
    assert "tea canister as the main subject" in image_service._enrich_prompt("茶海报", "fresh", "product")
    # 去末尾句号后补后缀，避免双句号
    assert image_service._enrich_prompt("海报。", "fresh", "closeup") == image_service._enrich_prompt("海报", "fresh", "closeup")
    assert image_service._enrich_prompt("Poster.", "fresh", "closeup") == image_service._enrich_prompt("Poster", "fresh", "closeup")
    # 空 prompt 原样返回（不拼后缀）
    assert image_service._enrich_prompt("", "fresh", "closeup") == ""
    assert image_service._enrich_prompt("   ", "fresh", "closeup") == ""


# ---------------------------------------------------------------------------
# 缓存命中
# ---------------------------------------------------------------------------


def test_generate_image_cache_hit(monkeypatch):
    """已缓存且 ≤29 天 → 命中、不触网、不新增缓存。"""
    _patch_get_settings(monkeypatch, ENABLED_SETTINGS)
    # 先预写一条新鲜的缓存（created_at = now，必在 29 天内）
    now_iso = datetime.now(timezone.utc).isoformat()
    input_hash = output_store.compute_input_hash(
        ImageResult, "海报prompt", "1024x1024", image_service.DEFAULT_STYLE, image_service.DEFAULT_SCENE
    )
    output_store.persist(
        output_type="image",
        tea_id=None,
        route_id=None,
        input_hash=input_hash,
        content={
            "url": "https://example.com/cached.png",
            "model": "cogview-4",
            "size": "1024x1024",
            "style": image_service.DEFAULT_STYLE,
            "scene": image_service.DEFAULT_SCENE,
            "created_at": now_iso,  # 刚写，新鲜
        },
    )
    assert output_store.count_rows() == 1

    # 若被触网就抛错
    monkeypatch.setattr(
        image_service, "_client",
        lambda: (_ for _ in ()).throw(AssertionError("应命中缓存不触网")),
    )
    result, status = image_service.generate_image(prompt="海报prompt")
    assert status == "ok"
    assert result["url"] == "https://example.com/cached.png"
    assert output_store.count_rows() == 1, "命中缓存不应新增行"


def test_generate_image_cache_expired(monkeypatch):
    """缓存 >29 天 → 判 miss → 触网重生、覆盖。"""
    _patch_get_settings(monkeypatch, ENABLED_SETTINGS)
    # 预写一条 40 天前的缓存（已过期）
    expired_iso = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
    input_hash = output_store.compute_input_hash(
        ImageResult, "旧海报", "1024x1024", image_service.DEFAULT_STYLE, image_service.DEFAULT_SCENE
    )
    output_store.persist(
        output_type="image",
        tea_id=None,
        route_id=None,
        input_hash=input_hash,
        content={
            "url": "https://example.com/stale.png",
            "model": "cogview-4",
            "size": "1024x1024",
            "created_at": expired_iso,  # 40 天前
        },
    )

    class _FakeImages:
        def generate(self, **kw):
            return _fake_images_response("https://example.com/fresh.png")

    monkeypatch.setattr(image_service, "_client", lambda: type("C", (), {"images": _FakeImages()})())
    result, status = image_service.generate_image(prompt="旧海报")
    assert status == "ok"
    assert result["url"] == "https://example.com/fresh.png", "过期缓存应被新生图覆盖"


# ---------------------------------------------------------------------------
# 各类失败
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reason", ["timeout", "network_error", "gateway_error", "parse_error"])
def test_generate_image_failure(monkeypatch, reason):
    """各类失败 → (None, status)，且不写缓存。"""
    _patch_get_settings(monkeypatch, ENABLED_SETTINGS)

    class _FakeImages:
        def generate(self, **kw):
            if reason == "timeout":
                raise APITimeoutError("timeout")
            if reason == "network_error":
                raise APIConnectionError(request=None)
            if reason == "gateway_error":
                resp = httpx.Response(
                    status_code=429,
                    request=httpx.Request("POST", "https://x.com"),
                    text='{"error":"x"}',
                )
                raise APIStatusError(message="429", response=resp, body=None)
            # parse_error: data 为空
            return ImagesResponse(id="x", created=1, model="cogview-4", data=[])

    monkeypatch.setattr(image_service, "_client", lambda: type("C", (), {"images": _FakeImages()})())
    result, status = image_service.generate_image(prompt="x")
    assert result is None
    assert status == reason
    assert output_store.count_rows() == 0, "失败不应写缓存"
