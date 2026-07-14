"""生图服务：调智谱 CogView-4 文生图。

设计要点（镜像 llm_service.generate）：
- 基于 openai SDK（OpenAI 兼容），智谱 CogView 端点 {base_url}/images/generations
  与 GLM 共用智谱 key + base_url。但本服务凭证独立走 IMAGE_*（当前 LLM_* 多半
  指向 DeepSeek，不覆盖智谱生图端点）—— 不回退 LLM_*，必须独立配 IMAGE_*。
- 同步调用，与现有同步 service 风格一致（FastAPI 把同步 handler 丢线程池跑）。
- 失败永不抛：未启用 / 网络 / 超时 / 解析失败统一返回降级状态，由路由层走 fallback。
  生图无 seed 兜底（没有预置图），与文本三接口"退回 seed"不同。

prompt 富化：marketing-asset.image_prompt 通常是一句话的精短描述（人读友好、
确定），直接喂 CogView 易出图虚。本服务在发图前套一段确定性质量后缀（光照 /
构图 / 镜头 / 画质 / 负面词），不加 LLM 调用、零幻觉、确定性——marketing-asset
契约的 image_prompt 字段仍保持精短，富化只在生图内部发生，对前端透明。

quality（hd/standard）走 openai SDK 的 quality 参数；watermark_enabled=false
是智谱扩展参数、SDK 不暴露，经 extra_body 透传。

返回 (result | None, status)：
  status ∈ "ok" / "disabled" / "network_error" / "timeout"
         / "parse_error" / "gateway_error"
  result = {"url": str, "model": str, "size": str}

缓存（镜像 intent_service）：按 prompt + size 算 input_hash，命中且 created_at
≤29 天（智谱图片临时链接 30 天有效）即复用、跳过 CogView 调用；否则调 CogView、
成功后写回。缓存命中仍标 success（对前端透明）。注意：富化后缀不计入缓存键——
固定后缀不产生新维度，避免同一精短 prompt 因后缀微调产生多份缓存。
"""

import logging
from datetime import datetime, timedelta, timezone

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
)

from app.config import get_settings
from app.llm_schemas import ImageResult
from app.services import output_store

logger = logging.getLogger("app.image")

# 生图缓存有效期：智谱图片临时链接 30 天，留 1 天裕量提前判 miss 重生。
_CACHE_TTL = timedelta(days=29)

# status 取值（与 llm_service 对齐，便于路由层统一处理）
FALLBACK_DISABLED = "disabled"
FALLBACK_NETWORK = "network_error"
FALLBACK_TIMEOUT = "timeout"
FALLBACK_PARSE = "parse_error"
FALLBACK_GATEWAY = "gateway_error"

# 确定性质量后缀 + 风格片段：富化 CogView 出图，加在用户 prompt 之后。
# 不依赖 LLM、零幻觉、确定性。quality / watermark_enabled 走请求体参数。
#
# 三层职责切分（避免光照在 seed 与片段里打架）：
#   - seed / LLM image_prompt：物体 + 构图（9:16 / 中下部 / 文字安全区）+ 负面词
#   - _STYLE_FRAGMENTS[style]：光照 + 色调 + 氛围 + 背景（风格化轴）
#   - _TECHNICAL_SUFFIX：构图护栏 + 画质 + 负面词（对任意 prompt 都成立的兜底）
# P1 已证明：prompt 残留 "Professional commercial product photography / elegant
# composition" 这类企业画册美学词会把 CogView 拽向商务老气风。故默认走 fresh
# 清新风，business 风格片段才显式给商务信号（要商务时调用方显式传 style=business）。

DEFAULT_STYLE = "fresh"

# 风格片段：只写光照 / 色调 / 氛围 / 背景，不写构图与画质（避免与 seed / 技术后缀
# 冲突）。每个片段是一段英文短语，不含句末标点（由 _enrich_prompt 统一拼接）。
_STYLE_FRAGMENTS: dict[str, str] = {
    "fresh": (
        "soft diffused morning daylight, fresh green and ivory color palette, "
        "airy clean background with light wood and fresh foliage, bright clean "
        "and natural mood"
    ),
    "business": (
        "dramatic low-key studio lighting, dark charcoal and warm gold color "
        "palette, dark premium background with deep walnut wood and subtle gold "
        "accents, serious authoritative luxury commercial mood"
    ),
}

# 技术后缀：构图护栏 + 画质 + 负面词。刻意不含"专业商品摄影 / elegant composition"
# 这类企业画册美学词（实测把出图拽向商务老气风）。风格化由 _STYLE_FRAGMENTS 负责。
_TECHNICAL_SUFFIX = (
    ". Vertical 9:16 mobile poster composition, main subject in the lower-middle "
    "frame, clean uncluttered text-safe area occupying about 25-35% of the upper "
    "frame for later headline overlay, shallow depth of field, sharp focus on the "
    "subject, high detail, 8k, photorealistic. "
    "No text, no watermark, no generated text, no logo, no distorted proportions, no extra objects."
)


def _client() -> OpenAI:
    """构造 OpenAI 兼容 client（指向配置的 IMAGE_BASE_URL）。"""
    s = get_settings()
    api_key, base_url = s.image_credentials()
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=s.image_timeout,
        max_retries=0,  # 不静默延长延迟；失败即降级
    )


def _normalize_style(style: str | None) -> str:
    """归一化 style：lower / strip；未知或 None → DEFAULT_STYLE。

    未知 style 不抛、走默认 + log（生图无 seed 兜底，不能因 style 拼错而白屏）。
    """
    if not style:
        return DEFAULT_STYLE
    s = style.strip().lower()
    if s not in _STYLE_FRAGMENTS:
        logger.warning("未知 style=%r，回退默认 %s", style, DEFAULT_STYLE)
        return DEFAULT_STYLE
    return s


def _enrich_prompt(prompt: str, style: str) -> str:
    """给精短 prompt 套风格片段 + 确定性技术后缀（构图/画质/负面词）。

    marketing-asset.image_prompt 是画面物体描述（茶具/茶汤/道具/场景/构图），
    光照/色调/氛围由 style 片段注入——这样同一茶 prompt × N 风格，无 seed 爆炸、
    不调 LLM、确定性。零 LLM、零幻觉。
    """
    prompt = (prompt or "").strip()
    if not prompt:
        return prompt
    # 去掉末尾句号再补后缀，避免双句号
    if prompt.endswith(("。", ".")):
        prompt = prompt[:-1]
    frag = _STYLE_FRAGMENTS[style]
    return f"{prompt}, {frag}{_TECHNICAL_SUFFIX}"


def generate_image(*, prompt: str, size: str | None = None, style: str | None = None) -> tuple[dict | None, str]:
    """调 CogView-4 生图。

    Args:
        prompt: 图片生成 prompt（通常来自 marketing-asset.image_prompt，画面物体描述）
        size: 输出尺寸，空则用配置默认 image_size
        style: 风格（fresh / business）；空或未知 → DEFAULT_STYLE（fresh）

    Returns:
        (result | None, status)。
        成功 → ({"url","model","size","style"}, "ok")；否则 → (None, fallback_reason)。
    """
    s = get_settings()
    if not s.image_enabled:
        return None, FALLBACK_DISABLED

    used_size = size or s.image_size
    effective_style = _normalize_style(style)
    enriched_prompt = _enrich_prompt(prompt, effective_style)

    # 先查缓存：命中且未过期即复用，跳过 CogView 调用。
    # 缓存键用原始 prompt + size + style（技术后缀固定，不计入键；style 必须计入，
    # 否则切风格会命中旧风格缓存——缓存投毒）。
    input_hash = output_store.compute_input_hash(
        ImageResult, prompt, used_size, effective_style
    )
    cached = output_store.get_cached(input_hash)
    if cached is not None and _cache_fresh(cached):
        return _build_result(cached, s.image_model, used_size), "ok"

    try:
        resp = _client().images.generate(
            model=s.image_model,
            prompt=enriched_prompt,
            n=1,
            size=used_size,
            quality=s.image_quality,
            # watermark_enabled 是智谱扩展参数，SDK 不暴露，经 extra_body 透传
            extra_body={"watermark_enabled": False},
        )
    except APITimeoutError:
        logger.warning("生图超时 model=%s timeout=%s", s.image_model, s.image_timeout)
        return None, FALLBACK_TIMEOUT
    except APIConnectionError as e:
        # APITimeoutError 是 APIConnectionError 子类，必须放它之前
        logger.warning("生图连接失败 model=%s err=%s", s.image_model, e)
        return None, FALLBACK_NETWORK
    except APIStatusError as e:
        body = ""
        try:
            body = e.response.text[:500]
        except Exception:
            body = ""
        logger.warning(
            "生图网关错误 model=%s status=%s body=%s err=%s",
            s.image_model, e.status_code, body, e,
        )
        return None, FALLBACK_GATEWAY
    except Exception as e:  # 其他未预期异常，兜底为 network_error
        logger.warning("生图调用失败（未分类）model=%s err=%s", s.image_model, e)
        return None, FALLBACK_NETWORK

    # 取图片 URL；代理可能返回非标准 ImagesResponse 形状，统一兜住。
    try:
        url = resp.data[0].url
    except (AttributeError, IndexError, TypeError) as e:
        logger.warning(
            "生图响应非标准 ImagesResponse 形状，降级 model=%s type=%s err=%s",
            s.image_model, type(resp).__name__, e,
        )
        return None, FALLBACK_PARSE
    if not url:
        logger.warning("生图响应 data[0].url 为空，降级 model=%s", s.image_model)
        return None, FALLBACK_PARSE

    now = datetime.now(timezone.utc).isoformat()
    content = {
        "url": url, "model": s.image_model, "size": used_size,
        "style": effective_style, "created_at": now,
    }
    output_store.persist(
        output_type="image",
        tea_id=None,
        route_id=None,
        input_hash=input_hash,
        content=content,
    )
    logger.info("生图成功 model=%s size=%s style=%s quality=%s prompt_chars=%d",
                s.image_model, used_size, effective_style, s.image_quality, len(enriched_prompt))
    return _build_result(content, s.image_model, used_size), "ok"


def _cache_fresh(cached: dict) -> bool:
    """缓存是否在有效期内（created_at ≤29 天）。

    created_at 缺失或解析失败视为过期（强制重生，避免死链）。
    """
    created = cached.get("created_at")
    if not created:
        return False
    try:
        ts = datetime.fromisoformat(created)
    except (ValueError, TypeError):
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - ts < _CACHE_TTL


def _build_result(content: dict, model: str, size: str) -> dict:
    """从缓存内容组装返回结果（命中缓存时模型/尺寸/风格沿用缓存值）。"""
    return {
        "url": content["url"],
        "model": content.get("model") or model,
        "size": content.get("size") or size,
        "style": content.get("style"),
    }
