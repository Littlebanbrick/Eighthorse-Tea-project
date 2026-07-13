"""Demo 路径 + 茶品 + 知识卡片 + 风味坐标路由（第 1、2 层 + demo-routes）。"""

from fastapi import APIRouter

from app import data_loader, responses

router = APIRouter(prefix="/api", tags=["teas"])


@router.get("/demo-routes")
def list_demo_routes():
    """获取当前支持的 Demo 路径（国内链 + 跨文化链）。"""
    return responses.success(data_loader.list_demo_routes())


@router.get("/teas")
def list_teas():
    """获取茶品列表。"""
    return responses.success(data_loader.list_teas())


@router.get("/teas/{tea_id}/knowledge")
def get_knowledge(tea_id: str):
    """获取茶品知识卡片（第 1 层：知识 / 证据层）。"""
    knowledge = data_loader.get_knowledge(tea_id)
    if knowledge is None:
        return responses.error("TEA_NOT_FOUND", "未找到对应茶品")
    return responses.success(knowledge)


@router.get("/teas/{tea_id}/flavor-profile")
def get_flavor_profile(tea_id: str):
    """获取风味坐标卡（第 2 层：风味结构化层）。"""
    profile = data_loader.get_flavor_profile(tea_id)
    if profile is None:
        return responses.error("TEA_NOT_FOUND", "未找到对应茶品")
    return responses.success(profile)


@router.get("/teas/{tea_id}/component-flavor")
def get_component_flavor(tea_id: str):
    """成分 → 口感 映射（第 1→2 层桥接；UI"成分追溯"板块数据源）。

    返回该茶的成分→口感论断列表，每条挂证据溯源（evidence_ids 指向论文）。
    注意：这是第 1→2 层桥接关系的物化，不是纵向追溯链的一层——"追溯"指
    每条论断的证据溯源，不进 trace_links 表、不新增 level。
    """
    if data_loader.get_tea(tea_id) is None:
        return responses.error("TEA_NOT_FOUND", "未找到对应茶品")
    links = data_loader.list_component_flavor_links(tea_id)
    return responses.success({"tea_id": tea_id, "links": links})
