"""
C线模块：DeltaMerger
将多轮对话中的 delta 合并到当前 PlotSpec，并填充可选字段默认值。
"""

from __future__ import annotations

from schema import OPTIONAL_DEFAULTS, REQUIRED_FIELDS


def merge_delta(current_spec: dict, delta: dict) -> dict:
    """
    将 delta 中的字段覆盖到 current_spec。

    Args:
        current_spec: 当前完整 PlotSpec dict（不被修改）。
        delta:        仅含变更字段的 dict。

    Returns:
        合并后的新 PlotSpec dict。
    """
    merged = dict(current_spec)
    merged.update(delta)
    return merged


def strip_defaults(spec: dict) -> dict:
    """
    移除 spec 中与 OPTIONAL_DEFAULTS 相同的默认值字段，只保留必填字段和实际非默认的可选字段。
    data_source 始终排除。
    是 fill_defaults 的逆操作，用于将完整 PlotSpec 压缩为紧凑形式以节省 token。

    Args:
        spec: 完整 PlotSpec dict（可能含 data_source）。

    Returns:
        压缩后的 PlotSpec dict。
    """
    required_no_source = {f for f in REQUIRED_FIELDS if f != "data_source"}
    return {
        k: v for k, v in spec.items()
        if k in required_no_source
        or (k in OPTIONAL_DEFAULTS and v != OPTIONAL_DEFAULTS[k])
    }


def fill_defaults(spec: dict) -> dict:
    """
    将 OPTIONAL_DEFAULTS 中存在但 spec 中缺失的字段填入默认值。
    同时对 data_y 做规范化：单元素列表自动拆包为字符串，
    避免渲染器走"多列直接绘制"分支而忽略 data_group_by。

    Args:
        spec: 可能缺少 optional 字段的 PlotSpec dict（不被修改）。

    Returns:
        填充了所有 optional 默认值的完整 PlotSpec dict。
    """
    result = dict(spec)
    for key, default_val in OPTIONAL_DEFAULTS.items():
        if key not in result:
            result[key] = default_val

    # data_y 单元素列表 → 字符串，防止 group_by 失效
    data_y = result.get("data_y")
    if isinstance(data_y, list) and len(data_y) == 1:
        result["data_y"] = data_y[0]

    return result
