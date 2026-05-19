"""
tests/test_generator.py
验证 generate_spec() 的接口契约。
有 DEEPSEEK_API_KEY 时发起真实调用；无 key 时跳过 API 测试。
"""

import os

import pytest

from model.generator import generate_spec
from schema import REQUIRED_FIELDS

_HAS_KEY = bool(os.environ.get("DEEPSEEK_API_KEY"))
_skip_no_key = pytest.mark.skipif(not _HAS_KEY, reason="未设置 DEEPSEEK_API_KEY，跳过真实 API 测试")

_MOCK_DATA_CONTEXT = """\
数据摘要：
- 形状：24行 × 4列
- 列信息：
  · method（类别型，唯一值6个）：BERT-base, RoBERTa-base
  · dataset（类别型，唯一值4个）：SST-2, MR, CR, CoLA
  · accuracy（数值型，范围55.1~95.1）
  · std（数值型，范围0.3~1.1）
- 前2行预览：[["BERT-base","SST-2",93.5,0.3]]
- 缓存key：cache://a1b2c3d4"""

_CURRENT_SPEC = {
    "chart_type": "bar",
    "data_source": "cache://a1b2c3d4",
    "data_x": "method",
    "data_y": "accuracy",
    "style_theme": "nature",
}


@_skip_no_key
def test_first_round_contract():
    """首轮调用应返回包含所有 REQUIRED_FIELDS 的 dict。"""
    result = generate_spec("画柱状图对比各模型准确率，nature风格", _MOCK_DATA_CONTEXT)
    assert isinstance(result, dict)
    for field in REQUIRED_FIELDS:
        assert field in result, f"缺少必要字段：{field}"


@_skip_no_key
def test_first_round_data_fields_are_strings():
    """首轮 data_x 必须是字符串，data_y 必须是字符串或字符串列表（不能含数值）。"""
    result = generate_spec("画柱状图对比各模型准确率", _MOCK_DATA_CONTEXT)
    assert isinstance(result.get("data_x"), str), "data_x 应为字符串列名"
    dy = result.get("data_y")
    assert isinstance(dy, (str, list)), "data_y 应为字符串或字符串列表"
    if isinstance(dy, list):
        assert all(isinstance(v, str) for v in dy), "data_y 列表中每个元素应为字符串"


@_skip_no_key
def test_first_round_enums_valid():
    """首轮返回的 chart_type 和 style_theme 应在合法枚举内。"""
    from schema import CHART_TYPES, STYLE_THEMES
    result = generate_spec("画柱状图，vivid风格", _MOCK_DATA_CONTEXT)
    assert result.get("chart_type") in CHART_TYPES
    assert result.get("style_theme") in STYLE_THEMES


@_skip_no_key
def test_delta_round_contract():
    """修改轮应返回非空 dict，且只包含变更字段。"""
    result = generate_spec("换成ieee风格", _MOCK_DATA_CONTEXT, current_spec=_CURRENT_SPEC)
    assert isinstance(result, dict)
    assert len(result) > 0, "delta 不应为空 dict"


@_skip_no_key
def test_delta_round_style_change():
    """修改风格需求，返回的 delta 应包含 style_theme。"""
    result = generate_spec("换成ieee风格", _MOCK_DATA_CONTEXT, current_spec=_CURRENT_SPEC)
    assert "style_theme" in result, "风格修改的 delta 应包含 style_theme"
