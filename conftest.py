"""
pytest 配置：为所有测试注入 matplotlib 字体 fallback warning 的过滤器。
matplotlib 在使用中文字体 fallback 时会发出 "Glyph missing" UserWarning，
属正常行为，此处统一屏蔽，避免与 -W error::UserWarning 冲突。
"""

import warnings

import pytest


@pytest.fixture(autouse=True)
def suppress_mpl_font_warnings():
    warnings.filterwarnings("ignore", message=r"Glyph \d+ .* missing from font")
    yield
