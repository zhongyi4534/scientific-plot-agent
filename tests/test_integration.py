"""
tests/test_integration.py
端到端集成测试：用真实 API 跑通完整流水线。
有 DEEPSEEK_API_KEY 时发起真实调用；无 key 时跳过 API 测试。
"""

import os

import pytest

from system.agent import AgentResponse, PlotAgent

_HAS_KEY = bool(os.environ.get("DEEPSEEK_API_KEY"))
_skip_no_key = pytest.mark.skipif(not _HAS_KEY, reason="未设置 DEEPSEEK_API_KEY，跳过真实 API 测试")


@_skip_no_key
def test_full_pipeline_first_round():
    """首轮对话：load_data + process_input 应返回 status='ok'。"""
    agent = PlotAgent()
    agent.load_data("data/example_bar.csv")

    response = agent.process_input("画一张柱状图，对比各模型准确率，nature 风格")

    assert isinstance(response, AgentResponse)
    assert response.status == "ok", (
        f"期望 status='ok'，实际：{response.status}，"
        f"message={response.message}，question={response.question}"
    )
    assert response.image_path is not None, "ok 状态应包含 image_path"
    assert response.current_spec is not None, "ok 状态应包含 current_spec"


@_skip_no_key
def test_full_pipeline_delta_round():
    """两轮对话：第二轮应合并 delta 并返回 status='ok'。"""
    agent = PlotAgent()
    agent.load_data("data/example_bar.csv")

    r1 = agent.process_input("画一张柱状图，nature 风格")
    assert r1.status == "ok", f"首轮失败：{r1.message}"

    r2 = agent.process_input("换成 ieee 风格")
    assert r2.status == "ok", f"修改轮失败：{r2.message}"
    assert r2.current_spec is not None
    assert r2.current_spec.get("style_theme") == "ieee"


def test_reset_clears_state():
    """reset() 后状态应全部清空，不依赖 API。"""
    agent = PlotAgent()
    agent.current_spec = {"chart_type": "bar"}
    agent.current_cache_key = "cache://test"
    agent.data_context = "摘要"

    agent.reset()

    assert agent.current_spec is None
    assert agent.current_cache_key is None
    assert agent.data_context is None


def test_pipeline_no_key_returns_error():
    """未设置 API key 时，process_input 应返回 status='error' 而不是崩溃。"""
    if _HAS_KEY:
        pytest.skip("已设置 API key，跳过此测试")

    agent = PlotAgent()
    agent.load_data("data/example_bar.csv")
    response = agent.process_input("画一张折线图")

    assert response.status == "error"
    assert response.message is not None
