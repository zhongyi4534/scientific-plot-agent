"""
scripts/gen_supplement.py

针对训练数据内容缺陷补充高质量首轮 (user_input, plotspec) 配对：
  P1：rococo / macaron 主题样本严重不足（目标各补充至 15+ 条）
  P2：13 个字段从未在首轮出现（重点补充 7 个高价值字段，各 5-8 条）

调用 LLM 生成满足约束的样本，经四级校验后追加到 data/pairs/valid_pairs.jsonl。
运行完成后需重新执行 pack_finetune.py 重建 train/val.jsonl。

用法：
    python scripts/gen_supplement.py            # 默认：P1 + P2 全补
    python scripts/gen_supplement.py --p1-only
    python scripts/gen_supplement.py --p2-only
    python scripts/gen_supplement.py --dry-run  # 只打印场景列表，不调用 LLM
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

from model.prompts import SYSTEM_FIRST_FINETUNE
from scripts.validate_pairs import validate_one
from tools.loader import DataLoadError, load_data

load_dotenv(Path(__file__).parent.parent / ".env")

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

PAIRS_DIR = Path("data/pairs")
TRAIN_DIR = Path("data/train")
VALID_PAIRS_PATH = PAIRS_DIR / "valid_pairs.jsonl"

MAX_RETRIES = 3
RETRY_DELAY = 5.0
_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
_DEFAULT_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

_RNG = random.Random(99)

# 主题中文描述
_THEME_DESC: dict[str, str] = {
    "rococo":  "洛可可低对比淡雅色调，清淡柔和",
    "macaron": "马卡龙轻快糖果色，明亮活泼",
}


# ---------------------------------------------------------------------------
# 场景定义
# ---------------------------------------------------------------------------

@dataclass
class P1Scenario:
    theme: str
    chart_type: str
    n_target: int


@dataclass
class P2Scenario:
    field: str
    chart_types: list[str]
    n_target: int
    field_desc: str
    value_example: str


P1_SCENARIOS: list[P1Scenario] = [
    P1Scenario("rococo", "bar",     5),
    P1Scenario("rococo", "line",    4),
    P1Scenario("rococo", "scatter", 3),
    P1Scenario("rococo", "heatmap", 3),
    P1Scenario("macaron", "bar",     5),
    P1Scenario("macaron", "heatmap", 5),
    P1Scenario("macaron", "box",     4),
    P1Scenario("macaron", "line",    4),
    P1Scenario("macaron", "scatter", 3),
]

P2_SCENARIOS: list[P2Scenario] = [
    P2Scenario(
        field="style_hatch",
        chart_types=["bar"],
        n_target=8,
        field_desc="柱子纹理填充，字符串如\"/\"或列表[\"/\",\"\\\\\",\"|\"]，黑白打印时用于区分不同分组",
        value_example='示例值："/" 或 ["/", "\\\\", "|"]（注意JSON中反斜杠需写成"\\\\"）',
    ),
    P2Scenario(
        field="style_font_size",
        chart_types=["bar", "line", "scatter", "box", "heatmap"],
        n_target=6,
        field_desc="字体大小（整数磅），覆盖主题默认字号，常见值12或14",
        value_example='示例值：14',
    ),
    P2Scenario(
        field="style_figure_width",
        chart_types=["bar", "line", "scatter", "box", "heatmap"],
        n_target=5,
        field_desc="图幅最小宽度（英寸），适合论文中精确控制图幅大小",
        value_example='示例值：7.0 或 8.0',
    ),
    P2Scenario(
        field="style_spines",
        chart_types=["bar", "line", "scatter", "box"],
        n_target=5,
        field_desc="保留的轴脊列表，去掉上边框和右边框只保留左轴和下轴",
        value_example='示例值：["left","bottom"]',
    ),
    P2Scenario(
        field="style_line_width",
        chart_types=["line", "scatter", "box"],
        n_target=5,
        field_desc="线宽（磅），加粗线条用于投影展示或期刊要求",
        value_example='示例值：2.5',
    ),
    P2Scenario(
        field="params_marker_size",
        chart_types=["line", "scatter"],
        n_target=5,
        field_desc="数据点标记大小（数值），null时按数据密度自动计算",
        value_example='示例值：4 或 6',
    ),
    P2Scenario(
        field="style_dpi",
        chart_types=["bar", "line", "scatter", "box", "heatmap"],
        n_target=4,
        field_desc="输出分辨率（dpi），期刊通常要求300 dpi",
        value_example='示例值：300',
    ),
]


# ---------------------------------------------------------------------------
# LLM 调用
# ---------------------------------------------------------------------------

def _get_client() -> tuple[OpenAI, str]:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("未找到 DEEPSEEK_API_KEY 环境变量，请在 .env 中配置")
    return OpenAI(api_key=api_key, base_url=_DEEPSEEK_BASE_URL), _DEFAULT_MODEL


def _build_p1_system(theme: str, chart_type: str) -> str:
    theme_desc = _THEME_DESC.get(theme, theme)
    return (
        f"{SYSTEM_FIRST_FINETUNE}\n\n"
        f"⚠️ 本次生成的额外约束（优先级高于上述规则）：\n"
        f'- chart_type 必须是："{chart_type}"\n'
        f'- style_theme 必须是："{theme}"（{theme_desc}）\n'
        f"- user_input 要自然地提到希望使用该风格，长度 15-70 字，用中文\n"
        f'- 输出格式：{{"user_input": "...", "plotspec": {{...完整PlotSpec，不含data_source...}}}}\n'
        f"- 只输出 JSON 对象，无任何解释文字"
    )


def _build_p2_system(scenario: P2Scenario, chart_type: str) -> str:
    return (
        f"{SYSTEM_FIRST_FINETUNE}\n\n"
        f"⚠️ 本次生成的额外约束（优先级高于上述规则）：\n"
        f'- chart_type 选择："{chart_type}"\n'
        f'- plotspec 中必须包含字段 "{scenario.field}"（{scenario.field_desc}）\n'
        f"- {scenario.value_example}\n"
        f"- user_input 要自然地描述需要 {scenario.field} 的使用场景，长度 20-80 字，用中文\n"
        f'- 输出格式：{{"user_input": "...", "plotspec": {{...完整PlotSpec，不含data_source...}}}}\n'
        f"- 只输出 JSON 对象，无任何解释文字"
    )


def _build_user_message(data_context: str) -> str:
    return (
        f"{data_context}\n\n"
        "请根据上述数据生成一条 (user_input, plotspec) 配对样本。\n"
        "输出：/no_think"
    )


def _call_llm(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
) -> dict | None:
    """调用 LLM，返回解析后的 dict（含 user_input 和 plotspec），失败返回 None。"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.85,
                max_tokens=1500,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            raw = resp.choices[0].message.content.strip()
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "user_input" in parsed and "plotspec" in parsed:
                return parsed
            print(f"    [格式错误] 响应缺少 user_input 或 plotspec 字段")
            return None
        except RateLimitError:
            wait = RETRY_DELAY * attempt
            print(f"    [限速] 等待 {wait:.0f}s 后重试（{attempt}/{MAX_RETRIES}）...")
            time.sleep(wait)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"    [解析失败 {attempt}/{MAX_RETRIES}] {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"    [API错误 {attempt}/{MAX_RETRIES}] {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    return None


# ---------------------------------------------------------------------------
# 场景执行逻辑
# ---------------------------------------------------------------------------

def _get_all_csvs() -> list[Path]:
    csvs = sorted(TRAIN_DIR.glob("*.csv"))
    _RNG.shuffle(csvs)
    return csvs


def _try_generate_one(
    client: OpenAI,
    model: str,
    system: str,
    csv_path: Path,
    do_render: bool,
    scenario_id: str,
    required_field: Optional[str] = None,
) -> dict | None:
    """
    对给定 CSV 调用 LLM 生成一条样本，经四级校验后返回 enriched 记录。
    若指定 required_field，则验证该字段确实出现在生成的 plotspec 中。
    校验失败或 LLM 调用失败时返回 None。
    """
    try:
        data_context, _ = load_data(str(csv_path))
    except DataLoadError as e:
        print(f"      load_data 失败：{e}")
        return None

    user_msg = _build_user_message(data_context)
    result = _call_llm(client, model, system, user_msg)
    if result is None:
        return None

    user_input = result.get("user_input", "").strip()
    plotspec = result.get("plotspec", {})
    if not user_input or not isinstance(plotspec, dict):
        print(f"      LLM 输出格式无效")
        return None

    # P2 专用：验证 LLM 确实输出了目标字段
    if required_field and required_field not in plotspec:
        print(f"      LLM 未输出目标字段 {required_field}，跳过")
        return None

    record = {
        "id": scenario_id,
        "record_type": "first",
        "csv_path": str(csv_path),
        "user_input": user_input,
        "plotspec": plotspec,
    }

    ok, reason, enriched = validate_one(record, do_render=do_render)
    if not ok:
        print(f"      校验失败：{reason}")
        return None

    return enriched


def _run_scenario(
    client: OpenAI,
    model: str,
    system: str,
    n_target: int,
    scenario_label: str,
    fout,
    do_render: bool,
    required_field: Optional[str] = None,
) -> int:
    """
    对一个场景反复尝试不同 CSV，直到写出 n_target 条有效样本。
    required_field 不为 None 时，每次生成后额外验证该字段确实出现在 plotspec 中。
    返回实际写出条数。
    """
    csvs = _get_all_csvs()
    written = 0
    tried = 0

    for csv_path in csvs:
        if written >= n_target:
            break
        tried += 1
        scenario_id = f"supp_{scenario_label}_{tried}"
        print(f"    尝试 {csv_path.name} ...")
        enriched = _try_generate_one(
            client, model, system, csv_path, do_render, scenario_id,
            required_field=required_field,
        )
        if enriched is not None:
            fout.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1
            print(f"    ✓ 写入第 {written}/{n_target} 条")

    if written < n_target:
        print(f"    ⚠ CSV 已用完，只写出 {written}/{n_target} 条")

    return written


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

def run(
    do_p1: bool,
    do_p2: bool,
    dry_run: bool,
    model: str,
    do_render: bool,
) -> None:
    if dry_run:
        print("=== dry-run 模式，只打印场景列表 ===\n")
        if do_p1:
            print("P1 场景：")
            for s in P1_SCENARIOS:
                print(f"  theme={s.theme} chart_type={s.chart_type} target={s.n_target}")
        if do_p2:
            print("\nP2 场景：")
            for s in P2_SCENARIOS:
                print(f"  field={s.field} charts={s.chart_types} target={s.n_target}")
        return

    client, model = _get_client()

    total_written = 0
    with VALID_PAIRS_PATH.open("a", encoding="utf-8") as fout:

        # ── P1：低频主题补充 ───────────────────────────────────────────
        if do_p1:
            print("\n===== P1：低频主题补充（rococo / macaron）=====\n")
            for s in P1_SCENARIOS:
                print(f"  场景：theme={s.theme}  chart_type={s.chart_type}  target={s.n_target}")
                system = _build_p1_system(s.theme, s.chart_type)
                label = f"p1_{s.theme}_{s.chart_type}"
                n = _run_scenario(client, model, system, s.n_target, label, fout, do_render)
                total_written += n
                print()

        # ── P2：缺失字段补充 ────────────────────────────────────────────
        if do_p2:
            print("\n===== P2：缺失字段补充 =====\n")
            for s in P2_SCENARIOS:
                chart_type = _RNG.choice(s.chart_types)
                print(f"  场景：field={s.field}  chart_type={chart_type}  target={s.n_target}")
                system = _build_p2_system(s, chart_type)
                label = f"p2_{s.field}"
                n = _run_scenario(
                    client, model, system, s.n_target, label, fout, do_render,
                    required_field=s.field,
                )
                total_written += n
                print()

    print(f"\n补充完成：共写入 {total_written} 条新样本 → {VALID_PAIRS_PATH}")
    print("请运行 python scripts/pack_finetune.py 重建 train/val.jsonl")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="补充低频主题和缺失字段的训练数据")
    parser.add_argument("--p1-only",  action="store_true", help="只运行 P1（低频主题）")
    parser.add_argument("--p2-only",  action="store_true", help="只运行 P2（缺失字段）")
    parser.add_argument("--dry-run",  action="store_true", help="只打印场景列表，不调用 LLM")
    parser.add_argument("--no-render", action="store_true", help="跳过渲染校验（快速模式）")
    parser.add_argument(
        "--model", type=str, default=_DEFAULT_MODEL,
        help=f"DeepSeek 模型（默认：{_DEFAULT_MODEL}）",
    )
    args = parser.parse_args()

    do_p1 = not args.p2_only
    do_p2 = not args.p1_only

    run(
        do_p1=do_p1,
        do_p2=do_p2,
        dry_run=args.dry_run,
        model=args.model,
        do_render=not args.no_render,
    )


if __name__ == "__main__":
    main()
