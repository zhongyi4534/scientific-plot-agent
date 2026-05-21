"""
scripts/gen_delta.py

从 data/pairs/valid_pairs.jsonl 的首轮记录出发，生成修改轮 (user_input, delta) 训练数据。
输出到 data/pairs/delta_pairs.jsonl。

两种生成策略：
  1. 规则驱动（快速）：随机挑选一个简单字段变更，拼接中文请求模板
  2. LLM 生成（多样）：调用 DeepSeek API，给定当前 spec 后让模型想象各种合理的修改需求

用法：
    python scripts/gen_delta.py              # 默认：LLM + 规则，目标 200 条
    python scripts/gen_delta.py --target 400  # 目标 400 条
    python scripts/gen_delta.py --no-llm     # 只用规则驱动
    python scripts/gen_delta.py --no-rule    # 只用 LLM
    python scripts/gen_delta.py --limit 20   # 只使用前 20 条种子（测试模式）
    python scripts/gen_delta.py --append     # 追加到已有 delta_pairs.jsonl

环境变量：
    DEEPSEEK_API_KEY   DeepSeek API 密钥（--no-llm 时可不设）
    DEEPSEEK_MODEL     模型名称（默认 deepseek-v4-pro）
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.prompts import _FIELD_SPEC
from schema import (
    CHART_TYPES,
    OPTIONAL_DEFAULTS,
    PALETTE_OVERRIDES,
    REQUIRED_FIELDS,
    STYLE_THEMES,
)
from system.merger import fill_defaults, merge_delta
from system.validator import validate
from tools.loader import DataLoadError, _CACHE, load_data
from scripts.validate_pairs import _check_col_exists, _check_semantic_validity

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).parent.parent / ".env")

PAIRS_DIR = Path("data/pairs")
VALID_PAIRS_PATH      = PAIRS_DIR / "valid_pairs.jsonl"
DELTA_PAIRS_PATH      = PAIRS_DIR / "delta_pairs.jsonl"
DELTA_REJECT_LOG_PATH = PAIRS_DIR / "delta_reject_log.jsonl"

N_LLM_PER_SPEC = 4   # LLM 模式每条种子 spec 最多生成的候选对数
N_RULE_PER_SPEC = 1  # 规则模式每条种子 spec 最多采用的候选对数
MAX_PER_SEED = 5     # 每条种子最多写出的修改轮样本数
TARGET_DEFAULT = 200  # 默认目标总条数
MAX_CYCLES = 5        # 种子集最多循环几轮

MAX_RETRIES = 3
RETRY_DELAY = 5.0

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
_DEFAULT_MODEL = "deepseek-v4-pro"

# 允许出现在 delta 中的字段（排除 data_source）
_VALID_DELTA_FIELDS: frozenset[str] = frozenset(
    set(OPTIONAL_DEFAULTS.keys()) | {"chart_type", "data_x", "data_y", "style_theme"}
)

# 主题中文描述（用于规则模板）
_THEME_ZH: dict[str, str] = {
    "normal":  "简洁常规",
    "morandi": "莫兰迪低饱和",
    "macaron": "马卡龙轻快",
    "bright":  "高对比鲜艳",
    "rococo":  "洛可可淡色",
    "earth":   "大地色系",
    "science": "Science期刊冷色",
    "nature":  "Nature期刊暖色",
}

_RNG = random.Random(42)


def _pick(options: list) -> object:
    return _RNG.choice(options)


# ---------------------------------------------------------------------------
# 规则驱动生成
# ---------------------------------------------------------------------------

def _rule_candidates(
    spec: dict,
    df: pd.DataFrame,
) -> list[tuple[dict, str]]:
    """
    从当前 spec 生成简单规则变更候选，返回随机打乱的 (delta, user_input) 列表。
    调用者按需截取前 N_RULE_PER_SPEC 条通过校验的候选。
    """
    candidates: list[tuple[dict, str]] = []
    chart_type = spec.get("chart_type", "")
    current_theme = spec.get("style_theme", "")

    # 1. 主题切换
    other_themes = [t for t in STYLE_THEMES if t != current_theme]
    if other_themes:
        new_theme = _pick(other_themes)
        desc = _THEME_ZH.get(new_theme, new_theme)
        candidates.append((
            {"style_theme": new_theme},
            _pick([
                f"换成{desc}风格", f"改用{desc}主题", f"用{desc}配色方案",
                f"主题改为{desc}", f"切换到{desc}配色", f"配色方案换成{desc}",
            ]),
        ))

    # 2. 网格线切换
    current_grid = spec.get("style_grid")
    if current_grid is True:
        candidates.append((
            {"style_grid": False},
            _pick(["去掉网格线", "隐藏背景网格", "不显示网格", "把网格线关掉", "背景网格不要了", "取消网格显示"]),
        ))
    else:
        candidates.append((
            {"style_grid": True},
            _pick(["加上网格线", "显示网格线", "背景加网格", "把网格线打开", "需要背景网格辅助读数", "加一下参考网格线"]),
        ))

    # 3. 标题设置 / 清除
    current_title = spec.get("label_title") or ""
    if current_title:
        candidates.append((
            {"label_title": None},
            _pick(["删掉标题", "去掉图表标题", "标题留空", "不需要标题了", "把标题去掉", "标题删了吧"]),
        ))
    else:
        y_col = spec.get("data_y", "")
        title = f"{y_col} 对比分析" if isinstance(y_col, str) else "实验结果对比"
        candidates.append((
            {"label_title": title},
            _pick([
                f'加一个标题"{title}"', f'标题设为"{title}"', f"给图加标题：{title}",
                f'图表标题改成"{title}"', f"标题写{title}",
            ]),
        ))

    # 4. Y轴标签
    current_label_y = spec.get("label_y") or ""
    if not current_label_y:
        y_col = spec.get("data_y", "")
        label_y = str(y_col) if isinstance(y_col, str) else "数值"
        candidates.append((
            {"label_y": label_y},
            _pick([
                f'Y轴标签设为"{label_y}"', f'Y轴加标注"{label_y}"', f"把Y轴标签改为{label_y}",
                f"Y轴标题写{label_y}", f"Y轴描述加上{label_y}",
            ]),
        ))
    else:
        candidates.append((
            {"label_y": None},
            _pick(["去掉Y轴标签", "Y轴标签留空", "Y轴不要标注了", "删掉Y轴标题", "Y轴标签清空"]),
        ))

    # 5. Y轴范围（需要数值型 Y 列）
    y_col = spec.get("data_y")
    if isinstance(y_col, str) and y_col in df.columns and pd.api.types.is_numeric_dtype(df[y_col]):
        col_min = float(df[y_col].min())
        col_max = float(df[y_col].max())
        span = col_max - col_min
        if span > 0:
            nice_min = round(col_min - span * 0.05, 2)
            candidates.append((
                {"axes_y_min": nice_min},
                _pick([
                    f"Y轴从{nice_min}开始", f"Y轴下限设为{nice_min}", f"Y轴最小值改为{nice_min}",
                    f"纵轴起始值设成{nice_min}", f"Y轴底部从{nice_min}截断",
                ]),
            ))

    # 6. X轴刻度旋转
    current_rot = spec.get("axes_x_tick_rotation", 0)
    if current_rot == 0:
        angle = _pick([30, 45])
        candidates.append((
            {"axes_x_tick_rotation": angle},
            _pick([
                f"X轴标签旋转{angle}度", f"X轴刻度旋转{angle}°", f"把X轴标注斜{angle}度",
                f"X轴文字旋转{angle}度方便阅读", f"横轴刻度标签倾斜{angle}度",
            ]),
        ))
    else:
        candidates.append((
            {"axes_x_tick_rotation": 0},
            _pick(["X轴标签恢复水平", "把X轴刻度旋转角度去掉", "X轴标签改回水平", "取消X轴旋转", "横轴标签放平"]),
        ))

    # 7. 图例位置
    current_loc = spec.get("legend_loc")
    if current_loc != "outside_right":
        candidates.append((
            {"legend_loc": "outside_right"},
            _pick([
                "图例放到图外右侧", "把图例移到右侧图外", "legend放在图的外部右方",
                "图例挪到图的右边外侧", "legend放图外右边", "把图例放到右侧外部",
            ]),
        ))
    if current_loc != "none":
        candidates.append((
            {"legend_loc": "none"},
            _pick([
                "隐藏图例", "不显示图例", "去掉legend",
                "把图例去掉", "legend不要了", "图例隐藏掉",
            ]),
        ))
    if current_loc in ("outside_right", "none"):
        candidates.append((
            {"legend_loc": None},
            _pick(["图例放回图内", "图例恢复默认位置", "legend放回图里", "图例移回图内", "恢复图例默认"]),
        ))

    # 8. 字体大小
    current_fs = spec.get("style_font_size")
    if current_fs is None or int(current_fs) <= 10:
        new_fs = _pick([12, 14])
        candidates.append((
            {"style_font_size": new_fs},
            _pick([
                f"字号调大到{new_fs}", f"把字体改成{new_fs}号", f"字体大小设为{new_fs}",
                f"文字大小改为{new_fs}pt", f"所有字号调整为{new_fs}",
            ]),
        ))
    else:
        candidates.append((
            {"style_font_size": None},
            _pick(["字号恢复默认", "字体大小重置为主题默认", "字体大小恢复原来的", "字号恢复主题设置", "取消字号修改"]),
        ))

    # 9. 对数坐标
    current_scale = spec.get("axes_y_scale", "linear")
    if current_scale == "linear":
        candidates.append((
            {"axes_y_scale": "log"},
            _pick([
                "Y轴换成对数刻度", "Y轴用log坐标", "改成对数坐标轴",
                "纵轴改为对数尺度", "Y轴切换到log scale", "用对数坐标显示Y轴",
            ]),
        ))
    else:
        candidates.append((
            {"axes_y_scale": "linear"},
            _pick(["Y轴恢复线性刻度", "取消对数坐标", "Y轴换回线性刻度", "恢复线性Y轴", "纵轴改回普通线性坐标"]),
        ))

    # 10. 配色方案
    candidates.append((
        {"style_palette_override": _pick(PALETTE_OVERRIDES)},
        _pick([
            "换一套配色方案", "换成tab10配色", "颜色配色改成莫兰迪色系", "换个更鲜明的颜色方案",
            "配色换成nature_d", "改用coolwarm色板", "用不同的颜色主题", "换个配色试试",
        ]),
    ))

    # 11. 图表专属参数
    if chart_type == "bar":
        if not spec.get("params_sort"):
            candidates.append((
                {"params_sort": "desc"},
                _pick([
                    "柱子按Y值从大到小排序", "按数值降序排列柱子", "排一下顺序，最高的放左边",
                    "柱状图按高低排列", "从高到低排列各组柱子",
                ]),
            ))
        else:
            candidates.append((
                {"params_sort": None},
                _pick(["取消排序", "柱子恢复原始顺序", "去掉排序", "恢复原来的柱子顺序", "不需要排序了"]),
            ))
        show_val = spec.get("params_show_values", False)
        candidates.append((
            {"params_show_values": not show_val},
            _pick(["柱顶显示数值", "把每根柱子的数值标出来", "每根柱上标注具体数字", "柱子上面加数值标签"]) if not show_val
            else _pick(["去掉柱顶数值", "不显示柱子数值标注", "柱上数字标注去掉", "取消柱子数值显示", "数值标注不要了"]),
        ))

    elif chart_type == "line":
        smooth = spec.get("params_smooth", False)
        if not smooth:
            candidates.append((
                {"params_smooth": True},
                _pick(["折线改成平滑曲线", "加平滑插值", "线条平滑一下", "用平滑样条曲线", "折线平滑处理一下"]),
            ))
        else:
            candidates.append((
                {"params_smooth": False},
                _pick(["取消平滑，改回折线", "直接连线，不用平滑", "去掉平滑效果", "恢复折线", "不要平滑插值了"]),
            ))
        current_ls = spec.get("params_linestyle", "solid")
        if current_ls == "solid":
            new_ls = _pick(["dashed", "dotted", "dashdot"])
            candidates.append((
                {"params_linestyle": new_ls},
                _pick(["线型改成虚线", "用点划线", "改用虚线样式", "线条换成虚线", "线型改为虚线看起来更清晰"]),
            ))
        else:
            candidates.append((
                {"params_linestyle": "solid"},
                _pick(["线型恢复实线", "换回实线", "线型改回实线", "恢复实线样式", "用实线连接数据点"]),
            ))
        show_markers = spec.get("params_show_markers", True)
        candidates.append((
            {"params_show_markers": not show_markers},
            _pick(["不显示数据点标记", "去掉线上的标记点", "隐藏折线上的数据点", "标记点不要了", "取消数据点标注"]) if show_markers
            else _pick(["加上数据点标记", "在数据点上显示标记", "给每个数据点加标记", "折线上标出数据点", "显示数据点marker"]),
        ))

    elif chart_type == "scatter":
        alpha = spec.get("params_alpha", 0.8)
        new_alpha = 0.5 if float(alpha) > 0.6 else 0.9
        candidates.append((
            {"params_alpha": new_alpha},
            _pick([
                f"透明度调整为{new_alpha}", f"点的透明度改为{new_alpha}",
                f"散点透明度设为{new_alpha}", f"点透明度改成{new_alpha}", f"alpha值设为{new_alpha}",
            ]),
        ))
        show_reg = spec.get("params_show_regression", False)
        x_col = spec.get("data_x", "")
        x_is_numeric = (
            isinstance(x_col, str)
            and x_col in df.columns
            and pd.api.types.is_numeric_dtype(df[x_col])
        )
        if not show_reg and x_is_numeric:
            candidates.append((
                {"params_show_regression": True},
                _pick(["加一条回归线", "拟合回归直线", "添加线性回归", "画出回归拟合线", "加上线性趋势线"]),
            ))
        elif show_reg:
            candidates.append((
                {"params_show_regression": False},
                _pick(["去掉回归线", "不显示拟合线", "删掉回归拟合线", "取消回归线", "回归线不要了"]),
            ))

    elif chart_type == "box":
        show_pts = spec.get("params_show_points", "outliers")
        if show_pts != "all":
            candidates.append((
                {"params_show_points": "all"},
                _pick(["显示所有数据点", "把散点全画出来", "除箱线外也显示原始数据点", "每个数据点都画出来", "jitter点全部显示"]),
            ))
        if show_pts != "none":
            candidates.append((
                {"params_show_points": "none"},
                _pick(["不显示任何数据点", "隐藏散点，只保留箱线", "散点全部隐藏", "只显示箱线图轮廓", "数据点都隐藏掉"]),
            ))
        notch = spec.get("params_notch", False)
        candidates.append((
            {"params_notch": not notch},
            _pick(["换成缺口箱线图", "加上缺口notch", "改为notch箱线图", "箱线图加缺口", "使用带notch的箱线图"]) if not notch
            else _pick(["取消缺口箱线图", "恢复普通箱线图", "去掉notch缺口", "改回标准箱线图", "取消箱线图notch"]),
        ))

    elif chart_type == "heatmap":
        annot = spec.get("params_annot", True)
        candidates.append((
            {"params_annot": not annot},
            _pick(["热力图格子里不显示数值", "隐藏数值标注", "格子里数字去掉", "取消热力图数值显示", "不要在格子里标数字"]) if annot
            else _pick(["热力图格子显示数值", "加上数值标注", "格子里显示具体数值", "每个格子标注数值", "在热力图上显示数字"]),
        ))
        current_fmt = spec.get("params_annot_fmt", ".2f")
        if current_fmt == ".2f":
            candidates.append((
                {"params_annot_fmt": ".1f"},
                _pick(["数值保留一位小数", "精度改为1位小数", "小数位数改为1位", "数字精度调整为一位小数", "改成保留一位小数"]),
            ))
        else:
            candidates.append((
                {"params_annot_fmt": ".2f"},
                _pick(["数值保留两位小数", "精度改为2位小数", "改为2位小数精度", "数字保留两位小数", "小数位数改为2位"]),
            ))

    _RNG.shuffle(candidates)
    return candidates


# ---------------------------------------------------------------------------
# LLM 生成
# ---------------------------------------------------------------------------

def _build_delta_system_prompt() -> str:
    return f"""\
你是科研绘图训练数据生成专家，任务是为修改轮生成 (user_input, delta) 配对样本。

给定一个已渲染成功的 PlotSpec（当前图表配置）和对应的数据摘要，你需要想象用户看到图后可能
提出的修改需求，并输出对应的 delta JSON（只含需修改的字段）。

=== 字段参考 ===

{_FIELD_SPEC}

=== delta 格式规则 ===
- delta 只包含需要改变的字段，未变更的字段一律不输出
- 若需将字段恢复为默认值（如删除标题、取消过滤、关闭误差棒），将该字段值设为 null
- data_source 字段永远不能出现在 delta 中
- delta 不能是空字典（必须至少包含一个字段）

=== 多样性要求（{N_LLM_PER_SPEC} 个样本整体满足）===
1. 修改类型多样：覆盖至少 3 类——样式修改（主题/配色/字体/网格）、标注修改（标题/轴标签/范围）、参数修改（图表专属参数）
2. 语气多样：口语化（1-2条，如"换个深色背景""排个序"）和学术正式（1-2条，明确指定参数）
3. delta 字段数量多样：单字段变更（1-2条）和多字段同时变更（1-2条）都要有
4. user_input 用中文，15-100字，禁止使用重复的句式开头

=== 额外约束 ===
- scatter + 类别型 data_x（如 model/method/dataset）：delta 中禁止出现 params_show_regression: true
- line + 类别型 data_x：delta 中禁止出现 params_smooth: true
- box 图：delta 中的 data_y 禁止是列表

=== 输出格式 ===
只输出合法 JSON 数组，不含解释文字：
[
  {{
    "user_input": "用户的中文修改请求",
    "delta": {{"field1": value1, "field2": value2}}
  }},
  ...
]"""


def _build_delta_user_message(
    data_context: str,
    current_spec: dict,
    csv_name: str,
) -> str:
    spec_json = json.dumps(current_spec, ensure_ascii=False, separators=(",", ":"))
    return (
        f"数据文件：{csv_name}\n"
        f"数据摘要：\n{data_context}\n\n"
        f"当前PlotSpec：{spec_json}\n\n"
        f"请生成 {N_LLM_PER_SPEC} 个多样化的修改请求配对样本。"
    )


def _get_client(model: str) -> tuple[OpenAI, str]:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未找到 DEEPSEEK_API_KEY 环境变量。请设置：\n"
            "  $env:DEEPSEEK_API_KEY='your-key'  (PowerShell)\n"
            "  export DEEPSEEK_API_KEY='your-key'  (bash)"
        )
    client = OpenAI(api_key=api_key, base_url=_DEEPSEEK_BASE_URL)
    return client, model


def _call_api_for_deltas(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
) -> list[dict] | None:
    """调用 API 获取 delta 样本列表。全部失败返回 None。"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.8,
                max_tokens=3000,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            raw = resp.choices[0].message.content.strip()
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
            for v in parsed.values():
                if isinstance(v, list):
                    return v
            raise ValueError(f"无法从响应中提取 list，原始：{raw[:200]}")
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
# delta 校验
# ---------------------------------------------------------------------------

def _validate_delta(
    delta: dict,
    current_spec: dict,
    cache_key: str,
    df: pd.DataFrame,
) -> str | None:
    """
    校验一条 delta 的合法性。
    返回 None 表示通过，返回错误描述字符串表示失败。
    """
    if not isinstance(delta, dict) or not delta:
        return "delta 不是非空 dict"

    if "data_source" in delta:
        return "delta 含有 data_source（禁止）"

    unknown = set(delta.keys()) - _VALID_DELTA_FIELDS
    if unknown:
        return f"delta 含非法字段：{unknown}"

    # 确认至少一个字段实际改变
    changed = False
    for k, v in delta.items():
        old_v = current_spec.get(k, OPTIONAL_DEFAULTS.get(k))
        if v != old_v:
            changed = True
            break
    if not changed:
        return "delta 没有实际改变任何字段（no-op）"

    # 合并后完整校验
    merged = merge_delta(current_spec, delta)
    spec_with_source = fill_defaults({**merged, "data_source": cache_key})
    result = validate(spec_with_source)
    if not result.ok:
        parts = []
        if result.missing_required:
            parts.append(f"缺少必填字段：{result.missing_required}")
        if result.type_errors:
            parts.append(f"类型错误：{result.type_errors}")
        return "合并后 validate 失败：" + "；".join(parts)

    df_cols = set(df.columns)
    col_err = _check_col_exists(spec_with_source, df_cols)
    if col_err:
        return col_err

    semantic_err = _check_semantic_validity(spec_with_source, df)
    if semantic_err:
        return f"语义检查失败：{semantic_err}"

    return None


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

def generate_delta_pairs(
    target: int,
    model: str,
    source_limit: Optional[int],
    use_llm: bool,
    use_rule: bool,
    append: bool,
) -> None:
    """
    从 valid_pairs.jsonl 的首轮记录出发，生成修改轮训练数据，
    写入 DELTA_PAIRS_PATH。
    """
    if not VALID_PAIRS_PATH.exists():
        print(f"✗ 找不到 {VALID_PAIRS_PATH}，请先运行 validate_pairs.py")
        return

    # 加载首轮（first）记录作为种子
    all_seeds: list[dict] = []
    with VALID_PAIRS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if rec.get("record_type", "first") == "first":
                    all_seeds.append(rec)

    if not all_seeds:
        print("✗ valid_pairs.jsonl 中没有首轮（first）记录")
        return

    if source_limit:
        all_seeds = all_seeds[:source_limit]

    print(f"读取 {len(all_seeds)} 条首轮记录作为种子")
    print(f"目标：{target} 条修改轮样本  LLM={'开' if use_llm else '关'}  规则={'开' if use_rule else '关'}\n")

    llm_client: OpenAI | None = None
    if use_llm:
        try:
            llm_client, model = _get_client(model)
        except RuntimeError as e:
            print(f"⚠ LLM 初始化失败：{e}")
            print("  将回退到仅使用规则驱动生成")
            use_llm = False

    delta_system_prompt = _build_delta_system_prompt()
    written_count = 0
    total_rejected = 0
    seed_written: dict[str, int] = {}  # 每条种子已写出的样本数

    mode = "a" if append else "w"
    seed_records = list(all_seeds)
    _RNG.shuffle(seed_records)
    seed_idx = 0
    n_cycles = 0

    with (
        DELTA_PAIRS_PATH.open(mode, encoding="utf-8") as fout,
        DELTA_REJECT_LOG_PATH.open(mode, encoding="utf-8") as frej,
    ):
        while written_count < target:
            # 循环种子
            if seed_idx >= len(seed_records):
                # 若所有种子都已达写出上限，无需继续循环
                if all(seed_written.get(s.get("id", ""), 0) >= MAX_PER_SEED for s in seed_records):
                    remaining = target - written_count
                    print(f"  ⚠ 所有种子已达上限（{MAX_PER_SEED} 条/种子），仍差 {remaining} 条，停止")
                    break
                n_cycles += 1
                if n_cycles >= MAX_CYCLES:
                    remaining = target - written_count
                    print(f"  ⚠ 已循环 {n_cycles} 轮种子，仍差 {remaining} 条，停止")
                    break
                seed_idx = 0
                _RNG.shuffle(seed_records)

            source = seed_records[seed_idx]
            seed_idx += 1
            source_id = source.get("id", f"seed_{seed_idx}")

            # 跳过已达上限的种子
            if seed_written.get(source_id, 0) >= MAX_PER_SEED:
                continue
            csv_path = source.get("csv_path", "")
            current_spec = {
                k: v for k, v in source.get("plotspec", {}).items()
                if k != "data_source"
            }
            data_context = source.get("data_context", "")

            # 加载 CSV
            try:
                _, cache_key = load_data(csv_path)
            except DataLoadError as e:
                print(f"  ⚠ {source_id}: load_data 失败：{e}")
                continue
            df = _CACHE[cache_key]

            round_candidates: list[tuple[dict, str]] = []

            # ── 规则驱动 ──────────────────────────────────────────────
            if use_rule:
                rule_cands = _rule_candidates(current_spec, df)
                accepted = 0
                for delta, user_input in rule_cands:
                    if accepted >= N_RULE_PER_SPEC:
                        break
                    err = _validate_delta(delta, current_spec, cache_key, df)
                    if err is None:
                        round_candidates.append((delta, user_input))
                        accepted += 1
                    else:
                        frej.write(json.dumps({
                            "record_type": "delta_reject",
                            "source_id": source_id,
                            "user_input": user_input,
                            "delta": delta,
                            "reject_reason": err,
                        }, ensure_ascii=False) + "\n")
                        total_rejected += 1

            # ── LLM 生成 ──────────────────────────────────────────────
            if use_llm and llm_client is not None:
                csv_name = Path(csv_path).name if csv_path else "unknown.csv"
                user_msg = _build_delta_user_message(data_context, current_spec, csv_name)
                pairs = _call_api_for_deltas(llm_client, model, delta_system_prompt, user_msg)
                if pairs is None:
                    print(f"  ⚠ {source_id}: LLM API 全部失败，跳过 LLM 部分")
                else:
                    for pair in pairs:
                        if not isinstance(pair, dict):
                            frej.write(json.dumps({
                                "record_type": "delta_reject",
                                "source_id": source_id,
                                "user_input": "",
                                "delta": {},
                                "reject_reason": f"LLM返回格式非dict：{str(pair)[:120]}",
                            }, ensure_ascii=False) + "\n")
                            total_rejected += 1
                            continue
                        delta = pair.get("delta", {})
                        user_input = pair.get("user_input", "")
                        if not user_input or not isinstance(delta, dict):
                            frej.write(json.dumps({
                                "record_type": "delta_reject",
                                "source_id": source_id,
                                "user_input": user_input,
                                "delta": delta if isinstance(delta, dict) else {},
                                "reject_reason": "缺少 user_input 或 delta 字段",
                            }, ensure_ascii=False) + "\n")
                            total_rejected += 1
                            continue
                        err = _validate_delta(delta, current_spec, cache_key, df)
                        if err is None:
                            round_candidates.append((delta, user_input))
                        else:
                            frej.write(json.dumps({
                                "record_type": "delta_reject",
                                "source_id": source_id,
                                "user_input": user_input,
                                "delta": delta,
                                "reject_reason": err,
                            }, ensure_ascii=False) + "\n")
                            total_rejected += 1

            # ── 写出本轮有效样本 ────────────────────────────────────
            for delta, user_input in round_candidates:
                if written_count >= target:
                    break
                if seed_written.get(source_id, 0) >= MAX_PER_SEED:
                    break
                record = {
                    "id": f"{source_id}_d{written_count}",
                    "record_type": "delta",
                    "csv_path": csv_path,
                    "user_input": user_input,
                    "current_spec": current_spec,
                    "plotspec": delta,
                    "data_context": data_context,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                written_count += 1
                seed_written[source_id] = seed_written.get(source_id, 0) + 1

            if written_count > 0 and written_count % 20 == 0:
                print(f"  进度：{written_count}/{target} 条已写入 ...")

    print(f"\n完成：写入 {written_count} 条修改轮样本，{total_rejected} 条被拒绝")
    print(f"输出：{DELTA_PAIRS_PATH}")
    if total_rejected > 0:
        print(f"拒绝日志：{DELTA_REJECT_LOG_PATH}（{total_rejected} 条）")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="生成修改轮 (user_input, delta) 训练数据")
    parser.add_argument(
        "--target", type=int, default=TARGET_DEFAULT,
        help=f"目标生成条数（默认 {TARGET_DEFAULT}）",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="只使用前 N 条 valid_pairs 作为种子（测试模式）",
    )
    parser.add_argument(
        "--model", type=str,
        default=os.environ.get("DEEPSEEK_MODEL", _DEFAULT_MODEL),
        help=f"DeepSeek 模型名称（默认：{_DEFAULT_MODEL}）",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="跳过 LLM 生成，只用规则驱动",
    )
    parser.add_argument(
        "--no-rule", action="store_true",
        help="跳过规则驱动，只用 LLM",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="追加到已有的 delta_pairs.jsonl，而非覆盖",
    )
    args = parser.parse_args()

    if args.no_llm and args.no_rule:
        print("✗ --no-llm 和 --no-rule 不能同时指定")
        return

    generate_delta_pairs(
        target=args.target,
        model=args.model,
        source_limit=args.limit,
        use_llm=not args.no_llm,
        use_rule=not args.no_rule,
        append=args.append,
    )


if __name__ == "__main__":
    main()
