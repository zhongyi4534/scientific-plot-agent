"""
scripts/gen_large_data.py

生成两个规模较大的训练 CSV，并为它们生成训练数据对（追加到 manual_pairs.jsonl）。

  data/train/s45_llm_benchmark.csv   (240行 6列)
      30个LLM × 8基准测试，包含参数量、发布年份
      适合: data_filter柱状图/散点图、热力图、分组图

  data/train/s46_hparam_search.csv   (180行 7列)
      NLP超参数搜索实验日志（5架构×3优化器×4学习率×部分组合）
      适合: 箱线图、散点图、带data_filter的柱状图、ask_user(val_acc vs val_f1)

用法：
    python scripts/gen_large_data.py              # 生成 CSV + 训练对
    python scripts/gen_large_data.py --csv-only   # 仅生成 CSV
    python scripts/gen_large_data.py --pairs-only # 仅生成训练对（CSV 必须已存在）
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.loader import load_data

OUT_DIR = Path("data/train")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MANUAL_PAIRS_PATH = Path("data/pairs/manual_pairs.jsonl")
S45_CSV = "data/train/s45_llm_benchmark.csv"
S46_CSV = "data/train/s46_hparam_search.csv"


# ---------------------------------------------------------------------------
# CSV 生成
# ---------------------------------------------------------------------------

def gen_s45() -> None:
    """30个LLM × 8基准 = 240行"""

    MODELS: list[tuple] = [
        # GPT 系列
        ("GPT-3.5-turbo",   "GPT",      175.0,  2022, [70.0, 85.5, 47.0, 57.1, 48.1, 85.2, 34.1, 70.1]),
        ("GPT-4",           "GPT",     1760.0,  2023, [86.4, 95.3, 59.0, 92.0, 67.0, 96.3, 52.6, 86.7]),
        ("GPT-4o",          "GPT",      200.0,  2024, [88.7, 95.8, 65.4, 95.9, 90.2, 97.0, 76.6, 90.2]),
        # Claude 系列
        ("Claude-2",        "Claude",    70.0,  2023, [78.5, 91.2, 54.3, 79.6, 47.0, 90.5, 32.5, 78.2]),
        ("Claude-3-Haiku",  "Claude",    20.0,  2024, [75.2, 88.0, 56.1, 76.9, 40.0, 88.1, 38.9, 73.5]),
        ("Claude-3-Sonnet", "Claude",    70.0,  2024, [86.7, 93.5, 59.8, 92.3, 73.0, 93.2, 40.5, 82.9]),
        ("Claude-3-Opus",   "Claude",    70.0,  2024, [88.2, 95.4, 61.2, 95.0, 84.9, 96.4, 60.1, 86.8]),
        # Gemini 系列
        ("Gemini-Pro",      "Gemini",    37.0,  2023, [71.8, 84.7, 53.0, 74.4, 32.5, 82.3, 28.6, 75.0]),
        ("Gemini-1.5-Pro",  "Gemini",   340.0,  2024, [81.9, 92.5, 60.1, 90.7, 71.9, 92.1, 58.5, 85.3]),
        ("Gemini-Ultra",    "Gemini",  1700.0,  2024, [90.0, 97.7, 61.9, 94.4, 74.4, 97.3, 53.2, 89.7]),
        # Llama-2 系列
        ("Llama-2-7B",      "Llama-2",   7.0,  2023, [45.3, 77.2, 29.3, 13.5, 12.8, 61.0,  2.5, 35.2]),
        ("Llama-2-13B",     "Llama-2",  13.0,  2023, [54.8, 80.7, 36.8, 28.7, 18.3, 67.3,  7.0, 39.4]),
        ("Llama-2-70B",     "Llama-2",  70.0,  2023, [68.9, 87.3, 44.8, 56.8, 29.9, 79.8, 13.5, 58.1]),
        # Llama-3 系列
        ("Llama-3-8B",      "Llama-3",   8.0,  2024, [66.6, 82.0, 43.9, 79.6, 62.2, 78.6, 29.8, 62.7]),
        ("Llama-3-70B",     "Llama-3",  70.0,  2024, [82.0, 93.1, 52.8, 93.0, 81.7, 92.9, 50.4, 81.3]),
        ("Llama-3.1-405B",  "Llama-3", 405.0,  2024, [88.6, 95.0, 56.8, 96.8, 89.0, 96.1, 73.8, 88.5]),
        # Mistral 系列
        ("Mistral-7B",      "Mistral",   7.0,  2023, [62.5, 81.3, 42.1, 52.2, 30.5, 72.3, 11.0, 56.3]),
        ("Mixtral-8x7B",    "Mistral",  47.0,  2024, [70.6, 89.1, 48.5, 74.4, 40.2, 85.4, 28.4, 68.0]),
        ("Mixtral-8x22B",   "Mistral", 141.0,  2024, [77.8, 91.8, 51.2, 84.6, 59.0, 90.1, 41.8, 76.9]),
        # Qwen 系列
        ("Qwen1.5-7B",      "Qwen",      7.0,  2024, [61.0, 78.5, 41.0, 62.5, 36.0, 74.0, 20.3, 52.0]),
        ("Qwen1.5-72B",     "Qwen",     72.0,  2024, [77.2, 91.3, 54.0, 85.7, 60.0, 89.3, 40.5, 74.4]),
        ("Qwen2-72B",       "Qwen",     72.0,  2024, [84.2, 94.2, 57.5, 91.1, 64.6, 93.3, 51.1, 82.4]),
        # DeepSeek 系列
        ("DeepSeek-7B",     "DeepSeek",  7.0,  2023, [48.2, 75.1, 34.0, 17.4, 26.0, 59.2,  6.0, 37.0]),
        ("DeepSeek-67B",    "DeepSeek", 67.0,  2023, [71.3, 89.1, 46.5, 63.1, 43.0, 82.0, 18.7, 66.2]),
        ("DeepSeek-V2",     "DeepSeek",236.0,  2024, [78.5, 92.3, 52.3, 79.2, 48.8, 88.1, 43.6, 77.8]),
        # Phi 系列
        ("Phi-2",           "Phi",       2.7,  2023, [57.3, 73.1, 44.1, 57.7, 47.8, 69.5,  3.5, 42.8]),
        ("Phi-3-mini",      "Phi",       3.8,  2024, [68.8, 83.0, 50.1, 86.5, 62.3, 85.0, 37.9, 68.4]),
        ("Phi-3-medium",    "Phi",      14.0,  2024, [78.0, 91.5, 52.6, 91.0, 62.1, 91.0, 53.8, 75.1]),
        # Falcon 系列
        ("Falcon-7B",       "Falcon",    7.0,  2023, [27.8, 74.5, 25.2,  6.8,  5.5, 47.3,  1.5, 29.2]),
        ("Falcon-40B",      "Falcon",   40.0,  2023, [55.4, 85.3, 36.6, 19.6, 14.7, 71.8,  4.6, 46.1]),
    ]

    BENCHMARKS = ["MMLU", "HellaSwag", "TruthfulQA", "GSM8K", "HumanEval", "ARC-Challenge", "MATH", "BBH"]
    rng = random.Random(2024)

    rows = []
    for model, family, params_b, year, base_scores in MODELS:
        for i, benchmark in enumerate(BENCHMARKS):
            score = round(base_scores[i] + rng.uniform(-0.3, 0.3), 1)
            rows.append({
                "model":         model,
                "family":        family,
                "benchmark":     benchmark,
                "accuracy":      score,
                "params_B":      params_b,
                "year_released": year,
            })

    df = pd.DataFrame(rows)
    out = Path(S45_CSV)
    df.to_csv(out, index=False)
    print(f"✓ {out}  shape={df.shape}")
    print(f"  models: {df['model'].nunique()}  benchmarks: {df['benchmark'].nunique()}")
    print(f"  accuracy range: {df['accuracy'].min()}~{df['accuracy'].max()}")


def gen_s46() -> None:
    """NLP微调超参数搜索实验日志，约180行"""

    ARCHITECTURES = {
        "BERT-base":     {"base_acc": 88.0, "base_f1": 87.2, "base_hours": 1.2},
        "BERT-large":    {"base_acc": 90.5, "base_f1": 89.8, "base_hours": 2.8},
        "RoBERTa-base":  {"base_acc": 90.2, "base_f1": 89.5, "base_hours": 1.3},
        "RoBERTa-large": {"base_acc": 93.1, "base_f1": 92.5, "base_hours": 3.2},
        "DistilBERT":    {"base_acc": 85.3, "base_f1": 84.6, "base_hours": 0.7},
    }
    OPTIMIZERS = ["AdamW", "Adam", "SGD"]
    LEARNING_RATES = [1e-5, 2e-5, 5e-5, 1e-4]
    BATCH_SIZES = [16, 32, 64]
    WEIGHT_DECAYS = [0.0, 0.01]

    OPT_BONUS = {"AdamW": 0.8, "Adam": 0.0, "SGD": -3.2}
    LR_BONUS  = {1e-5: -1.2, 2e-5: 0.0, 5e-5: -0.5, 1e-4: -2.8}
    BS_BONUS  = {16: 0.3, 32: 0.0, 64: -0.4}

    rng = random.Random(42)
    rows = []

    for arch, arch_cfg in ARCHITECTURES.items():
        for opt in OPTIMIZERS:
            for lr in LEARNING_RATES:
                for bs in BATCH_SIZES:
                    wd = rng.choice(WEIGHT_DECAYS)
                    noise = rng.gauss(0, 0.4)
                    val_acc = round(
                        arch_cfg["base_acc"] + OPT_BONUS[opt] + LR_BONUS[lr] + BS_BONUS[bs] + noise, 2
                    )
                    val_f1 = round(
                        arch_cfg["base_f1"] + OPT_BONUS[opt] + LR_BONUS[lr] + BS_BONUS[bs] + rng.gauss(0, 0.4), 2
                    )
                    hours = round(arch_cfg["base_hours"] * (32 / bs) + rng.gauss(0, 0.05), 2)
                    rows.append({
                        "architecture":   arch,
                        "optimizer":      opt,
                        "learning_rate":  lr,
                        "batch_size":     bs,
                        "weight_decay":   wd,
                        "val_acc":        val_acc,
                        "val_f1":         val_f1,
                        "training_hours": max(0.1, hours),
                    })

    df = pd.DataFrame(rows)
    out = Path(S46_CSV)
    df.to_csv(out, index=False)
    print(f"✓ {out}  shape={df.shape}")
    print(f"  val_acc range: {df['val_acc'].min()}~{df['val_acc'].max()}")
    print(f"  val_f1 range: {df['val_f1'].min()}~{df['val_f1'].max()}")


# ---------------------------------------------------------------------------
# 训练对辅助函数
# ---------------------------------------------------------------------------

def _make_first(
    record_id: str, csv_path: str, data_context: str, user_input: str, plotspec: dict,
) -> dict:
    return {
        "id":           record_id,
        "record_type":  "first",
        "csv_path":     csv_path.replace("/", "\\"),
        "user_input":   user_input,
        "data_context": data_context,
        "plotspec":     plotspec,
    }


def _make_delta(
    record_id: str, csv_path: str, data_context: str,
    user_input: str, current_spec: dict, plotspec: dict,
) -> dict:
    return {
        "id":           record_id,
        "record_type":  "delta",
        "csv_path":     csv_path.replace("/", "\\"),
        "user_input":   user_input,
        "data_context": data_context,
        "current_spec": current_spec,
        "plotspec":     plotspec,
    }


def _make_ask_first(
    record_id: str, csv_path: str, data_context: str, user_input: str, question: str,
) -> dict:
    return {
        "id":           record_id,
        "record_type":  "ask_first",
        "csv_path":     csv_path.replace("/", "\\"),
        "user_input":   user_input,
        "data_context": data_context,
        "plotspec":     {"question": question},
    }


def _make_ask_delta(
    record_id: str, csv_path: str, data_context: str,
    user_input: str, current_spec: dict, question: str,
) -> dict:
    return {
        "id":           record_id,
        "record_type":  "ask_delta",
        "csv_path":     csv_path.replace("/", "\\"),
        "user_input":   user_input,
        "data_context": data_context,
        "current_spec": current_spec,
        "plotspec":     {"question": question},
    }


# ---------------------------------------------------------------------------
# 训练对构建
# ---------------------------------------------------------------------------

def _build_s45_records(ctx: str) -> list[dict]:
    """10 条 s45_llm_benchmark.csv 训练对"""
    csv = S45_CSV
    records: list[dict] = []

    records.append(_make_first(
        "manual_s45_01", csv, ctx,
        "比较各模型在MMLU上的准确率，按模型系列分组显示，从高到低排列",
        {
            "chart_type":           "bar",
            "data_x":               "model",
            "data_y":               "accuracy",
            "data_group_by":        "family",
            "data_filter":          "benchmark == 'MMLU'",
            "style_theme":          "science",
            "params_sort":          "desc",
            "axes_x_rotate_labels": True,
            "label_title":          "各模型MMLU准确率对比（按系列分组）",
            "label_x":              "模型",
            "label_y":              "准确率 (%)",
        },
    ))

    records.append(_make_first(
        "manual_s45_02", csv, ctx,
        "用散点图展示模型参数量与MMLU准确率的关系，按模型系列上色",
        {
            "chart_type":    "scatter",
            "data_x":        "params_B",
            "data_y":        "accuracy",
            "data_group_by": "family",
            "data_filter":   "benchmark == 'MMLU'",
            "style_theme":   "nature",
            "label_x":       "参数量 (B)",
            "label_y":       "准确率 (%)",
        },
    ))

    records.append(_make_first(
        "manual_s45_03", csv, ctx,
        "参数量与代码生成能力（HumanEval）的关系，加一条趋势回归线",
        {
            "chart_type":             "scatter",
            "data_x":                 "params_B",
            "data_y":                 "accuracy",
            "data_group_by":          "family",
            "data_filter":            "benchmark == 'HumanEval'",
            "style_theme":            "bright",
            "params_show_regression": True,
            "label_title":            "参数量与HumanEval代码生成能力",
            "label_x":                "参数量 (B)",
            "label_y":                "HumanEval Pass@1 (%)",
        },
    ))

    records.append(_make_first(
        "manual_s45_04", csv, ctx,
        "画一张热力图，展示GPT、Claude和Llama-3系列在所有基准上的成绩，列为基准，行为模型",
        {
            "chart_type":           "heatmap",
            "data_x":               "benchmark",
            "data_y":               "model",
            "data_filter":          "family in ['GPT', 'Claude', 'Llama-3']",
            "params_heatmap_value": "accuracy",
            "style_theme":          "nature",
            "params_annot":         True,
            "params_annot_fmt":     ".1f",
            "label_title":          "GPT / Claude / Llama-3 各基准准确率热力图",
        },
    ))

    records.append(_make_first(
        "manual_s45_05", csv, ctx,
        "各模型在GSM8K数学推理基准上的得分，从高到低排列，earth 风格",
        {
            "chart_type":           "bar",
            "data_x":               "model",
            "data_y":               "accuracy",
            "data_filter":          "benchmark == 'GSM8K'",
            "style_theme":          "earth",
            "params_sort":          "desc",
            "axes_x_rotate_labels": True,
            "label_title":          "各模型GSM8K数学推理准确率",
            "label_y":              "准确率 (%)",
        },
    ))

    records.append(_make_first(
        "manual_s45_06", csv, ctx,
        "探索模型参数量与TruthfulQA真实性评测成绩的关系，按系列标注颜色",
        {
            "chart_type":    "scatter",
            "data_x":        "params_B",
            "data_y":        "accuracy",
            "data_group_by": "family",
            "data_filter":   "benchmark == 'TruthfulQA'",
            "style_theme":   "macaron",
            "label_x":       "参数量 (B)",
            "label_y":       "TruthfulQA准确率 (%)",
        },
    ))

    records.append(_make_first(
        "manual_s45_07", csv, ctx,
        "只展示Llama-3系列各模型在MMLU上的表现，显示数值标注",
        {
            "chart_type":         "bar",
            "data_x":             "model",
            "data_y":             "accuracy",
            "data_filter":        "benchmark == 'MMLU' and family == 'Llama-3'",
            "style_theme":        "nature",
            "params_show_values": True,
            "label_title":        "Llama-3系列MMLU准确率",
            "label_y":            "准确率 (%)",
        },
    ))

    records.append(_make_first(
        "manual_s45_08", csv, ctx,
        "画柱状图展示各模型在MATH数学基准上的得分，从低到高排列，加个标题",
        {
            "chart_type":           "bar",
            "data_x":               "model",
            "data_y":               "accuracy",
            "data_filter":          "benchmark == 'MATH'",
            "style_theme":          "bright",
            "params_sort":          "asc",
            "axes_x_rotate_labels": True,
            "label_title":          "各模型数学推理能力（MATH基准）",
            "label_x":              "模型",
            "label_y":              "准确率 (%)",
        },
    ))

    records.append(_make_delta(
        "manual_s45_delta_01", csv, ctx,
        "给这张图加一个标题'参数量与MMLU准确率关系'，Y轴标注改为'MMLU准确率 (%)'",
        {
            "chart_type":    "scatter",
            "data_x":        "params_B",
            "data_y":        "accuracy",
            "data_group_by": "family",
            "data_filter":   "benchmark == 'MMLU'",
            "style_theme":   "nature",
            "label_x":       "参数量 (B)",
            "label_y":       "准确率 (%)",
        },
        {
            "label_title": "参数量与MMLU准确率关系",
            "label_y":     "MMLU准确率 (%)",
        },
    ))

    records.append(_make_ask_first(
        "manual_s45_ask_01", csv, ctx,
        "给我画个散点图，看看数据里的规律",
        (
            "请问散点图的横轴想展示哪个指标？"
            "数据中有两个数值型候选列："
            "params_B（模型参数量，单位B，范围2.7~1760.0）"
            "或 year_released（发布年份，范围2022~2024）。"
            "建议选 params_B，分析模型规模与准确率的关系。"
            "另外，是否需要筛选某个基准（如MMLU、GSM8K）？"
        ),
    ))

    return records


def _build_s46_records(ctx: str) -> list[dict]:
    """10 条 s46_hparam_search.csv 训练对"""
    csv = S46_CSV
    records: list[dict] = []

    records.append(_make_first(
        "manual_s46_01", csv, ctx,
        "用箱线图展示各架构模型在所有超参数组合下的验证准确率分布",
        {
            "chart_type":  "box",
            "data_x":      "architecture",
            "data_y":      "val_acc",
            "style_theme": "normal",
            "label_x":     "模型架构",
            "label_y":     "验证准确率 (%)",
        },
    ))

    records.append(_make_first(
        "manual_s46_02", csv, ctx,
        "比较不同优化器在各架构上val_acc的分布，按架构上色",
        {
            "chart_type":    "box",
            "data_x":        "optimizer",
            "data_y":        "val_acc",
            "data_group_by": "architecture",
            "style_theme":   "morandi",
            "label_x":       "优化器",
            "label_y":       "验证准确率 (%)",
        },
    ))

    records.append(_make_first(
        "manual_s46_03", csv, ctx,
        "用AdamW优化器的所有实验画散点图，看val_acc和val_f1的相关性，按架构上色",
        {
            "chart_type":    "scatter",
            "data_x":        "val_acc",
            "data_y":        "val_f1",
            "data_group_by": "architecture",
            "data_filter":   "optimizer == 'AdamW'",
            "style_theme":   "bright",
            "label_title":   "AdamW实验：val_acc vs val_f1",
            "label_x":       "验证准确率 (%)",
            "label_y":       "验证F1 (%)",
        },
    ))

    records.append(_make_first(
        "manual_s46_04", csv, ctx,
        "画散点图展示RoBERTa-large用AdamW时学习率与val_f1的关系，加回归线，science风格",
        {
            "chart_type":             "scatter",
            "data_x":                 "learning_rate",
            "data_y":                 "val_f1",
            "data_filter":            "architecture == 'RoBERTa-large' and optimizer == 'AdamW'",
            "style_theme":            "science",
            "params_show_regression": True,
            "label_title":            "RoBERTa-large+AdamW：学习率与val_f1",
            "label_x":                "学习率",
            "label_y":                "验证F1 (%)",
        },
    ))

    records.append(_make_first(
        "manual_s46_05", csv, ctx,
        "只看BERT-base + AdamW + batch_size=32的实验，展示不同学习率的val_acc，从高到低排序",
        {
            "chart_type":         "bar",
            "data_x":             "learning_rate",
            "data_y":             "val_acc",
            "data_filter":        "architecture == 'BERT-base' and optimizer == 'AdamW' and batch_size == 32",
            "style_theme":        "nature",
            "params_sort":        "desc",
            "params_show_values": True,
            "label_title":        "BERT-base+AdamW+BS32：学习率效果对比",
            "label_x":            "学习率",
            "label_y":            "验证准确率 (%)",
        },
    ))

    records.append(_make_first(
        "manual_s46_06", csv, ctx,
        "在batch_size=32的条件下，比较各架构搭配不同优化器的验证准确率，分组展示",
        {
            "chart_type":           "bar",
            "data_x":               "architecture",
            "data_y":               "val_acc",
            "data_group_by":        "optimizer",
            "data_filter":          "batch_size == 32",
            "style_theme":          "morandi",
            "axes_x_rotate_labels": True,
            "label_title":          "BS=32条件下各架构与优化器组合的val_acc",
            "label_x":              "架构",
            "label_y":              "验证准确率 (%)",
        },
    ))

    records.append(_make_first(
        "manual_s46_07", csv, ctx,
        "用散点图看训练时长与验证准确率的关系，按架构上色，分析性价比",
        {
            "chart_type":    "scatter",
            "data_x":        "training_hours",
            "data_y":        "val_acc",
            "data_group_by": "architecture",
            "style_theme":   "nature",
            "label_title":   "训练时长与验证准确率的关系",
            "label_x":       "训练时长 (小时)",
            "label_y":       "验证准确率 (%)",
        },
    ))

    records.append(_make_first(
        "manual_s46_08", csv, ctx,
        "不同batch_size对验证准确率有什么影响？用箱线图展示分布，显示所有数据点",
        {
            "chart_type":         "box",
            "data_x":             "batch_size",
            "data_y":             "val_acc",
            "style_theme":        "macaron",
            "params_show_points": "all",
            "label_title":        "batch_size对验证准确率的影响",
            "label_x":            "Batch Size",
            "label_y":            "验证准确率 (%)",
        },
    ))

    records.append(_make_delta(
        "manual_s46_delta_01", csv, ctx,
        "把排序改成从低到高，标题改成'BERT-base学习率敏感性分析（从差到好）'",
        {
            "chart_type":         "bar",
            "data_x":             "learning_rate",
            "data_y":             "val_acc",
            "data_filter":        "architecture == 'BERT-base' and optimizer == 'AdamW' and batch_size == 32",
            "style_theme":        "nature",
            "params_sort":        "desc",
            "params_show_values": True,
            "label_title":        "BERT-base+AdamW+BS32：学习率效果对比",
            "label_x":            "学习率",
            "label_y":            "验证准确率 (%)",
        },
        {
            "params_sort": "asc",
            "label_title": "BERT-base学习率敏感性分析（从差到好）",
        },
    ))

    records.append(_make_ask_first(
        "manual_s46_ask_01", csv, ctx,
        "画一张柱状图，分析各架构的模型性能",
        (
            "请问Y轴希望展示哪个性能指标？"
            "数据中有两个性能列：val_acc（验证准确率，范围78.7~94.5）"
            "或 val_f1（验证F1值，范围78.29~93.71）。"
            "建议用val_acc展示整体性能。"
        ),
    ))

    return records


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

def generate_csvs() -> None:
    print("── 生成训练 CSV ──")
    gen_s45()
    gen_s46()
    print()


def generate_pairs() -> None:
    print("── 生成训练数据对 ──")

    for csv_path in (S45_CSV, S46_CSV):
        if not Path(csv_path).exists():
            print(f"✗ 找不到 {csv_path}，请先运行 --csv-only 生成 CSV")
            return

    print(f"加载 {S45_CSV} ...")
    ctx45, _ = load_data(S45_CSV)
    print(f"加载 {S46_CSV} ...")
    ctx46, _ = load_data(S46_CSV)

    existing_ids: set[str] = set()
    if MANUAL_PAIRS_PATH.exists():
        with MANUAL_PAIRS_PATH.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_ids.add(json.loads(line).get("id", ""))

    all_records: list[dict] = []
    all_records.extend(_build_s45_records(ctx45))
    all_records.extend(_build_s46_records(ctx46))

    appended = 0
    skipped = 0
    with MANUAL_PAIRS_PATH.open("a", encoding="utf-8") as f:
        for rec in all_records:
            if rec["id"] in existing_ids:
                print(f"  跳过（已存在）：{rec['id']}")
                skipped += 1
            else:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                appended += 1

    print(f"\n完成！新增 {appended} 条，跳过 {skipped} 条。")
    print(f"manual_pairs.jsonl 现共约 {len(existing_ids) + appended} 条记录。")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成大规模训练 CSV 及配对数据")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--csv-only",   action="store_true", help="仅生成 CSV，不生成训练对")
    group.add_argument("--pairs-only", action="store_true", help="仅生成训练对（CSV 必须已存在）")
    args = parser.parse_args()

    if args.csv_only:
        generate_csvs()
    elif args.pairs_only:
        generate_pairs()
    else:
        generate_csvs()
        generate_pairs()


if __name__ == "__main__":
    main()
