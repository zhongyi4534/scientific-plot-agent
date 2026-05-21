"""
scripts/gen_large_csv_pairs.py

为 s45_llm_benchmark.csv 和 s46_hparam_search.csv 生成训练数据对，
并追加到 data/pairs/manual_pairs.jsonl。

重点覆盖：
  - data_filter (pandas query) 过滤大表
  - data_group_by 按类别上色
  - scatter + regression
  - heatmap (long format, 带 data_filter 选取子集模型)
  - box chart (超参数效果对比)
  - ask_user 触发（列名歧义）
  - delta 修改

生成 20 条记录：s45 × 10 + s46 × 10

用法：
    python scripts/gen_large_csv_pairs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.loader import load_data

MANUAL_PAIRS_PATH = Path("data/pairs/manual_pairs.jsonl")
S45_CSV = "data/train/s45_llm_benchmark.csv"
S46_CSV = "data/train/s46_hparam_search.csv"


def make_first(
    record_id: str,
    csv_path: str,
    data_context: str,
    user_input: str,
    plotspec: dict,
) -> dict:
    return {
        "id": record_id,
        "record_type": "first",
        "csv_path": csv_path.replace("/", "\\"),
        "user_input": user_input,
        "data_context": data_context,
        "plotspec": plotspec,
    }


def make_delta(
    record_id: str,
    csv_path: str,
    data_context: str,
    user_input: str,
    current_spec: dict,
    plotspec: dict,
) -> dict:
    return {
        "id": record_id,
        "record_type": "delta",
        "csv_path": csv_path.replace("/", "\\"),
        "user_input": user_input,
        "data_context": data_context,
        "current_spec": current_spec,
        "plotspec": plotspec,
    }


def make_ask_first(
    record_id: str,
    csv_path: str,
    data_context: str,
    user_input: str,
    question: str,
) -> dict:
    return {
        "id": record_id,
        "record_type": "ask_first",
        "csv_path": csv_path.replace("/", "\\"),
        "user_input": user_input,
        "data_context": data_context,
        "plotspec": {"question": question},
    }


def make_ask_delta(
    record_id: str,
    csv_path: str,
    data_context: str,
    user_input: str,
    current_spec: dict,
    question: str,
) -> dict:
    return {
        "id": record_id,
        "record_type": "ask_delta",
        "csv_path": csv_path.replace("/", "\\"),
        "user_input": user_input,
        "data_context": data_context,
        "current_spec": current_spec,
        "plotspec": {"question": question},
    }


def build_s45_records(ctx: str) -> list[dict]:
    """10 条 s45_llm_benchmark.csv 训练对"""
    csv = S45_CSV
    records = []

    # ── 首轮 1：按 family 分组的 MMLU 柱状图，降序 ──
    records.append(make_first(
        "manual_s45_01", csv, ctx,
        "比较各模型在MMLU上的准确率，按模型系列分组显示，从高到低排列",
        {
            "chart_type":        "bar",
            "data_x":            "model",
            "data_y":            "accuracy",
            "data_group_by":     "family",
            "data_filter":       "benchmark == 'MMLU'",
            "style_theme":       "science",
            "params_sort":       "desc",
            "axes_x_rotate_labels": True,
            "label_title":       "各模型MMLU准确率对比（按系列分组）",
            "label_x":           "模型",
            "label_y":           "准确率 (%)",
        },
    ))

    # ── 首轮 2：参数量 vs MMLU 散点图 ──
    records.append(make_first(
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

    # ── 首轮 3：代码生成能力（HumanEval）× 参数量，加回归线 ──
    records.append(make_first(
        "manual_s45_03", csv, ctx,
        "参数量与代码生成能力（HumanEval）的关系，加一条趋势回归线",
        {
            "chart_type":           "scatter",
            "data_x":               "params_B",
            "data_y":               "accuracy",
            "data_group_by":        "family",
            "data_filter":          "benchmark == 'HumanEval'",
            "style_theme":          "bright",
            "params_show_regression": True,
            "label_title":          "参数量与HumanEval代码生成能力",
            "label_x":              "参数量 (B)",
            "label_y":              "HumanEval Pass@1 (%)",
        },
    ))

    # ── 首轮 4：热力图，只选 GPT/Claude/Llama-3 三个系列 ──
    records.append(make_first(
        "manual_s45_04", csv, ctx,
        "画一张热力图，展示GPT、Claude和Llama-3系列在所有基准上的成绩，列为基准，行为模型",
        {
            "chart_type":          "heatmap",
            "data_x":              "benchmark",
            "data_y":              "model",
            "data_filter":         "family in ['GPT', 'Claude', 'Llama-3']",
            "params_heatmap_value": "accuracy",
            "style_theme":         "nature",
            "params_annot":        True,
            "params_annot_fmt":    ".1f",
            "label_title":         "GPT / Claude / Llama-3 各基准准确率热力图",
        },
    ))

    # ── 首轮 5：GSM8K 数学推理，降序排列 ──
    records.append(make_first(
        "manual_s45_05", csv, ctx,
        "各模型在GSM8K数学推理基准上的得分，从高到低排列，earth 风格",
        {
            "chart_type":         "bar",
            "data_x":             "model",
            "data_y":             "accuracy",
            "data_filter":        "benchmark == 'GSM8K'",
            "style_theme":        "earth",
            "params_sort":        "desc",
            "axes_x_rotate_labels": True,
            "label_title":        "各模型GSM8K数学推理准确率",
            "label_y":            "准确率 (%)",
        },
    ))

    # ── 首轮 6：TruthfulQA × 参数量散点，按系列上色 ──
    records.append(make_first(
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

    # ── 首轮 7：只看 Llama-3 系列的 MMLU，显示数值标注 ──
    records.append(make_first(
        "manual_s45_07", csv, ctx,
        "只展示Llama-3系列各模型在MMLU上的表现，显示数值标注",
        {
            "chart_type":       "bar",
            "data_x":           "model",
            "data_y":           "accuracy",
            "data_filter":      "benchmark == 'MMLU' and family == 'Llama-3'",
            "style_theme":      "nature",
            "params_show_values": True,
            "label_title":      "Llama-3系列MMLU准确率",
            "label_y":          "准确率 (%)",
        },
    ))

    # ── 首轮 8：MATH 基准，升序排列，标题 ──
    records.append(make_first(
        "manual_s45_08", csv, ctx,
        "画柱状图展示各模型在MATH数学基准上的得分，从低到高排列，加个标题",
        {
            "chart_type":         "bar",
            "data_x":             "model",
            "data_y":             "accuracy",
            "data_filter":        "benchmark == 'MATH'",
            "style_theme":        "bright",
            "params_sort":        "asc",
            "axes_x_rotate_labels": True,
            "label_title":        "各模型数学推理能力（MATH基准）",
            "label_x":            "模型",
            "label_y":            "准确率 (%)",
        },
    ))

    # ── delta：在 #2 散点图基础上补充标题和Y轴标注 ──
    current_s45_02 = {
        "chart_type":    "scatter",
        "data_x":        "params_B",
        "data_y":        "accuracy",
        "data_group_by": "family",
        "data_filter":   "benchmark == 'MMLU'",
        "style_theme":   "nature",
        "label_x":       "参数量 (B)",
        "label_y":       "准确率 (%)",
    }
    records.append(make_delta(
        "manual_s45_delta_01", csv, ctx,
        "给这张图加一个标题'参数量与MMLU准确率关系'，Y轴标注改为'MMLU准确率 (%)'",
        current_s45_02,
        {
            "label_title": "参数量与MMLU准确率关系",
            "label_y":     "MMLU准确率 (%)",
        },
    ))

    # ── ask_user：数据有多个数值型列，散点图横轴不明确 ──
    records.append(make_ask_first(
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


def build_s46_records(ctx: str) -> list[dict]:
    """10 条 s46_hparam_search.csv 训练对"""
    csv = S46_CSV
    records = []

    # ── 首轮 1：各架构 val_acc 分布箱线图 ──
    records.append(make_first(
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

    # ── 首轮 2：不同优化器 × 架构的 val_acc 分布 ──
    records.append(make_first(
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

    # ── 首轮 3：AdamW 下 val_acc vs val_f1 散点，按架构上色 ──
    records.append(make_first(
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

    # ── 首轮 4：RoBERTa-large+AdamW 下学习率 vs val_f1，加回归线 ──
    records.append(make_first(
        "manual_s46_04", csv, ctx,
        "画散点图展示RoBERTa-large用AdamW时学习率与val_f1的关系，加回归线，science风格",
        {
            "chart_type":           "scatter",
            "data_x":               "learning_rate",
            "data_y":               "val_f1",
            "data_filter":          "architecture == 'RoBERTa-large' and optimizer == 'AdamW'",
            "style_theme":          "science",
            "params_show_regression": True,
            "label_title":          "RoBERTa-large+AdamW：学习率与val_f1",
            "label_x":              "学习率",
            "label_y":              "验证F1 (%)",
        },
    ))

    # ── 首轮 5：BERT-base+AdamW+BS32 的不同学习率效果，降序 ──
    records.append(make_first(
        "manual_s46_05", csv, ctx,
        "只看BERT-base + AdamW + batch_size=32的实验，展示不同学习率的val_acc，从高到低排序",
        {
            "chart_type":       "bar",
            "data_x":           "learning_rate",
            "data_y":           "val_acc",
            "data_filter":      "architecture == 'BERT-base' and optimizer == 'AdamW' and batch_size == 32",
            "style_theme":      "nature",
            "params_sort":      "desc",
            "params_show_values": True,
            "label_title":      "BERT-base+AdamW+BS32：学习率效果对比",
            "label_x":          "学习率",
            "label_y":          "验证准确率 (%)",
        },
    ))

    # ── 首轮 6：batch_size=32 下各架构 × 优化器的 val_acc 分组柱状图 ──
    records.append(make_first(
        "manual_s46_06", csv, ctx,
        "在batch_size=32的条件下，比较各架构搭配不同优化器的验证准确率，分组展示",
        {
            "chart_type":       "bar",
            "data_x":           "architecture",
            "data_y":           "val_acc",
            "data_group_by":    "optimizer",
            "data_filter":      "batch_size == 32",
            "style_theme":      "morandi",
            "axes_x_rotate_labels": True,
            "label_title":      "BS=32条件下各架构与优化器组合的val_acc",
            "label_x":          "架构",
            "label_y":          "验证准确率 (%)",
        },
    ))

    # ── 首轮 7：训练时长 vs val_acc 散点，按架构上色 ──
    records.append(make_first(
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

    # ── 首轮 8：不同 batch_size 对 val_acc 的影响，显示所有数据点 ──
    records.append(make_first(
        "manual_s46_08", csv, ctx,
        "不同batch_size对验证准确率有什么影响？用箱线图展示分布，显示所有数据点",
        {
            "chart_type":       "box",
            "data_x":           "batch_size",
            "data_y":           "val_acc",
            "style_theme":      "macaron",
            "params_show_points": "all",
            "label_title":      "batch_size对验证准确率的影响",
            "label_x":          "Batch Size",
            "label_y":          "验证准确率 (%)",
        },
    ))

    # ── delta：在 #5 基础上改为升序，更新标题 ──
    current_s46_05 = {
        "chart_type":       "bar",
        "data_x":           "learning_rate",
        "data_y":           "val_acc",
        "data_filter":      "architecture == 'BERT-base' and optimizer == 'AdamW' and batch_size == 32",
        "style_theme":      "nature",
        "params_sort":      "desc",
        "params_show_values": True,
        "label_title":      "BERT-base+AdamW+BS32：学习率效果对比",
        "label_x":          "学习率",
        "label_y":          "验证准确率 (%)",
    }
    records.append(make_delta(
        "manual_s46_delta_01", csv, ctx,
        "把排序改成从低到高，标题改成'BERT-base学习率敏感性分析（从差到好）'",
        current_s46_05,
        {
            "params_sort": "asc",
            "label_title": "BERT-base学习率敏感性分析（从差到好）",
        },
    ))

    # ── ask_user：val_acc 还是 val_f1 作为 Y 轴，歧义触发 ──
    records.append(make_ask_first(
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


def main() -> None:
    # 生成两个 CSV 的 data_context
    print("加载 s45_llm_benchmark.csv ...")
    ctx45, _ = load_data(S45_CSV)

    print("加载 s46_hparam_search.csv ...")
    ctx46, _ = load_data(S46_CSV)

    # 检查现有记录（去重用）
    existing_ids: set[str] = set()
    if MANUAL_PAIRS_PATH.exists():
        with MANUAL_PAIRS_PATH.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    existing_ids.add(rec.get("id", ""))

    # 构建所有记录
    all_records: list[dict] = []
    all_records.extend(build_s45_records(ctx45))
    all_records.extend(build_s46_records(ctx46))

    # 追加（跳过已存在的 id）
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


if __name__ == "__main__":
    main()
