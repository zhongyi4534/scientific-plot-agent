"""
scripts/_add_training_v2.py

补充第二批手动训练数据，解决微调后发现的三类问题：
1. 宽表热力图格式混淆（用 smoke_test/wide_heatmap.csv 添加正例）
2. 模型对颜色/风格请求调用 ask_user（添加直接修改的 delta 正例）
3. 渲染报错后模型应直接修正 PlotSpec（添加修正 delta 正例）

运行：python scripts/_add_training_v2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.loader import load_data

MANUAL_PATH = Path("data/pairs/manual_pairs.jsonl")


def _load_ctx(csv_path: str) -> str:
    ctx, _ = load_data(csv_path)
    return ctx


def build_records() -> list[dict]:
    records: list[dict] = []

    # ── 宽表热力图 smoke_test/wide_heatmap.csv ─────────────────────────────
    ctx_wide = _load_ctx("data/smoke_test/wide_heatmap.csv")

    records.append({
        "id": "manual_v2_hm_wide_smoke_01",
        "record_type": "first",
        "csv_path": "data/smoke_test/wide_heatmap.csv",
        "user_input": "帮我画一张热力图，展示各模型在不同数据集上的准确率",
        "plotspec": {
            "chart_type": "heatmap",
            "data_x": "dataset",
            "data_y": "model",
            "style_theme": "science",
            "params_annot": True,
        },
        "data_context": ctx_wide,
    })

    records.append({
        "id": "manual_v2_hm_wide_smoke_02",
        "record_type": "first",
        "csv_path": "data/smoke_test/wide_heatmap.csv",
        "user_input": "用热力图比较各 NLP 模型在五个 benchmark 上的得分，加注数值",
        "plotspec": {
            "chart_type": "heatmap",
            "data_x": "benchmark",
            "data_y": "model",
            "style_theme": "nature",
            "params_annot": True,
            "params_annot_fmt": ".1f",
        },
        "data_context": ctx_wide,
    })

    records.append({
        "id": "manual_v2_hm_wide_smoke_03",
        "record_type": "first",
        "csv_path": "data/smoke_test/wide_heatmap.csv",
        "user_input": "画一个热图，Science 风格，显示各模型在各数据集上的数值，不需要注释",
        "plotspec": {
            "chart_type": "heatmap",
            "data_x": "dataset",
            "data_y": "model",
            "style_theme": "science",
            "params_annot": False,
        },
        "data_context": ctx_wide,
    })

    records.append({
        "id": "manual_v2_hm_wide_smoke_04",
        "record_type": "first",
        "csv_path": "data/smoke_test/wide_heatmap.csv",
        "user_input": "用 morandi 风格画热力图，显示模型和数据集之间的关系，数值保留两位小数",
        "plotspec": {
            "chart_type": "heatmap",
            "data_x": "dataset",
            "data_y": "model",
            "style_theme": "morandi",
            "params_annot": True,
            "params_annot_fmt": ".2f",
        },
        "data_context": ctx_wide,
    })

    # ── 添加 delta 示例：直接执行颜色/风格修改，不调用 ask_user ─────────────
    # 使用已有的训练 CSV
    ctx_nlp = _load_ctx("data/train/s01a_nlp_benchmark_long.csv")
    ctx_s25 = _load_ctx("data/train/s25_wide_model_dataset.csv")

    # 换颜色 → 直接用 style_custom_palette，不问用户
    records.append({
        "id": "manual_v2_delta_color_01",
        "record_type": "delta",
        "csv_path": "data/train/s01a_nlp_benchmark_long.csv",
        "user_input": "换个颜色，用偏暖色的配色",
        "current_spec": {
            "chart_type": "bar",
            "data_x": "benchmark",
            "data_y": "score",
            "data_group_by": "baseline",
            "style_theme": "normal",
        },
        "plotspec": {
            "style_custom_palette": ["#E64B35", "#F5A623", "#E8A838", "#D9534F"],
        },
        "data_context": ctx_nlp,
    })

    # 换深色背景 → 直接修改 style_theme，不问用户
    records.append({
        "id": "manual_v2_delta_color_02",
        "record_type": "delta",
        "csv_path": "data/train/s01a_nlp_benchmark_long.csv",
        "user_input": "换成深色风格背景",
        "current_spec": {
            "chart_type": "bar",
            "data_x": "benchmark",
            "data_y": "score",
            "data_group_by": "baseline",
            "style_theme": "normal",
        },
        "plotspec": {
            "style_theme": "earth",
        },
        "data_context": ctx_nlp,
    })

    # 接近白色背景 → 直接设置 style_bg_color
    records.append({
        "id": "manual_v2_delta_color_03",
        "record_type": "delta",
        "csv_path": "data/train/s01a_nlp_benchmark_long.csv",
        "user_input": "背景改成接近白色的浅色",
        "current_spec": {
            "chart_type": "line",
            "data_x": "benchmark",
            "data_y": "score",
            "data_group_by": "baseline",
            "style_theme": "normal",
        },
        "plotspec": {
            "style_bg_color": "#f5f5f5",
        },
        "data_context": ctx_nlp,
    })

    # 换成 Science 期刊配色 → 直接改 style_theme
    records.append({
        "id": "manual_v2_delta_color_04",
        "record_type": "delta",
        "csv_path": "data/train/s25_wide_model_dataset.csv",
        "user_input": "换成 Science 期刊风格",
        "current_spec": {
            "chart_type": "heatmap",
            "data_x": "dataset",
            "data_y": "model",
            "style_theme": "normal",
        },
        "plotspec": {
            "style_theme": "science",
        },
        "data_context": ctx_s25,
    })

    # 蓝色系 → 直接用 style_palette_override
    records.append({
        "id": "manual_v2_delta_color_05",
        "record_type": "delta",
        "csv_path": "data/train/s01a_nlp_benchmark_long.csv",
        "user_input": "换成蓝色系的配色方案",
        "current_spec": {
            "chart_type": "bar",
            "data_x": "benchmark",
            "data_y": "score",
            "data_group_by": "baseline",
            "style_theme": "nature",
        },
        "plotspec": {
            "style_palette_override": "coolwarm",
        },
        "data_context": ctx_nlp,
    })

    # 用户模糊说"好看点" → 直接选合理配色，不问
    records.append({
        "id": "manual_v2_delta_color_06",
        "record_type": "delta",
        "csv_path": "data/train/s01a_nlp_benchmark_long.csv",
        "user_input": "颜色好看点",
        "current_spec": {
            "chart_type": "bar",
            "data_x": "benchmark",
            "data_y": "score",
            "data_group_by": "baseline",
            "style_theme": "normal",
        },
        "plotspec": {
            "style_theme": "macaron",
        },
        "data_context": ctx_nlp,
    })

    # ── delta 修正示例：渲染报错后直接修正，不问用户 ─────────────────────────
    # 模型在渲染时发现 data_y 列不存在 → 修正为正确列名
    ctx_s01a = _load_ctx("data/train/s01a_nlp_benchmark_long.csv")

    records.append({
        "id": "manual_v2_delta_fix_01",
        "record_type": "delta",
        "csv_path": "data/train/s01a_nlp_benchmark_long.csv",
        "user_input": "渲染报错，请修正PlotSpec：data_y 列 'value' 不存在于数据中",
        "current_spec": {
            "chart_type": "bar",
            "data_x": "benchmark",
            "data_y": "value",
            "data_group_by": "baseline",
            "style_theme": "science",
        },
        "plotspec": {
            "data_y": "score",
        },
        "data_context": ctx_s01a,
    })

    records.append({
        "id": "manual_v2_delta_fix_02",
        "record_type": "delta",
        "csv_path": "data/train/s01a_nlp_benchmark_long.csv",
        "user_input": "渲染报错，请修正PlotSpec：data_filter 解析失败，列 'acc' 不存在",
        "current_spec": {
            "chart_type": "bar",
            "data_x": "benchmark",
            "data_y": "score",
            "data_group_by": "baseline",
            "style_theme": "nature",
            "data_filter": "acc > 0.9",
        },
        "plotspec": {
            "data_filter": None,
        },
        "data_context": ctx_s01a,
    })

    records.append({
        "id": "manual_v2_delta_fix_03",
        "record_type": "delta",
        "csv_path": "data/train/s25_wide_model_dataset.csv",
        "user_input": "渲染报错，请修正PlotSpec：data_y 列 'SST-2' 不存在于数据中",
        "current_spec": {
            "chart_type": "heatmap",
            "data_x": "dataset",
            "data_y": ["SST-2", "MR", "CoLA", "RTE", "QNLI"],
            "style_theme": "science",
        },
        "plotspec": {
            "data_y": "model",
        },
        "data_context": ctx_s25,
    })

    return records


def main() -> None:
    new_records = build_records()
    # 检查是否已存在
    existing_ids: set[str] = set()
    if MANUAL_PATH.exists():
        with MANUAL_PATH.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        existing_ids.add(json.loads(line)["id"])
                    except Exception:
                        pass

    to_add = [r for r in new_records if r["id"] not in existing_ids]
    if not to_add:
        print("所有记录已存在，无需追加。")
        return

    with MANUAL_PATH.open("a", encoding="utf-8") as f:
        for rec in to_add:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"追加 {len(to_add)} 条记录到 {MANUAL_PATH}")
    for r in to_add:
        print(f"  {r['id']}")


if __name__ == "__main__":
    main()
