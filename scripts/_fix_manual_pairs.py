"""临时脚本：修复 manual_pairs.jsonl 中格式损坏的第 42 行"""
import json
from pathlib import Path

path = Path("data/pairs/manual_pairs.jsonl")

with path.open(encoding="utf-8") as f:
    raw_lines = f.readlines()

# Line 42 (index 41) 的 plotspec.label_title 被错误写成非字符串值
# 重构该行为合法 JSON
rec42 = {
    "id": "manual_hm_wide_s26_03",
    "record_type": "delta",
    "csv_path": "data\\train\\s26_wide_method_metric.csv",
    "user_input": "把数值格式改为保留两位小数，加个标题Method Evaluation Results",
    "current_spec": {
        "chart_type": "heatmap",
        "data_x": "metric",
        "data_y": "method",
        "style_theme": "science",
    },
    "plotspec": {
        "params_annot_fmt": ".2f",
        "label_title": "Method Evaluation Results",
    },
    "data_context": (
        "数据摘要：\n"
        "- 形状：5行 × 6列\n"
        "- 列信息：\n"
        "  · method（类别型，唯一值5个）：Ours, Baseline-A, Baseline-B, State-of-the-art, Ablation\n"
        "  · score（数值型，范围84.36~91.16）\n"
        "  · macro_f1（数值型，范围81.6~88.53）\n"
        "  · BLEU_score（数值型，范围27.78~35.27）\n"
        "  · ROUGE-L（数值型，范围50.91~58.3）\n"
        "  · EM（数值型，范围67.54~75.1）\n"
        "- 前2行预览：[['Ours', 90.03, 87.81, 34.27, 57.26, 74.34], "
        "['Baseline-A', 88.4, 85.93, 32.27, 55.31, 71.81]]\n"
        "- 缓存key：cache://m26w0003"
    ),
}

new_line42 = json.dumps(rec42, ensure_ascii=False) + "\n"

# 验证
try:
    json.loads(new_line42.strip())
    print("Line 42 fixed OK")
except Exception as e:
    print(f"Still broken: {e}")
    raise

raw_lines[41] = new_line42

with path.open("w", encoding="utf-8") as f:
    f.writelines(raw_lines)

# 最终验证全文件
errors = 0
with path.open(encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            json.loads(line)
        except Exception as e:
            print(f"  Line {i} still broken: {e}")
            errors += 1

if errors == 0:
    print("All lines valid. Done.")
else:
    print(f"{errors} lines still have errors.")
