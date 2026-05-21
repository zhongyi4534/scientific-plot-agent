"""
scripts/gen_large_csvs.py

生成两个规模较大的训练用 CSV 文件：

  data/train/s45_llm_benchmark.csv   (240行 6列)
      30个LLM × 8个基准测试，包含参数量、发布年份
      适合: 带 data_filter 的柱状图/散点图、热力图、分组图

  data/train/s46_hparam_search.csv   (180行 7列)
      NLP超参数搜索实验日志（5架构×3优化器×4学习率×部分组合）
      适合: 箱线图分布、散点图、带data_filter的柱状图、ask_user(val_acc vs val_f1)

用法：
    python scripts/gen_large_csvs.py
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

OUT_DIR = Path("data/train")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# s45_llm_benchmark.csv
# ---------------------------------------------------------------------------

def gen_s45() -> None:
    """30个LLM × 8基准 = 240行"""

    # (model, family, params_B, year_released, base_scores_dict)
    # base_scores: MMLU, HellaSwag, TruthfulQA, GSM8K, HumanEval, ARC, MATH, BBH
    MODELS: list[tuple] = [
        # GPT 系列
        ("GPT-3.5-turbo",  "GPT",      175.0,  2022, [70.0, 85.5, 47.0, 57.1, 48.1, 85.2, 34.1, 70.1]),
        ("GPT-4",          "GPT",     1760.0,  2023, [86.4, 95.3, 59.0, 92.0, 67.0, 96.3, 52.6, 86.7]),
        ("GPT-4o",         "GPT",      200.0,  2024, [88.7, 95.8, 65.4, 95.9, 90.2, 97.0, 76.6, 90.2]),
        # Claude 系列
        ("Claude-2",       "Claude",    70.0,  2023, [78.5, 91.2, 54.3, 79.6, 47.0, 90.5, 32.5, 78.2]),
        ("Claude-3-Haiku", "Claude",    20.0,  2024, [75.2, 88.0, 56.1, 76.9, 40.0, 88.1, 38.9, 73.5]),
        ("Claude-3-Sonnet","Claude",    70.0,  2024, [86.7, 93.5, 59.8, 92.3, 73.0, 93.2, 40.5, 82.9]),
        ("Claude-3-Opus",  "Claude",    70.0,  2024, [88.2, 95.4, 61.2, 95.0, 84.9, 96.4, 60.1, 86.8]),
        # Gemini 系列
        ("Gemini-Pro",     "Gemini",    37.0,  2023, [71.8, 84.7, 53.0, 74.4, 32.5, 82.3, 28.6, 75.0]),
        ("Gemini-1.5-Pro", "Gemini",   340.0,  2024, [81.9, 92.5, 60.1, 90.7, 71.9, 92.1, 58.5, 85.3]),
        ("Gemini-Ultra",   "Gemini",  1700.0,  2024, [90.0, 97.7, 61.9, 94.4, 74.4, 97.3, 53.2, 89.7]),
        # Llama-2 系列
        ("Llama-2-7B",    "Llama-2",    7.0,  2023, [45.3, 77.2, 29.3, 13.5, 12.8, 61.0, 2.5,  35.2]),
        ("Llama-2-13B",   "Llama-2",   13.0,  2023, [54.8, 80.7, 36.8, 28.7, 18.3, 67.3, 7.0,  39.4]),
        ("Llama-2-70B",   "Llama-2",   70.0,  2023, [68.9, 87.3, 44.8, 56.8, 29.9, 79.8, 13.5, 58.1]),
        # Llama-3 系列
        ("Llama-3-8B",    "Llama-3",    8.0,  2024, [66.6, 82.0, 43.9, 79.6, 62.2, 78.6, 29.8, 62.7]),
        ("Llama-3-70B",   "Llama-3",   70.0,  2024, [82.0, 93.1, 52.8, 93.0, 81.7, 92.9, 50.4, 81.3]),
        ("Llama-3.1-405B","Llama-3",  405.0,  2024, [88.6, 95.0, 56.8, 96.8, 89.0, 96.1, 73.8, 88.5]),
        # Mistral 系列
        ("Mistral-7B",    "Mistral",    7.0,  2023, [62.5, 81.3, 42.1, 52.2, 30.5, 72.3, 11.0, 56.3]),
        ("Mixtral-8x7B",  "Mistral",   47.0,  2024, [70.6, 89.1, 48.5, 74.4, 40.2, 85.4, 28.4, 68.0]),
        ("Mixtral-8x22B", "Mistral",  141.0,  2024, [77.8, 91.8, 51.2, 84.6, 59.0, 90.1, 41.8, 76.9]),
        # Qwen 系列
        ("Qwen1.5-7B",    "Qwen",       7.0,  2024, [61.0, 78.5, 41.0, 62.5, 36.0, 74.0, 20.3, 52.0]),
        ("Qwen1.5-72B",   "Qwen",      72.0,  2024, [77.2, 91.3, 54.0, 85.7, 60.0, 89.3, 40.5, 74.4]),
        ("Qwen2-72B",     "Qwen",      72.0,  2024, [84.2, 94.2, 57.5, 91.1, 64.6, 93.3, 51.1, 82.4]),
        # DeepSeek 系列
        ("DeepSeek-7B",   "DeepSeek",   7.0,  2023, [48.2, 75.1, 34.0, 17.4, 26.0, 59.2, 6.0,  37.0]),
        ("DeepSeek-67B",  "DeepSeek",  67.0,  2023, [71.3, 89.1, 46.5, 63.1, 43.0, 82.0, 18.7, 66.2]),
        ("DeepSeek-V2",   "DeepSeek", 236.0,  2024, [78.5, 92.3, 52.3, 79.2, 48.8, 88.1, 43.6, 77.8]),
        # Phi 系列
        ("Phi-2",         "Phi",        2.7,  2023, [57.3, 73.1, 44.1, 57.7, 47.8, 69.5, 3.5,  42.8]),
        ("Phi-3-mini",    "Phi",        3.8,  2024, [68.8, 83.0, 50.1, 86.5, 62.3, 85.0, 37.9, 68.4]),
        ("Phi-3-medium",  "Phi",       14.0,  2024, [78.0, 91.5, 52.6, 91.0, 62.1, 91.0, 53.8, 75.1]),
        # Falcon 系列
        ("Falcon-7B",     "Falcon",     7.0,  2023, [27.8, 74.5, 25.2, 6.8,  5.5,  47.3, 1.5,  29.2]),
        ("Falcon-40B",    "Falcon",    40.0,  2023, [55.4, 85.3, 36.6, 19.6, 14.7, 71.8, 4.6,  46.1]),
    ]

    BENCHMARKS = ["MMLU", "HellaSwag", "TruthfulQA", "GSM8K", "HumanEval", "ARC-Challenge", "MATH", "BBH"]

    rng = random.Random(2024)

    rows = []
    for model, family, params_b, year, base_scores in MODELS:
        for i, benchmark in enumerate(BENCHMARKS):
            # 添加轻微随机扰动（±0.3%）保持真实感
            score = round(base_scores[i] + rng.uniform(-0.3, 0.3), 1)
            rows.append({
                "model":          model,
                "family":         family,
                "benchmark":      benchmark,
                "accuracy":       score,
                "params_B":       params_b,
                "year_released":  year,
            })

    df = pd.DataFrame(rows)
    out = OUT_DIR / "s45_llm_benchmark.csv"
    df.to_csv(out, index=False)
    print(f"✓ {out}  shape={df.shape}")
    print(f"  models: {df['model'].nunique()}  benchmarks: {df['benchmark'].nunique()}")
    print(f"  accuracy range: {df['accuracy'].min()}~{df['accuracy'].max()}")


# ---------------------------------------------------------------------------
# s46_hparam_search.csv
# ---------------------------------------------------------------------------

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

    # 优化器效果修正
    OPT_BONUS = {"AdamW": 0.8, "Adam": 0.0, "SGD": -3.2}
    # 学习率效果（相对最优学习率2e-5的偏差）
    LR_BONUS = {1e-5: -1.2, 2e-5: 0.0, 5e-5: -0.5, 1e-4: -2.8}
    # batch_size 效果
    BS_BONUS = {16: 0.3, 32: 0.0, 64: -0.4}

    rng = random.Random(42)
    rows = []

    for arch, arch_cfg in ARCHITECTURES.items():
        for opt in OPTIMIZERS:
            for lr in LEARNING_RATES:
                for bs in BATCH_SIZES:
                    # 随机选一个 weight_decay
                    wd = rng.choice(WEIGHT_DECAYS)

                    noise = rng.gauss(0, 0.4)
                    val_acc = round(
                        arch_cfg["base_acc"]
                        + OPT_BONUS[opt]
                        + LR_BONUS[lr]
                        + BS_BONUS[bs]
                        + noise,
                        2,
                    )
                    val_f1 = round(
                        arch_cfg["base_f1"]
                        + OPT_BONUS[opt]
                        + LR_BONUS[lr]
                        + BS_BONUS[bs]
                        + rng.gauss(0, 0.4),
                        2,
                    )
                    # 训练时长：与 batch_size 反比，与模型大小正比
                    hours = round(
                        arch_cfg["base_hours"] * (32 / bs) + rng.gauss(0, 0.05),
                        2,
                    )

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
    out = OUT_DIR / "s46_hparam_search.csv"
    df.to_csv(out, index=False)
    print(f"✓ {out}  shape={df.shape}")
    print(f"  val_acc range: {df['val_acc'].min()}~{df['val_acc'].max()}")
    print(f"  val_f1 range: {df['val_f1'].min()}~{df['val_f1'].max()}")


if __name__ == "__main__":
    gen_s45()
    gen_s46()
    print("\n两个大规模 CSV 已生成。")
