"""
scripts/run_synthesis.py

训练数据合成流水线的一键运行脚本。按顺序执行以下步骤：

  步骤 1    gen_csv.py          —— 生成场景 CSV（含列名随机化）
  步骤 2    gen_pairs.py        —— 调用 DeepSeek API 合成首轮 (user_input, plotspec) 配对
  步骤 3    validate_pairs.py   —— 四级校验（字段 + 列名 + 语义 + 渲染）→ valid_pairs.jsonl
  步骤 3.5  gen_pairs.py --supplement  —— 补充低频主题 / 缺失字段样本（可选，需指定 --with-supplement）
  步骤 4    gen_delta.py        —— 生成修改轮 (user_input, delta) 配对 → delta_pairs.jsonl
  步骤 5    pack_finetune.py    —— 合并首轮 + 修改轮，打包为 Qwen3 ChatML 微调 JSONL

前置条件：
  - 在 .env 或环境变量中设置 DEEPSEEK_API_KEY
  - pip install -r requirements.txt

用法示例：

  # 快速测试（只合成 2 个 CSV 的配对，跳过渲染校验，修改轮只用规则驱动）
  python scripts/run_synthesis.py --limit 2 --no-render --delta-no-llm

  # 完整运行（指定模型）
  python scripts/run_synthesis.py --model deepseek-chat

  # 跳过 CSV 生成（data/train/ 已存在时）
  python scripts/run_synthesis.py --skip-csv --model deepseek-chat

  # 跳过步骤1-3，只重新生成修改轮并打包
  python scripts/run_synthesis.py --skip-to-delta

  # 只重新打包（valid_pairs.jsonl 和 delta_pairs.jsonl 均已存在）
  python scripts/run_synthesis.py --only-pack
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# 项目根目录（scripts/ 的上一级）
ROOT = Path(__file__).parent.parent
SCRIPTS = Path(__file__).parent

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")


def _run(label: str, script: str, extra_args: list[str]) -> bool:
    """
    用当前 Python 解释器运行 scripts/<script>，传入额外参数。
    返回 True 表示成功（returncode == 0）。
    """
    _banner(f"步骤：{label}")
    cmd = [sys.executable, str(SCRIPTS / script)] + extra_args
    print(f"命令：{' '.join(cmd)}\n")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0
    ok = result.returncode == 0
    status = "✓ 完成" if ok else "✗ 失败"
    print(f"\n{status}  用时 {elapsed:.1f}s")
    return ok


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="训练数据合成一键脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="gen_pairs 只处理前 N 个 CSV（快速测试用）",
    )
    parser.add_argument(
        "--model", type=str, default="deepseek-chat",
        help="DeepSeek 模型名（默认：deepseek-chat）",
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="validate_pairs 跳过渲染校验（更快，但不保证 PlotSpec 可真实出图）",
    )
    parser.add_argument(
        "--skip-csv", action="store_true",
        help="跳过步骤1（data/train/ 已有 CSV 时使用）",
    )
    parser.add_argument(
        "--skip-pairs", action="store_true",
        help="跳过步骤2（raw_pairs.jsonl 已存在时）",
    )
    parser.add_argument(
        "--skip-to-delta", action="store_true",
        help="跳过步骤1-3，直接从步骤4（gen_delta）开始",
    )
    parser.add_argument(
        "--only-pack", action="store_true",
        help="只执行步骤5（valid_pairs.jsonl 和 delta_pairs.jsonl 均已存在时）",
    )
    parser.add_argument(
        "--delta-target", type=int, default=200,
        help="gen_delta 目标生成条数（默认 200）",
    )
    parser.add_argument(
        "--delta-no-llm", action="store_true",
        help="gen_delta 只用规则驱动，不调用 LLM（更快，适合测试）",
    )
    parser.add_argument(
        "--skip-delta", action="store_true",
        help="跳过步骤4（delta_pairs.jsonl 已存在或不需要修改轮数据时）",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.10,
        help="验证集比例（默认 0.10）",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="gen_pairs 追加写入（而非覆盖）raw_pairs.jsonl",
    )
    parser.add_argument(
        "--with-supplement", action="store_true",
        help="步骤3.5：校验后运行补充数据生成（P1低频主题 + P2缺失字段）",
    )
    parser.add_argument(
        "--supplement-p1-only", action="store_true",
        help="步骤3.5：只补充低频主题（P1: rococo/macaron）",
    )
    parser.add_argument(
        "--supplement-p2-only", action="store_true",
        help="步骤3.5：只补充缺失字段（P2）",
    )
    args = parser.parse_args()

    # 打印摘要
    print("\n训练数据合成流水线")
    print(f"  模型        : {args.model}")
    print(f"  CSV 限制    : {'全量' if args.limit is None else f'前 {args.limit} 个'}")
    print(f"  渲染校验    : {'关闭' if args.no_render else '开启'}")
    print(f"  修改轮目标  : {args.delta_target} 条{'（仅规则驱动）' if args.delta_no_llm else ''}")
    print(f"  验证集比例  : {args.val_ratio:.0%}")

    t_start = time.time()
    success_steps: list[str] = []
    failed_step: str | None = None

    skip_early = args.only_pack or args.skip_to_delta

    # ── 步骤 1：生成 CSV ──────────────────────────────────────────────────
    if skip_early or args.skip_csv or args.skip_pairs:
        print("\n[步骤1] 跳过 CSV 生成")
    else:
        ok = _run("生成训练 CSV（gen_csv.py）", "gen_csv.py", [])
        if ok:
            success_steps.append("步骤1 生成CSV")
        else:
            failed_step = "步骤1 gen_csv"
            _print_summary(success_steps, failed_step, t_start)
            sys.exit(1)

    # ── 步骤 2：合成配对 ──────────────────────────────────────────────────
    if skip_early or args.skip_pairs:
        print("\n[步骤2] 跳过配对合成")
    else:
        extra: list[str] = ["--model", args.model]
        if args.limit:
            extra += ["--limit", str(args.limit)]
        if args.append:
            extra.append("--append")
        ok = _run("合成配对（gen_pairs.py）", "gen_pairs.py", extra)
        if ok:
            success_steps.append("步骤2 合成配对")
        else:
            failed_step = "步骤2 gen_pairs"
            _print_summary(success_steps, failed_step, t_start)
            sys.exit(1)

    # ── 步骤 3：校验过滤 ──────────────────────────────────────────────────
    if skip_early:
        print("\n[步骤3] 跳过校验")
    else:
        extra = ["--no-render"] if args.no_render else []
        ok = _run("校验过滤（validate_pairs.py）", "validate_pairs.py", extra)
        if ok:
            success_steps.append("步骤3 校验过滤")
        else:
            failed_step = "步骤3 validate_pairs"
            _print_summary(success_steps, failed_step, t_start)
            sys.exit(1)

    # ── 步骤 3.5：补充数据（可选）────────────────────────────────────────
    if skip_early or not args.with_supplement:
        if not skip_early:
            print("\n[步骤3.5] 跳过补充数据（使用 --with-supplement 启用）")
    else:
        extra = ["--supplement", "--model", args.model]
        if args.no_render:
            extra.append("--no-render")
        if args.supplement_p1_only:
            extra.append("--p1-only")
        elif args.supplement_p2_only:
            extra.append("--p2-only")
        ok = _run("补充数据（gen_pairs.py --supplement）", "gen_pairs.py", extra)
        if ok:
            success_steps.append("步骤3.5 补充数据")
        else:
            failed_step = "步骤3.5 gen_pairs --supplement"
            _print_summary(success_steps, failed_step, t_start)
            sys.exit(1)

    # ── 步骤 4：生成修改轮数据 ────────────────────────────────────────────
    if args.only_pack or args.skip_delta:
        print("\n[步骤4] 跳过修改轮数据生成")
    else:
        extra = ["--target", str(args.delta_target), "--model", args.model]
        if args.limit:
            extra += ["--limit", str(args.limit)]
        if args.delta_no_llm:
            extra.append("--no-llm")
        ok = _run("生成修改轮数据（gen_delta.py）", "gen_delta.py", extra)
        if ok:
            success_steps.append("步骤4 修改轮数据")
        else:
            failed_step = "步骤4 gen_delta"
            _print_summary(success_steps, failed_step, t_start)
            sys.exit(1)

    # ── 步骤 5：打包 JSONL ────────────────────────────────────────────────
    extra = ["--val-ratio", str(args.val_ratio)]
    ok = _run("打包微调数据（pack_finetune.py）", "pack_finetune.py", extra)
    if ok:
        success_steps.append("步骤5 打包JSONL")
    else:
        failed_step = "步骤5 pack_finetune"

    _print_summary(success_steps, failed_step, t_start)
    if failed_step:
        sys.exit(1)


def _print_summary(
    success_steps: list[str],
    failed_step: str | None,
    t_start: float,
) -> None:
    total = time.time() - t_start
    _banner("流水线结束")
    for s in success_steps:
        print(f"  ✓  {s}")
    if failed_step:
        print(f"  ✗  {failed_step}  ← 在此停止")
    print(f"\n总用时：{total:.1f}s")
    if not failed_step:
        print("\n输出文件：")
        for p in [
            Path("data/pairs/delta_pairs.jsonl"),
            Path("data/pairs/delta_reject_log.jsonl"),
            Path("data/finetune/train.jsonl"),
            Path("data/finetune/val.jsonl"),
        ]:
            full = ROOT / p
            if full.exists():
                size_kb = full.stat().st_size / 1024
                n_lines = sum(1 for _ in full.open(encoding="utf-8"))
                print(f"  {p}  ({n_lines} 条, {size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
