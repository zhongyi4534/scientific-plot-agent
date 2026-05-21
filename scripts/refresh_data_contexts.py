"""
scripts/refresh_data_contexts.py

当 tools/loader._build_context() 格式变更后，用此脚本刷新所有配对文件中
存储的 data_context 字符串，使其与当前格式保持一致。

处理文件（存在才处理）：
    data/pairs/valid_pairs.jsonl
    data/pairs/delta_pairs.jsonl
    data/pairs/manual_pairs.jsonl

用法：
    python scripts/refresh_data_contexts.py
    python scripts/refresh_data_contexts.py --dry-run   # 只打印差异，不写入
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.loader import DataLoadError, load_data

PROJECT_ROOT = Path(__file__).parent.parent
PAIRS_DIR = PROJECT_ROOT / "data" / "pairs"

TARGET_FILES: list[Path] = [
    PAIRS_DIR / "valid_pairs.jsonl",
    PAIRS_DIR / "delta_pairs.jsonl",
    PAIRS_DIR / "manual_pairs.jsonl",
]


def refresh_file(path: Path, dry_run: bool) -> tuple[int, int, int]:
    """
    刷新单个 JSONL 文件中所有记录的 data_context 字段。

    Args:
        path:    JSONL 文件路径。
        dry_run: True 时只打印差异，不写回文件。

    Returns:
        (total, updated, skipped) 三元组。
    """
    if not path.exists():
        print(f"  跳过（文件不存在）：{path.name}")
        return 0, 0, 0

    records: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    total = len(records)
    updated = 0
    skipped = 0

    for rec in records:
        csv_path_str: str = rec.get("csv_path", "")
        if not csv_path_str:
            print(f"  ⚠ 记录 {rec.get('id', '?')} 无 csv_path，跳过")
            skipped += 1
            continue

        # csv_path 存储为相对路径（相对于项目根目录），支持正斜杠和反斜杠
        csv_path = PROJECT_ROOT / Path(csv_path_str.replace("\\", "/"))
        if not csv_path.exists():
            print(f"  ⚠ CSV 不存在，跳过 {rec.get('id', '?')}：{csv_path_str}")
            skipped += 1
            continue

        try:
            new_context, _ = load_data(str(csv_path))
        except DataLoadError as exc:
            print(f"  ⚠ 加载失败，跳过 {rec.get('id', '?')}：{exc}")
            skipped += 1
            continue

        old_context: str = rec.get("data_context", "")
        if old_context != new_context:
            if dry_run:
                print(f"  [dry-run] 将更新：{rec.get('id', '?')}")
            else:
                rec["data_context"] = new_context
            updated += 1

    if not dry_run:
        if updated > 0:
            with path.open("w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  ✓ {path.name}：{updated}/{total} 条已更新，{skipped} 条跳过")
        else:
            print(f"  ✓ {path.name}：全部 {total} 条无需更新")
    else:
        print(f"  [dry-run] {path.name}：{updated} 条需更新，{skipped} 条跳过，共 {total} 条")

    return total, updated, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="刷新配对文件中存储的 data_context 字段")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只打印哪些记录需要更新，不实际写入文件",
    )
    args = parser.parse_args()

    mode = "[dry-run] " if args.dry_run else ""
    print(f"{mode}刷新 data_context（基于当前 _build_context() 格式）\n")

    total_all = updated_all = skipped_all = 0
    for path in TARGET_FILES:
        print(f"处理 {path.name}：")
        t, u, s = refresh_file(path, args.dry_run)
        total_all += t
        updated_all += u
        skipped_all += s

    print(
        f"\n{'完成' if not args.dry_run else 'dry-run 结束'}："
        f"共 {total_all} 条，"
        f"{'已更新' if not args.dry_run else '需更新'} {updated_all} 条，"
        f"跳过 {skipped_all} 条。"
    )


if __name__ == "__main__":
    main()
