"""临时验证脚本"""
import json
from collections import Counter
from pathlib import Path

records = []
with open("data/pairs/manual_pairs.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            records.append(r)
        except Exception as e:
            print(f"Line {i} INVALID: {e}")

print(f"Total records: {len(records)}")
rt = Counter(r.get("record_type") for r in records)
print(f"record_type: {dict(rt)}")

# 展示新加的 s45/s46 记录
print("\nNew s45/s46 records:")
for r in records:
    if "s45" in r.get("id", "") or "s46" in r.get("id", ""):
        spec = r.get("plotspec", {})
        df = spec.get("data_filter", "")
        ct = spec.get("chart_type", "ask_user")
        q  = spec.get("question", "")
        print(f"  [{r['record_type']:10}] {r['id']}: {ct}  filter={df}")
        if q:
            print(f"              question={q[:60]}...")
