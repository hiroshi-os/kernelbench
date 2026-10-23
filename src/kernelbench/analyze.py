"""Turn a measured GPU report into a comparison table (never invent results)."""

import argparse
import json
import math
from pathlib import Path


def markdown(report):
    if report.get("status") != "gpu_benchmark":
        raise ValueError("Only measured GPU benchmark reports can produce a performance table")
    device = str(report["device"]).replace("\n", " ")
    lines = [f"Device: {device}; dtype: {report['dtype']}", "",
             "| Operation | Shape | Backend | Median ms | vs eager | vs compile |",
             "|---|---|---|---:|---|---|"]
    for row in report["results"]:
        for backend, result in row["backends"].items():
            median = result["median_ms"]
            if not math.isfinite(median) or median <= 0:
                raise ValueError("Report contains invalid measured latency")
            def cell(key):
                value = result.get(key)
                if value is None:
                    return "—"
                return f"{value['speedup']:.3f}x ({value['outcome']})"
            shape = "x".join(str(n) for n in row["shape"])
            lines.append(f"| {row['operation']} | {shape} | {backend} | {median:.6f} | {cell('versus_eager')} | {cell('versus_compile')} |")
    lines.extend(["", "Labels use a 5% descriptive threshold; inspect p10/p90 and repeat runs."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    try:
        print(markdown(json.loads(args.report.read_text(encoding="utf-8"))), end="")
    except (ValueError, KeyError) as exc:
        parser.exit(2, f"Cannot summarize report: {exc}\n")


if __name__ == "__main__":
    main()
