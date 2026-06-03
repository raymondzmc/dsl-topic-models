"""Print a console summary of locally-stored results (a verification helper).

Walks ``results/<dataset>/<run_name>/averaged_results.json`` (and ``retrieval.json``
when present) and tabulates the key metrics for quick inspection. This is intentionally
NOT a paper-table builder; it only echoes the JSON that ``dsl-train`` already wrote.
"""
import os
import json
import glob
import argparse

METRIC_KEYS = ["cv_wiki", "llm_rating", "inverted_rbo", "topic_diversity", "purity"]
RETR_KEYS = ["precision@1", "precision@5", "precision@10"]


def _load(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def main():
    parser = argparse.ArgumentParser(description="Summarize local DSL-Topic results")
    parser.add_argument("results_dir", nargs="?", default="results",
                        help="Root results directory (default: results/)")
    parser.add_argument("--dataset", default=None, help="Restrict to results_dir/<dataset>/")
    parser.add_argument("--metrics", default=None,
                        help="Comma-separated metric keys to show (default: common metrics)")
    args = parser.parse_args()

    keys = args.metrics.split(",") if args.metrics else METRIC_KEYS
    pattern = os.path.join(args.results_dir, args.dataset if args.dataset else "*", "*")

    rows = []
    for run_dir in sorted(glob.glob(pattern)):
        avg = _load(os.path.join(run_dir, "averaged_results.json"))
        if not avg:
            continue
        retr = _load(os.path.join(run_dir, "retrieval.json"))
        rows.append((os.path.relpath(run_dir, args.results_dir), avg, retr))

    if not rows:
        print(f"No averaged_results.json found under {args.results_dir}. "
              f"Train models first with dsl-train.")
        return

    show_retr = any(retr for _, _, retr in rows)
    cols = keys + (RETR_KEYS if show_retr else [])
    name_w = max([len(r[0]) for r in rows] + [3])

    header = "run".ljust(name_w) + "  " + "  ".join(c.rjust(9) for c in cols)
    print(header)
    print("-" * len(header))
    for rel, avg, retr in rows:
        cells = []
        for c in cols:
            v = retr.get(c) if c in RETR_KEYS else avg.get(c)
            cells.append(f"{v:9.3f}" if isinstance(v, (int, float)) else " " * 9)
        print(rel.ljust(name_w) + "  " + "  ".join(cells))


if __name__ == "__main__":
    main()
