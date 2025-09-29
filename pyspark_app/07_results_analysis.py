# 07_results_analysis.py
# Goal: Read the scaling results from training/evaluation runs,
#       compute Speedup and Efficiency, save summarized CSVs,
#       and print a neat table to the console.

import csv
import io
import os
from typing import List, Dict, Tuple

# I keep all run artifacts here
RUNS_DIR = "../runs"
TRAIN_IN = os.path.join(RUNS_DIR, "scaling_results.csv")          # from training loops
EVAL_IN  = os.path.join(RUNS_DIR, "scaling_eval_results.csv")     # from evaluation loops
TRAIN_OUT = os.path.join(RUNS_DIR, "training_scaling_summary.csv")
EVAL_OUT  = os.path.join(RUNS_DIR, "eval_scaling_summary.csv")

# Helper: read CSV flexibly and validate required columns
def _read_rows(path: str, must_cols: List[str]) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")

    # I read the whole file so I can strip BOM if it exists
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        raw = f.read()

    f2 = io.StringIO(raw)
    reader = csv.reader(f2)

    try:
        header = next(reader)
    except StopIteration:
        raise RuntimeError(f"{path} is empty.")

    # Normalize header to avoid small formatting issues
    norm_hdr = [h.strip().lstrip("\ufeff").lower() for h in header]
    need = [c.strip().lower() for c in must_cols]
    missing = [c for c in need if c not in norm_hdr]
    if missing:
        raise RuntimeError(f"{path} is missing required columns {missing}")

    # Build a list of dict rows (keyed by normalized header)
    rows: List[Dict[str, str]] = []
    for row in reader:
        if not row or all((c is None or str(c).strip() == "") for c in row):
            continue
        d = {norm_hdr[i]: row[i] if i < len(norm_hdr) else "" for i in range(len(norm_hdr))}
        rows.append(d)
    return rows

# Small converters (they won't crash if the cell is messy)
def _to_int(x, default=0) -> int:
    try:
        return int(float(str(x).strip()))
    except Exception:
        return default

def _to_float(x, default=0.0) -> float:
    try:
        return float(str(x).strip())
    except Exception:
        return default

# Baseline is the smallest 'cores' row (e.g., 2 cores)
def _pick_baseline(rows: List[Dict[str, str]]) -> Tuple[int, float]:
    if not rows:
        raise RuntimeError("No rows to pick a baseline from.")
    sorted_rows = sorted(rows, key=lambda r: _to_int(r.get("cores", 0), 0))
    base = sorted_rows[0]
    return _to_int(base.get("cores", 0), 0), _to_float(base.get("duration_sec", 0.0), 0.0)

# Compute Speedup and Efficiency for each row 
# Speedup(c)   = T_baseline / T_c
# Efficiency(c)= Speedup(c) / (c / baseline_cores)
def _add_speedup_efficiency(rows: List[Dict[str, str]],
                            base_cores: int,
                            base_time: float) -> List[Dict[str, str]]:
    out = []
    for r in rows:
        c = _to_int(r.get("cores", 0), 0)
        t = _to_float(r.get("duration_sec", 0.0), 0.0)
        sp = (base_time / t) if (t > 0) else 0.0
        denom = (c / base_cores) if (base_cores > 0) else 0.0
        eff = (sp / denom) if denom > 0 else 0.0

        rr = dict(r)           # keep original columns
        rr["speedup"] = f"{sp:.4f}"
        rr["eff"] = f"{eff:.4f}"
        out.append(rr)

    # I sort by cores to make the output table nice
    out = sorted(out, key=lambda r: _to_int(r.get("cores", 0), 0))
    return out

# Write a tiny CSV with the columns I care about 
def _write_csv(path: str, rows: List[Dict[str, str]], cols: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])

# Pretty console table (just for quick inspection)
def _pretty_print(title: str, rows: List[Dict[str, str]]) -> None:
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))
    print(f"{'cores':>5} | {'duration(s)':>12} | {'speedup':>7} | {'eff':>6}")
    print("-"*40)
    for r in rows:
        print(f"{_to_int(r.get('cores')):>5} | {_to_float(r.get('duration_sec')):>12.2f} | "
              f"{_to_float(r.get('speedup')):>7.3f} | {_to_float(r.get('eff')):>6.3f}")

# Main flow: summarize training and evaluation scaling
def main():
    # 1) Training scaling summary
    train_cols = ["cores", "duration_sec"]  # I only need these two to compute scaling
    train_rows = _read_rows(TRAIN_IN, train_cols)
    base_c_train, base_t_train = _pick_baseline(train_rows)
    train_sum = _add_speedup_efficiency(train_rows, base_c_train, base_t_train)

    _write_csv(
        TRAIN_OUT,
        train_sum,
        cols=["cores", "duration_sec", "speedup", "eff"]
    )
    _pretty_print("Training Scalability (from scaling_results.csv)", train_sum)

    # 2) Evaluation scaling summary (plus accuracy metrics for context)
    eval_cols = ["cores", "duration_sec", "accuracy", "f1",
                 "weighted_precision", "weighted_recall", "auc"]
    eval_rows = _read_rows(EVAL_IN, eval_cols)
    base_c_eval, base_t_eval = _pick_baseline(eval_rows)
    eval_sum = _add_speedup_efficiency(eval_rows, base_c_eval, base_t_eval)

    _write_csv(
        EVAL_OUT,
        eval_sum,
        cols=["cores", "duration_sec", "speedup", "eff",
              "accuracy", "f1", "weighted_precision", "weighted_recall", "auc"]
    )
    _pretty_print("Evaluation Scalability (from scaling_eval_results.csv)", eval_sum)

    print(f"\nWrote training summary -> {TRAIN_OUT}")
    print(f"Wrote eval summary -> {EVAL_OUT}")

if __name__ == "__main__":
    main()
