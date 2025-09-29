import os
import io
import pandas as pd
import matplotlib.pyplot as plt

RUNS_DIR  = "../runs"
PLOTS_DIR = os.path.join(RUNS_DIR, "plots")
TRAIN_SUM = os.path.join(RUNS_DIR, "training_scaling_summary.csv")
EVAL_SUM  = os.path.join(RUNS_DIR, "eval_scaling_summary.csv")

os.makedirs(PLOTS_DIR, exist_ok=True)

def read_csv_robust(path):
    if not os.path.exists(path):
        print(f"[!] Not found: {path}")
        return None
    with open(path, "r", encoding="utf-8-sig") as f:
        text = f.read()
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ["cores", "duration_sec", "speedup", "eff"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "cores" in df.columns:
        df = df.dropna(subset=["cores"]).sort_values("cores")
    return df

def ensure_speedup_efficiency(df):
    need_compute = False
    if "speedup" not in df.columns or df["speedup"].isna().all():
        need_compute = True
    if "eff" not in df.columns or df["eff"].isna().all():
        need_compute = True

    if not need_compute:
        return df

    if "cores" not in df.columns or "duration_sec" not in df.columns:
        print("Cannot compute speedup/eff: missing 'cores' or 'duration_sec'.")
        return df

    df = df.copy()
    df = df.dropna(subset=["cores", "duration_sec"]).sort_values("cores")
    base_row = df.iloc[0]
    base_cores = float(base_row["cores"])
    base_time  = float(base_row["duration_sec"])

    sp = []
    ef = []
    for _, r in df.iterrows():
        c = float(r["cores"])
        t = float(r["duration_sec"])
        s = (base_time / t) if t > 0 else 0.0
        ideal = (c / base_cores) if base_cores > 0 else 0.0
        e = (s / ideal) if ideal > 0 else 0.0
        sp.append(s)
        ef.append(e)

    df["speedup"] = sp
    df["eff"] = ef
    return df

def plot_line(x, y, title, xlabel, ylabel, out_path, marker="o"):
    plt.figure()
    plt.plot(x, y, marker=marker)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def make_all_plots(df, tag):
    if "duration_sec" in df.columns:
        plot_line(
            df["cores"], df["duration_sec"],
            f"{tag.capitalize()} — Duration vs Cores",
            "Cores", "Duration (sec)",
            os.path.join(PLOTS_DIR, f"{tag}_duration.png"),
            marker="o"
        )
    else:
        print(f"[!] '{tag}': missing duration_sec")
    if "speedup" in df.columns:
        plot_line(
            df["cores"], df["speedup"],
            f"{tag.capitalize()} — Speedup vs Cores",
            "Cores", "Speedup",
            os.path.join(PLOTS_DIR, f"{tag}_speedup.png"),
            marker="s"
        )
    else:
        print(f"[!] '{tag}': missing speedup")

    if "eff" in df.columns:
        plot_line(
            df["cores"], df["eff"],
            f"{tag.capitalize()} — Efficiency vs Cores",
            "Cores", "Efficiency",
            os.path.join(PLOTS_DIR, f"{tag}_efficiency.png"),
            marker="^"
        )
    else:
        print(f"[!] '{tag}': missing eff")

def main():
    train_df = read_csv_robust(TRAIN_SUM)
    if train_df is not None and not train_df.empty:
        train_df = ensure_speedup_efficiency(train_df)
        make_all_plots(train_df, "training")
    else:
        print("Skipped training plots (no data).")

    eval_df = read_csv_robust(EVAL_SUM)
    if eval_df is not None and not eval_df.empty:
        eval_df = ensure_speedup_efficiency(eval_df)
        make_all_plots(eval_df, "evaluation")
    else:
        print("Skipped evaluation plots (no data).")

if __name__ == "__main__":
    main()
