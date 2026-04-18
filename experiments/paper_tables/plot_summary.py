"""Plot helper for paper table summary.csv."""

import argparse
import os
import csv
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot summary.csv")
    parser.add_argument("--summary", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    rows = []
    with open(args.summary, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        raise ValueError("summary.csv is empty")

    names = [r.get("name", "") for r in rows]
    top1 = [float(r.get("final_top1", "nan")) for r in rows]
    top5 = [float(r.get("final_top5", "nan")) for r in rows]

    x = range(len(names))
    plt.figure(figsize=(10, 4))
    plt.bar(x, top1, label="Top-1")
    plt.xticks(x, names, rotation=30, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.title("Summary Top-1")
    plt.tight_layout()

    out = args.out or os.path.join(os.path.dirname(args.summary), "summary_top1.png")
    plt.savefig(out, dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.bar(x, top5, label="Top-5", color="orange")
    plt.xticks(x, names, rotation=30, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.title("Summary Top-5")
    plt.tight_layout()

    out2 = args.out or os.path.join(os.path.dirname(args.summary), "summary_top5.png")
    plt.savefig(out2, dpi=150)
    plt.close()

    print(f"Saved plots: {out}, {out2}")


if __name__ == "__main__":
    main()
