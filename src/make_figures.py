"""Draws the main figure from results/results.csv. Never edit a figure by hand.

    python src/make_figures.py
    python src/make_figures.py --norm l2
    python src/make_figures.py --n 500
    python src/make_figures.py --layout row   (lower figure, if space is short)

One panel per attack, one curve per model: accuracy under attack against the
perturbation budget. The figure is drawn at the width of one IEEE column
(3.5 inches), so the text keeps its size in the paper.
"""
import argparse
import os
import matplotlib
matplotlib.use("Agg")   # draw into a file, not on screen
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# names as they appear in the paper tables
MODEL_NAMES = {"resnet50": "ResNet-50", "efficientnet": "EfficientNet-B0",
               "vit": "ViT-B/16", "convnext": "ConvNeXt-T"}
MODEL_ORDER = ["resnet50", "efficientnet", "vit", "convnext"]
# only attacks that use a budget; DeepFool has no eps, so it has no curve
ATTACKS = ["fgsm", "bim", "pgd"]

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="imagenette")
ap.add_argument("--norm", default="linf")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--layout", default="grid", choices=["grid", "row"])
args = ap.parse_args()

table = pd.read_csv(os.path.join(ROOT, "results", "results.csv"))
table = table[(table["dataset"] == args.dataset)
              & (table["norm"] == args.norm)
              & (table["n"] == args.n)
              & (table["attack"].isin(ATTACKS))]
if len(table) == 0:
    raise SystemExit(f"no rows for {args.dataset} {args.norm} n={args.n} in results.csv")

# "4/255" is text; turn it into a number so it can go on an axis
table["eps_number"] = table["eps"].map(lambda s: eval(s))

# if a run was repeated, keep the last one
table = table.drop_duplicates(subset=["model", "attack", "eps"], keep="last")

attacks = [a for a in ATTACKS if a in set(table["attack"])]
models = [m for m in MODEL_ORDER if m in set(table["model"])]

plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "legend.fontsize": 7})

if args.layout == "grid":
    # 2 x 2 grid: one panel per attack, the last free panel holds the legend
    figure, grid = plt.subplots(2, 2, figsize=(3.5, 3.3), sharey=True)
    panels = list(grid.flatten())
    first_column = [grid[0][0], grid[1][0]]
else:
    # one row of panels, legend below them; about half the height of the grid
    plt.rcParams.update({"font.size": 6, "axes.titlesize": 7, "legend.fontsize": 6})
    figure, row = plt.subplots(1, len(ATTACKS), figsize=(3.5, 1.75), sharey=True)
    panels = list(row)
    first_column = [row[0]]

for panel, attack in zip(panels, attacks):
    for model in models:
        curve = table[(table["attack"] == attack) & (table["model"] == model)]
        curve = curve.sort_values("eps_number")
        if len(curve) > 0:
            x = curve["eps_number"] * 255 if args.norm == "linf" else curve["eps_number"]
            panel.plot(x, curve["robust_acc"] * 100, marker="o", markersize=3,
                       linewidth=1, label=MODEL_NAMES.get(model, model))
    panel.set_title(attack.upper())
    panel.set_xscale("log", base=2)
    panel.set_ylim(0, 100)
    panel.grid(alpha=0.3)
    if args.norm == "linf":
        panel.set_xlabel(r"$\epsilon$ ($\times 1/255$)")
        ticks = sorted(set(table["eps_number"] * 255))
        panel.set_xticks(ticks)
        panel.set_xticklabels([f"{t:g}" for t in ticks])
    else:
        panel.set_xlabel(r"$\epsilon$ ($L_2$)")

for panel in first_column:
    panel.set_ylabel("accuracy under attack (%)")

handles, labels = panels[0].get_legend_handles_labels()
for panel in panels[len(attacks):]:
    panel.axis("off")
if args.layout == "row":
    # legend on one line under the panels
    figure.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
                  handlelength=1.5, columnspacing=1.0)
    figure.tight_layout(rect=(0, 0.1, 1, 1))
elif len(attacks) < len(panels):
    # legend in the free panel of the grid
    panels[-1].legend(handles, labels, loc="center", frameon=False)
    figure.tight_layout()
else:
    panels[-1].legend(fontsize=6)
    figure.tight_layout()

os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
suffix = "" if args.layout == "grid" else "_row"
output = os.path.join(ROOT, "figures", f"robust_acc_{args.dataset}_{args.norm}_n{args.n}{suffix}.png")
figure.savefig(output, dpi=300)
print("written:", output)
