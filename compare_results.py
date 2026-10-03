"""
compare_results.py
==================
Continual-learning report: SAC baseline vs SAC + CMC on T1 push -> T2 push-wall -> T3 shelf-place.

Reads every results/<before_cmc|after_cmc>/seed_<k>/ folder, uses the seeds
present in BOTH arms (paired), checks that the evaluation protocol was
identical, and reports (primary metric = success rate, secondary = return):

  1. evaluation matrix per arm      (after T1 / after T2 / after T3) x (T1, T2, T3)
  2. new-task learning              T1 after T1, T2 after T2, T3 after T3
  3. previous-task retention        T1 after T2, T1 after T3, T2 after T3
  4. forgetting                     learned value - later value
  5. CMC minus baseline for each of the above (mean +- std over seeds)

Outputs in --out-dir: report.md, metrics.json, matrix_success.png,
stage_success.png, forgetting.png, learning_curves.png (if curves exist).

    python compare_results.py --results results --out-dir results/comparison
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

from cmc_rl.tasks import TASK_SEQUENCE, TASK_LABELS
from cmc_rl.metrics import build_matrix, cl_metrics

ARMS = [("before_cmc", "SAC baseline"), ("after_cmc", "SAC + CMC")]
ARM_COLOR = {"before_cmc": "#2a78d6", "after_cmc": "#eb6834"}          # validated pair
TASK_COLOR = {"T1": "#2a78d6", "T2": "#eb6834", "T3": "#1baf7a"}        # validated trio
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"
PROTOCOL_KEYS = ["tasks", "goal_mode", "steps_per_task", "eval_episodes", "curve_episodes",
                 "eval_every", "batch_size", "warmup", "gamma", "layer_norm", "reward_scale",
                 "demo_episodes", "demo_noise", "demo_ratio"]


# ───────────────────────────────────────────────────────────────── loading
def load_runs(results):
    runs = {}
    for arm, _ in ARMS:
        for d in sorted((Path(results) / arm).glob("seed_*")):
            if not (d / "results.json").exists():
                continue
            seed = int(d.name.split("_")[1])
            runs.setdefault(arm, {})[seed] = {
                "rows": json.load(open(d / "results.json")),
                "config": json.load(open(d / "config.json")) if (d / "config.json").exists() else {},
                "curves": json.load(open(d / "learning_curves.json"))
                if (d / "learning_curves.json").exists() else [],
            }
    return runs


def check_protocol(runs, seeds):
    problems = []
    ref = None
    for arm, _ in ARMS:
        for s in seeds:
            cfg = runs[arm][s]["config"]
            proto = {k: cfg.get("args", {}).get(k) for k in PROTOCOL_KEYS}
            proto["eval_episode_seeds"] = cfg.get("eval_episode_seeds")
            if ref is None:
                ref = (arm, s, proto)
            else:
                for k, v in proto.items():
                    if v != ref[2][k]:
                        problems.append(f"{arm}/seed_{s}: {k}={v!r} differs from "
                                        f"{ref[0]}/seed_{ref[1]} ({ref[2][k]!r})")
            rows = runs[arm][s]["rows"]
            n_eps = {r["n_episodes"] for r in rows}
            if len(n_eps) != 1:
                problems.append(f"{arm}/seed_{s}: unequal episode counts {n_eps}")
    return problems


def per_seed_metrics(runs, arm, seeds, key):
    return {s: cl_metrics(build_matrix(runs[arm][s]["rows"], key), TASK_LABELS) for s in seeds}


def mean_std(values):
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return float("nan"), float("nan"), 0
    return float(np.mean(v)), float(np.std(v)), len(v)


def ms(values, spec=".2f"):
    m, s, n = mean_std(values)
    if n == 0:
        return "n/a"
    return f"{m:{spec}} ± {s:{spec}}" if n > 1 else f"{m:{spec}}"


# ─────────────────────────────────────────────────────────────────── tables
def matrix_table(runs, arm, seeds, key, spec):
    lines = ["| after stage | " + " | ".join(f"{l} {t}" for l, t in zip(TASK_LABELS, TASK_SEQUENCE)) + " |",
             "|---|" + "---|" * len(TASK_LABELS)]
    for i, si in enumerate(TASK_LABELS):
        cells = []
        for j, tj in enumerate(TASK_LABELS):
            if j > i:
                cells.append("–")
                continue
            vals = [build_matrix(runs[arm][s]["rows"], key).get(si, {}).get(tj) for s in seeds]
            cell = ms(vals, spec)
            cells.append(f"**{cell}**" if i == j else cell)
        lines.append(f"| after {si} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def cl_rows():
    rows = [("New-task learning", f"{t} after {t}", "new_task", t) for t in TASK_LABELS]
    for j, tj in enumerate(TASK_LABELS[:-1]):
        for ti in TASK_LABELS[j + 1:]:
            rows.append(("Retained value", f"{tj} after {ti}", "matrix", (ti, tj)))
    for j, tj in enumerate(TASK_LABELS[:-1]):
        for ti in TASK_LABELS[j + 1:]:
            rows.append(("Forgetting", f"{tj}: after {tj} − after {ti}", "forgetting", f"{tj}_after_{ti}"))
    rows.append(("Summary", "Average final (all tasks after T3)", "avg_final", None))
    rows.append(("Summary", "Average forgetting (T1, T2 after T3)", "avg_forgetting_final", None))
    return rows


def metric_value(runs, arm, seed, met, kind, ref, key):
    if kind == "matrix":
        return build_matrix(runs[arm][seed]["rows"], key).get(ref[0], {}).get(ref[1])
    if kind in ("avg_final", "avg_forgetting_final"):
        return met[kind]
    return met[kind].get(ref)


def cl_table(runs, seeds, key, spec):
    mets = {arm: per_seed_metrics(runs, arm, seeds, key) for arm, _ in ARMS}
    lines = ["| group | metric | SAC baseline | SAC + CMC | CMC − baseline |",
             "|---|---|---|---|---|"]
    data = {}
    for group, name, kind, ref in cl_rows():
        vals = {arm: [metric_value(runs, arm, s, mets[arm][s], kind, ref, key) for s in seeds]
                for arm, _ in ARMS}
        diff = [None if (a is None or b is None) else b - a
                for a, b in zip(vals["before_cmc"], vals["after_cmc"])]
        data[name] = {"baseline": vals["before_cmc"], "cmc": vals["after_cmc"], "diff": diff}
        lines.append(f"| {group} | {name} | {ms(vals['before_cmc'], spec)} | "
                     f"{ms(vals['after_cmc'], spec)} | {ms(diff, spec)} |")
    return "\n".join(lines), data


# ──────────────────────────────────────────────────────────────────── plots
def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_matrices(runs, seeds, out):
    n = len(TASK_LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue", ["#f4f8fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
    for ax, (arm, name) in zip(axes, ARMS):
        mat = np.full((n, n), np.nan)
        for i, si in enumerate(TASK_LABELS):
            for j, tj in enumerate(TASK_LABELS[: i + 1]):
                mat[i, j] = mean_std([build_matrix(runs[arm][s]["rows"]).get(si, {}).get(tj)
                                      for s in seeds])[0]
        ax.imshow(np.ma.masked_invalid(mat), vmin=0, vmax=1, cmap=cmap)
        for i in range(n):
            for j in range(n):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=11,
                            color="white" if mat[i, j] > 0.55 else INK,
                            fontweight="bold" if i == j else "normal")
        ax.set_xticks(range(n), [f"{l}\n{t}" for l, t in zip(TASK_LABELS, TASK_SEQUENCE)], fontsize=9)
        ax.set_yticks(range(n), [f"after {l}" for l in TASK_LABELS], fontsize=9)
        ax.set_title(f"{name}: success rate", fontsize=11, color=INK, loc="left")
        for side in ax.spines.values():
            side.set_visible(False)
    fig.text(0.01, 0.01, f"Mean over {len(seeds)} seed(s). Diagonal (bold) = new-task learning; "
             "below diagonal = retention of earlier tasks.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_stage_success(runs, seeds, out):
    """Success of each task across stages: one panel per task, one line per arm."""
    fig, axes = plt.subplots(1, len(TASK_LABELS), figsize=(11, 3.6), sharey=True)
    for ax, (j, tj) in zip(axes, enumerate(TASK_LABELS)):
        stages = TASK_LABELS[j:]
        for arm, name in ARMS:
            m, s = [], []
            for si in stages:
                mu, sd, _ = mean_std([build_matrix(runs[arm][x]["rows"]).get(si, {}).get(tj)
                                      for x in seeds])
                m.append(mu); s.append(0 if math.isnan(sd) else sd)
            x = np.arange(len(stages))
            ax.plot(x, m, "-o", color=ARM_COLOR[arm], linewidth=2, markersize=7, label=name)
            if len(seeds) > 1:
                ax.fill_between(x, np.array(m) - s, np.array(m) + s, color=ARM_COLOR[arm],
                                alpha=0.12, linewidth=0)
        ax.set_xticks(range(len(stages)), [f"after {si}" for si in stages], fontsize=9)
        ax.set_xlim(-0.3, max(len(stages) - 0.7, 0.3))
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"{tj} {TASK_SEQUENCE[j]}", fontsize=10, color=INK, loc="left")
        _style(ax)
    axes[0].set_ylabel("success rate", color=MUTED)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_forgetting(cl_data, seeds, out):
    names = [n for n in cl_data if ": after" in n]
    x = np.arange(len(names))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for k, (arm, name) in enumerate(ARMS):
        vals = [mean_std(cl_data[n]["baseline" if arm == "before_cmc" else "cmc"])
                for n in names]
        mu = [v[0] for v in vals]
        sd = [0 if math.isnan(v[1]) else v[1] for v in vals]
        ax.bar(x + (k - 0.5) * w, mu, w - 0.03, yerr=sd if len(seeds) > 1 else None,
               color=ARM_COLOR[arm], label=name, capsize=3, error_kw={"ecolor": MUTED})
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(x, [f"{n.split(':')[0]} after {n.split('after ')[-1]}" for n in names],
                  fontsize=9)
    ax.set_ylabel("forgetting (success drop)", color=MUTED)
    ax.set_title("Forgetting of earlier tasks (lower is better)", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_curves(runs, seeds, out):
    if not all(runs[arm][s]["curves"] for arm, _ in ARMS for s in seeds):
        return False
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, (arm, name) in zip(axes, ARMS):
        steps_per_task = runs[arm][seeds[0]]["config"].get("args", {}).get("steps_per_task")
        for tj in TASK_LABELS:
            by_step = {}
            for s in seeds:
                for pt in runs[arm][s]["curves"]:
                    if tj in pt:
                        by_step.setdefault(pt["global_step"], []).append(pt[tj]["success"])
            if not by_step:
                continue
            xs = sorted(by_step)
            ax.plot(xs, [np.mean(by_step[x]) for x in xs], color=TASK_COLOR[tj], linewidth=2,
                    label=f"{tj} {TASK_SEQUENCE[TASK_LABELS.index(tj)]}")
        if steps_per_task:
            for k in range(1, len(TASK_LABELS)):
                ax.axvline(k * steps_per_task, color=MUTED, linewidth=0.8, linestyle=":")
            for k, tl in enumerate(TASK_LABELS):
                ax.text((k + 0.5) * steps_per_task, 1.06, f"training {tl}", ha="center",
                        fontsize=8, color=MUTED)
        ax.set_ylim(-0.03, 1.12)
        ax.set_title(name, fontsize=11, color=INK, loc="left", pad=16)
        ax.set_xlabel("environment steps", color=MUTED)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1e3:g}k"))
        _style(ax)
    axes[0].set_ylabel("success rate (curve episodes)", color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=len(labels),
               loc="lower center")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return True


# ───────────────────────────────────────────────────────────────────── main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--out-dir", default="results/comparison")
    p.add_argument("--learned-threshold", type=float, default=0.8)
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs = load_runs(a.results)
    seeds = sorted(set(runs.get("before_cmc", {})) & set(runs.get("after_cmc", {})))
    if not seeds:
        raise SystemExit("No seed has results for BOTH before_cmc and after_cmc.")
    problems = check_protocol(runs, seeds)

    md = ["# Continual learning: SAC baseline vs SAC + CMC", "",
          f"Tasks: {' → '.join(f'{l} {t}' for l, t in zip(TASK_LABELS, TASK_SEQUENCE))}  ",
          f"Paired seeds: {seeds} (values are mean ± std over seeds)", ""]
    if problems:
        md += ["## ⚠ Protocol mismatch — comparison is NOT valid", ""] + [f"- {x}" for x in problems] + [""]
    else:
        md += ["Evaluation protocol identical in both arms (same tasks, budget, episode count "
               "and evaluation episodes).", ""]

    # Was each task actually learned? Forgetting is only meaningful if it was.
    learned_note = []
    for arm, name in ARMS:
        for tj in TASK_LABELS:
            mu = mean_std([build_matrix(runs[arm][s]["rows"]).get(tj, {}).get(tj) for s in seeds])[0]
            if not mu >= a.learned_threshold:
                learned_note.append(f"- {name}: {tj} reached only {mu:.2f} success right after "
                                    f"training (< {a.learned_threshold}).")
    if learned_note:
        md += ["## ⚠ Tasks not reliably learned", "",
               "A task that was never reliably learned cannot meaningfully be forgotten; read its numbers with care.", ""] \
              + learned_note + [""]

    md += ["## Success rate (primary)", ""]
    for arm, name in ARMS:
        md += [f"### {name}", "", matrix_table(runs, arm, seeds, "mean_success", ".2f"), ""]
    succ_table, succ_data = cl_table(runs, seeds, "mean_success", ".2f")
    md += ["### Continual-learning metrics (success)", "", succ_table, "",
           "Forgetting > 0 means success dropped after later training; CMC − baseline < 0 "
           "for forgetting means CMC forgot less.", ""]

    md += ["## Return (secondary)", ""]
    for arm, name in ARMS:
        md += [f"### {name}", "", matrix_table(runs, arm, seeds, "mean_return", ".0f"), ""]
    ret_table, ret_data = cl_table(runs, seeds, "mean_return", ".0f")
    md += ["### Continual-learning metrics (return)", "", ret_table, ""]

    plot_matrices(runs, seeds, out / "matrix_success.png")
    plot_stage_success(runs, seeds, out / "stage_success.png")
    plot_forgetting(succ_data, seeds, out / "forgetting.png")
    has_curves = plot_curves(runs, seeds, out / "learning_curves.png")
    md += ["## Figures", "", "![matrix](matrix_success.png)", "",
           "![per-task success across stages](stage_success.png)", "",
           "![forgetting](forgetting.png)", ""]
    if has_curves:
        md += ["![learning curves](learning_curves.png)", ""]
    if len(seeds) < 3:
        md += [f"_Only {len(seeds)} seed(s): differences between arms are not statistically "
               "supported. Run at least 3 seeds per arm before drawing conclusions._", ""]

    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    with open(out / "metrics.json", "w") as f:
        json.dump({"seeds": seeds, "protocol_problems": problems,
                   "success": succ_data, "return": ret_data}, f, indent=2)
    print("\n".join(md))
    print(f"\nSaved report and figures to {out}")


if __name__ == "__main__":
    main()
