"""Continual-learning metrics computed from the evaluation matrix.

A[stage][task] = mean success (or return) on `task` evaluated right after
training on `stage`. Only tasks already trained (task <= stage) are evaluated.

    new-task learning   L_j        = A[j][j]
    after-stage value   A[i][j]    for every later stage i > j
    forgetting          F_j(i)     = A[j][j] - A[i][j]        (positive = forgot)
    retention           R_j(i)     = A[i][j] / A[j][j]        (undefined if A[j][j] == 0)
    final value         A[last][j]
    average forgetting  mean_j F_j(last) over tasks learned before the last stage
                        and with A[j][j] > 0 (a task never learned cannot be forgotten)
"""

import math


def build_matrix(rows, key="mean_success"):
    """rows: list of result dicts with trained_task_label / eval_task_label."""
    mat = {}
    for r in rows:
        mat.setdefault(r["trained_task_label"], {})[r["eval_task_label"]] = r[key]
    return mat


def cl_metrics(matrix, labels):
    """labels: stage labels in training order, e.g. ['T1','T2','T3']."""
    out = {"new_task": {}, "final": {}, "forgetting": {}, "retention": {}}
    last = labels[-1]
    for j, tj in enumerate(labels):
        base = matrix.get(tj, {}).get(tj)
        if base is None:
            continue
        out["new_task"][tj] = base
        out["final"][tj] = matrix.get(last, {}).get(tj)
        for ti in labels[j + 1:]:
            cur = matrix.get(ti, {}).get(tj)
            if cur is None:
                continue
            out["forgetting"][f"{tj}_after_{ti}"] = base - cur
            out["retention"][f"{tj}_after_{ti}"] = (cur / base) if base > 0 else float("nan")

    learned_before_last = [tj for tj in labels[:-1]
                           if out["new_task"].get(tj, 0) > 0
                           and f"{tj}_after_{last}" in out["forgetting"]]
    out["avg_forgetting_final"] = (
        sum(out["forgetting"][f"{tj}_after_{last}"] for tj in learned_before_last)
        / len(learned_before_last) if learned_before_last else float("nan"))
    out["tasks_counted_in_avg_forgetting"] = learned_before_last
    finals = [v for v in out["final"].values() if v is not None]
    out["avg_final"] = sum(finals) / len(finals) if finals else float("nan")
    return out


def fmt(v, spec=".2f"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return format(v, spec)
