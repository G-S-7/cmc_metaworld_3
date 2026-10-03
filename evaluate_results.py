import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt


def forgetting(df):
    rows = []
    for task_id, g in df.groupby("evaluated_task_id"):
        g = g.sort_values("stage")
        first = g.iloc[0]["success"]
        final = g.iloc[-1]["success"]
        best = g["success"].max()
        rows.append({
            "task_id": task_id,
            "task": g.iloc[0]["evaluated_task"],
            "first_success": first,
            "best_success": best,
            "final_success": final,
            "forgetting": best - final,
        })
    return pd.DataFrame(rows)


p = argparse.ArgumentParser()
p.add_argument("results_json")
p.add_argument("--save-prefix", default="cmc")
args = p.parse_args()

with open(args.results_json) as f:
    data = json.load(f)

df = pd.DataFrame(data)
df.to_csv(args.save_prefix + "_results.csv", index=False)

fg = forgetting(df)
fg.to_csv(args.save_prefix + "_forgetting.csv", index=False)

print("\nForgetting:")
print(fg.to_string(index=False))

# Heatmap-like task retention matrix
pivot = df.pivot_table(
    index="stage", columns="evaluated_task_id",
    values="success", aggfunc="mean"
)

plt.figure(figsize=(10, 6))
plt.imshow(pivot.values, aspect="auto", interpolation="nearest")
plt.colorbar(label="Success rate")
plt.xlabel("Evaluated task ID")
plt.ylabel("Training stage")
plt.title("Continual RL retention matrix")
plt.tight_layout()
plt.savefig(args.save_prefix + "_retention.png", dpi=180)
plt.close()
