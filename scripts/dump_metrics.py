"""Minimal script: print epoch metrics for latest chess_v2_10k aim run."""
from aim import Repo

repo = Repo(".aim")
runs = sorted(
    [r for r in repo.iter_runs() if r.experiment == "chess_v2_10k"],
    key=lambda r: r.end_time or 0,
    reverse=True,
)
if not runs:
    raise SystemExit("No runs found")

run = runs[0]
va = {m.step: m.value for m in run.metrics("val_accuracy")}
ta = {m.step: m.value for m in run.metrics("train_accuracy")}
vl = {m.step: m.value for m in run.metrics("val_loss")}

out = []
for ep in sorted(va):
    out.append(
        f"E{ep:02d} ta={ta.get(ep,0):.4f} va={va[ep]:.4f} vl={vl.get(ep,0):.4f}"
    )
with open("metrics_out.txt", "w") as f:
    f.write("\n".join(out))
print("done")
