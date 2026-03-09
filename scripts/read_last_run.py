"""Print epoch metrics for the most recent chess_v2_10k aim run."""
from aim import Repo

repo = Repo(".aim")
runs = sorted(
    [r for r in repo.iter_runs() if r.experiment == "chess_v2_10k"],
    key=lambda r: r.end_time or 0,
    reverse=True,
)
if not runs:
    print("No chess_v2_10k runs found")
    raise SystemExit(1)

run = runs[0]
print(f"Run: {run.hash}  ended: {run.end_time}")

val_acc   = {m.step: m.value for m in run.metrics("val_accuracy")}
train_acc = {m.step: m.value for m in run.metrics("train_accuracy")}
val_loss  = {m.step: m.value for m in run.metrics("val_loss")}

for ep in sorted(val_acc):
    ta = train_acc.get(ep, 0)
    va = val_acc[ep]
    vl = val_loss.get(ep, 0)
    print(
        f"Epoch {ep:02d}: train_acc={ta:.4f} "
        f"val_acc={va:.4f} val_loss={vl:.4f} "
        f"gap={va - ta:+.4f}"
    )

best_va = max(val_acc.values())
best_vl = min(val_loss.values())
final_ep = max(val_acc)
final_gap = val_acc[final_ep] - train_acc.get(final_ep, 0)
print(f"\nBest val_acc={best_va:.4f} | Best val_loss={best_vl:.4f} | Final gap={final_gap:+.4f}")
