import sys
import torch

sys.path.insert(0, "experiments/E4c-lite-mst-margins")
sys.path.insert(0, "experiments/E4b-lite-noise-aware-synthetic")
sys.path.insert(0, "experiments/E4-lite-noise-aware-rr")
sys.path.insert(0, "experiments")
import e4c_mst as m

torch.manual_seed(0)
gen = torch.Generator().manual_seed(0)
task = m.draw_tasks(2, gen)
x, y = m.draw_rows(task, 2000, gen)
real = torch.corrcoef(torch.stack([x[0, :, j] for j in range(m.D)]))
py1, px1, pair, edges = m.release(x, y, torch.full((2,), 1e6))
xs, ys = m.synth_tree(py1, px1, pair, edges, 2000, gen)
got = torch.corrcoef(torch.stack([xs[0, :, j] for j in range(m.D)]))
off = ~torch.eye(m.D, dtype=torch.bool)
print("real corr %.3f" % real[off].abs().mean())
print("tree corr %.3f" % got[off].abs().mean())
print("label real %.3f synth %.3f" % (y[0].float().mean(), ys[0].float().mean()))
print("edges", edges[:3])
u, v = edges[0]
print("corr real (%d,%d) %.3f synth %.3f" % (u, v, real[u, v], got[u, v]))
