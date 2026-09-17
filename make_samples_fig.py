#!/usr/bin/env python3
# make_samples_fig.py — 拼接 MNIST16 三列样本图 + 测量 base/rho2 逐像素差
import sys, glob, os, numpy as np
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))   # 脚本所在目录, 与工作目录无关
os.chdir(BASE)

def find(pat):
    hits = sorted(glob.glob(os.path.join(BASE, pat)))
    return hits[0] if hits else None

d0 = find("out_mnist_base*")
d2 = find("out_mnist_rho2*")
dd = find("out_mnist_dsim*")
out = (sys.argv[1:] + ["fig_samples_mnist16.png"])[0]

missing = [n for n, d in [("base(omega=0)", d0), ("rho2(omega=2,闭式)", d2), ("dsim(直接模拟)", dd)] if d is None]
if missing:
    print("!! 未找到目录:", ", ".join(missing))
    print("!! BASE =", BASE)
    print("!! 候选:", [d for d in os.listdir(BASE) if d.startswith("out_")])
    sys.exit(1)
print("使用目录:", os.path.basename(d0), "|", os.path.basename(d2), "|", os.path.basename(dd))

for d in (d0, d2, dd):
    if not os.path.exists(f"{d}/samples.png"):
        print(f"!! {d}/samples.png 不存在"); sys.exit(1)

imgs = [Image.open(f"{d}/samples.png") for d in (d0, d2, dd)]
h = min(im.height for im in imgs)
imgs = [im.resize((int(im.width*h/im.height), h)) for im in imgs]
W = sum(im.width for im in imgs) + 20*(len(imgs)-1)
canvas = Image.new("RGB", (W, h+28), "white")
x = 0
for im in imgs:
    canvas.paste(im, (x, 28)); x += im.width + 20
canvas.save(out)
print("saved", out)

# FILLVAL
import torch, importlib.util
mod = find("heatball_mnist16*.py")
if mod is None:
    print("!! 未找到 heatball_mnist16*.py, 跳过 FILLVAL"); sys.exit(0)
if not (os.path.exists(f"{d0}/model_final.pt") and os.path.exists(f"{d2}/model_final.pt")):
    print("!! 未找到 model_final.pt, 跳过 FILLVAL"); sys.exit(0)
print("使用模块:", os.path.basename(mod))
spec = importlib.util.spec_from_file_location("h", mod)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
device = torch.device("cpu"); dtype = torch.float32
rng = np.random.default_rng(0)
ds = m.MNIST16(train=True, download=False)
raw = ds.sample(10000, rng)
mean = raw.mean(0, keepdims=True); std = np.maximum(raw.std(0, keepdims=True), 0.05)
data_t = torch.tensor((raw-mean)/std, device=device, dtype=dtype)
var_vec = data_t.var(dim=0).clamp_min(1e-2); var_scalar = float(var_vec.mean())
data_dim = data_t.shape[1]
class A: pass
gargs = A()
gargs.n_samples = 64; gargs.sample_steps = 500; gargs.s_floor = 1e-5
gargs.data_dim = data_dim
gargs.score_cv = 0; gargs.score_est = "moment"; gargs.n_kappa_eval = 512
gargs.r_frac = 0.02; gargs.sample_auto_tail = 0.0
kap = m.KappaSampler(d=data_dim)
outs = []
for omega, d in [(0.0, d0), (2.0, d2)]:
    m.ROT = m.RotationGen(data_dim, "const", omega, 3.0)
    gargs.omega = omega
    torch.manual_seed(0)
    model = m.PsiNet(data_dim, var_vec).to(device)
    model.load_state_dict(torch.load(f"{d}/model_final.pt", map_location=device))
    model.eval()
    g, _, _ = m.sample(model, gargs, device, dtype, var_scalar, kap)
    outs.append(g)
diff = float(np.abs(outs[0]-outs[1]).max())
print(f"FILLVAL = max|x_base - x_rho2| = {diff:.4f}  (含 MPS 训练权重差 + 残差旋转)")