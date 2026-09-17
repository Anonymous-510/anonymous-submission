"""
heatball_mnist16.py — 热球 score 表示（标量头 + Kaiming 初始化 + hb 训练分支）

═══ 主线 ═══
[1] OU-FP 经代换 y=e^τx, s=(e^{2τ}−1)/2, u=e^{dτ}w 精确化为 ∂_s w = Δ_y w。
    前向闭式：y = x0 + √(2s)·eps。psi = log w（d=256 时 w~e^{-1280}，必须走 log 空间）。
[2] score 的热球表示（主角）：κ 换元后积分域固定，y0 只在被积函数里，
    直接对积分号内求导，分母用 Watson 恒等式替换：
        score(y0,s0) = ∬∇w dκ / ∬w dκ = Σᵢ ωᵢ ∇psi(yᵢ,sᵢ),  ω=softmax(psi_i)
    与 DSM 的条件期望表示 ∇log p_t = E[∇log p(x_t|x_0)|x_t] 本质不同。
[3] 监督来自前向转移核（DSM 只是数据入口）：大 s 用零方差精确目标
        −(y − Σ ω_m x_m)/(2s)，小 s 退回单样本 −eps/√(2s)。
[4] L = ||S_热球 − t||²·2s/d + w_hb·L_hb + w_bnd·L_bnd。密度约束是辅助，
    目的是让 psi 更 caloric、从而让 score 更准。

═══ 本版改动 ═══
1. 新增 --score_est_train hb（默认）。实测（conv, r_frac=0.02, n_eval=16）：
       hb     相对误差 0.0003~0.0065  耗时 ~500ms
       moment 相对误差 2.57~5.46      耗时 ~140ms
   精度高 400~18000 倍、只贵 4 倍。原因：hb 只有权重 ω 是 MC 估的，∇psi 本身精确；
   moment 要从函数值差出整个 d 维梯度，误差 ~√(d/h)。
   ω 未发生高维塌缩（实测 r_frac=0.02 时 ESS=53~62/64）。
2. l_hb 复用 score_hb 已算出的球内 psi/base，只多一次 y0 前向。
   实测边际成本 +6%。两者是同一恒等式的"导数"与"本身"，互相加强、无退化解：
   l_hb 满足得越好 ⇒ psi 越 caloric ⇒ score_hb 越接近真 ∇psi。
   score_est_train 非 hb 时走原独立路径（可用多半径 GL）。
3. nn-check 判读修正：比值 = (生成→训练集)/(训练集内部)，**≈1.0 是理想**
   （新样本间距 = 训练点典型间距），≪1.0 才是记忆，≫1.0 是未贴合流形。
   原提示"前者明显更小 ⇒ 记忆"会把 conv 那轮的 1.04（好结果）误判。
   新增 test→train 作绝对参照。
4. --sample_auto_tail 默认 0（全程热球）。设 >0 会让小 s 段切回 autograd，
   破坏"采样端免求导"的主张，仅作对照用。
5. y_p 提前构造，消掉三处重复；精确目标的 cdist 只在 m_ex 行上算。
6. 分步存盘 model_step{k}.pt 每个 eval 点都存（移出 best 判断，供考古）；
   w_hb=0 时 l_hb 不进 loss 图（消 0 权重反向）；[cfg] 补 n_kappa_dsm 等字段。

═══ 关键数值事实（实测，不要重复踩） ═══
· PyTorch 默认 nn.Linear 用 kaiming_uniform_(a=√5)，gain 0.577 而非 SiLU 该有的 1.414，
  depth=4 时输入梯度衰减 208 倍 ⇒ |∇_yn h|~0.002，修正项对 score 只占 0.8%，
  l_score 对末层缩放**精确不敏感**，梯度全被 boundary 主导（比 500:1）。
  必须显式 kaiming_normal_ 初始化。这是"训练一步不动"的真正病根。
· eps 二次头 corr = −0.5·gain·scale·|g|² ≤ 0 恒成立 ⇒ psi ≤ base 处处成立，
  但真解在数据聚集处 psi > base（GMM 闭式验证：s=0.01 时 +33.95，100% 的点）。
  故必须用标量头 corr = gain·scale·h（可正可负）。
· 热球/一阶矩误差的闭式标度律（三组配置验证，偏差 0.0~1.0%）：
      相对误差 = √(d/h) · ||∇(psi−base)|| / ||∇psi||
  第二个因子是 caloric 控制变量的降噪倍数的倒数；模型学好后修正项占 99%，
  控制变量失效，误差回到 √(d/h)。
· DSM 权重必须 weight=2s（无 cap），使损失恒为 O(d)。l_score 再除以 d 消掉维度依赖。
· 训练用一阶矩时梯度**无偏**（50 次平均后与 autograd 余弦 h=8→0.878, h=32→0.972），
  故 h 可以小，噪声由 SGD 平均；但采样时噪声逐步累积，那里才需要大 n_kappa_eval。
· sliced_w/mmd 只测逐像素边缘分布，方差匹配的高斯就能拿 0.15；看 nn-check、[hist]、图。
· conv 显著优于 mlp：score 0.109 vs 0.2575，[hist] 中间段 14.0% vs 22.4%（真实 14.8%），
  nn-check 比值 1.04 vs 1.56。但 sliced_w 在 ~4000 步后过拟合，建议早停。
· 仍未解决：gen std 偏低（conv 0.640 vs 目标 0.830，方差比 0.581），亮像素只有应有的 57%。
"""


import argparse, json, math, os, time, uuid

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance


# ---------------------------------------------------------------------------
# 旋转结构（256D 共形档）: J(tau) = r(tau)·J0, J0 = d/2 个块对角 2×2 旋转生成元
#   共形判据 [gamma I, J]=0 ⇒ Phi = e^{-tau}R(tau), R 正交 ⇒ 被动坐标完整吸收旋转;
#   热坐标系 (z,s) 内训练/采样/评估与 omega 完全无关 (可自行验证: 两次运行日志逐位一致)。
#   旋转唯一出口 = x 系反拉 x = e^{-tau}R z, 见 sample() 末尾。
#   注: 盲偏差标度律需解析真值, 属 2D 闭式; 本文件验证 d=256 不变性 + kappa 采样器健康。
# ---------------------------------------------------------------------------
class RotationGen:
    """d 维块对角旋转（相邻坐标对等速率）. profile: const r=w / osc r=w sin(k tau) / ramp r=w tanh(k tau).
    交换族 ⇒ theta(t)=∫r 闭式, R 为块对角正交阵."""
    def __init__(self, d, profile="const", omega=0.0, k=3.0):
        assert d % 2 == 0, "RotationGen 需要偶数维"
        self.d, self.profile, self.omega, self.k = d, profile, omega, k
    def r(self, tau):
        if self.profile == "const": return self.omega
        if self.profile == "osc":   return self.omega * math.sin(self.k * tau)
        if self.profile == "ramp":  return self.omega * math.tanh(self.k * tau)
        raise ValueError(self.profile)
    def theta(self, tau):
        if self.profile == "const": return self.omega * tau
        if self.profile == "osc":   return self.omega * (1.0 - math.cos(self.k * tau)) / self.k
        if self.profile == "ramp":  return self.omega * math.log(math.cosh(self.k * tau)) / self.k
        raise ValueError(self.profile)
    def apply_RT_torch(self, y, tau, device, dtype):
        """行向量右乘 R^T（= 列向量左乘 R）: y (n,d) -> (n,d). 所有块共享同一 2x2 旋转."""
        th = self.theta(tau); c, s = math.cos(th), math.sin(th)
        R2T = torch.tensor([[c, s], [-s, c]], device=device, dtype=dtype)   # R^T
        n = y.shape[0]
        return (y.view(n, self.d // 2, 2) @ R2T).view(n, self.d)

ROT = RotationGen(2, "const", 0.0)   # main() 中按 data_dim 重建 (global)

# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
class MNIST16:
    def __init__(self, root='./data', train=True, download=True):
        tf = transforms.Compose([
            transforms.Resize(16), transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))])
        self.dataset = torchvision.datasets.MNIST(root=root, train=train,
                                                  download=download, transform=tf)

    def sample(self, n, rng):
        idx = rng.integers(0, len(self.dataset), n)
        return np.stack([self.dataset[i][0].numpy() for i in idx], axis=0)


# ---------------------------------------------------------------------------
# 时间表：s = (e^{2tau}-1)/2
# ---------------------------------------------------------------------------
TAU_MAX = 0.5 * (0.1 + 9.95)
S_MAX = 0.5 * (math.exp(2 * TAU_MAX) - 1)

def s_of_tau(tau):
    return 0.5 * torch.expm1(2 * tau)

def tau_of_s(s):
    return 0.5 * torch.log1p(2 * s)


# ---------------------------------------------------------------------------
# psi = 解析 caloric 基线 + 标量修正（B0）
#   psi(y,s) = logN(y;0,(var+2s)I) + corr_gain*scale*h(yn,s)
#   score = -y/(var+2s) + corr_gain*scale*∇h/√vv
# ---------------------------------------------------------------------------
class PsiNet(nn.Module):
    """标量头：psi = base + corr_gain * scale * h(yn,s)

    h 是标量场，可正可负，因此 psi 可大于或小于 base，不再有符号约束。
    末层输出 1 维，使用默认初始化即可（标量头无二次梯度消失问题）。
    历史教训（勿回退）：二次头 corr=-0.5*scale*|g|^2 已证双重死刑（v3 RC-0：
    符号锁 psi≤base vs 真 ψ−base≈+446 nat；梯度∝g 的棘轮）。
    """
    def __init__(self, data_dim, var_vec, width=256, depth=4, n_freq=8,
                 corr_gain=1.0, backbone="conv", conv_width=32):
        super().__init__()
        self.data_dim = data_dim
        self.n_freq = n_freq
        self.corr_gain = corr_gain
        self.backbone = backbone
        self.register_buffer('var_vec', var_vec)
        t_dim = 2 * n_freq + 1

        if backbone == "conv":
            side = int(math.isqrt(data_dim))
            if side * side != data_dim:
                raise ValueError(f"backbone=conv 需要 data_dim 为完全平方数，当前 {data_dim}")
            self.side = side
            c = conv_width
            self.enc = nn.Sequential(
                nn.Conv2d(1, c, 3, padding=1), nn.SiLU(),
                nn.Conv2d(c, c, 3, padding=1), nn.SiLU(),
                nn.Conv2d(c, 2 * c, 3, padding=1, stride=2), nn.SiLU(),
                nn.Conv2d(2 * c, 2 * c, 3, padding=1), nn.SiLU(),
                nn.Conv2d(2 * c, 4 * c, 3, padding=1, stride=2), nn.SiLU(),
                nn.Conv2d(4 * c, 4 * c, 3, padding=1), nn.SiLU(),
            )
            self.film_idx = 5
            self.t_film = nn.Sequential(nn.Linear(t_dim, width), nn.SiLU(),
                                        nn.Linear(width, 4 * c))
            feat = 4 * c * (side // 4) ** 2
            self.head = nn.Sequential(nn.Linear(feat + t_dim, width), nn.SiLU(),
                                      nn.Linear(width, width), nn.SiLU(),
                                      nn.Linear(width, 1))      # 输出 1 维标量
        else:
            layers, dd = [], data_dim + t_dim
            for _ in range(depth):
                layers += [nn.Linear(dd, width), nn.SiLU()]
                dd = width
            layers += [nn.Linear(dd, 1)]      # 输出 1 维标量
            self.net = nn.Sequential(*layers)

        # ★ v4 修复（RC-6，统一根因）：PyTorch 默认 init 等效 gain=√(2/6)=0.577
        # （SiLU 应 √2=1.414），输入梯度随深度指数衰减——depth=4 实测 ~204×，
        # 导致 l_score 梯度仅 8e-5、被 boundary（4e-2，505:1）通吃，全部 run 不学。
        # Kaiming 初始化恢复输入梯度后 l_score 梯度 ~1e-2 量级。
        for mo in self.modules():
            if isinstance(mo, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(mo.weight, nonlinearity='relu')
                nn.init.zeros_(mo.bias)

    def time_feat(self, s):
        u = (tau_of_s(s) / TAU_MAX).unsqueeze(-1)
        k = torch.arange(1, self.n_freq + 1, device=s.device, dtype=s.dtype)
        ang = 2 * math.pi * u * k
        return torch.cat([u, torch.sin(ang), torch.cos(ang)], dim=-1)

    def forward(self, y, s, return_corr=False, return_base=False):
        v = (2.0 * s).clamp_min(1e-8)
        vv = self.var_vec + v.unsqueeze(-1)
        yn = y / torch.sqrt(vv)
        base = -0.5 * ((yn ** 2).sum(-1) + torch.log(2 * math.pi * vv).sum(-1))
        scale = torch.sqrt(vv.mean(-1) / v)
        t = self.time_feat(s)
        if self.backbone == "conv":
            x = yn.view(-1, 1, self.side, self.side)
            for i, layer in enumerate(self.enc):
                x = layer(x)
                if i == self.film_idx:
                    gg, bb = self.t_film(t).chunk(2, dim=-1)
                    x = x * (1 + gg[:, :, None, None]) + bb[:, :, None, None]
            h = self.head(torch.cat([x.flatten(1), t], dim=-1)).squeeze(-1)
        else:
            h = self.net(torch.cat([yn, t], dim=-1)).squeeze(-1)
        corr = self.corr_gain * scale * h        # 标量修正，可正可负
        psi = base + corr
        if return_corr and return_base:
            return psi, corr, base
        if return_corr:
            return psi, corr
        if return_base:
            return psi, base
        return psi

    def base_logp(self, y, s):
        vv = self.var_vec + (2.0 * s).clamp_min(1e-8).unsqueeze(-1)
        return -0.5 * ((y ** 2 / vv).sum(-1) + torch.log(2 * math.pi * vv).sum(-1))

    def base_score(self, y, s):
        return -y / (self.var_vec + (2.0 * s).clamp_min(1e-8).unsqueeze(-1))


# ---------------------------------------------------------------------------
# kappa 测度采样（热球）
# ---------------------------------------------------------------------------
class KappaSampler:
    def __init__(self, grid=8192, d=256):
        self.d = d
        v = np.linspace(-30.0, 0.0, grid)
        lp = ((d + 2) / 2.0) * np.log(-v + 1e-12) + (d / 2.0) * v
        pdf = np.exp(lp - lp.max())
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(v))])
        cdf /= cdf[-1]
        self._v, self._c, self._cache = v, cdf, {}
        a = d / 2.0
        self.m2 = (a + 1) / (2.0 * math.pi * a) * (a / (a + 1)) ** (a + 3)

    def _tables(self, device, dtype):
        key = (device, dtype)
        if key not in self._cache:
            self._cache[key] = (torch.as_tensor(self._v, device=device, dtype=dtype),
                                torch.as_tensor(self._c, device=device, dtype=dtype))
        return self._cache[key]

    def sample(self, n, device, dtype):
        vg, cdf = self._tables(device, dtype)
        q = torch.rand(n, device=device, dtype=dtype)
        i = torch.searchsorted(cdf, q).clamp(1, cdf.numel() - 1)
        c0, c1 = cdf[i - 1], cdf[i]
        v0, v1 = vg[i - 1], vg[i]
        v = v0 + (q - c0) / (c1 - c0).clamp(min=1e-12) * (v1 - v0)
        sigma = torch.exp(v) / (4 * math.pi)
        R = torch.sqrt(-2 * self.d * sigma * torch.log(4 * math.pi * sigma))
        t = torch.rand(n, device=device, dtype=dtype) ** (2.0 / (self.d + 2))
        g = torch.randn(n, self.d, device=device, dtype=dtype)
        zhat = g / g.norm(dim=1, keepdim=True).clamp(min=1e-30)
        return torch.sqrt(t).unsqueeze(1) * R.unsqueeze(1) * zhat, sigma


def gl_nodes_log(n, lo, hi):
    """log(r²/r²_hi) 上的 Gauss-Legendre 节点。"""
    if n <= 1:
        return [(1.0, 1.0)]
    xi, wi = np.polynomial.legendre.leggauss(n)
    t = 0.5 * (hi - lo) * xi + 0.5 * (hi + lo)
    w = 0.5 * (hi - lo) * wi
    return [(float(np.exp(t[i])), float(w[i])) for i in range(n)]


def logmeanexp(a, dim=-1):
    m = a.max(dim=dim, keepdim=True).values
    return (m + (a - m).exp().mean(dim=dim, keepdim=True).log()).squeeze(-1)


# ---------------------------------------------------------------------------
# 热球 score 表示（一阶矩 + caloric 控制变量）
# ---------------------------------------------------------------------------
def _moment_core(kap, z, r, pc, pp, pm):
    dp = pp - pc.unsqueeze(1); dm = pm - pc.unsqueeze(1)
    mx = torch.cat([dp, dm], 1).max(1, keepdim=True).values.detach()
    Rp = (dp - mx).exp(); Rm = (dm - mx).exp()
    return (1.0 / kap.m2 / r) * ((Rp - Rm).unsqueeze(-1) * z).sum(1) \
           / (Rp + Rm).sum(1, keepdim=True).clamp(min=1e-30)


def _moment_once(model, y, s, h, r_frac, kap, device, dtype):
    """一次一阶矩估计（带 caloric 控制变量）。base 是解析式，零额外网络调用。"""
    n, d = y.shape
    z, sig = kap.sample(n * h, device, dtype)
    z = z.view(n, h, d); sig = sig.view(n, h)
    r2 = r_frac * s
    rt = r2.sqrt().view(-1, 1, 1)
    s_i = (s.view(-1, 1) - r2.view(-1, 1) * sig).reshape(-1)
    yp = (y.unsqueeze(1) + rt * z).reshape(-1, d)
    ym = (y.unsqueeze(1) - rt * z).reshape(-1, d)
    r = r2.sqrt().view(-1, 1).clamp(min=1e-4)
    S = _moment_core(kap, z, r, model(y, s),
                     model(yp, s_i).view(n, h), model(ym, s_i).view(n, h))
    with torch.no_grad():
        S0 = _moment_core(kap, z, r, model.base_logp(y, s),
                          model.base_logp(yp, s_i).view(n, h),
                          model.base_logp(ym, s_i).view(n, h))
    return torch.nan_to_num(S - S0 + model.base_score(y, s))


@torch.no_grad()
def score_moment(model, y, s, n_eval, r_frac, device, dtype, kap):
    return _moment_once(model, y, s, max(n_eval // 2, 1), r_frac, kap, device, dtype)


def moment_score_halves(model, y, s, h, r_frac, kap, device, dtype):
    """两个独立半批，供交叉积用。"""
    return (_moment_once(model, y, s, h, r_frac, kap, device, dtype),
            _moment_once(model, y, s, h, r_frac, kap, device, dtype))


# ---------------------------------------------------------------------------
# 热球 score 表示（主线）：score(y0) = ∬∇w dκ / ∬w dκ = Σᵢ ωᵢ ∇ψ(yᵢ,sᵢ)
# κ 换元后积分域是固定的单位热球，y0 只出现在被积函数里，故可直接对积分号内求导；
# 分母用 Watson 恒等式替换 ⇒ 自归一形式，ψ 整体平移不影响 ω。
# 只有权重 ω 是 MC 估的，∇ψ 本身精确，故误差远小于一阶矩：
#   实测（conv, r_frac=0.02, n_eval=16）：hb 相对误差 0.0003~0.0065 / 耗时 ~500ms
#                                        moment 相对误差 2.57~5.46  / 耗时 ~140ms
#   精度高 400~18000 倍，只贵 4 倍 ⇒ 训练用 hb。
# ω 的有效样本数是健康的（实测 r_frac=0.02 时 ESS=53~62 / 64），未发生高维塌缩。
# ---------------------------------------------------------------------------
def hb_score_once(model, y, s, h, r_frac, kap, device, dtype, need_graph=True):
    """返回 (S, psi_i, base_i)。后两者供 l_hb 复用，避免重采一批球。"""
    n, d = y.shape
    z, sig = kap.sample(n * h, device, dtype)
    z = z.view(n, h, d); sig = sig.view(n, h)
    r2 = r_frac * s
    yi = (y.unsqueeze(1) + r2.sqrt().view(-1, 1, 1) * z).reshape(-1, d)
    si = (s.view(-1, 1) - r2.view(-1, 1) * sig).reshape(-1)
    yi = yi.detach().requires_grad_(True)
    psi_i, base_i = model(yi, si, return_base=True)
    g_i = torch.autograd.grad(psi_i.sum(), yi, create_graph=need_graph)[0]
    om = torch.softmax(psi_i.detach().view(n, h), dim=1)
    S = (om.unsqueeze(-1) * g_i.view(n, h, d)).sum(1)
    return S, psi_i.view(n, h), base_i.view(n, h)


def hb_score_halves(model, y, s, h, r_frac, kap, device, dtype):
    """两个独立半批，供交叉积去掉 Var(S) 噪声地板。"""
    return (hb_score_once(model, y, s, h, r_frac, kap, device, dtype),
            hb_score_once(model, y, s, h, r_frac, kap, device, dtype))


@torch.no_grad()
def score_autograd(model, y, s):
    """仅供对照/诊断，不是主线。"""
    y = y.detach().requires_grad_(True)
    with torch.enable_grad():
        return torch.autograd.grad(model(y, s).sum(), y)[0]


# ---------------------------------------------------------------------------
# 诊断
# ---------------------------------------------------------------------------
@torch.no_grad()
def arch_check(model, data_t, tag, device, kap, args):
    """判断模型有没有跳出"各向同性高斯"平凡解，并分离估计器噪声。"""
    d = data_t.shape[1]; n = 128
    print(f"[arch-check {tag}]", flush=True)
    for sv in [1.0, 0.05, 1e-4]:
        x = data_t[torch.randint(0, len(data_t), (n,), device=device)]
        y = x + math.sqrt(2 * sv) * torch.randn_like(x)
        s_vec = torch.full((n,), sv, device=device, dtype=data_t.dtype)
        g_auto = score_autograd(model, y, s_vec)          # 无噪声真实 score
        g_hb   = score_moment(model, y, s_vec, args.n_kappa_eval, args.r_frac,
                              device, data_t.dtype, kap) # 热球估计
        gb     = model.base_score(y, s_vec)               # 基线
        cos    = torch.nn.functional.cosine_similarity(g_auto, -y, dim=-1).mean()  # 改用 g_auto
        rel    = ((g_hb - g_auto).norm(dim=-1) / g_auto.norm(dim=-1).clamp(min=1e-8)).mean()
        ga_corr = (g_auto - gb).norm(dim=-1).mean()       # 修正贡献（无噪声）
        print(f"   s={sv:<8.0e} |auto|={g_auto.norm(dim=-1).mean():8.2f} "
              f"|hb|={g_hb.norm(dim=-1).mean():8.2f} "
              f"|修正贡献(auto)|={ga_corr:8.2f} "
              f"(纯基线={gb.norm(dim=-1).mean():7.2f}) "
              f"cos(score,-y)={cos:+.4f}  热球vs autograd 相对差={rel:.4f}", flush=True)



# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def train(model, data_t, test_t, args, device, dtype, kap, eval_args, data_mean, data_std):
    # 优化器分组：末层（g输出层）不加 weight decay
    last_module = model.head[-1] if args.backbone == "conv" else model.net[-1]
    head_params = list(last_module.parameters())
    other_params = [p for p in model.parameters() if all(p is not hp for hp in head_params)]
    opt = torch.optim.Adam([
        {'params': other_params, 'weight_decay': args.weight_decay},
        {'params': head_params, 'weight_decay': 0.0}
    ], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=0.1 * args.lr)
    rng = np.random.default_rng(args.seed)
    var_scalar = float(data_t.var(dim=0).mean().item())
    gl_r = gl_nodes_log(args.n_r, math.log(args.r_lo), 0.0)

    hist = {k: [] for k in ("total", "score", "hb", "hb_diag", "boundary", "sgn")}
    t0 = time.perf_counter()
    best_sw, best_step = float('inf'), 0

    # 配置落盘：每次 run 先核对这一行
    print(f"[cfg] score_est={args.score_est_train} device={args.device} "
          f"w_hb={args.w_hb} w_boundary={args.w_boundary} s_exact_min={args.s_exact_min} "
          f"n_exact_ref={args.n_exact_ref} batch={args.batch} lr={args.lr} "
          f"corr_gain={args.corr_gain} r_frac={args.r_frac} "
          f"backbone={args.backbone} n_kappa_dsm={args.n_kappa_dsm} "
          f"n_kappa_eval={args.n_kappa_eval} auto_tail={getattr(args,'sample_auto_tail',0.0)}",
          flush=True)

    arch_check(model, data_t, "before_training", device, kap, args)

    for step in range(args.steps):
        x0 = data_t[rng.integers(0, len(data_t), args.batch)]

        # ---- 前向轨迹点（闭式，热坐标下的纯扩散）。热球与 DSM 共用 ----
        s_fwd = torch.empty(args.batch, device=device, dtype=dtype).uniform_(
            math.log(args.s_dsm_min), math.log(S_MAX)).exp()
        eps_fwd = torch.randn_like(x0)
        rt_fwd = (2 * s_fwd).sqrt().unsqueeze(1)
        m_hb = s_fwd >= args.s_hb_min          # 热球要求 r² ≲ 4πs0，锚点避开 s→0

        # ================= 主线：热球表示的 score + 转移核监督 =================
        y_p = x0 + rt_fwd * eps_fwd          # 前向轨迹点，三处共用
        tp = -eps_fwd / rt_fwd
        n_ex, ess_ex = 0, 0.0
        if args.s_exact_min > 0:
            m_ex = s_fwd >= args.s_exact_min
            if bool(m_ex.any()):
                with torch.no_grad():
                    # 修复：使用 torch.randint 采样索引
                    idx_ref = torch.randint(0, len(data_t), (args.n_exact_ref,), device=device)
                    Xs = data_t[idx_ref]
                    v2 = (2 * s_fwd).unsqueeze(1)
                    yy = y_p
                    ye, ve = yy[m_ex], v2[m_ex]
                    om = torch.softmax(-torch.cdist(ye, Xs) ** 2 / (2 * ve), dim=1)
                    tp = tp.clone()
                    tp[m_ex] = -(ye - om @ Xs) / ve
                    ess_ex = float((1.0 / (om ** 2).sum(1)).mean()) / args.n_exact_ref
                    n_ex = int(m_ex.sum())

        # weight=2s 使 2s||S-t||² = ||√(2s)S + eps||² = ||eps_hat-eps||²，损失恒为 O(d)
        w = (2 * s_fwd).unsqueeze(1)
        reuse_pi = reuse_bi = None            # hb 分支下供 l_hb 复用的球内 psi
        if args.score_est_train == "hb":
            (Sa, pa, ba), (Sb, pb, bb) = hb_score_halves(
                model, y_p, s_fwd, args.n_kappa_dsm, args.r_frac, kap, device, dtype)
            mse_per = (w * (Sa - tp) * (Sb - tp)).sum(-1)     # 交叉积
            reuse_pi, reuse_bi = (pa, pb), (ba, bb)
        elif args.score_est_train == "moment":
            Sa, Sb = moment_score_halves(model, y_p, s_fwd, args.n_kappa_dsm,
                                         args.r_frac, kap, device, dtype)
            mse_per = (w * (Sa - tp) * (Sb - tp)).sum(-1)
        else:                                                  # autograd 对照
            yg = y_p.detach().requires_grad_(True)
            g = torch.autograd.grad(model(yg, s_fwd).sum(), yg, create_graph=True)[0]
            mse_per = (w * (g - tp).pow(2)).sum(-1)
        l_score = mse_per.mean() / args.data_dim               # 除以 d，消掉维度依赖

        # ================= 辅助：密度热球（让 psi 更接近 caloric） =================
        # score_hb 与 l_hb 是同一恒等式的"导数"与"本身"，互相加强、无退化解：
        # l_hb 满足得越好 ⇒ psi 越 caloric ⇒ score_hb 越接近真 ∇psi。
        # 且两者所需的球内 psi 完全重叠，复用后边际成本实测仅 +6%。
        l_hb = torch.zeros((), device=device, dtype=dtype)
        l_hb_diag = sgn_val = 0.0
        need_hb = (args.w_hb != 0.0) or ((step + 1) % 500 == 0)
        hb_ctx = torch.enable_grad() if args.w_hb != 0.0 else torch.no_grad()

        if need_hb and reuse_pi is not None and int(m_hb.sum()) >= 8:
            # --- 复用路径：直接用 score_hb 已算出的球内 psi，只多一次 y0 的前向 ---
            with hb_ctx:
                psi_c, base_c = model(y_p[m_hb], s_fwd[m_hb], return_base=True)
                (pa, pb), (ba, bb) = reuse_pi, reuse_bi
                # caloric 控制变量：base 自身 caloric，其热球残差期望为 0；
                # 同批样本减掉可消去相关 MC 噪声（实测 l_hb 降 24 倍）
                ra = (psi_c - logmeanexp(pa[m_hb], 1)) - (base_c - logmeanexp(ba[m_hb], 1))
                rb = (psi_c - logmeanexp(pb[m_hb], 1)) - (base_c - logmeanexp(bb[m_hb], 1))
                l_hb = (ra * rb).mean()                            # 交叉积，可为负
                l_hb_diag = float((0.5 * (ra + rb)).pow(2).mean().detach())
                sgn_val = float((0.5 * (ra + rb)).mean().detach())

        elif need_hb and int(m_hb.sum()) >= 8:
            # --- 独立路径：score_est_train 非 hb 时，另采一批球（可多半径 GL） ---
            ctx = torch.enable_grad() if args.w_hb != 0.0 else torch.no_grad()
            with ctx:
                s0 = s_fwd[m_hb]; y0 = y_p[m_hb]
                Bh = y0.shape[0]
                z, sig = kap.sample(Bh * args.n_kappa, device, dtype)
                z = z.view(Bh, args.n_kappa, args.data_dim); sig = sig.view(Bh, args.n_kappa)
                psi_c, base_c = model(y0, s0, return_base=True)
                hh = args.n_kappa // 2
                r2_hi = args.r_frac * s0
                acc = acc_a = acc_b = 0.0; wsum = 0.0
                for rs, rw in gl_r:
                    r2 = r2_hi * rs
                    yi = (y0.unsqueeze(1) + r2.sqrt().view(-1, 1, 1) * z).reshape(-1, args.data_dim)
                    si = (s0.view(-1, 1) - r2.view(-1, 1) * sig).reshape(-1)
                    pi, bi = model(yi, si, return_base=True)
                    pi = pi.view(Bh, args.n_kappa); bi = bi.view(Bh, args.n_kappa)
                    rf = (psi_c - logmeanexp(pi, 1)) - (base_c - logmeanexp(bi, 1))
                    ra = (psi_c - logmeanexp(pi[:, :hh], 1)) - (base_c - logmeanexp(bi[:, :hh], 1))
                    rb = (psi_c - logmeanexp(pi[:, hh:], 1)) - (base_c - logmeanexp(bi[:, hh:], 1))
                    acc = acc + rw * rf; acc_a = acc_a + rw * ra; acc_b = acc_b + rw * rb
                    wsum += rw
                acc = acc / wsum; acc_a = acc_a / wsum; acc_b = acc_b / wsum
                l_hb = (acc_a * acc_b).mean()
                l_hb_diag = float((acc ** 2).mean().detach())
                sgn_val = float(acc.mean().detach())

        # ================= 辅助：大 s 边界（修正项 -> 0） =================
        idx_b = rng.integers(0, len(data_t), args.boundary_batch)
        y_b = data_t[idx_b] + math.sqrt(2 * S_MAX) * torch.randn(
            args.boundary_batch, args.data_dim, device=device, dtype=dtype)
        s_b = torch.full((args.boundary_batch,), S_MAX, device=device, dtype=dtype)
        _, corr_b = model(y_b, s_b, return_corr=True)
        l_boundary = (corr_b ** 2).mean()

        if args.w_hb != 0.0:
            loss = args.w_score_scale * l_score + args.w_hb * l_hb + args.w_boundary * l_boundary
        else:
            loss = args.w_score_scale * l_score + args.w_boundary * l_boundary

        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True); continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step(); sched.step()

        for k, v in [("total", loss), ("score", l_score), ("hb", l_hb), ("boundary", l_boundary)]:
            hist[k].append(float(v.detach()))
        hist["hb_diag"].append(l_hb_diag); hist["sgn"].append(sgn_val)

        if args.mag_steps > 0 and args.mag_every > 0 and step < args.mag_steps \
                and (step + 1) % args.mag_every == 0:
            wd = args.mag_every
            m = {k: float(np.mean(hist[k][-wd:])) for k in ("score", "boundary")}
            hbw = [v for v in hist["hb_diag"][-wd:] if v != 0.0]
            hb_m = (float(np.mean(hist["hb"][-wd:])) * wd / len(hbw)) if hbw else 0.0
            print(f"[mag] step={step+1} score={m['score']:.4g} hb={hb_m:+.4g} "
                  f"(diag={float(np.mean(hbw)) if hbw else 0.0:.4g}) bnd={m['boundary']:.4g} "
                  f"| 精确目标 n={n_ex}/{args.batch} ESS/M={ess_ex:.3f}", flush=True)

        if (step + 1) % 500 == 0:
            wm = {k: float(np.mean(hist[k][-500:])) for k in ("total", "score", "boundary")}
            hbw = [v for v in hist["hb_diag"][-500:] if v != 0.0]
            sgw = [v for v in hist["sgn"][-500:] if v != 0.0]
            hb_diag = float(np.mean(hbw)) if hbw else 0.0
            hb_sys = abs(float(np.mean(sgw))) if sgw else 0.0
            hb_m = (float(np.mean(hist["hb"][-500:])) * 500 / len(hbw)) if hbw else 0.0
            print(f"step {step+1}/{args.steps} loss={wm['total']:.4e} score={wm['score']:.4e} "
                  f"hb={hb_m:.4e} (diag={hb_diag:.4e} n={len(hbw)}) bnd={wm['boundary']:.4e} "
                  f"hb_sys={hb_sys:.2e} hb_noise={math.sqrt(max(hb_diag-hb_sys**2,0)):.2e}", flush=True)
            print(f"  [weighted] score={args.w_score_scale*wm['score']:.4f} "
                  f"hb={args.w_hb*hb_m:.4f} bnd={args.w_boundary*wm['boundary']:.4f}", flush=True)

        if (step + 1) % args.eval_every == 0 and test_t is not None:
            model.eval()
            te = time.perf_counter()
            with torch.no_grad():
                gen_e, _ = sample(model, eval_args, device, dtype, var_scalar, kap)
            cur_sw = sliced_wasserstein(gen_e, test_t.cpu().numpy())
            model.train()
            torch.save(model.state_dict(), os.path.join(args.outdir, f"model_step{step+1}.pt"))
            if cur_sw < best_sw:
                best_sw, best_step = cur_sw, step + 1
                torch.save(model.state_dict(), os.path.join(args.outdir, "model_best.pt"))
            print(f"[eval] step {step+1} mmd={mmd_rbf(gen_e, test_t.cpu().numpy()):.4f} "
                  f"sliced_w={cur_sw:.4f} (best={best_sw:.4f}@{best_step}) "
                  f"{time.perf_counter()-te:.1f}s  gen std={gen_e.std():.4f} "
                  f"(应≈{math.sqrt(var_scalar):.3f})", flush=True)
            arch_check(model, data_t, f"step{step+1}", device, kap, args)
            if data_mean is not None:
                save_image_grid(np.clip(gen_e[:16] * data_std + data_mean, 0, 1),
                                os.path.join(args.outdir, f"eval_step{step+1}.png"), nrow=4)

    arch_check(model, data_t, "after_training", device, kap, args)
    return hist, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# 采样：反向 SDE，score 全程由热球一阶矩给出（不调 autograd）
# ---------------------------------------------------------------------------
def sample(model, args, device, dtype, var_scalar, kap):
    n = args.n_samples
    y = math.sqrt(var_scalar + 2 * S_MAX) * torch.randn(n, args.data_dim, device=device, dtype=dtype)
    n_tail = max(1, int(round(args.sample_steps * 0.2)))
    n_main = max(1, args.sample_steps - n_tail)
    ss = torch.cat([
        s_of_tau(torch.linspace(TAU_MAX, float(tau_of_s(torch.tensor(0.3))),
                                n_main + 1, device=device, dtype=dtype))[:-1],
        torch.exp(torch.linspace(math.log(0.3), math.log(args.s_floor),
                                 n_tail + 1, device=device, dtype=dtype))])

    # 读取尾部切换阈值
    sample_auto_tail = getattr(args, "sample_auto_tail", 0.0)

    t0 = time.perf_counter()
    for k in range(len(ss) - 1):
        s_hi = ss[k]; ds = float(s_hi - ss[k + 1])
        s_vec = torch.full((n,), float(s_hi), device=device, dtype=dtype)

        # 尾段开关：小 s 处热球估计器噪声过大，切换为 autograd 精确 score
        if float(s_hi) < sample_auto_tail:
            sc = score_autograd(model, y, s_vec)
            tag = "auto"
        else:
            sc = score_moment(model, y, s_vec, args.n_kappa_eval, args.r_frac, device, dtype, kap)
            tag = "hb  "

        if k % 40 == 0 or k == len(ss) - 2:
            print(f"    [traj:{tag}] k={k} s={float(s_hi):.4g} |score|={sc.norm(dim=-1).mean():.4f} "
                  f"y_std={y.std():.3f} (理论≈{math.sqrt(var_scalar+2*float(s_hi)):.3f})", flush=True)

        y = y + 2.0 * sc * ds + math.sqrt(2 * ds) * torch.randn_like(y)
        if not torch.isfinite(y).all():
            print("NaN detected, stopping"); break
    if ROT.omega != 0.0:
        tau_end = 0.5 * math.log1p(2 * float(ss[-1]))
        y = math.exp(-tau_end) * ROT.apply_RT_torch(y, tau_end, device, dtype)
    return y.detach().cpu().numpy(), time.perf_counter() - t0



# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def mmd_rbf(x, y):
    x = torch.as_tensor(x); y = torch.as_tensor(y)
    with torch.no_grad():
        dxy = torch.cdist(torch.cat([x, y]), torch.cat([x, y])) ** 2
        k = (-dxy / (2 * dxy.median().clamp(min=1e-6))).exp()
        n, m = len(x), len(y)
        return float((k[:n, :n].sum() - k[:n, :n].diagonal().sum()) / (n * (n - 1)) +
                     (k[n:, n:].sum() - k[n:, n:].diagonal().sum()) / (m * (m - 1)) -
                     2 * k[:n, n:].mean())


def sliced_wasserstein(x, y, n_proj=50, seed=0):
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((n_proj, x.shape[1]))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return float(np.mean([wasserstein_distance(x @ d, y @ d) for d in dirs]))


def nn_distance_check(gen, data_np, tag="", ref_np=None):
    """比值 = (生成→训练集最近邻) / (训练集内部最近邻)。
    ≈1.0 是理想：新样本与训练点的间距 = 训练点彼此的典型间距。
    ≪1.0 才是记忆（生成的就是训练点本身，距离趋于 0）。
    ≫1.0 说明样本还没落到数据流形上。
    实测：mlp 10000 步 1.56；conv 10000 步 1.04。
    若给了 ref_np（如 test 集），额外报 test→train 作为绝对参照。"""
    g = torch.as_tensor(gen, dtype=torch.float32)
    d = torch.as_tensor(data_np[:5000], dtype=torch.float32)
    with torch.no_grad():
        dg = torch.cdist(g, d).min(dim=1).values
        dd = torch.cdist(d[:1000], d); dd.fill_diagonal_(float('inf'))
        dt_ = dd.min(dim=1).values
        extra = ""
        if ref_np is not None:
            dr = torch.cdist(torch.as_tensor(ref_np[:1000], dtype=torch.float32), d)
            extra = f"  test->train={dr.min(dim=1).values.median():.4f}(绝对参照)"
    ratio = float(dg.median() / dt_.median())
    verdict = "理想" if 0.8 <= ratio <= 1.3 else ("疑似记忆" if ratio < 0.8 else "未贴合流形")
    print(f"[nn-check {tag}] 生成->训练集={dg.median():.4f} 训练集内部={dt_.median():.4f} "
          f"比值={ratio:.3f} [{verdict}]{extra}", flush=True)
    return float(dg.median()), float(dt_.median())


def save_image_grid(images, path, nrow=8, size=16):
    n = len(images)
    fig, axes = plt.subplots(max(n // nrow, 1), nrow, figsize=(nrow, max(n // nrow, 1)))
    for i, ax in enumerate(np.atleast_1d(axes).flatten()):
        if i < n:
            ax.imshow(images[i].reshape(size, size), cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    run_id = str(uuid.uuid4())[:8]
    print(f"[RUN-ID] {run_id}", flush=True)
    ap = argparse.ArgumentParser()
    # 训练
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    ap.add_argument("--outdir", default="outputs_hb_b0")
    ap.add_argument("--omega", type=float, default=0.0, help="旋转强度 rho=omega/gamma (gamma=1); 0=无旋转")
    ap.add_argument("--rot_profile", default="const", choices=["const", "osc", "ramp"], help="旋转轮廓")
    ap.add_argument("--rot_k", type=float, default=3.0, help="osc/ramp 频率 k")
    # 网络
    ap.add_argument("--backbone", default="conv", choices=["mlp", "conv"])
    ap.add_argument("--conv_width", type=int, default=32)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--corr_gain", type=float, default=1.0, help="修正项强度系数")
    # 热球 score 表示（主线）
    ap.add_argument("--score_est_train", default="hb", choices=["moment", "moment", "autograd"],
                    help="训练 score 来源：hb=热球权重×精确局部梯度（默认，精度最高）；"
                         "moment=一阶矩差商（免求导，噪声大）；autograd=逐点梯度（对照）")
    ap.add_argument("--n_kappa_dsm", type=int, default=16,
                    help="hb/moment 模式下每半批的 kappa 数；autograd 模式下不用")

    ap.add_argument("--n_kappa_eval", type=int, default=512,
                    help="采样/诊断 kappa 数。rel≈√(d/h)·修正份额：d=256 时 512→rel~1.0，2048→~0.5")
    ap.add_argument("--r_frac", type=float, default=0.02,
                    help="热球半径 r²=r_frac*s。0.125 恰在一阶矩分辨率边界（κ偏移≈场变化尺度），0.02 偏置可忽略")
    # 监督目标
    ap.add_argument("--s_dsm_min", type=float, default=1e-2)
    ap.add_argument("--s_exact_min", type=float, default=0.3,
                    help="s>=该值改用转移核精确目标（零方差）")
    ap.add_argument("--n_exact_ref", type=int, default=4096)
    ap.add_argument("--w_score_scale", type=float, default=1.0)
    # 辅助约束
    ap.add_argument("--w_hb", type=float, default=10.0)
    ap.add_argument("--n_kappa", type=int, default=64)
    ap.add_argument("--n_r", type=int, default=1)
    ap.add_argument("--r_lo", type=float, default=0.1)
    ap.add_argument("--s_hb_min", type=float, default=0.1)
    ap.add_argument("--w_boundary", type=float, default=0.01,
                    help="v4：验证期保持 0；学习确认后长跑用 0.05~0.1（505:1 教训，勿回 1.0）")
    ap.add_argument("--boundary_batch", type=int, default=64)
    # 采样与评估
    ap.add_argument("--n_samples", type=int, default=128)
    ap.add_argument("--sample_steps", type=int, default=500)
    ap.add_argument("--s_floor", type=float, default=0.01,
                    help="采样尾部下限，应与 s_dsm_min 对齐")
    ap.add_argument("--sample_auto_tail", type=float, default=0.0,
                    help="采样时 s 低于该值改用 autograd score（0=禁用，1e6=全 autograd 上界参考）")
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--n_eval_samples", type=int, default=32)
    ap.add_argument("--eval_sample_steps", type=int, default=150)
    ap.add_argument("--mag_steps", type=int, default=200)
    ap.add_argument("--mag_every", type=int, default=20)
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dtype = torch.float32
    args.outdir = f"{args.outdir}_{run_id}"
    os.makedirs(args.outdir, exist_ok=True)

    ds = MNIST16(train=True, download=True)
    rng = np.random.default_rng(args.seed)
    raw = ds.sample(10000, rng)
    data_mean = raw.mean(axis=0, keepdims=True)
    data_std = np.maximum(raw.std(axis=0, keepdims=True), 0.05)
    data_np = (raw - data_mean) / data_std
    args.data_dim = data_np.shape[1]
    global ROT
    ROT = RotationGen(args.data_dim, args.rot_profile, args.omega, args.rot_k)
    if args.omega != 0.0:
        _th = ROT.theta(TAU_MAX)
        print(f"[rot] d={args.data_dim} profile={args.rot_profile} omega={args.omega} k={args.rot_k} "
              f"theta(TAU_MAX)={_th:.3f} (块正交性由构造保证)", flush=True)
    data_t = torch.tensor(data_np, device=device, dtype=dtype)
    var_vec = data_t.var(dim=0).clamp_min(1e-2)
    var_scalar = float(var_vec.mean().item())
    print(f"[cfg] d={args.data_dim} var_vec mean={var_scalar:.4f}", flush=True)

    model = PsiNet(args.data_dim, var_vec, width=args.width, depth=args.depth,
                   corr_gain=args.corr_gain, backbone=args.backbone,
                   conv_width=args.conv_width).to(device)
    print(f"[cfg] backbone={args.backbone} 参数量="
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    kap = KappaSampler(d=args.data_dim)

    test_np = (MNIST16(train=False, download=True).sample(2000, rng) - data_mean) / data_std
    test_t = torch.tensor(test_np, device=device, dtype=dtype)

    eval_args = argparse.Namespace(**vars(args))
    eval_args.n_samples = args.n_eval_samples
    eval_args.sample_steps = args.eval_sample_steps

    hist, t_train = train(model, data_t, test_t, args, device, dtype, kap,
                          eval_args, data_mean, data_std)
    torch.save(model.state_dict(), os.path.join(args.outdir, "model_final.pt"))

    gen, t_sample = sample(model, args, device, dtype, var_scalar, kap)
    gen_un = gen * data_std + data_mean
    real_un = data_np * data_std + data_mean
    print(f"[final] gen std={gen.std():.3f} (应≈{math.sqrt(var_scalar):.3f}) "
          f"越界={float((np.abs(gen_un-0.5)>0.5).mean()*100):.1f}%", flush=True)
    print(f"[hist] 生成: <0.2 {(gen_un<0.2).mean()*100:.1f}%  >0.8 {(gen_un>0.8).mean()*100:.1f}%  "
          f"中间 {((gen_un>=0.2)&(gen_un<=0.8)).mean()*100:.1f}%", flush=True)
    print(f"       真实: <0.2 {(real_un<0.2).mean()*100:.1f}%  >0.8 {(real_un>0.8).mean()*100:.1f}%  "
          f"中间 {((real_un>=0.2)&(real_un<=0.8)).mean()*100:.1f}%", flush=True)
    gv, dv = gen.var(axis=0), data_np.var(axis=0)
    print(f"[var-check] 生成/数据 方差比={gv.mean()/dv.mean():.3f}", flush=True)
    nn_g, nn_d = nn_distance_check(gen, data_np, "final", ref_np=test_np)

    save_image_grid(np.clip(gen_un[:64], 0, 1), os.path.join(args.outdir, "samples.png"))
    save_image_grid(np.clip(data_np[:64] * data_std + data_mean, 0, 1),
                    os.path.join(args.outdir, "real.png"))

    json.dump(dict(run_id=run_id, steps=args.steps,
                   rotation=dict(profile=args.rot_profile, omega=args.omega, rot_k=args.rot_k),
                   gen_std=float(gen.std()),
                   mmd=mmd_rbf(gen, test_np), sliced_w=sliced_wasserstein(gen, test_np),
                   nn_gen=nn_g, nn_data=nn_d,
                   runtime=dict(train=t_train, sample=t_sample), args=vars(args)),
              open(os.path.join(args.outdir, "metrics.json"), "w"), indent=2)

    plt.figure()
    for k in ("total", "score", "hb_diag", "boundary"):
        plt.plot(np.maximum(np.asarray(hist[k]), 1e-12), label=k, lw=0.7)
    plt.yscale('log'); plt.legend(); plt.title(f"loss (run={run_id})")
    plt.savefig(os.path.join(args.outdir, "loss.png")); plt.close()
    print(f"Done. Results in {args.outdir}", flush=True)


if __name__ == "__main__":
    main()