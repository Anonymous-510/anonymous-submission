#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
heatball_toy2d.py — 热方程生成模型框架（DDPM 特例）在 2D toy 数据上的实现
修订版：修复前向加噪均值项、归一化系数、评分函数数值稳定性
"""
import argparse, json, math, os, time

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

# ---------------------------------------------------------------------------
# 旋转结构: J(tau) = r(tau) * J0, 共形档 —— 被动坐标完整吸收
# ---------------------------------------------------------------------------
class Rotation2D:
    """J(tau) = r(tau) * J0, J0 = [[0,-1],[1,0]]; R(tau) = e^{theta(tau) J0}（2D 闭式）.
    profile: 'const' r=w ; 'osc' r=w sin(k tau) ; 'ramp' r=w tanh(k tau). 净角度 theta=∫r."""
    def __init__(self, profile="const", omega=0.0, k=3.0):
        self.profile, self.omega, self.k = profile, omega, k
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
    def R(self, tau):
        th = self.theta(tau); c, s = math.cos(th), math.sin(th)
        return np.array([[c, -s], [s, c]])

ROT = Rotation2D("const", 0.0)  # main() 中用 --rot_profile/--omega 覆盖 (global)

def rotation_blind_bias(ds, omega, taus=None, n=4000):
    """R2 实验的闭式版本: 用 J=0 的解析 score 估计含旋转前向的 score.
    修复: 使用全局 ROT (含 --rot_profile/--rot_k), 此前硬编码 const 导致 osc 下结果错误.
    细 tau 网格 (默认 25 点, 上限 2.5) 用于分辨峰位 tau* ~ pi/(2*rho)."""
    if ds.modes is None: return None
    if taus is None: taus = np.linspace(0.05, 2.5, 25)
    rot = ROT
    rng = np.random.default_rng(123); rows = []
    for tau in taus:
        tau = float(tau)
        x0 = ds.sample(n, rng); e2 = math.exp(-2 * tau); R = rot.R(tau)
        x = math.exp(-tau) * (x0 @ R.T) + math.sqrt(1 - e2) * rng.standard_normal(x0.shape)
        s_true  = analytic_score_gmm(ds, x, tau, R)          # 旋转感知 (真值)
        s_blind = analytic_score_gmm(ds, x, tau, np.eye(2))  # 旋转盲 (J=0 公式)
        mb = float(((s_true - s_blind) ** 2).sum(1).mean())
        mt = float((s_true ** 2).sum(1).mean())
        rows.append(dict(tau=tau, theta=rot.theta(tau),
                         mse_blind=mb, mse_mag_true=mt,
                         rel_bias=mb / max(mt, 1e-12)))
    return rows

# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------
class GMM2D:
    def __init__(self, weights, means, covs, name):
        self.name = name
        self.w = np.asarray(weights, dtype=np.float64)
        self.mu = np.asarray(means, dtype=np.float64)
        self.cov = np.asarray(covs, dtype=np.float64)
        self.prec = np.linalg.inv(self.cov)
        self.logdet = np.linalg.slogdet(self.cov)[1]
        self.modes = self.mu.copy()

    def sample(self, n, rng):
        idx = rng.choice(len(self.w), size=n, p=self.w)
        x = np.empty((n, 2))
        for k in range(len(self.w)):
            m = idx == k
            if m.any():
                L = np.linalg.cholesky(self.cov[k])
                x[m] = self.mu[k] + rng.standard_normal((m.sum(), 2)) @ L.T
        return x

    def logpdf(self, x):
        x = np.atleast_2d(x)
        out = np.empty((len(self.w), x.shape[0]))
        for k in range(len(self.w)):
            d = x - self.mu[k]
            q = np.einsum("ni,ij,nj->n", d, self.prec[k], d)
            out[k] = np.log(self.w[k]) - 0.5 * (2 * np.log(2 * np.pi) + self.logdet[k] + q)
        m = out.max(axis=0)
        return m + np.log(np.exp(out - m).sum(axis=0))

class Ring2D:
    def __init__(self):
        self.name = "ring"
        self.R = 2.5
        self.sig = 0.12
        self.modes = None

    def sample(self, n, rng):
        th = rng.uniform(0, 2 * np.pi, n)
        r = self.R + self.sig * rng.standard_normal(n)
        return np.stack([r * np.cos(th), r * np.sin(th)], axis=1)

    def logpdf(self, x):
        x = np.atleast_2d(x)
        r = np.linalg.norm(x, axis=1)
        log_pr = -0.5 * ((r - self.R) / self.sig) ** 2 - np.log(self.sig * np.sqrt(2 * np.pi))
        return log_pr - np.log(2 * np.pi * np.maximum(r, 1e-12))

def make_dataset(name):
    if name == "sparse_dense_gmm":
        return GMM2D([0.7, 0.2, 0.1], [(0, 0), (4, 3), (-4, 3)], [np.eye(2) * 0.35**2] * 3, name)
    if name == "asym_gmm":
        return GMM2D([0.5, 0.3, 0.2], [(0, 0), (3, 0), (-3, -1)],
                     [np.diag([0.5**2, 0.5**2]), np.diag([1.5**2, 0.3**2]), np.diag([0.4**2, 1.2**2])], name)
    if name == "ring":
        return Ring2D()
    raise ValueError(name)

# ---------------------------------------------------------------------------
# 时间表
# ---------------------------------------------------------------------------
TAU_MAX = 0.5 * (0.1 + 9.95)
S_MAX = 0.5 * (math.exp(2 * TAU_MAX) - 1)

def s_of_tau(tau):
    return 0.5 * torch.expm1(2 * tau)

def tau_of_s(s):
    return 0.5 * torch.log1p(2 * s)

# ---------------------------------------------------------------------------
# 网络（修改：传入 var_scalar，修正归一化）
# ---------------------------------------------------------------------------
class PsiNet(nn.Module):
    def __init__(self, var_scalar, width=256, depth=4, n_freq=8):
        super().__init__()
        self.var_scalar = var_scalar  # 数据方差，用于归一化
        self.n_freq = n_freq
        in_dim = 2 + 2 * n_freq + 1
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]
            d = width
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def featurize(self, y, s):
        u = (tau_of_s(s) / TAU_MAX).unsqueeze(-1)
        k = torch.arange(1, self.n_freq + 1, device=y.device, dtype=y.dtype)
        ang = 2 * math.pi * u * k
        return torch.cat([y, u, torch.sin(ang), torch.cos(ang)], dim=-1)

    def forward(self, y, s):
        # 修正：使用数据实际方差 var_scalar + 2s 进行归一化
        yn = y / (self.var_scalar + 2 * s).sqrt().unsqueeze(-1)
        return self.net(self.featurize(yn, s)).squeeze(-1)

# ---------------------------------------------------------------------------
# kappa 采样器
# ---------------------------------------------------------------------------
class KappaSampler:
    def __init__(self, grid=8192, d=2):
        self.d = d
        v = np.linspace(-30.0, 0.0, grid)
        pdf = v ** 2 * np.exp(v)
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(v))])
        cdf /= cdf[-1]
        self._v_grid_np, self._cdf_np = v, cdf
        self._cache = {}
        if d == 2:
            self.m2 = 1.0 / (16.0 * math.pi)
        else:
            raise NotImplementedError

    def _tables(self, device, dtype):
        key = (device, dtype)
        t = self._cache.get(key)
        if t is None:
            v_grid = torch.as_tensor(self._v_grid_np, device=device, dtype=dtype)
            cdf = torch.as_tensor(self._cdf_np, device=device, dtype=dtype)
            t = (v_grid, cdf)
            self._cache[key] = t
        return t

    def sample(self, n, device, dtype):
        v_grid, cdf = self._tables(device, dtype)
        q = torch.rand(n, device=device, dtype=dtype)
        idx = torch.searchsorted(cdf, q).clamp(1, cdf.numel() - 1)
        c0, c1 = cdf[idx - 1], cdf[idx]
        v0, v1 = v_grid[idx - 1], v_grid[idx]
        frac = (q - c0) / (c1 - c0).clamp(min=1e-12)
        v = v0 + frac * (v1 - v0)
        sigma = torch.exp(v) / (4 * math.pi)
        R = torch.sqrt(-4 * sigma * torch.log(4 * math.pi * sigma))
        t = torch.rand(n, device=device, dtype=dtype) ** 0.5
        g = torch.randn(n, 2, device=device, dtype=dtype)
        zhat = g / g.norm(dim=1, keepdim=True).clamp(min=1e-30)
        z = torch.sqrt(t).unsqueeze(1) * R.unsqueeze(1) * zhat
        return z, sigma

class UniformBallSampler:
    def __init__(self, grid=8192):
        v = np.linspace(-30.0, 0.0, grid)
        pdf = -v * np.exp(2 * v)
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(v))])
        cdf /= cdf[-1]
        self.v_grid, self.cdf = v, cdf

    def sample(self, n, device, dtype):
        q = np.random.rand(n)
        v = np.interp(q, self.cdf, self.v_grid)
        sigma = torch.tensor(np.exp(v) / (4 * math.pi), device=device, dtype=dtype)
        R = torch.sqrt(-4 * sigma * torch.log(4 * math.pi * sigma))
        t = torch.rand(n, device=device, dtype=dtype) ** 0.5
        g = torch.randn(n, 2, device=device, dtype=dtype)
        zhat = g / g.norm(dim=1, keepdim=True)
        z = t.unsqueeze(1) * R.unsqueeze(1) * zhat
        return z, sigma

    @staticmethod
    def kernel_logwt(z, sigma):
        return torch.log((z ** 2).sum(-1).clamp(min=1e-30)) - 2.0 * torch.log(sigma)

def logmeanexp(a, dim=-1):
    m = a.max(dim=dim, keepdim=True).values
    return (m + (a - m).exp().mean(dim=dim, keepdim=True).log()).squeeze(-1)

# ---------------------------------------------------------------------------
# 熵约束 v2
# ---------------------------------------------------------------------------
ES_KAPPA = 1.0 / (32.0 * math.pi)
SIG_KAPPA_MAX = 1.0 / (4.0 * math.pi)

def entropy_gap_v2(model, y0, s0, r2, n_pairs, kap, psi_cut, cmax, q=None):
    B = y0.shape[0]
    yg = y0.detach().requires_grad_(True)
    psi_c = model(yg, s0)
    g_c = torch.autograd.grad(psi_c.sum(), yg, retain_graph=True)[0]
    gsq = g_c.pow(2).sum(-1).detach()
    r2a = torch.minimum(r2, (cmax / gsq.sqrt().clamp(min=1e-8)) ** 2)
    r2a = torch.minimum(r2a, 0.9 * s0 / SIG_KAPPA_MAX)
    h = max(int(n_pairs), 4)
    z, sig = kap.sample(B * h, y0.device, y0.dtype)
    z = z.view(B, h, 2); sig = sig.view(B, h)
    sp = (s0.unsqueeze(1) - r2a.unsqueeze(1) * sig).reshape(-1)
    yp = (yg.unsqueeze(1) + r2a.sqrt().view(-1, 1, 1) * z).reshape(-1, 2)
    ym = (yg.unsqueeze(1) - r2a.sqrt().view(-1, 1, 1) * z).reshape(-1, 2)
    psi_p = model(yp, sp).view(B, h)
    psi_m = model(ym, sp).view(B, h)
    if q is None:
        wb_p = psi_p * psi_p.clamp(max=30).exp()
        wb_m = psi_m * psi_m.clamp(max=30).exp()
        w0 = psi_c.clamp(max=30).exp()
        wb_c = psi_c * w0
        norm = (r2a * ES_KAPPA * w0).unsqueeze(1)
    else:
        wb_p = (q * psi_p).clamp(max=30).exp()
        wb_m = (q * psi_m).clamp(max=30).exp()
        wb_c = (q * psi_c).clamp(max=30).exp()
        norm = (r2a * ES_KAPPA * (q * (q - 1.0)) * wb_c).unsqueeze(1)
    g = ((wb_p + wb_m) * 0.5 - wb_c.unsqueeze(1)) / norm
    G = g.mean(1)
    v2 = g.var(1, unbiased=True).clamp(min=0.0) / h
    mask = (psi_c.detach().abs() <= psi_cut)
    mden = mask.sum().clamp(min=1)
    dz = torch.relu(-G - 2.0 * v2.sqrt()) * mask
    l_hinge = (dz ** 2).sum() / mden
    l_fisher = (((G - gsq) ** 2 - v2) * mask).sum() / mden
    act = ((dz > 0).float().sum() / mden).detach()
    return l_hinge, l_fisher, act, mask.float().mean().detach()

# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# R1 直接模拟对照 (direct-simulation control)
#   不用闭式前向 z=x0+sqrt(2s)eps; 改为: x-系欧拉-丸山积分 dx=(-I+r(t)J0)x dt+sqrt(2)dW,
#   再拉回 z=e^tau R^T x, 有效噪声 eps_eff=(z-x0)/sqrt(2s) 送入原训练循环 (其余一行不改).
#   若共形化定理成立且变换实现正确, 本对照与闭式运行的指标须一致;
#   若 R/R^T 写反等实现 bug 存在, 模拟数据分布错误, 指标会系统性偏离 —— 这正是本对照的目的.
# ---------------------------------------------------------------------------
def build_sim_table(data_t, args, device, dtype):
    n = min(args.sim_n, len(data_t)); K = args.sim_k
    s_lo = min(getattr(args, "s_dsm_min", 1e-2), getattr(args, "s_hb_min", 0.1), getattr(args, "s_min", 0.05))
    s_grid = torch.exp(torch.linspace(math.log(s_lo), math.log(S_MAX), K, device=device, dtype=dtype))
    tau_grid = tau_of_s(s_grid).cpu().numpy()
    x = data_t[:n].clone()
    ztab = torch.empty(n, K, data_t.shape[1], device=device, dtype=dtype)
    t = 0.0; k_next = 0
    while k_next < K:
        tau_next = float(tau_grid[k_next])
        while t < tau_next - 1e-12:
            h = min(args.sim_dt, tau_next - t)
            xr = x.view(n, -1, 2)
            Jx = torch.stack([-xr[..., 1], xr[..., 0]], dim=-1).reshape(n, -1)
            x = x + h * (-x + ROT.r(t) * Jx) + math.sqrt(2 * h) * torch.randn_like(x)
            t += h
        th = ROT.theta(tau_next); c, s_ = math.cos(th), math.sin(th)
        xr = x.view(n, -1, 2)
        zr = torch.stack([c * xr[..., 0] + s_ * xr[..., 1],
                          -s_ * xr[..., 0] + c * xr[..., 1]], dim=-1)
        ztab[:, k_next] = math.exp(tau_next) * zr.reshape(n, -1)
        k_next += 1
    return ztab, s_grid

def train(model, data_t, args, device, dtype, ev=None, probe=None, kap=None):
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=0.1 * args.lr)
    if kap is None:
        kap = KappaSampler()
    uni = UniformBallSampler() if args.hb_measure == "uniform" else None
    rng = np.random.default_rng(args.seed)

    var_scalar = float(data_t.var(dim=0).mean().item())
    K = args.n_anchor_layers
    s_layers = torch.exp(torch.linspace(math.log(args.s_min), math.log(S_MAX), K, device=device, dtype=dtype))
    tau_layers = tau_of_s(s_layers)
    # 修正：提议分布方差为 var_scalar + 2s
    v_layers = var_scalar + 2 * s_layers
    b_k = max(args.batch // K, 32)

    x_lo = data_t.min(dim=0).values; x_hi = data_t.max(dim=0).values
    box_lo = x_lo - args.ent_box_pad; box_hi = x_hi + args.ent_box_pad

    hist = {"total": [], "anchor": [], "hb": [], "shb": [], "score": [],
            "ent": [], "entl": [], "sgn": [], "entf": [], "entfl": [],
            "enta": [], "entm": [], "bb": [], "entq": [], "entqf": [],
            "essb": [], "acv": []}
    hist_eval, probe_hist = [], []
    a_ema = None; a_floor = None
    t0 = time.perf_counter()
    for step in range(args.steps):
        idx0 = rng.integers(0, len(data_t), args.batch)
        x0 = data_t[idx0]

        # (1) 多层 MLE 锚定
        idx_a = rng.integers(0, len(data_t), (K, b_k))
        xk = data_t[idx_a.reshape(-1)].view(K, b_k, 2)
        # 修正：y_a = xk + sqrt(2*s_layers)*noise
        y_a = xk + (2 * s_layers).sqrt().view(K, 1, 1) * torch.randn_like(xk)
        s_a = s_layers.view(K, 1).expand(K, b_k)
        psi_a = model(y_a.reshape(-1, 2), s_a.reshape(-1)).view(K, b_k)

        y_q = (v_layers.sqrt().view(K, 1, 1) * torch.randn(K, args.snis_m, 2, device=device, dtype=dtype))
        s_q = s_layers.view(K, 1).expand(K, args.snis_m)
        psi_q = model(y_q.reshape(-1, 2), s_q.reshape(-1)).view(K, args.snis_m)
        logq = -0.5 * (2 * math.log(2 * math.pi) + torch.log(v_layers).view(K, 1) + (y_q ** 2).sum(-1) / v_layers.view(K, 1))

        logZ = torch.logsumexp(psi_q - logq, dim=1) - math.log(args.snis_m)
        l_anchor = (-psi_a.mean(dim=1) + logZ).mean()

        acv_on = 0.0
        if args.anchor_cv:
            with torch.no_grad():
                Xf = torch.stack([(y_q ** 2).sum(-1) + 4.0 * s_layers.view(K, 1), y_q[..., 0], y_q[..., 1], torch.ones_like(psi_q)], dim=-1)
                Gm = Xf.transpose(-1, -2) @ Xf + 1e-6 * torch.eye(4, device=device, dtype=dtype)
                coef = torch.linalg.solve(Gm, (Xf.transpose(-1, -2) @ psi_q.detach().unsqueeze(-1))).squeeze(-1)
            a_ = coef[:, 0]; b_ = coef[:, 1:3]; c_ = coef[:, 3]
            concave_pre = torch.isfinite(coef).all(dim=1)
            concave = (a_ < -1e-3) & concave_pre
            if bool(concave.any()):
                s_col = s_layers
                g_q = (a_.unsqueeze(1) * ((y_q ** 2).sum(-1) + 4.0 * s_col.view(K, 1)) + (y_q * b_.unsqueeze(1)).sum(-1) + c_.unsqueeze(1))
                logIg = (c_ + 4.0 * a_ * s_col + torch.log(math.pi / (-a_)) + b_.pow(2).sum(-1) / (-4.0 * a_))
                l1 = torch.logsumexp(psi_q - logq, dim=1) - math.log(args.snis_m)
                l2 = torch.logsumexp(g_q - logq, dim=1) - math.log(args.snis_m)
                z_hat = l1.exp() - l2.exp() + logIg.exp()
                logz_cv = z_hat.clamp(min=1e-30).log()
                ok = concave & torch.isfinite(logz_cv) & (logz_cv.abs() < 8.0)
                acv_on = float(ok.float().mean())
                logz_cv = torch.nan_to_num(logz_cv, nan=0.0, posinf=0.0, neginf=0.0)
                anchor_cv = -psi_a.mean(dim=1) + logz_cv
                anchor_snis = -psi_a.mean(dim=1) + logZ
                l_anchor = torch.where(ok, anchor_cv, anchor_snis).mean()
            else:
                acv_on = 0.0

        # (2) 热球 caloric 约束
        log_s0 = torch.empty(args.batch, device=device, dtype=dtype).uniform_(math.log(args.s_hb_min), math.log(S_MAX))
        s0 = log_s0.exp()
        tau0 = tau_of_s(s0)
        if getattr(args, "direct_sim", 0):
            _lo, _hi = math.log(float(args.sim_s[0])), math.log(float(args.sim_s[-1]))
            gi = ((torch.log(s0) - _lo) / (_hi - _lo) * args.sim_s.numel()).long().clamp(0, args.sim_s.numel() - 1)
            rows = torch.as_tensor(idx0, device=s0.device) % args.sim_z.shape[0]
            eps0 = (args.sim_z[rows, gi] - x0) / (2 * s0).sqrt().unsqueeze(1)
        else:
            eps0 = torch.randn_like(x0)
        # 修正：y0 = x0 + sqrt(2*s0)*eps
        y0 = x0 + (2 * s0).sqrt().unsqueeze(1) * eps0
        rj = math.exp(rng.uniform(-math.log(args.r_jitter), math.log(args.r_jitter))) if args.r_jitter > 1.0 else 1.0
        r2 = args.r_frac * s0 * rj
        if args.hb_cmax > 0:
            with torch.enable_grad():
                y0g = y0.detach().requires_grad_(True)
                _pc = model(y0g, s0)
                _gc = torch.autograd.grad(_pc.sum(), y0g)[0]
            _gsq = _gc.pow(2).sum(-1).detach()
            r2 = torch.minimum(r2, (args.hb_cmax / _gsq.sqrt().clamp(min=1e-8)) ** 2)
            r2 = torch.minimum(r2, 0.9 * s0 / SIG_KAPPA_MAX)
        if uni is None:
            z, sig = kap.sample(args.batch * args.n_kappa, device, dtype)
            logk = None
        else:
            z, sig = uni.sample(args.batch * args.n_kappa, device, dtype)
            logk = UniformBallSampler.kernel_logwt(z, sig)
        z = z.view(args.batch, args.n_kappa, 2)
        sig = sig.view(args.batch, args.n_kappa)
        y_i = y0.unsqueeze(1) + r2.sqrt().view(-1, 1, 1) * z
        s_i = s0.view(-1, 1) - r2.view(-1, 1) * sig
        psi_c = model(y0, s0)
        psi_i = model(y_i.reshape(-1, 2), s_i.reshape(-1)).view(args.batch, args.n_kappa)
        ess_bad_frac = 0.0
        if args.train_ess and logk is None:
            with torch.no_grad():
                _w = torch.softmax(psi_i, dim=1)
                _ess = 1.0 / (_w ** 2).sum(1).clamp(min=1e-30)
                _bad = _ess < args.ess_thr * args.n_kappa
            ess_bad_frac = float(_bad.float().mean())
            if bool(_bad.any()):
                nb = int(_bad.sum())
                z2, sig2 = kap.sample(nb * 2 * args.n_kappa, device, dtype)
                z2 = z2.view(nb, 2 * args.n_kappa, 2)
                sig2 = sig2.view(nb, 2 * args.n_kappa)
                y_i2 = y0[_bad].unsqueeze(1) + r2[_bad].sqrt().view(-1, 1, 1) * z2
                s_i2 = s0[_bad].view(-1, 1) - r2[_bad].view(-1, 1) * sig2
                psi_i2 = model(y_i2.reshape(-1, 2), s_i2.reshape(-1)).view(nb, 2 * args.n_kappa)
                lme_bad = logmeanexp(torch.cat([psi_i[_bad], psi_i2], dim=1), dim=1)
                _lme = logmeanexp(psi_i, dim=1).clone()
                _lme[_bad] = lme_bad
                lme = _lme
            else:
                lme = logmeanexp(psi_i, dim=1)
        elif logk is None:
            lme = logmeanexp(psi_i, dim=1)
        else:
            logk = logk.view(args.batch, args.n_kappa)
            lme = torch.logsumexp(psi_i + logk, dim=1) - torch.logsumexp(logk, dim=1)
        l_hb = ((psi_c - lme) ** 2).mean()

        # (2a) 多尺度球-球一致性
        l_bb = torch.zeros((), device=device, dtype=dtype)
        if args.w_bb > 0:
            r2b = r2 * args.bb_ratio
            if uni is None:
                zb, sigb = kap.sample(args.batch * args.n_kappa, device, dtype)
            else:
                zb, sigb = uni.sample(args.batch * args.n_kappa, device, dtype)
            zb = zb.view(args.batch, args.n_kappa, 2)
            sigb = sigb.view(args.batch, args.n_kappa)
            y_ib = y0.unsqueeze(1) + r2b.sqrt().view(-1, 1, 1) * zb
            s_ib = s0.view(-1, 1) - r2b.view(-1, 1) * sigb
            psi_ib = model(y_ib.reshape(-1, 2), s_ib.reshape(-1)).view(args.batch, args.n_kappa)
            if uni is None:
                lme_b = logmeanexp(psi_ib, dim=1)
            else:
                logk_b = UniformBallSampler.kernel_logwt(zb.reshape(-1, 2), sigb.reshape(-1)).view(args.batch, -1)
                lme_b = torch.logsumexp(psi_ib + logk_b, dim=1) - torch.logsumexp(logk_b, dim=1)
            l_bb = ((lme - lme_b) ** 2).mean()

        # (2b) score 热球一致性
        l_shb = torch.zeros((), device=device, dtype=dtype)
        if args.w_shb > 0:
            yg = y_i.reshape(-1, 2).detach().requires_grad_(True)
            sg = s_i.reshape(-1)
            psi_g = model(yg, sg)
            g_i = torch.autograd.grad(psi_g.sum(), yg, create_graph=True)[0]
            yc = y0.detach().requires_grad_(True)
            psi_cc = model(yc, s0)
            g_c = torch.autograd.grad(psi_cc.sum(), yc, create_graph=True)[0]
            omega_logits = (psi_g.view(args.batch, args.n_kappa) - psi_cc.view(args.batch, 1))
            if logk is not None:
                omega_logits += logk
            omega = torch.softmax(omega_logits.detach(), dim=1)
            s_avg = (omega.unsqueeze(-1) * g_i.view(args.batch, args.n_kappa, 2)).sum(1)
            l_shb = ((g_c - s_avg) ** 2).sum(-1).mean()

        # (2c) 可选：前向条件 score 约束
        l_score = torch.zeros((), device=device, dtype=dtype)
        if args.w_mse > 0 or args.w_dir > 0:
            mask = s0 >= args.dir_s_min
            if mask.any():
                x0_m = x0[mask]; eps0_m = eps0[mask]; s0_m = s0[mask]; tau0_m = tau0[mask]
                # 修正：y_p = x0_m + sqrt(2s)*eps
                y_p = (x0_m + (2 * s0_m).sqrt().unsqueeze(1) * eps0_m).requires_grad_(True)
                y_m = (x0_m - (2 * s0_m).sqrt().unsqueeze(1) * eps0_m).requires_grad_(True)
                psi_p = model(y_p, s0_m); psi_m = model(y_m, s0_m)
                g_p = torch.autograd.grad(psi_p.sum(), y_p, create_graph=True)[0]
                g_m = torch.autograd.grad(psi_m.sum(), y_m, create_graph=True)[0]
                target_p = -eps0_m / (2 * s0_m).sqrt().unsqueeze(1)
                target_m =  eps0_m / (2 * s0_m).sqrt().unsqueeze(1)

                weight = (2 * s0_m).clamp(max=args.mse_weight_cap).unsqueeze(1)
                mse_p = (weight * (g_p - target_p).pow(2)).sum(-1)
                mse_m = (weight * (g_m - target_m).pow(2)).sum(-1)
                mse_per = 0.5 * (mse_p + mse_m)

                cos_p = torch.nn.functional.cosine_similarity(g_p, target_p, dim=-1, eps=1e-8)
                cos_m = torch.nn.functional.cosine_similarity(g_m, target_m, dim=-1, eps=1e-8)
                dir_per = 0.5 * ((1.0 - cos_p) + (1.0 - cos_m))

                norm_target = target_p.norm(dim=-1, keepdim=True).squeeze(-1)
                w_mse_dyn = torch.clamp(args.dir_norm_threshold / (norm_target + 1e-8), min=0.1, max=1.0)
                w_dir_dyn = torch.clamp(norm_target / args.dir_norm_threshold, min=0.1, max=1.0)
                loss_per = args.w_mse * w_mse_dyn * mse_per + args.w_dir * w_dir_dyn * dir_per
                l_score = loss_per.mean()

        # (3) 熵约束
        l_ent = torch.zeros((), device=device, dtype=dtype)
        l_entl = torch.zeros((), device=device, dtype=dtype)
        l_entf = torch.zeros((), device=device, dtype=dtype)
        l_entfl = torch.zeros((), device=device, dtype=dtype)
        l_entq2 = torch.zeros((), device=device, dtype=dtype)
        l_entfq2 = torch.zeros((), device=device, dtype=dtype)
        l_entq2l = torch.zeros((), device=device, dtype=dtype)
        l_entfq2l = torch.zeros((), device=device, dtype=dtype)
        ent_act = torch.zeros((), device=device, dtype=dtype)
        ent_mask = torch.zeros((), device=device, dtype=dtype)
        w_ent = 0.0
        if args.lam_ent > 0:
            w_ent = args.lam_ent * min(1.0, step / max(1, args.ent_warmup))
            if args.ent_ver == 2:
                l_ent, l_entf, ent_act, ent_mask = entropy_gap_v2(model, y0, s0, r2, args.n_ent_pairs, kap, args.ent_psi_cut, args.ent_cmax)
                if args.ent_q2 != 0.0:
                    l_entq2, l_entfq2 = entropy_gap_v2(model, y0, s0, r2, args.n_ent_pairs, kap, args.ent_q2_cut, args.ent_cmax, q=args.ent_q2)[:2]
            else:
                wb_c = psi_c * psi_c.clamp(max=30).exp()
                wb_i = psi_i * psi_i.clamp(max=30).exp()
                if logk is None: wb_bar = wb_i.mean(dim=1)
                else:
                    wk = torch.softmax(logk, dim=1); wb_bar = (wk * wb_i).sum(dim=1)
                l_ent = (torch.relu(wb_c - wb_bar) ** 2).mean()

        # ---- 权重整形 ----
        a_now = float(l_anchor.detach())
        a_ema = a_now if a_ema is None else 0.98 * a_ema + 0.02 * a_now
        a_floor = a_ema if a_floor is None else min(a_floor, a_ema)
        in_warmup = step < args.anchor_warmup
        if in_warmup:
            ramp = 0.0
        else:
            ramp = min(1.0, (step - args.anchor_warmup) / max(1, args.ramp_steps))
        gate = math.exp(-max(0.0, a_ema - a_floor)) if args.gate else 1.0

        shb_ramp = 0.0 if in_warmup else min(1.0, max(0.0, (step - args.anchor_warmup - args.shb_delay) / max(1, args.ramp_steps)))
        w_hb_eff   = args.w_hb   * gate * ramp
        w_shb_eff  = args.w_shb  * gate * shb_ramp
        w_bb_eff   = args.w_bb   * gate * ramp
        w_score_eff = args.w_score_scale * gate * ramp

        if in_warmup:
            w_ent = 0.0
        else:
            w_ent = w_ent * gate * ramp

        loss = (args.w_anchor * l_anchor + w_hb_eff * l_hb + w_shb_eff * l_shb
                + w_score_eff * l_score + w_bb_eff * l_bb
                + w_ent * (l_ent + l_entl + args.w_entf * (l_entf + l_entfl)
                           + args.w_entq2 * (l_entq2 + l_entq2l + args.w_entf * (l_entfq2 + l_entfq2l))))

        if not torch.isfinite(loss):
            if os.environ.get("HBDEBUG"): print(f"[dbg] step {step}: loss 非有限")
            opt.zero_grad(set_to_none=True); continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        _gok = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if not _gok:
            if os.environ.get("HBDEBUG"): print(f"[dbg] step {step}: 梯度非有限")
            opt.zero_grad(set_to_none=True); continue
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step(); sched.step()

        for k, v in [("total", loss), ("anchor", l_anchor), ("hb", l_hb), ("shb", l_shb), ("score", l_score),
                     ("ent", l_ent), ("entl", l_entl), ("entf", l_entf), ("entfl", l_entfl),
                     ("enta", ent_act), ("entm", ent_mask), ("bb", l_bb), ("entq", l_entq2), ("entqf", l_entfq2)]:
            hist[k].append(float(v.detach()))
        hist["essb"].append(ess_bad_frac); hist["acv"].append(acv_on)
        hist["sgn"].append(float((psi_c - lme).mean().detach()))
        if (step + 1) % 500 == 0:
            wmean = {k: float(np.mean(hist[k][-500:])) if hist[k] else float("nan")
                     for k in ("total", "anchor", "hb", "shb", "score", "ent", "entl", "entf", "enta", "entm", "bb", "entq", "entqf", "essb", "acv")}
            sgn_win = np.asarray(hist["sgn"][-500:]); hb_sys = abs(float(sgn_win.mean()))
            hb_var = max(wmean["hb"] - hb_sys ** 2, 0.0)
            line = (f"step {step + 1}/{args.steps}  loss*={wmean['total']:.4e} anchor*={wmean['anchor']:.4e} "
                    f"hb*={wmean['hb']:.4e} shb*={wmean['shb']:.4e} score*={wmean['score']:.4e} "
                    f"ent*={wmean['ent']:.4e} entl*={wmean['entl']:.4e} gate={gate:.2f} "
                    f"hb_sys={hb_sys:.2e} hb_noise={math.sqrt(hb_var):.2e}")
            if args.lam_ent > 0 and args.ent_ver == 2:
                line += f" entf*={wmean['entf']:.4e} enta={wmean['enta']:.3f} entm={wmean['entm']:.2f}"
            if args.w_bb > 0: line += f" bb*={wmean['bb']:.4e}"
            if args.train_ess: line += f" essb={wmean['essb']:.3f}"
            if args.anchor_cv: line += f" acv={wmean['acv']:.2f}"
            extra = []
            if probe is not None:
                res = hb_residual(model, probe); probe_hist.append(dict(step=step + 1, res=res)); extra.append(f"hb_res={res:.4e}")
            if ev is not None and ev.ok:
                m = ev.mse(model, 0.3, device, dtype)
                hist_eval.append(dict(step=step + 1, mse_global=m["global_"], mse_high=m["high"], mse_low=m["low"],
                                      rel_global=m["global_rel"], rel_high=m["high_rel"], rel_low=m["low_rel"]))
                extra.append(f"mse_x={m['global_']:.4e} (h={m['high']:.4e} l={m['low']:.4e}) "
                             f"rel_x={m['global_rel']:.4e} (h={m['high_rel']:.4e} l={m['low_rel']:.4e})")
            print(line + ("  " + "  ".join(extra) if extra else ""), flush=True)
    return hist, hist_eval, probe_hist, time.perf_counter() - t0

# ---------------------------------------------------------------------------
# 采样、评估、绘图等函数
# ---------------------------------------------------------------------------
@torch.no_grad()
def score(model, y, s):
    y = y.detach().requires_grad_(True)
    with torch.enable_grad():
        psi = model(y, s)
        return torch.autograd.grad(psi.sum(), y)[0]

@torch.no_grad()
def score_hb(model, y, s, n_eval, r_frac, var_scalar, device, dtype, kap, ess_thr=0.5, cv=False):
    n = len(y)
    z, sig = kap.sample(n * n_eval, device, dtype)
    z = z.view(n, n_eval, 2); sig = sig.view(n, n_eval)
    r2 = r_frac * s
    shrinks = (1.0, 0.25, 0.0625)
    ess_list = []
    for sh in shrinks:
        rr = r2 * sh
        y_i = y.unsqueeze(1) + rr.sqrt().view(-1, 1, 1) * z
        s_i = s.view(-1, 1) - rr.view(-1, 1) * sig
        p = model(y_i.reshape(-1, 2), s_i.reshape(-1)).view(n, n_eval)
        # 稳定化 exp
        p_shifted = p - p.max(dim=1, keepdim=True).values
        e = p_shifted.clamp(max=50).exp()
        ess_list.append(e.sum(1) ** 2 / (e ** 2).sum(1).clamp(min=1e-30))
    choice = torch.zeros(n, dtype=torch.long, device=device)
    for j in range(1, len(shrinks)):
        move = (ess_list[j - 1] < ess_thr * n_eval) & (choice == j - 1)
        choice = torch.where(move, torch.full_like(choice, j), choice)
    psi_c = model(y, s)
    S = torch.zeros(n, 2, device=device, dtype=dtype)
    res_all = torch.zeros(n, device=device, dtype=dtype)
    ess_all = torch.zeros(n, device=device, dtype=dtype)
    for j, sh in enumerate(shrinks):
        idx = (choice == j).nonzero(as_tuple=True)[0]
        if idx.numel() == 0: continue
        rr = (r2[idx] * sh)
        zz, sg = z[idx], sig[idx]
        y_g = (y[idx].unsqueeze(1) + rr.sqrt().view(-1, 1, 1) * zz).reshape(-1, 2)
        s_g = (s[idx].view(-1, 1) - rr.view(-1, 1) * sg).reshape(-1)
        y_g = y_g.detach().requires_grad_(True)
        with torch.enable_grad():
            psi_i = model(y_g, s_g)
            g_i = torch.autograd.grad(psi_i.sum(), y_g)[0]
        psi_i = psi_i.view(-1, n_eval)
        # softmax with stable exp
        logits = psi_i - psi_i.max(dim=1, keepdim=True).values
        omega = torch.exp(logits.clamp(max=50)) / (torch.exp(logits.clamp(max=50)).sum(dim=1, keepdim=True) + 1e-30)
        if not cv:
            S[idx] = (omega.unsqueeze(-1) * g_i.view(-1, n_eval, 2)).sum(1)
        else:
            with torch.enable_grad():
                yc = y[idx].detach().requires_grad_(True)
                psi_cc = model(yc, s[idx])
                g_c = torch.autograd.grad(psi_cc.sum(), yc, create_graph=True)[0]
                H0 = torch.autograd.grad(g_c[:, 0].sum(), yc, retain_graph=True)[0]
                H1 = torch.autograd.grad(g_c[:, 1].sum(), yc)[0]
                Hc = torch.stack([H0, H1], dim=1)
            dpsi = psi_i - psi_c[idx].unsqueeze(1)
            logD = torch.logsumexp(dpsi, dim=1)
            Dinv = (-logD).exp()
            rt = rr.sqrt().view(-1, 1)
            a1s = (rt * torch.einsum("mnj,mj->mn", zz, g_c)).sum(1)
            b1s = (rt.unsqueeze(-1) * torch.einsum("mnj,mjk->mnk", zz, Hc)).sum(1)
            Snum = (omega.unsqueeze(-1) * g_i.view(-1, n_eval, 2)).sum(1) \
                - g_c * (a1s * Dinv).unsqueeze(-1) - b1s * Dinv.unsqueeze(-1)
            Sden = 1.0 - a1s * Dinv
            S[idx] = Snum / Sden.clamp(min=1e-6).unsqueeze(-1)
        # 使用稳定的 logmeanexp
        log_mean = torch.logsumexp(psi_i, dim=1) - math.log(n_eval)
        res_all[idx] = log_mean - psi_c[idx]
        ess_all[idx] = ess_list[j][idx]
    S = torch.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
    res_all = torch.nan_to_num(res_all, nan=0.0, posinf=0.0, neginf=0.0)
    diag = dict(ess=float(ess_all.mean()),
                shrink1=float((choice == 1).float().mean()),
                shrink2=float((choice == 2).float().mean()),
                hb_res=float(res_all.abs().mean()))
    return S, diag

def score_moment(model, y, s, n_eval, r_frac, var_scalar, device, dtype, kap):
    n = len(y)
    h = max(n_eval // 2, 1)
    z, sig = kap.sample(n * h, device, dtype)
    z = z.view(n, h, 2); sig = sig.view(n, h)
    r2 = r_frac * s
    rt = r2.sqrt().view(-1, 1, 1)
    y_p = y.unsqueeze(1) + rt * z
    y_m = y.unsqueeze(1) - rt * z
    s_i = s.view(-1, 1) - r2.view(-1, 1) * sig
    with torch.no_grad():
        psi_c = model(y, s)
        psi_p = model(y_p.reshape(-1, 2), s_i.reshape(-1)).view(n, h)
        psi_m = model(y_m.reshape(-1, 2), s_i.reshape(-1)).view(n, h)
        # 稳定化 exp
        max_val = torch.cat([psi_p, psi_m], dim=1).max(dim=1, keepdim=True).values
        Rp = (psi_p - psi_c.unsqueeze(1) - max_val).clamp(max=50).exp()
        Rm = (psi_m - psi_c.unsqueeze(1) - max_val).clamp(max=50).exp()
        r = r2.sqrt().view(-1, 1)
        S = (1.0 / kap.m2 / r) * ((Rp - Rm).unsqueeze(-1) * z).sum(1) \
            / (Rp + Rm).sum(1, keepdim=True).clamp(min=1e-30)
        e = torch.cat([Rp, Rm], dim=1)
        ess = e.sum(1) ** 2 / (e ** 2).sum(1).clamp(min=1e-30)
        log_mean = torch.logsumexp(torch.cat([psi_p, psi_m], dim=1), dim=1) - math.log(2*h)
        res = log_mean - psi_c
    S = torch.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
    return S, dict(ess=float(ess.mean()), shrink1=0.0, shrink2=0.0, hb_res=float(res.abs().mean()))

def score_moment_richardson(model, y, s, n_eval, r_frac, var_scalar, device, dtype, kap):
    n = len(y)
    h = max(n_eval // 2, 1)
    z, sig = kap.sample(n * h, device, dtype)
    z = z.view(n, h, 2); sig = sig.view(n, h)
    r2 = r_frac * s
    r2_half = r2 / 4.0
    with torch.no_grad():
        psi_c = model(y, s).unsqueeze(1)
        rt = r2.sqrt().view(-1, 1, 1)
        y_p = y.unsqueeze(1) + rt * z
        y_m = y.unsqueeze(1) - rt * z
        s_i = s.view(-1, 1) - r2.view(-1, 1) * sig
        psi_p = model(y_p.reshape(-1, 2), s_i.reshape(-1)).view(n, h)
        psi_m = model(y_m.reshape(-1, 2), s_i.reshape(-1)).view(n, h)
        max_val = torch.cat([psi_p, psi_m], dim=1).max(dim=1, keepdim=True).values
        Rp = (psi_p - psi_c - max_val).clamp(max=50).exp()
        Rm = (psi_m - psi_c - max_val).clamp(max=50).exp()
        r = r2.sqrt().view(-1, 1)
        S_r = (1.0 / kap.m2 / r) * ((Rp - Rm).unsqueeze(-1) * z).sum(1) \
            / (Rp + Rm).sum(1, keepdim=True).clamp(min=1e-30)
        rt_half = r2_half.sqrt().view(-1, 1, 1)
        y_p_half = y.unsqueeze(1) + rt_half * z
        y_m_half = y.unsqueeze(1) - rt_half * z
        s_i_half = s.view(-1, 1) - r2_half.view(-1, 1) * sig
        psi_p_half = model(y_p_half.reshape(-1, 2), s_i_half.reshape(-1)).view(n, h)
        psi_m_half = model(y_m_half.reshape(-1, 2), s_i_half.reshape(-1)).view(n, h)
        max_val_half = torch.cat([psi_p_half, psi_m_half], dim=1).max(dim=1, keepdim=True).values
        Rp_half = (psi_p_half - psi_c - max_val_half).clamp(max=50).exp()
        Rm_half = (psi_m_half - psi_c - max_val_half).clamp(max=50).exp()
        r_half = r2_half.sqrt().view(-1, 1)
        S_r_half = (1.0 / kap.m2 / r_half) * ((Rp_half - Rm_half).unsqueeze(-1) * z).sum(1) \
            / (Rp_half + Rm_half).sum(1, keepdim=True).clamp(min=1e-30)
        S_rich = (4.0 * S_r_half - S_r) / 3.0
        e = torch.cat([Rp, Rm], dim=1)
        ess = e.sum(1) ** 2 / (e ** 2).sum(1).clamp(min=1e-30)
        log_mean = torch.logsumexp(torch.cat([psi_p, psi_m], dim=1), dim=1) - math.log(2*h)
        res = log_mean - psi_c.squeeze(1)
    S_rich = torch.nan_to_num(S_rich, nan=0.0, posinf=0.0, neginf=0.0)
    return S_rich, dict(ess=float(ess.mean()), shrink1=0.0, shrink2=0.0, hb_res=float(res.abs().mean()))

def sample(model, args, device, dtype, var_scalar, data_t=None, kap=None):
    n = args.n_samples
    if kap is None:
        kap = KappaSampler()
    init_mode = getattr(args, "sample_init", "data")
    if init_mode == "data" and data_t is not None:
        idx0 = torch.randint(0, len(data_t), (n,), device=device)
        x0 = data_t[idx0]
        # 修正：y = x0 + sqrt(2*S_MAX)*noise
        y = x0 + math.sqrt(2 * S_MAX) * torch.randn(n, 2, device=device, dtype=dtype)
    else:
        v_prior = var_scalar + 2 * S_MAX
        y = math.sqrt(v_prior) * torch.randn(n, 2, device=device, dtype=dtype)

    def build_schedule(tau_max, n_main, n_tail, s_tail_start, device, dtype):
        s_tail_start_tensor = torch.tensor(s_tail_start, device=device, dtype=dtype)
        tau_tail_start = tau_of_s(s_tail_start_tensor).item()
        taus_main = torch.linspace(tau_max, tau_tail_start, n_main + 1, device=device, dtype=dtype)
        ss_main = s_of_tau(taus_main)
        s_floor = 1e-5
        log_s_tail = torch.linspace(math.log(s_tail_start), math.log(s_floor), n_tail + 1,
                                    device=device, dtype=dtype)
        ss_tail = torch.exp(log_s_tail)
        ss_tail[-1] = 1e-5
        ss = torch.cat([ss_main[:-1], ss_tail])
        return ss

    n_tail = 100
    n_main = args.sample_steps - n_tail
    if n_main <= 0:
        raise ValueError("sample_steps 必须大于 100")
    s_tail_start = 0.3
    ss = build_schedule(TAU_MAX, n_main, n_tail, s_tail_start, device, dtype)

    t0 = time.perf_counter()
    diag_acc = []
    for k in range(len(ss) - 1):
        s_hi, s_lo = ss[k], ss[k + 1]
        ds = float(s_hi - s_lo)
        s_vec = torch.full((n,), float(s_hi), device=device, dtype=dtype)
        if args.score_est == "hb":
            sc, dg = score_hb(model, y, s_vec, args.n_kappa_eval, args.r_frac, var_scalar,
                              device, dtype, kap, ess_thr=args.ess_thr, cv=bool(args.score_cv))
            diag_acc.append(dg)
        elif args.score_est == "moment":
            sc, dg = score_moment(model, y, s_vec, args.n_kappa_eval, args.r_frac, var_scalar,
                                  device, dtype, kap)
            diag_acc.append(dg)
        elif args.score_est == "moment_richardson":
            sc, dg = score_moment_richardson(model, y, s_vec, args.n_kappa_eval, args.r_frac, var_scalar,
                                             device, dtype, kap)
            diag_acc.append(dg)
        else:
            sc = score(model, y, s_vec)
        y = y + 2.0 * sc * ds + math.sqrt(2 * ds) * torch.randn_like(y)
        # 诊断打印（每20步或遇到NaN）
        if (k % max(1, len(ss)//20) == 0) or not torch.isfinite(y).all():
            with torch.no_grad():
                psi_c = model(y, s_vec)
                y_norm = y.norm(dim=1).mean().item()
                psi_min, psi_max = psi_c.min().item(), psi_c.max().item()
            print(f"[sample] step {k:4d}, s={s_hi:.3e}, |y|_mean={y_norm:.3e}, psi∈[{psi_min:.2e}, {psi_max:.2e}]")
            if not torch.isfinite(y).all():
                print(">>> y 出现 NaN，停止采样")
                break
    if abs(args.omega) > 0.0:
        tau_end = 0.5 * math.log1p(2 * float(ss[-1]))
        Rend = torch.as_tensor(ROT.R(tau_end).T, device=device, dtype=dtype)
        y = math.exp(-tau_end) * (y @ Rend)
    diag = None
    if diag_acc:
        diag = {k: float(np.mean([d[k] for d in diag_acc])) for k in diag_acc[0]}
        print(f"[score_est={args.score_est} 诊断] 平均 ESS=%.1f/%d  沿线 HB 残差均值=%.4f"
              % (diag["ess"], args.n_kappa_eval, diag["hb_res"]), flush=True)
    return y.detach().cpu().numpy(), time.perf_counter() - t0, diag

# ---------------------------------------------------------------------------
# 指标与绘图
# ---------------------------------------------------------------------------
def mmd_rbf(x, y):
    x = torch.as_tensor(x, dtype=torch.float32); y = torch.as_tensor(y, dtype=torch.float32)
    with torch.no_grad():
        dxy = torch.cdist(torch.cat([x, y]), torch.cat([x, y])) ** 2
        med = dxy.median().clamp(min=1e-6)
        k = (-dxy / (2 * med)).exp()
        n, m = len(x), len(y)
        kxx = k[:n, :n]; kyy = k[n:, n:]; kxy = k[:n, n:]
        return float((kxx.sum() - kxx.diagonal().sum()) / (n * (n - 1))
                     + (kyy.sum() - kyy.diagonal().sum()) / (m * (m - 1))
                     - 2 * kxy.mean())

def sliced_wasserstein(x, y, n_proj=50, seed=0):
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((n_proj, 2)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return float(np.mean([wasserstein_distance(x @ d, y @ d) for d in dirs]))

def mode_metrics(ds, gen):
    if ds.modes is None: return None, None
    K = len(ds.w)
    sig = np.sqrt(np.array([np.mean(np.diag(c)) for c in ds.cov]))
    d2 = ((gen[:, None, :] - ds.modes[None]) ** 2).sum(-1)
    assign = d2.argmin(1)
    near = d2[np.arange(len(gen)), assign] < (3 * sig[assign]) ** 2
    counts = np.array([((assign == k) & near).sum() for k in range(K)], dtype=np.float64)
    prop = counts / max(counts.sum(), 1.0)
    eps = 1e-3
    p = np.clip(prop, eps, None); p /= p.sum()
    q = np.clip(ds.w, eps, None); q /= q.sum()
    kl = float(np.sum(q * np.log(q / p)))
    coverage = float(np.mean(counts > max(1, 0.005 * len(gen))))
    return coverage, kl

def lowdens_metrics(ds, data, gen):
    lp_data = ds.logpdf(data); thr = np.quantile(lp_data, 0.20); lp_gen = ds.logpdf(gen)
    return dict(lowdens_frac_gen=float(np.mean(lp_gen < thr)),
                lowdens_frac_data=float(np.mean(lp_data < thr)),
                tail_logpdf_gen=float(lp_gen[lp_gen < thr].mean()) if (lp_gen < thr).any() else float("nan"),
                tail_logpdf_data=float(lp_data[lp_data < thr].mean()))

def make_plots(ds, data, gen, hist, outdir, tag):
    os.makedirs(outdir, exist_ok=True)
    pad = 2.0; lo = data.min(0) - pad; hi = data.max(0) + pad
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 200), np.linspace(lo[1], hi[1], 200))
    zz = ds.logpdf(np.stack([gx.ravel(), gy.ravel()], 1)).reshape(gx.shape)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.contour(gx, gy, zz, levels=10, cmap="viridis", linewidths=0.8)
    ax.scatter(data[:, 0], data[:, 1], s=3, c="gray", alpha=0.15, label="data")
    ax.scatter(gen[:, 0], gen[:, 1], s=5, c="crimson", alpha=0.5, label="generated")
    ax.legend(); ax.set_title(f"Samples ({tag})")
    fig.tight_layout(); fig.savefig(f"{outdir}/samples.png", dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for k in hist:
        v = np.asarray(hist[k]); ax.plot(np.maximum(v, 1e-12), label=k)
    ax.set_yscale("log"); ax.legend(); ax.set_title("Training loss")
    fig.tight_layout(); fig.savefig(f"{outdir}/loss.png", dpi=140); plt.close(fig)
    if ds.modes is not None:
        sig = np.sqrt(np.array([np.mean(np.diag(c)) for c in ds.cov]))
        d2 = ((gen[:, None, :] - ds.modes[None]) ** 2).sum(-1); assign = d2.argmin(1)
        near = d2[np.arange(len(gen)), assign] < (3 * sig[assign]) ** 2
        K = len(ds.w); prop = np.array([((assign == k) & near).mean() for k in range(K)]); prop /= max(prop.sum(), 1e-12)
        fig, ax = plt.subplots(figsize=(5.5, 4)); xs = np.arange(K); wd = 0.38
        ax.bar(xs - wd/2, ds.w, wd, label="true"); ax.bar(xs + wd/2, prop, wd, label="generated")
        ax.set_xticks(xs); ax.legend(); ax.set_title("Mode proportions")
        fig.tight_layout(); fig.savefig(f"{outdir}/modes.png", dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ds.logpdf(data), bins=60, density=True, alpha=0.5, label="true logpdf @ data")
    ax.hist(ds.logpdf(gen), bins=60, density=True, alpha=0.5, label="true logpdf @ generated")
    ax.legend(); ax.set_title("Tail behavior")
    fig.tight_layout(); fig.savefig(f"{outdir}/tail.png", dpi=140); plt.close(fig)

@torch.no_grad()
def anchor_nll(model, data_t, args, device, dtype, var_scalar):
    tau_min = 0.5 * math.log1p(2 * args.s_min)
    v_q = var_scalar + 2 * args.s_min
    x0 = data_t[torch.randint(0, len(data_t), (2000,))]
    y_a = x0 + math.sqrt(2 * args.s_min) * torch.randn_like(x0)
    s_a = torch.full((len(x0),), args.s_min, device=device, dtype=dtype)
    M = 8192
    y_q = math.sqrt(v_q) * torch.randn(M, 2, device=device, dtype=dtype)
    s_q = torch.full((M,), args.s_min, device=device, dtype=dtype)
    logq = -0.5 * (2 * math.log(2 * math.pi) + math.log(v_q) + (y_q ** 2).sum(1) / v_q)
    logw_q = model(y_q, s_q) - logq; logZ = (logw_q - math.log(M)).logsumexp(0)
    return float(-(model(y_a, s_a).mean() - logZ))

def analytic_score_gmm(ds, x, tau, R=None):
    if R is None: R = np.eye(2)
    if ds.modes is None: return None
    e2 = math.exp(-2 * tau); x = np.atleast_2d(x); K = len(ds.w)
    logc = np.empty((K, len(x))); sc = np.zeros((K, len(x), 2))
    for k in range(K):
        Sig = e2 * (R @ ds.cov[k] @ R.T) + (1 - e2) * np.eye(2); P = np.linalg.inv(Sig); m = math.exp(-tau) * (ds.mu[k] @ R.T)
        d = x - m; q = np.einsum("ni,ij,nj->n", d, P, d)
        logc[k] = np.log(ds.w[k]) - 0.5 * (2 * np.log(2 * np.pi) + np.linalg.slogdet(Sig)[1] + q)
        sc[k] = -d @ P.T
    mx = logc.max(0); rho = np.exp(logc - mx); rho /= rho.sum(0)
    return np.einsum("kn,knj->nj", rho, sc)

def gmm_logpdf_at_tau(ds, x, tau, R=None):
    if R is None: R = ROT.R(tau)
    e2 = math.exp(-2 * tau); x = np.atleast_2d(x); K = len(ds.w)
    logc = np.empty((K, len(x)))
    for k in range(K):
        Sig = e2 * (R @ ds.cov[k] @ R.T) + (1 - e2) * np.eye(2); P = np.linalg.inv(Sig); m = math.exp(-tau) * (ds.mu[k] @ R.T)
        d = x - m; q = np.einsum("ni,ij,nj->n", d, P, d)
        logc[k] = np.log(ds.w[k]) - 0.5 * (2 * np.log(2 * np.pi) + np.linalg.slogdet(Sig)[1] + q)
    mx = logc.max(0); return mx + np.log(np.exp(logc - mx).sum(0))

def model_score_x(model, x, tau, device, dtype, var_scalar):
    s_val = 0.5 * math.expm1(2 * tau)
    e = math.exp(tau)
    xt = torch.as_tensor(x, device=device, dtype=dtype)
    # 坐标变换 y = e^τ R^T x（旋转动坐标），送入网络前需要归一化（与 forward 一致）
    Rt = torch.as_tensor(ROT.R(tau), device=device, dtype=dtype)
    y = (e * (xt @ Rt)).requires_grad_(True)
    s_full = torch.full((len(xt),), s_val, device=device, dtype=dtype)
    # 注意：PsiNet.forward 会对 y 做归一化，所以这里直接传入 y 即可
    with torch.enable_grad():
        psi = model(y, s_full)
        g = torch.autograd.grad(psi.sum(), y)[0]
    # 换算回 x 空间的得分: ∇_x log p = e^τ R ∇_y ψ
    return (e * (g @ Rt.T)).detach().cpu().numpy()

class ScoreEvalSet:
    def __init__(self, ds, data_np, taus=(0.1, 0.3, 0.5, 0.8), n=2000, seed=123):
        self.ok = ds.modes is not None
        if not self.ok: return
        rng = np.random.default_rng(seed); x0 = data_np[rng.integers(0, len(data_np), n)]
        self.per_tau = {}
        for tau in taus:
            e2 = math.exp(-2 * tau); R = ROT.R(tau); x = math.exp(-tau) * (x0 @ R.T) + math.sqrt(1 - e2) * rng.standard_normal(x0.shape)
            lp_tau = gmm_logpdf_at_tau(ds, x, tau); mask_low = lp_tau < np.quantile(lp_tau, 0.2)
            self.per_tau[tau] = (x, analytic_score_gmm(ds, x, tau, R), mask_low)
    def mse(self, model, tau, device, dtype):
        x, st_x, mask_low = self.per_tau[tau]
        e = math.exp(tau)
        xt = torch.as_tensor(x, device=device, dtype=dtype)
        Rt = torch.as_tensor(ROT.R(tau), device=device, dtype=dtype)
        y = (e * (xt @ Rt)).requires_grad_(True)
        s_val = 0.5 * math.expm1(2 * tau)
        with torch.enable_grad():
            psi = model(y, torch.full((len(xt),), s_val, device=device, dtype=dtype))
            g_y = torch.autograd.grad(psi.sum(), y)[0]
        sp_y = g_y.detach().cpu().numpy()
        Rnp = ROT.R(tau)
        st_y = math.exp(-tau) * (st_x @ Rnp)
        sp_x = e * (sp_y @ Rnp.T)
        err_x = ((sp_x - st_x) ** 2).sum(1)
        err_y = ((sp_y - st_y) ** 2).sum(1)
        eps = 1e-8
        rel_x = err_x / (np.sum(st_x ** 2, axis=1) + eps)
        return dict(global_=float(err_x.mean()), high=float(err_x[~mask_low].mean()), low=float(err_x[mask_low].mean()),
                    global_y=float(err_y.mean()), high_y=float(err_y[~mask_low].mean()), low_y=float(err_y[mask_low].mean()),
                    global_rel=float(rel_x.mean()), high_rel=float(rel_x[~mask_low].mean()), low_rel=float(rel_x[mask_low].mean()))

def make_hb_probe(model_dummy, data_t, args, device, dtype, var_scalar, n_probe=256, seed=7, kap=None):
    if kap is None: kap = KappaSampler()
    rng = np.random.default_rng(seed); x0 = data_t[rng.integers(0, len(data_t), n_probe)]
    s0 = torch.exp(torch.empty(n_probe, device=device, dtype=dtype).uniform_(math.log(args.s_hb_min), math.log(S_MAX)))
    tau0 = tau_of_s(s0); eps = torch.tensor(rng.standard_normal((n_probe, 2)), device=device, dtype=dtype)
    y0 = x0 + (2 * s0).sqrt().unsqueeze(1) * eps
    r2 = args.r_frac * s0; np.random.seed(seed); z, sig = kap.sample(n_probe * args.n_kappa, device, dtype); np.random.seed(args.seed)
    z = z.view(n_probe, args.n_kappa, 2); sig = sig.view(n_probe, args.n_kappa)
    y_i = y0.unsqueeze(1) + r2.sqrt().view(-1, 1, 1) * z; s_i = s0.view(-1, 1) - r2.view(-1, 1) * sig
    return y0, s0, y_i, s_i

@torch.no_grad()
def hb_residual(model, probe):
    y0, s0, y_i, s_i = probe; B, N = y_i.shape[0], y_i.shape[1]
    psi_c = model(y0, s0); psi_i = model(y_i.reshape(-1, 2), s_i.reshape(-1)).view(B, N)
    return float((psi_c - logmeanexp(psi_i, dim=1)).abs().mean())

def curl_div_grid(model, ds, data_np, tau, device, dtype, var_scalar, n_grid=50, h=0.05):
    pad = 1.5; lo = data_np.min(0) - pad; hi = data_np.max(0) + pad
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], n_grid), np.linspace(lo[1], hi[1], n_grid))
    base = np.stack([gx.ravel(), gy.ravel()], 1); offs = [np.array([h,0]), np.array([-h,0]), np.array([0,h]), np.array([0,-h])]
    s4 = [model_score_x(model, base + o, tau, device, dtype, var_scalar) for o in offs]
    ds1dx = (s4[0][:,0] - s4[1][:,0]) / (2*h); ds2dy = (s4[2][:,1] - s4[3][:,1]) / (2*h)
    curl = (s4[0][:,1] - s4[1][:,1]) / (2*h) - (s4[2][:,0] - s4[3][:,0]) / (2*h)
    div = ds1dx + ds2dy; rot_ratio = float(np.mean(np.abs(curl)) / (np.mean(np.abs(curl)) + np.mean(np.abs(div)) + 1e-12))
    return gx, gy, np.abs(curl).reshape(gx.shape), rot_ratio

def make_paper_figs(ds, data_np, gen, model, ev, hist_eval, probe_hist, metrics_sm, args, device, dtype, var_scalar, outdir):
    tau_show = 0.3; pad = 2.0; lo = data_np.min(0) - pad; hi = data_np.max(0) + pad
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 200), np.linspace(lo[1], hi[1], 200))
    zz = ds.logpdf(np.stack([gx.ravel(), gy.ravel()], 1)).reshape(gx.shape)
    sx, sy = np.meshgrid(np.linspace(lo[0], hi[0], 24), np.linspace(lo[1], hi[1], 24)); pts = np.stack([sx.ravel(), sy.ravel()], 1)
    sc = model_score_x(model, pts, tau_show, device, dtype, var_scalar)
    fig, ax = plt.subplots(figsize=(6, 5.5)); ax.contour(gx, gy, zz, levels=10, cmap="viridis", linewidths=0.8)
    ax.streamplot(sx, sy, sc[:,0].reshape(sx.shape), sc[:,1].reshape(sx.shape), color="k", density=0.9, linewidth=0.6, arrowsize=0.8)
    ax.set_title(f"Learned score field @ tau={tau_show} (heatball)"); fig.tight_layout(); fig.savefig(f"{outdir}/scorefield.png", dpi=140); plt.close(fig)
    if ev.ok:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4)); ax = axes[0]; steps_ax = [h["step"] for h in hist_eval]
        for key, lab in [("mse_global","global"),("mse_high","high-dens"),("mse_low","low-dens")]: ax.plot(steps_ax, [h[key] for h in hist_eval], label=lab)
        ax.set_yscale("log"); ax.legend(); ax.set_title(f"Score MSE during training @ tau=0.3"); ax = axes[1]
        taus = sorted(metrics_sm.keys()); xs = np.arange(len(taus)); wd = 0.27
        for j, (key, lab) in enumerate([("global_","global"),("high","high-dens"),("low","low-dens")]): ax.bar(xs+(j-1)*wd, [metrics_sm[t][key] for t in taus], wd, label=lab)
        ax.set_xticks(xs); ax.set_xticklabels([f"{t}" for t in taus]); ax.set_yscale("log"); ax.legend(); ax.set_title("Score MSE vs tau")
        fig.tight_layout(); fig.savefig(f"{outdir}/score_mse_regions.png", dpi=140); plt.close(fig)
    if ds.modes is not None:
        K = len(ds.w); sig = np.sqrt(np.array([np.mean(np.diag(c)) for c in ds.cov]))
        def count(pts):
            d2 = ((pts[:, None, :] - ds.modes[None]) ** 2).sum(-1); a = d2.argmin(1)
            near = d2[np.arange(len(pts)), a] < (3 * sig[a]) ** 2; cnt = np.array([((a == k) & near).sum() for k in range(K)])
            return int(np.sum(cnt > 0.01 * len(pts)))
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        for ax, pts, name in [(axes[0], data_np[:2000], "real"), (axes[1], gen, "generated")]:
            ax.contour(gx, gy, zz, levels=10, cmap="viridis", linewidths=0.8); ax.scatter(pts[:,0], pts[:,1], s=4, c="crimson", alpha=0.4)
            ax.set_title(f"{name}: recovered {count(pts)}/{K} modes")
        fig.tight_layout(); fig.savefig(f"{outdir}/modecount.png", dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4)); ax.plot([h["step"] for h in probe_hist], [h["res"] for h in probe_hist]); ax.set_yscale("log")
    ax.set_title(r"Heat-ball residual $|R_r[\psi_\theta]|$"); fig.tight_layout(); fig.savefig(f"{outdir}/hb_residual.png", dpi=140); plt.close(fig)

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sparse_dense_gmm", choices=["sparse_dense_gmm", "asym_gmm", "ring"])
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_samples", type=int, default=2000)
    ap.add_argument("--sample_steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="outputs_heatball")
    ap.add_argument("--device", default="cpu", choices=["mps", "cuda", "cpu"],
                    help="cpu / mps / cuda; 注意: MPS 的 float32 数值与 CPU 不同, 本管线在 MPS 下训练不稳定 (已多次踩坑), 请用 cpu")
    ap.add_argument("--s_min", type=float, default=0.05)
    ap.add_argument("--n_anchor_layers", type=int, default=8)
    ap.add_argument("--s_hb_min", type=float, default=0.1)
    ap.add_argument("--r_frac", type=float, default=0.125)
    ap.add_argument("--n_kappa", type=int, default=64)
    ap.add_argument("--hb_measure", default="kappa", choices=["kappa", "uniform"])
    ap.add_argument("--snis_m", type=int, default=256)
    ap.add_argument("--w_anchor", type=float, default=1.0)
    ap.add_argument("--w_hb", type=float, default=2.0)
    ap.add_argument("--w_shb", type=float, default=0.3, help="score 热球权重 (稳定化: 原 1.0 在锚未收敛时易爆炸)")
    ap.add_argument("--w_mse", type=float, default=0.05)
    ap.add_argument("--w_dir", type=float, default=2.0)
    ap.add_argument("--dir_s_min", type=float, default=0.01)
    ap.add_argument("--dir_norm_threshold", type=float, default=0.5)
    ap.add_argument("--low_frac", type=float, default=0.0)
    ap.add_argument("--w_curv",     type=float, default=0.0)
    ap.add_argument("--score_cv", type=int, default=0)
    ap.add_argument("--score_est", default="moment", choices=["direct", "hb", "moment", "moment_richardson", "hybrid"])
    ap.add_argument("--n_kappa_eval", type=int, default=64)
    ap.add_argument("--psi_low_thresh", type=float, default=-8.0)
    ap.add_argument("--low_estimator", default="hb", choices=["hb", "richardson"])
    ap.add_argument("--n_ent_low", type=int, default=0.0)
    ap.add_argument("--ent_box_pad", type=float, default=2.0)
    ap.add_argument("--lam_ent", type=float, default=0.0)
    ap.add_argument("--ent_warmup", type=int, default=1000)
    ap.add_argument("--ent_ver", type=int, default=2, choices=[1, 2])
    ap.add_argument("--w_entf", type=float, default=1.0)
    ap.add_argument("--n_ent_pairs", type=int, default=16)
    ap.add_argument("--ent_psi_cut", type=float, default=10.0)
    ap.add_argument("--ent_cmax", type=float, default=0.5)
    ap.add_argument("--ent_q2", type=float, default=0.0)
    ap.add_argument("--w_entq2", type=float, default=1.0)
    ap.add_argument("--ent_q2_cut", type=float, default=30.0)
    ap.add_argument("--hb_cmax", type=float, default=0.0)
    ap.add_argument("--w_bb", type=float, default=0.0)
    ap.add_argument("--bb_ratio", type=float, default=0.25)
    ap.add_argument("--train_ess", type=int, default=0)
    ap.add_argument("--ess_thr", type=float, default=0.5)
    ap.add_argument("--anchor_cv", type=int, default=1)
    ap.add_argument("--anchor_warmup", type=int, default=1000, help="锚定预热步数: 须足够锚先拟合 (seed 0 下 500 即可, 取 1000 留裕量)")
    ap.add_argument("--gate", type=int, default=1)
    ap.add_argument("--r_jitter", type=float, default=2)
    ap.add_argument("--grad_clip", type=float, default=10.0, help="梯度裁剪范数: 勿调小! 1.0 会拖垮锚定期 (已踩坑)")
    ap.add_argument("--sample_init", default="data", choices=["data", "gaussian"])
    ap.add_argument("--w_score_scale", type=float, default=0.1,
                    help="外部缩放系数，用于降低 score 损失的主导地位")
    ap.add_argument("--ramp_steps", type=int, default=2000,
                    help="warmup 后损失权重线性增加到完整值的步数")
    ap.add_argument("--shb_delay", type=int, default=1000,
                    help="anchor_warmup 之后额外延迟 shb 开启的步数 (稳定化: shb 对未校准模型最敏感)")
    ap.add_argument("--mse_weight_cap", type=float, default=50.0,
                    help="mse 项 (2s) 放大因子的上限，防止大 s 样本贡献无界梯度")
    ap.add_argument("--omega", type=float, default=1.0, help="旋转强度 r(tau)=omega (tau-时间单位), 0=无旋转")
    ap.add_argument("--rot_profile", default="osc", choices=["const", "osc", "ramp"], help="旋转轮廓")
    ap.add_argument("--rot_k", type=float, default=3.0, help="osc/ramp 轮廓的频率参数 k")
    ap.add_argument("--direct_sim", type=int, default=0, help="R1对照: 1=用 x-系直接模拟替换闭式前向")
    ap.add_argument("--sim_n", type=int, default=4096, help="模拟轨迹数")
    ap.add_argument("--sim_k", type=int, default=80, help="tau 检查点数")
    ap.add_argument("--sim_dt", type=float, default=2e-3, help="模拟步长")
    args = ap.parse_args()
    # ============ 运行参数覆盖区 (每次运行只改这里, 会覆盖命令行与默认值) ============
    #args.seed = 0               # 固定健康 seed (原覆盖区漏了这行, PyCharm 的 --seed 曾泄漏进来)
    # args.omega = 0.0            # 旋转强度 rho = omega (gamma=1); rho 扫描改这里   [dsim版已注释: 命令行参数生效]
    # args.rot_profile = "osc"    # const / osc / ramp; 注意: rho 扫描请用 "const"   [dsim版已注释: 命令行参数生效]
    # args.rot_k = 3.0            # osc/ramp 轮廓的频率 k   [dsim版已注释: 命令行参数生效]
    # args.outdir = "out_base"    # 每次运行必须改成不同目录, 否则 metrics.json/model.pt 互相覆盖!   [dsim版已注释: 命令行参数生效]
    # ==============================================================================

    global ROT
    ROT = Rotation2D(args.rot_profile, args.omega, args.rot_k)
    device = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dtype = torch.float32
    ds = make_dataset(args.dataset); rng = np.random.default_rng(args.seed)
    data_np = ds.sample(10000, rng); data_t = torch.tensor(data_np, device=device, dtype=dtype)
    var_scalar = float(data_t.var(dim=0).mean().item())

    # 传入 var_scalar 到网络
    model = PsiNet(var_scalar=var_scalar).to(device=device, dtype=dtype)
    ev = ScoreEvalSet(ds, data_np)
    kap = KappaSampler()
    if args.direct_sim:
        args.sim_z, args.sim_s = build_sim_table(data_t, args, device, dtype)
        print(f"[dsim] 模拟表就绪: {args.sim_z.shape} (dt={args.sim_dt})", flush=True)
    probe = make_hb_probe(model, data_t, args, device, dtype, var_scalar, kap=kap)
    hist, hist_eval, probe_hist, t_train = train(model, data_t, args, device, dtype, ev, probe, kap=kap)
    gen, t_sample, score_diag = sample(model, args, device, dtype, var_scalar, data_t=data_t, kap=kap)

    eval_data = ds.sample(2000, rng)
    metrics = dict(
        method="heatball", dataset=args.dataset, steps=args.steps, seed=args.seed,
        hb_measure=args.hb_measure, score_est=args.score_est,
        score_cv=args.score_cv, w_shb=args.w_shb, w_mse=args.w_mse, w_dir=args.w_dir,
        dir_s_min=args.dir_s_min, dir_norm_threshold=args.dir_norm_threshold,
        low_frac=args.low_frac, w_curv=args.w_curv, sample_init=args.sample_init,
        runtime=dict(train=t_train, sample=t_sample, total=t_train + t_sample),
        mmd=mmd_rbf(gen, eval_data),
        sliced_w=sliced_wasserstein(gen, eval_data),
        anchor_nll_smin=anchor_nll(model, data_t, args, device, dtype, var_scalar),
    )
    if score_diag is not None: metrics["score_est_diag"] = score_diag
    cov, kl = mode_metrics(ds, gen)
    if cov is not None: metrics.update(mode_coverage=cov, mode_prop_kl=kl)
    metrics.update(lowdens_metrics(ds, eval_data, gen))
    metrics["rotation"] = dict(profile=args.rot_profile, omega=args.omega)
    if args.omega != 0.0 and ds.modes is not None:
        metrics["rotation_blind_bias"] = rotation_blind_bias(ds, args.omega)

    metrics_sm = {}
    if ev.ok:
        for tau in (0.1, 0.3, 0.5, 0.8):
            m = ev.mse(model, tau, device, dtype); metrics_sm[tau] = m
        metrics["score_mse"] = {
            str(t): dict(global_=metrics_sm[t]["global_"], highdens=metrics_sm[t]["high"], lowdens=metrics_sm[t]["low"],
                         global_y=metrics_sm[t]["global_y"], highdens_y=metrics_sm[t]["high_y"], lowdens_y=metrics_sm[t]["low_y"],
                         global_rel=metrics_sm[t]["global_rel"], highdens_rel=metrics_sm[t]["high_rel"], lowdens_rel=metrics_sm[t]["low_rel"])
            for t in metrics_sm}
    metrics["rot_mass_ratio"] = None
    if ds.modes is not None:
        sig = np.sqrt(np.array([np.mean(np.diag(c)) for c in ds.cov]))
        d2 = ((gen[:, None, :] - ds.modes[None]) ** 2).sum(-1); a = d2.argmin(1)
        near = d2[np.arange(len(gen)), a] < (3 * sig[a]) ** 2
        cnt = np.array([((a == k) & near).sum() for k in range(len(ds.w))])
        metrics["modes_recovered"] = f"{int(np.sum(cnt > 0.01 * len(gen)))}/{len(ds.w)}"

    make_plots(ds, eval_data, gen, hist, args.outdir, "heatball")
    make_paper_figs(ds, data_np, gen, model, ev, hist_eval, probe_hist, metrics_sm,
                    args, device, dtype, var_scalar, args.outdir)
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f: json.dump(metrics, f, indent=2, ensure_ascii=False)
    model_path = os.path.join(args.outdir, "model.pt"); torch.save(model.state_dict(), model_path)
    print(f"模型权重已保存: {model_path}")
    print("\n===== heatball 结果 =====")
    print(f"训练 {t_train:.1f}s | 采样 {t_sample:.1f}s | 总计 {t_train + t_sample:.1f}s")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"输出目录: {args.outdir}")

if __name__ == "__main__":
    main()