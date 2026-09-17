#!/usr/bin/env python3
# empirical_blind_bias.py — 真实数据(经验测度)上的精确盲偏差扫描, 任意维度, 无需训练
# 原理: 前向边缘对经验测度 p0=(1/M)Σδ_{x_m} 是高斯混合:
#   u_tau    = (1/M)Σ_m N(e^{-tau}R(tau)x_m, (1-e^{-2tau})I)   <- 含旋转
#   p^bl_tau = (1/M)Σ_m N(e^{-tau}x_m,        (1-e^{-2tau})I)   <- 旋转盲
# 两者的 ∇log 都是 softmax 加权的线性场, 给定 y 解析可算;
# 用 (m, eps) 精确采样 y ~ u_tau, 得无偏估计 (score 精确, 唯一误差 = 有限 M 与 MC)。
# v2 修复: 距离用 ||y||^2+||c||^2-2yc^T 矩阵乘计算 (不再三维广播, 峰值内存 ~160MB);
#          加进度打印; MNIST 用 F.interpolate 批量缩放 (秒级, 不再逐张 transform)。
import argparse, math, time
import numpy as np

def gmm_score(y, centers, var):
    """等权高斯混合(公共协方差 var*I)的 score. y:(n,d), centers:(M,d).
    s(y) = (-y + Σ_m w_m mu_m)/var,  w = softmax(-||y-mu||^2/2var). 分块防内存."""
    n, d = y.shape; M = centers.shape[0]
    out = np.empty_like(y)
    c2 = (centers * centers).sum(1)                    # (M,)
    chunk = max(1, int(2e7 / M))                       # 峰值内存 chunk*M*8B <= 160MB
    for i in range(0, n, chunk):
        yc = y[i:i+chunk]
        dist2 = (yc * yc).sum(1)[:, None] + c2[None, :] - 2.0 * (yc @ centers.T)
        w = np.exp(-dist2 / (2 * var) - (-dist2 / (2 * var)).max(1, keepdims=True))
        w /= w.sum(1, keepdims=True)
        out[i:i+chunk] = (-yc + w @ centers) / var
    return out

def blind_bias_scan(X, rho, profile="const", k=3.0, taus=None, n_mc=100000, seed=0,
                    m_max=8192, verbose=True):
    if taus is None: taus = np.linspace(0.05, 2.5, 25)
    rng = np.random.default_rng(seed)
    M0, d = X.shape
    if M0 > m_max:
        X = X[rng.choice(M0, m_max, replace=False)]
        M0 = m_max
    Xc = X - X.mean(0)
    if verbose:
        print(f"[load] d={d} 成分数 M={M0}  n_mc={n_mc}  rho={rho} ({profile})", flush=True)
    res = []
    t0 = time.perf_counter()
    for i_t, tau in enumerate(taus):
        e = math.exp(-tau); e2 = e * e; var = 1 - e2
        if profile == "const": th = rho * tau
        elif profile == "osc": th = rho * (1 - math.cos(k * tau)) / k
        else: th = rho * math.log(math.cosh(k * tau)) / k
        c, s = math.cos(th), math.sin(th)
        R = np.eye(d)
        if d == 2:
            R = np.array([[c, -s], [s, c]])
        else:
            for p in range(0, d - 1, 2):
                R[p:p+2, p:p+2] = [[c, -s], [s, c]]
        idx = rng.integers(0, M0, n_mc)
        eps = rng.standard_normal((n_mc, d))
        mu_t = (Xc[idx] @ R.T) * e
        mu_b = Xc[idx] * e
        y = mu_t + math.sqrt(var) * eps
        st = gmm_score(y, mu_t, var)
        sb = gmm_score(y, mu_b, var)
        mb = float(((st - sb) ** 2).sum(1).mean())
        mt = float((st ** 2).sum(1).mean())
        res.append(dict(tau=float(tau), theta=th, mse_blind=mb,
                        mse_mag=mt, rel_bias=mb / max(mt, 1e-30)))
        if verbose and ((i_t + 1) % 5 == 0 or i_t == 0):
            print(f"[scan] {i_t+1}/{len(taus)} tau={tau:.2f} rel={res[-1]['rel_bias']:.4f} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="mnist16", choices=["mnist16", "synthetic"])
    ap.add_argument("--rho", type=float, default=2.0)
    ap.add_argument("--profile", default="const")
    ap.add_argument("--rot_k", type=float, default=3.0)
    ap.add_argument("--n_mc", type=int, default=100000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    t0 = time.perf_counter()
    if args.data == "mnist16":
        import torch, torch.nn.functional as F
        import torchvision
        ds = torchvision.datasets.MNIST('./data', train=True, download=True)
        x = torch.from_numpy(ds.data.numpy()[:, None]).float() / 255.0   # (60000,1,28,28)
        x = F.interpolate(x, size=16, mode='area').view(len(x), -1)      # 批量缩放, 秒级
        X = x.numpy()[:20000]
        # 与 heatball 训练管线一致的归一化: 去均值 + 除每像素标准差(下限 0.05)
        # (不归一化时像素方差仅 ~0.06, 非径向结构相对 N(0,I) 稳态过小, 偏差被低估两个量级)
        _std = np.maximum(X.std(0), 0.05)
        X = (X - X.mean(0)) / _std
        print(f"[load] MNIST16 下载+加载+归一化 {time.perf_counter()-t0:.0f}s "
              f"(每像素std 均值 {float(_std.mean()):.3f})", flush=True)
    else:
        rng = np.random.default_rng(0)
        d = 256
        X = rng.standard_normal((4000, d))
        X[:, 0] *= 1.5; X[:, 1] *= 0.2
        X[:, 2] *= 2.0; X[:, 3] *= 0.1
    rows = blind_bias_scan(X, args.rho, args.profile, args.rot_k, n_mc=args.n_mc)
    import json
    out = args.out or f"bb_{args.data}_rho{args.rho}_{args.profile}.json"
    json.dump(rows, open(out, "w"), indent=1)
    rel = [r["rel_bias"] for r in rows]; ipk = int(np.argmax(rel))
    print(f"RESULT data={args.data} rho={args.rho} {args.profile}: "
          f"峰 tau*={rows[ipk]['tau']:.2f} (theta*={rows[ipk]['theta']:.2f}) "
          f"rel峰={rel[ipk]:.3f}  总耗时={time.perf_counter()-t0:.0f}s -> {out}", flush=True)