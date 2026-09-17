#!/usr/bin/env python3
# speciation_time.py — 可见性窗口 tau* 与 speciation time t_S 的对照
# 判据 (Lu & Tang; Biroli-Mézard): lambda_min( Sigma_sto(t) - e^{-At} Sigma_B e^{-A^T t} ) = 0
# 本文归一化下闭式:  Sigma_sto(t) = (1-e^{-2t}) I,  e^{-At} = e^{-t}R(rho t)
#   => t_S = 0.5*ln(1 + lambda_max(Sigma_B)),  与 rho 无关
# Sigma_B 取 "数据协方差的对称破缺部分": 全协方差的迹less部分 Sigma_0 - tr(Sigma_0)/d * I
#   (另报 between-mode 散度变体以供对照)
import argparse, math, json
import numpy as np

def t_s_closed(lam_max):
    return 0.5 * math.log1p(lam_max)

def speciation_report(name, Sigma, tau_star):
    d = Sigma.shape[0]
    trless = Sigma - np.trace(Sigma)/d * np.eye(d)
    lam_tl = float(np.linalg.eigvalsh(trless)[-1])
    tS = t_s_closed(max(lam_tl, 0.0))
    return dict(dataset=name, lam_max_traceless=lam_tl, t_S=tS,
                tau_star=tau_star, ratio=tau_star/tS if tS>0 else float('nan'))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="all2d", choices=["all2d", "mnist16"])
    args = ap.parse_args()
    rows = []
    if args.data == "all2d":
        import importlib.util
        spec = importlib.util.spec_from_file_location("h", "heatball_toy2d_rot.py")
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        taus = np.linspace(0.05, 2.5, 25)
        sets = [("aniso_gauss", m.GMM2D([1.0],[(0,0)],[np.diag([2.25,0.09])],"a")),
                ("asym_gmm", m.make_dataset("asym_gmm")),
                ("sparse_dense", m.make_dataset("sparse_dense_gmm")),
                ("scaled_gmm", m.GMM2D([0.7,0.2,0.1],[(0,0),(2.,1.5),(-2.,1.5)],[np.eye(2)*0.35**2]*3,"s"))]
        for name, ds in sets:
            rng = np.random.default_rng(1)
            X = ds.sample(200000, rng)
            Sigma = np.cov(X.T)
            m.ROT = m.Rotation2D("const", 2.0, 3.0)
            bb = m.rotation_blind_bias(ds, 2.0, taus=taus)
            rel = np.array([r["mse_blind"]/max(r.get("mse_mag", r.get("mse_mag_true", 1.0)), 1e-30)
                            for r in bb])
            ipk = int(np.argmax(rel))
            rows.append(speciation_report(name, Sigma, float(taus[ipk])))
    else:
        import torch, torch.nn.functional as F
        import torchvision
        ds = torchvision.datasets.MNIST('./data', train=True, download=True)
        x = torch.from_numpy(ds.data.numpy()[:,None]).float()/255.
        x = F.interpolate(x, size=16, mode='area').view(len(x), -1).numpy()[:20000]
        std = np.maximum(x.std(0), 0.05); X = (x - x.mean(0))/std
        y = np.array(ds.targets.numpy()[:20000])
        # between-class 散度 (10 类) + 全协方差迹less, 两种 Sigma_B 都报
        Sigma = np.cov(X.T)
        means = np.stack([X[y==k].mean(0) for k in range(10)])
        w = np.array([(y==k).mean() for k in range(10)])
        mu = X.mean(0)
        B = sum(wk*np.outer(mk-mu, mk-mu) for wk, mk in zip(w, means))
        d = X.shape[1]
        trless = Sigma - np.trace(Sigma)/d*np.eye(d)
        tau_star_rho2 = 0.458   # 已实测 (H2)
        rows.append(speciation_report("mnist16 (between-class)", B, tau_star_rho2))
        rows.append(speciation_report("mnist16 (traceless)", trless, tau_star_rho2))
    for r in rows:
        print(f"{r['dataset']:28s} lambda_max={r['lam_max_traceless']:8.3f}  "
              f"t_S={r['t_S']:.3f}  tau*(rho=2)={r['tau_star']:.3f}  ratio={r['ratio']:.3f}")
    json.dump(rows, open(f"speciation_{args.data}.json", "w"), indent=1)
