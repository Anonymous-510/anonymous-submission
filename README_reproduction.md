# Reproduction Guide — Rotational Drift in Diffusion Models as a Gauge Freedom

This archive contains the code for all experiments in the paper.
Six scripts, two pipelines (2D toy / 256D MNIST), all results reproducible
from raw commands below. Tested on macOS (Apple Silicon) and Linux,
python 3.9+, CPU; the MNIST pipeline also runs on MPS.

## 0. Environment

    pip install numpy scipy torch torchvision matplotlib

No GPU is required. For the 2D pipeline use **CPU** (`--device cpu`, the
default after v2): MPS was found to be non-deterministic at the level that
triggers a pre-existing training instability of the constraint-loss balance
(Section 5 of the paper; all 2D numbers in the paper are CPU numbers).
The MNIST pipeline is MPS-safe (`--device mps` default).

## 1. File map

| file | pipeline | purpose | key outputs |
|---|---|---|---|
| `heatball_toy2d_rot.py`   | 2D | training + sampling, rotation-aware (closed form) | `out_*/metrics.json`, figures |
| `heatball_toy2d_dsim.py`  | 2D | direct-simulation control (`--direct_sim 1`) | same |
| `heatball_mnist16_rot.py` | 256D | MNIST16 training + sampling, rotation-aware | same |
| `heatball_mnist16_dsim.py`| 256D | direct-simulation control | same |
| `empirical_blind_bias.py` | both | exact blind-bias scan (no training), Eq. empirical-measure GMM | `bb_*.json` |
| `speciation_time.py`      | both | speciation time t_S (closed form) | console + `speciation_*.json` |
| `make_samples_fig.py`     | 256D | stitches the 3-panel MNIST sample figure + diff | `fig_samples_mnist16.png` |

MNIST is downloaded automatically by torchvision on first use (`./data`).

## 2. Quick sanity check (10 seconds, no training)

    python3 empirical_blind_bias.py --data synthetic --rho 2.0 --n_mc 5000
    # expect: peak tau* = 0.56, rel = 0.59, ~10 s

## 3. 2D experiments (sparse_dense_gmm; ~20 min per run, CPU)

### 3.1 Baseline and rho scan (Table 2, Fig. 1)

    python3 -u heatball_toy2d_rot.py --omega 0.0 --seed 0 --steps 6000 --outdir out_base
    for rho in 0.3 1.0 2.0; do
      python3 -u heatball_toy2d_rot.py --omega $rho --rot_profile const --seed 0 \
          --steps 6000 --outdir out_rho$rho
    done
    python3 -u heatball_toy2d_rot.py --omega 2.0 --rot_profile osc --rot_k 3.0 \
        --seed 0 --steps 6000 --outdir out_osc2

Training logs are bit-identical across rho (gauge invariance, R1).
Expect per run: `score_mse["0.3"]["global_"]` in 0.099-0.105 (seed 0),
MMD 0.00254, 3/3 modes; `rotation_blind_bias` peaks at tau* = 0.66/0.56/0.46.

### 3.2 Mechanism datasets (Fig. 2, Table 1)

Add to `make_dataset` (already present in the uploaded file):

    iso_gauss, aniso_gauss, ring          # plus asym_gmm, sparse_dense_gmm

    python3 empirical_blind_bias.py --data mnist16 2>&1 | head -2   # not needed here
    # closed-form scans used for the paper tables are produced by the same
    # rotation_blind_bias() inside any 2D training run (see metrics.json),
    # or standalone:
    python3 -c "
    import importlib.util, numpy as np
    spec = importlib.util.spec_from_file_location('m','heatball_toy2d_rot.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    for rho in [2.0]:
        m.ROT = m.Rotation2D('const', rho, 3.0)
        for ds in ['iso_gauss','aniso_gauss','asym_gmm','sparse_dense_gmm']:
            rows = m.rotation_blind_bias(m.make_dataset(ds), rho)
            rel=[r['rel_bias'] for r in rows]; i=int(np.argmax(rel))
            print(ds, 'tau*=%.2f peak=%.3f'%(rows[i]['tau'], rel[i]))
    "

### 3.3 Direct-simulation control and seed bands (R1, Table 2 caption)

    python3 -u heatball_toy2d_dsim.py --omega 2.0 --rot_profile const --seed 0 \
        --direct_sim 1 --steps 6000 --outdir out_2d_dsim
    for s in 1 2 3; do
      python3 -u heatball_toy2d_rot.py --omega 2.0 --rot_profile const --seed $s \
          --steps 6000 --outdir out_cf_s$s
      python3 -u heatball_toy2d_dsim.py --omega 2.0 --rot_profile const --seed $s \
          --direct_sim 1 --steps 6000 --outdir out_ds_s$s
    done
    # memorization-artifact check (table-enlargement, R1):
    python3 -u heatball_toy2d_dsim.py --omega 2.0 --rot_profile const --seed 2 \
        --direct_sim 1 --sim_n 16384 --steps 6000 --outdir out_ds_s2_big

Expect: closed-form band [0.105, 0.145]; direct-sim band [0.124, 0.306];
seed-2 with the enlarged table: 0.306 -> 0.136 (inside the closed-form band).

## 4. 256D experiments (MNIST16; ~15 min per run, MPS)

    python3 -u heatball_mnist16_rot.py --omega 0.0 --outdir out_mnist_base
    python3 -u heatball_mnist16_rot.py --omega 2.0 --rot_profile const --outdir out_mnist_rho2
    python3 -u heatball_mnist16_dsim.py --omega 2.0 --rot_profile const \
        --direct_sim 1 --outdir out_mnist_dsim

Expect: training curves identical to printed precision; best sliced-W
0.1887 / 0.1897; nn-check ratio ~1.06; gen std ~0.63 (the gen-std gap is a
pre-existing property of the backbone, unchanged by rotation).

### 4.1 Exact blind-bias scan at d=256 (H2, Fig. 4; ~6 min each)

    python3 empirical_blind_bias.py --data mnist16 --rho 2.0 --n_mc 30000
    python3 empirical_blind_bias.py --data mnist16 --rho 1.0 --n_mc 30000
    python3 empirical_blind_bias.py --data mnist16 --rho 0.3 --n_mc 30000
    # expect peaks: tau* = 0.46 / 0.66 / 0.87, depth 32.9% / 10.7% / 1.1%

### 4.2 Speciation times (Table 1 column t_S, Table 3; ~1 min)

    python3 speciation_time.py --data all2d
    python3 speciation_time.py --data mnist16
    # expect t_S: aniso 0.366, asym 0.613, sparse 0.480, scaled 0.169;
    # MNIST between-class 1.2298 (traceless variant 1.46)

## 5. Small verification scripts (seconds; numbers quoted in the paper)

Entropy production (H4, sigma = d rho^2): simulate (Heun, dt=2e-3, M=5000)
the stationary OU with J and evaluate the Stratonovich integral
(1/T) int rho J0 x_t \circ dx_t; expect 64.20/256.39/1025.6 at d=256 for
rho = 0.5/1/2.

Sampler variance (H3): the exact Prop. 14 identity E_kappa[(w-1)theta^2]
with theta = |z|^2/2 sigma and w = |S| kappa over the bounding cylinder;
expect the gap to grow as d^1.78 over d in {8,...,256}.

First-moment constant check (Appendix G): d=6, 4e6 kappa-directions,
r in {0.2, 0.1, 0.05} against the exact caloric gradient; relative error
~1.7e-3 (Monte-Carlo floor), confirming the constant 1/(m^2 r) with
m^2 = E_kappa|z|^2 / d (per-dimension second moment).

## 6. Known pitfalls (all hit during development; do not re-discover)

1. `--device mps` on the 2D pipeline: non-deterministic numerics trigger
   a seed-dependent instability (shb explosion around step 1000). Use CPU.
2. `heatball_toy2d_rot.py` contains an override block near the top of
   `main()`; only the `args.seed = 0` line is active by default. Comment it
   out when running seed sweeps, or `--seed` is silently ignored.
3. `empirical_blind_bias.py` needs the matmul-form distance (v2); the
   broadcast form allocates ~41 GB and appears to hang.
4. Direct-simulation runs pair each batch point with the trajectory of the
   same data index (`idx0` is drawn mod sim-table size); do not decouple them.
5. MNIST must be standardized per-pixel (mean removal, std division floored
   at 0.05) before the blind-bias scan, matching the training pipeline;
   raw 0-1 pixels underestimate the bias by two orders of magnitude.
6. The Kaiming note: `nn.Linear` defaults to `kaiming_uniform_(a=sqrt(5))`
   (effective gain 0.577); the networks here are re-initialized with
   `kaiming_normal_` (gain sqrt(2)), without which input gradients vanish
   for depth-4 MLPs (~200x attenuation).

## 7. Paper-number map (metrics.json fields)

- Table 2 (invariance): `score_mse` / `mmd` / `sliced_w` / `mode_coverage`
  / `lowdens_frac_gen` from out_base, out_rho*, out_2d_dsim, out_mnist_*.
- Fig. 1: `rotation_blind_bias` from out_rho0.3/1.0/2.0 and out_osc2.
- Fig. 2: closed-form scans of Sec. 3.2 at rho=2.
- Fig. 3 / Table 1: Sec. 3.2 + speciation_time.py.
- Fig. 4 / Table 3: `bb_mnist16_*.json` + speciation mnist16.
- R1 bands: out_cf_s1-3, out_ds_s1-3, out_ds_s2_big.
- H4 / H3 / Appendix G: Sec. 5 scripts.
