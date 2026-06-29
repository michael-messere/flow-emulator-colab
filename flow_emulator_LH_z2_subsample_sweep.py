#!/usr/bin/env python3
"""
Subsample sweep for the FIXED-z=2 flow q(DM | theta) -- NOT an Optuna search.

Fixed architecture (batch=8192, hidden=16, transforms=4, num_bins=5); the only
thing varied is SUBSAMPLE = 1000, 2000, ..., 10000 (pixels per sim).
Each subsample value trains its own flow and gets its own folder with the model,
an info file, and a 10x10 held-out panel (full 250^2 / subsample / flow fit).
A summary plot of val-loss & held-out-LL vs subsample is written at the top.

Auto-increments into optuna_run/run_NN (so it becomes run_02 after run_01).

Usage:  NPROC=8 python flow_emulator_LH_z2_subsample_sweep.py
"""

import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import re
import time
import csv
import numpy as np
import h5py
from multiprocessing import Pool

# ───────────────────────────── configuration ───────────────────────────────
# Paths are env-overridable so the same file runs on Binder (the /home/jovyan
# defaults below) and on Colab (set STACK_DIR / BASE_DIR in the notebook).
NPROC      = int(os.environ.get('NPROC', '8'))
STACK_DIR  = os.environ.get(
    'STACK_DIR', '/home/jovyan/home/frb-camels/25/stacking/stacks_LH')
BASE_DIR   = os.environ.get(
    'BASE_DIR',
    '/home/jovyan/home/frb-camels/25/LH_emulator/normalizing_flow/optuna_run')

INPUTS     = ['Omega_m', 'sigma_8', 'A_SN1', 'A_AGN1', 'A_SN2', 'A_AGN2']
Z_TARGET   = 2.0
CLIP_PCT   = float(os.environ.get('CLIP', '0.0'))
TEST_FRAC  = 0.1
SEED       = 0
FLOW       = 'nsf'
CACHE_MAX  = 100000                            # cache full 250^2 positive pixels/sim

# fixed flow hyperparameters for this sweep
BATCH      = 8192
HIDDEN     = 16
TRANSFORMS = 4
NUM_BINS   = 5
MAX_EPOCHS = int(os.environ.get('MAX_EPOCHS', '500'))   # full convergence (early-stops)

# the swept quantity
SUBSAMPLES = list(range(1000, 10001, 1000))    # 1000, 2000, ..., 10000 (900x10000=9M < 12M)


def load_one(fp):
    try:
        with h5py.File(fp, 'r') as hf:
            zgrid = hf['z'][:]
            iz = int(np.argmin(np.abs(zgrid - Z_TARGET)))
            dm = hf['dm'][iz].astype(np.float64)
            theta = np.array([hf.attrs[n] for n in INPUTS], np.float64)
            sim = str(hf.attrs.get('sim', os.path.basename(fp)[:-3]))
    except (OSError, KeyError):
        return None
    lg = np.log10(dm[np.isfinite(dm) & (dm > 0)].ravel())
    if lg.size == 0:
        return None
    if CLIP_PCT > 0.0:
        lo, hi = np.percentile(lg, [CLIP_PCT, 100 - CLIP_PCT])
        lg = lg[(lg >= lo) & (lg <= hi)]
    sid = int(re.sub(r'\D', '', sim) or 0)
    if lg.size > CACHE_MAX:
        lg = np.random.default_rng(sid).choice(lg, CACHE_MAX, replace=False)
    return theta.astype(np.float32), lg.astype(np.float32), sid


def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    RUN_NAME = os.environ.get('RUN_NAME')          # e.g. RUN_NAME=run_02 to target one
    if RUN_NAME:
        RUN_DIR = os.path.join(BASE_DIR, RUN_NAME)
    else:
        # resume the latest existing SWEEP folder (one that has sub_* subdirs);
        # otherwise start the next run_NN
        sweeps = []
        for d in os.listdir(BASE_DIR):
            p = os.path.join(BASE_DIR, d)
            if re.match(r'run_\d+$', d) and os.path.isdir(p) and \
               any(s.startswith('sub_') for s in os.listdir(p)):
                sweeps.append((int(re.search(r'run_(\d+)', d).group(1)), p))
        if sweeps:
            RUN_DIR = max(sweeps)[1]
        else:
            _ex = [int(m.group(1)) for d in os.listdir(BASE_DIR)
                   if (m := re.match(r'run_(\d+)$', d))]
            RUN_DIR = os.path.join(BASE_DIR, f'run_{(max(_ex) + 1) if _ex else 1:02d}')
    os.makedirs(RUN_DIR, exist_ok=True)
    INFO = os.path.join(RUN_DIR, 'run_info.txt')

    def log(msg):
        print(msg)
        with open(INFO, 'a') as fh:
            fh.write(str(msg) + '\n')

    log(f'# subsample sweep started {time.strftime("%Y-%m-%d %H:%M:%S")}')
    log(f'run_dir   : {RUN_DIR}')
    log(f'fixed     : batch={BATCH} hidden={HIDDEN} transforms={TRANSFORMS} '
        f'num_bins={NUM_BINS} clip={CLIP_PCT} z={Z_TARGET}')
    log(f'subsamples: {SUBSAMPLES}')

    files = sorted((os.path.join(STACK_DIR, f) for f in os.listdir(STACK_DIR)
                    if re.match(r'^LH_\d+\.h5$', f)),
                   key=lambda p: int(re.search(r'LH_(\d+)', p).group(1)))
    t0 = time.time()
    with Pool(NPROC) as pool:
        results = [r for r in pool.map(load_one, files) if r is not None]
    log(f'loaded {len(results)} sims (z=2) in {time.time()-t0:.1f}s')

    THETA = np.stack([r[0] for r in results])
    POOL  = [r[1] for r in results]
    SID   = np.array([r[2] for r in results])
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(results))
    n_test = max(1, int(TEST_FRAC * len(results)))
    test_set = set(order[:n_test].tolist())
    train_ix = np.array([i for i in range(len(results)) if i not in test_set])
    test_ix = np.array(sorted(test_set))
    log(f'train: {train_ix.size}   test: {test_ix.size}')

    # heavy imports AFTER the multiprocessing load: torch/tensorflow are not
    # fork-safe, so importing them before Pool() can segfault on fork.
    import torch
    torch.set_num_threads(NPROC)
    from sbi.inference import NLE
    from sbi.utils import BoxUniform
    from sbi.neural_nets import likelihood_nn
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

    # DEVICE=cuda on a Colab GPU runtime; falls back to CPU on Binder.
    DEVICE = os.environ.get('DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'device    : {DEVICE}')

    lo = torch.tensor(THETA.min(0) - 1e-3 * np.abs(THETA.min(0)),
                      dtype=torch.float32, device=DEVICE)
    hi = torch.tensor(THETA.max(0) + 1e-3 * np.abs(THETA.max(0)),
                      dtype=torch.float32, device=DEVICE)
    prior = BoxUniform(low=lo, high=hi, device=DEVICE)

    def build(ss, seed):
        r = np.random.default_rng(seed)
        conds, xs = [], []
        for i in train_ix:
            p = POOL[i]
            s = r.choice(p, ss, replace=False) if p.size > ss else p
            conds.append(np.repeat(THETA[i][None, :], s.size, axis=0))
            xs.append(s)
        return (np.concatenate(conds).astype(np.float32),
                np.concatenate(xs)[:, None].astype(np.float32))

    def flow_ll(est, xv, condv):
        with torch.no_grad():
            xt = torch.as_tensor(xv).to(DEVICE)
            ct = torch.as_tensor(condv).to(DEVICE)
            try:
                out = est.log_prob(xt, condition=ct)
            except TypeError:
                out = est.log_prob(xt, context=ct)
        return np.asarray(out.detach().cpu()).ravel()

    def heldout_ll(est):
        xs = np.concatenate([POOL[i][:, None] for i in test_ix]).astype(np.float32)
        cs = np.concatenate([np.repeat(THETA[i][None, :], POOL[i].size, 0)
                             for i in test_ix]).astype(np.float32)
        if xs.shape[0] > 200_000:
            sel = np.random.default_rng(0).choice(xs.shape[0], 200_000, replace=False)
            xs, cs = xs[sel], cs[sel]
        return float(flow_ll(est, xs, cs).mean())

    def make_panel(est, ss, out_png, title):
        ncols = 10; nrows = int(np.ceil(test_ix.size / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 2.0 * nrows))
        for ax, i in zip(np.atleast_1d(axes).ravel(), test_ix):
            lg = POOL[i]
            blo, bhi = np.percentile(lg, [0.5, 99.5])
            bins = np.linspace(blo, bhi, 50); grid = np.linspace(blo, bhi, 150)
            cond = np.repeat(THETA[i][None, :], grid.size, axis=0).astype(np.float32)
            pdf = np.exp(flow_ll(est, grid[:, None].astype(np.float32), cond))
            sub = np.random.default_rng(int(SID[i])).choice(lg, min(ss, lg.size), replace=False)
            ax.hist(lg, bins=bins, density=True, color='0.7', label='true (250$^2$)')
            ax.hist(sub, bins=bins, density=True, histtype='step', color='C0',
                    lw=1.0, label=f'subsample ({ss})')
            ax.plot(grid, pdf, 'C1', lw=1.2, label='flow fit')
            ax.set_title(f'LH_{SID[i]}', fontsize=6); ax.set_xticks([]); ax.set_yticks([])
        np.atleast_1d(axes).ravel()[0].legend(fontsize=5, loc='upper right')
        for ax in np.atleast_1d(axes).ravel()[test_ix.size:]:
            ax.axis('off')
        fig.suptitle(title, y=1.0); plt.tight_layout()
        plt.savefig(out_png, dpi=130, bbox_inches='tight'); plt.close(fig)

    def train_flow(ss):
        torch.manual_seed(SEED)
        cond, x = build(ss, SEED)
        de = likelihood_nn(model=FLOW, hidden_features=HIDDEN,
                           num_transforms=TRANSFORMS, num_bins=NUM_BINS)
        inf = NLE(prior=prior, density_estimator=de, device=DEVICE)
        inf.append_simulations(torch.tensor(cond), torch.tensor(x))
        est = inf.train(training_batch_size=BATCH, max_num_epochs=MAX_EPOCHS,
                        show_train_summary=False)
        summ = dict(getattr(inf, 'summary', None) or getattr(inf, '_summary', {}))
        val = np.asarray(summ.get('validation_loss', [np.nan]), float)
        return est, float(np.nanmin(val)) if val.size else np.inf

    # ── sweep ─────────────────────────────────────────────────────────────────
    def read_done(tdir):
        txt = open(os.path.join(tdir, 'info.txt')).read()
        v = float(re.search(r'val_loss=([-\d.eE]+)', txt).group(1))
        l = float(re.search(r'heldout_ll_per_pixel=([-\d.eE]+)', txt).group(1))
        return v, l

    rows = []
    for ss in SUBSAMPLES:
        tdir = os.path.join(RUN_DIR, f'sub_{ss:05d}')
        if os.path.exists(os.path.join(tdir, 'model.pt')):     # already done -> resume
            try:
                v, l = read_done(tdir); rows.append((ss, v, l))
                log(f'sub={ss:5d} already done (val={v:.4f}), skip'); continue
            except Exception:
                log(f'sub={ss:5d} present but unreadable -> redo')
        os.makedirs(tdir, exist_ok=True)
        t1 = time.time()
        est, vbest = train_flow(ss)
        te_ll = heldout_ll(est)
        torch.save({'estimator': est, 'inputs': INPUTS, 'x_transform': 'log10(DM)',
                    'z_target': Z_TARGET, 'clip_pct': CLIP_PCT,
                    'params': dict(subsample=ss, batch=BATCH, hidden=HIDDEN,
                                   transforms=TRANSFORMS, num_bins=NUM_BINS),
                    'val_loss': vbest, 'heldout_ll': te_ll,
                    'test_sids': SID[test_ix].tolist()}, os.path.join(tdir, 'model.pt'))
        with open(os.path.join(tdir, 'info.txt'), 'w') as fh:
            fh.write(f'subsample={ss} batch={BATCH} hidden={HIDDEN} '
                     f'transforms={TRANSFORMS} num_bins={NUM_BINS}\n'
                     f'val_loss={vbest:.4f}\nheldout_ll_per_pixel={te_ll:.4f}\n')
        make_panel(est, ss, os.path.join(tdir, 'heldout_panel.png'),
                   f'subsample={ss} | val={vbest:.3f} LL={te_ll:.3f}')
        rows.append((ss, vbest, te_ll))
        log(f'sub={ss:5d} -> val={vbest:.4f}  heldout_LL={te_ll:.3f}  '
            f'({(time.time()-t1)/60:.1f} min)')

    # ── summary table + plot ────────────────────────────────────────────────
    with open(os.path.join(RUN_DIR, 'sweep.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['subsample', 'val_loss', 'heldout_ll'])
        w.writerows(rows)
    ssv = np.array([r[0] for r in rows]); vlv = np.array([r[1] for r in rows])
    llv = np.array([r[2] for r in rows])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.plot(ssv, vlv, 'o-'); a1.set_xlabel('subsample'); a1.set_ylabel('val loss')
    a1.grid(alpha=0.2, ls='-.')
    a2.plot(ssv, llv, 's-', color='C1'); a2.set_xlabel('subsample')
    a2.set_ylabel('held-out LL / pixel'); a2.grid(alpha=0.2, ls='-.')
    fig.suptitle(f'z={Z_TARGET} subsample sweep (batch={BATCH}, h={HIDDEN}, '
                 f't={TRANSFORMS}, nb={NUM_BINS})')
    plt.tight_layout(); plt.savefig(os.path.join(RUN_DIR, 'sweep_vs_subsample.png'), dpi=130)
    log(f'best subsample (val): {ssv[int(np.argmin(vlv))]}')
    log(f'# sweep finished {time.strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == '__main__':
    main()
