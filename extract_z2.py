#!/usr/bin/env python3
"""Extract the z=2 slice from each LH_*.h5 stack into a compact, gzip-compressed
copy under data/stacks_LH/, small enough to commit to GitHub.

Run this in Binder, where the full stacks live:

    SRC=/home/jovyan/home/frb-camels/25/stacking/stacks_LH \
    DST=data/stacks_LH \
    python extract_z2.py

The output files keep the exact structure the sweep loader expects
(`z`, `dm[iz]`, and the theta/sim attrs), so
flow_emulator_LH_z2_subsample_sweep.py runs unchanged with
STACK_DIR=data/stacks_LH.
"""
import os
import glob
import numpy as np
import h5py

SRC = os.environ.get('SRC', '/home/jovyan/home/frb-camels/25/stacking/stacks_LH')
DST = os.environ.get('DST', 'data/stacks_LH')
Z_TARGET = 2.0

os.makedirs(DST, exist_ok=True)
files = sorted(glob.glob(os.path.join(SRC, 'LH_*.h5')))
print(f'{len(files)} source files in {SRC}')

src_bytes = out_bytes = 0
for fp in files:
    with h5py.File(fp, 'r') as hf:
        z = hf['z'][:]
        iz = int(np.argmin(np.abs(z - Z_TARGET)))
        dm = np.asarray(hf['dm'][iz])
        attrs = {k: hf.attrs[k] for k in hf.attrs}
    out = os.path.join(DST, os.path.basename(fp))
    with h5py.File(out, 'w') as hf:
        # store the single chosen redshift so loader's argmin -> iz=0
        hf.create_dataset('z', data=np.array([z[iz]], dtype=np.float32))
        hf.create_dataset('dm', data=dm[None].astype(np.float32),
                          compression='gzip', compression_opts=4)
        for k, v in attrs.items():
            hf.attrs[k] = v
    src_bytes += os.path.getsize(fp)
    out_bytes += os.path.getsize(out)

print(f'wrote {len(files)} files to {DST}')
print(f'source total : {src_bytes/1e6:.1f} MB')
print(f'output total : {out_bytes/1e6:.1f} MB')
