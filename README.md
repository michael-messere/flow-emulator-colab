# flow-emulator-colab

Subsample sweep for the fixed-z=2 normalizing flow `q(DM | theta)`, packaged to
run on a **Google Colab GPU** runtime as well as on Binder.

## Files
- `flow_emulator_LH_z2_subsample_sweep.py` — the sweep. Paths and device are now
  read from environment variables, so the same file runs unchanged on Binder and
  Colab.
- `run_sweep_colab.ipynb` — opens the script on a Colab GPU: checks the GPU,
  clones this repo, installs deps, sets env vars, runs the sweep, shows the plots.
- `requirements.txt` — extra deps (`sbi`, `h5py`); Colab already ships
  torch+CUDA, numpy, matplotlib.

## Environment variables the script honors
| var | meaning | default |
|-----|---------|---------|
| `STACK_DIR` | folder with the `LH_*.h5` stacks | Binder path |
| `BASE_DIR` | where `run_NN/` output folders go | Binder path |
| `DEVICE` | `cuda` or `cpu` | auto-detect |
| `NPROC` | workers for the parallel HDF5 load | 8 |
| `MAX_EPOCHS` | max training epochs (early-stops) | 500 |
| `RUN_NAME` | target a specific `run_NN` folder | auto |

## Run on Colab
1. Push this folder to GitHub.
2. Open `run_sweep_colab.ipynb` in Colab via
   `https://colab.research.google.com/github/<your-username>/flow-emulator-colab/blob/main/run_sweep_colab.ipynb`.
3. **Runtime → Change runtime type → GPU**.
4. Edit `REPO_URL` (cell 2) and `STACK_DIR` (cell 4), then Run all.

## Getting the data to Colab
The `LH_*.h5` stacks are large. Options:
- **Push them into this repo** (e.g. a `data/stacks_LH/` folder) from your Binder
  account — `git lfs` recommended if files are big. Then set
  `STACK_DIR=data/stacks_LH`. Remove `*.h5` from `.gitignore` first.
- **Google Drive**: upload the stacks, mount Drive in the notebook (cell after
  step 4), and point `STACK_DIR` at the Drive path. This avoids bloating the repo.

## Run on Binder (unchanged behavior)
The old defaults still apply, so on Binder just:
```
NPROC=8 python flow_emulator_LH_z2_subsample_sweep.py
```
