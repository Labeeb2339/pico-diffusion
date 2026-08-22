# Evaluation reproducibility

## What the published numbers mean

The two pixel-space image-quality numbers currently quoted in `README.md` are
**current-harness evaluation evidence for named legacy checkpoint bytes**. The
sequential `--no-cache` run is published under
[`evaluation_runs/20260822-162514Z`](evaluation_runs/20260822-162514Z). Sampler
and latent-model numbers remain historical pre-hardening development evidence.
None of these numbers is canonical 50,000-sample CIFAR-10 FID.

The historical stdout files record that the runs reported completion and include
their scores. They do **not** bind every run to a Git commit, full command, seed,
checkpoint digest, dependency lock, or generated-array digest. The local
checkpoint and log hashes recovered on 2026-08-22 are recorded in
`receipts/historical-fid-pre-hardening.json`; they cannot retroactively establish
facts that were not captured when the runs happened.

The surviving checkpoints have legacy training-loop provenance. Their original
training loop applied EMA weights directly to the live model at each sampling
interval and then continued optimization from those weights. The current loop
uses EMA weights temporarily and restores the live training weights afterward.
Current-harness receipts bind evaluations to the exact legacy checkpoint hashes;
they do not claim that the current training code produced those checkpoints or
that the original training run is reproducible from the surviving metadata.

| Claim in README | Evidence status | Current-harness rerun |
|---|---|---|
| Pixel DDIM: 54.4094 at n=2,048 | Schema-v2 receipt + stdout log + checksums; checkpoint `66b2558a…86ae` | Current for fixed checkpoint |
| Conditional + CFG: 38.6036 at n=2,048 | Schema-v2 receipt + stdout log + checksums; checkpoint `40dab3fd…63f7` | Current for fixed checkpoint |
| DDIM/DPM-Solver++ sampler table | Historical single runs; only DPM 20 has a surviving standalone log | Pending |
| Latent: 143.43 before normalization | Historical log + surviving checkpoints | Pending |
| Latent: 105.36 after normalization | Historical log + surviving checkpoints/stats | Pending |

The current `fid.py` harness improves the evidence boundary by recording:

- the exact Git commit and whether the worktree is dirty;
- the SHA-256 and byte size of the checkpoint and harness;
- Python, NumPy, PyTorch, torchvision, CUDA, cuDNN, and GPU metadata;
- all score-affecting options, including both batch sizes and seed;
- the canonical CIFAR-10 backend, full-dataset hash, seeded subset hash, and
  every selected sample hash, independent of ImageFolder/archive ordering;
- SHA-256 hashes of generated images, selected real images, and both activation arrays;
- whether each array came from cache; and
- the internal score plus the real-vs-real sanity score.

A fixed seed supports controlled reruns. It does not imply bitwise-identical CUDA
results across different GPUs, drivers, or library versions, which is why the
receipt also binds the environment and produced arrays.

## Reproducibility boundary

The historical checkpoints are gitignored and are not currently distributed by
this repository. The protocol below is therefore **owner-machine
reproducibility** for the preserved local artifacts. A third party can inspect
the source, install the environment, run the tests, and train a new model, but
cannot reproduce the recorded checkpoint scores from a clean clone alone.

If model artifacts are published separately, their downloaded SHA-256 digests
must match the checkpoint digests in the corresponding receipts before the run
can be described as independently reproducible.

## Clean rerun protocol

Run these commands from the repository root in **PowerShell**, in this order.
The two n=2,048 jobs are intentionally sequential so they do not compete for GPU
memory. `--no-cache` forces regeneration even when old arrays are present.

The runner defaults to `.\.venv\Scripts\python.exe`. Create that environment as
shown below, or pass another interpreter explicitly with `-Python`; do not encode
a user-specific absolute path in published commands or receipts.

The checked-in runner performs the clean-worktree guard, tests, both evaluations,
baseline receipt checks, and final hashes. Independently confirm every acceptance
condition listed below before promoting a number:

```powershell
python -m venv .venv
$python = ".\.venv\Scripts\python.exe"
& $python -m pip install --upgrade pip
& $python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
& $python -m pip install -r requirements.txt

.\scripts\run_fid_receipts.ps1 -Python $python
```

The expanded protocol is shown below so the run is auditable without trusting
the wrapper:

```powershell
$python = ".\.venv\Scripts\python.exe"

$status = git status --porcelain
if ($LASTEXITCODE -ne 0) { throw "git status failed" }
if ($status) { throw "worktree must be clean" }
$commit = git rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw "git rev-parse HEAD failed" }

& $python -m pytest -q --basetemp ".pytest-tmp\fid-evidence"
if ($LASTEXITCODE -ne 0) { throw "tests failed" }

Get-FileHash -Algorithm SHA256 out_cifar\ckpt.pt
Get-FileHash -Algorithm SHA256 out_cifar_cond\ckpt.pt

New-Item -ItemType Directory -Force evaluation_runs | Out-Null

& $python fid.py `
  --ckpt out_cifar\ckpt.pt `
  --n 2048 `
  --channels 3 `
  --image-size 32 `
  --steps 50 `
  --sampler ddim `
  --order 2 `
  --batch-size 64 `
  --sample-batch-size 64 `
  --seed 0 `
  --no-cache `
  --receipt evaluation_runs\fid-unconditional-ddim50-n2048.json `
  2>&1 | Tee-Object evaluation_runs\fid-unconditional-ddim50-n2048.log
if ($LASTEXITCODE -ne 0) { throw "unconditional evaluation failed" }

Get-FileHash -Algorithm SHA256 evaluation_runs\fid-unconditional-ddim50-n2048.json
Get-FileHash -Algorithm SHA256 evaluation_runs\fid-unconditional-ddim50-n2048.log

& $python fid.py `
  --ckpt out_cifar_cond\ckpt.pt `
  --n 2048 `
  --channels 3 `
  --image-size 32 `
  --steps 50 `
  --sampler ddim `
  --order 2 `
  --batch-size 64 `
  --sample-batch-size 64 `
  --seed 0 `
  --num-classes 10 `
  --cfg-scale 2.0 `
  --no-cache `
  --receipt evaluation_runs\fid-conditional-cfg2-ddim50-n2048.json `
  2>&1 | Tee-Object evaluation_runs\fid-conditional-cfg2-ddim50-n2048.log
if ($LASTEXITCODE -ne 0) { throw "conditional evaluation failed" }

Get-FileHash -Algorithm SHA256 evaluation_runs\fid-conditional-cfg2-ddim50-n2048.json
Get-FileHash -Algorithm SHA256 evaluation_runs\fid-conditional-cfg2-ddim50-n2048.log

if ((git rev-parse HEAD) -ne $commit) { throw "HEAD changed during evaluation" }
if (git status --porcelain) { throw "worktree changed during evaluation" }
```

A clean clone can create `.venv` and install the CUDA PyTorch wheel plus
`requirements.txt` as shown above and in `README.md`. It still needs the
hash-matched checkpoint artifacts to reproduce the recorded scores. The run
recorded Python 3.11.15, NumPy 2.4.6, PyTorch 2.11.0+cu128, torchvision
0.26.0+cu128, CUDA 12.8, cuDNN version code 91900, NVIDIA driver 592.15, and an
RTX 5070 Laptop GPU. Treat that as context rather than a substitute for the
generated receipt.

## Acceptance rule

Promote a rerun number from "historical" to **current-harness evaluation evidence
for the named legacy checkpoint** only when:

1. the full test suite passes;
2. the evaluation used `--no-cache` and produced a JSON receipt;
3. the schema-v2 receipt says `n=2048`, `seed=0`, and
   `no_cache_requested=true`;
4. the canonical dataset identity contains the complete 10,000-image test-set
   hash, the 2,048-image subset hash, and 2,048 selected-sample hashes, and is
   identical across the sequential unconditional and conditional runs;
5. all and only the four required array records exist and say `cache_hit=false`;
6. the checkpoint hash matches the intended artifact;
7. the receipt records the tested Git commit and `repository.dirty=false`, and
   the runner reconfirms HEAD, source hashes, checkpoint hashes, and cleanliness
   after every external command;
8. `real_vs_real_sanity_score` is finite and effectively zero; and
9. the JSON receipt and stdout log hashes are saved with the reported result.

This promotion validates the evaluation of fixed checkpoint bytes. It does not
upgrade the legacy training run into a controlled or reproducible training
experiment. In particular, the conditioning/CFG and latent-normalization tables
remain observational comparisons between separately trained checkpoints.

Do not compare this repository-specific metric directly with paper FID tables.
For a publication-grade comparison, add a canonical 50,000-sample implementation
and validate it against a reference FID package first.
