# Fixed-checkpoint FID-style evaluation receipts

This directory publishes the clean sequential evaluations completed on
2026-08-22 UTC from source commit
`32f10ab97e2a20b4b75b207f210180d8849c1f15`.

| Checkpoint | Evaluation | Internal FID-style score | Sanity |
|---|---|---:|---:|
| Unconditional, `66b2558a…86ae` | DDIM 50, n=2,048, seed 0 | 54.4094 | 0.0 |
| Class-conditioned, `40dab3fd…63f7` | DDIM 50, CFG 2.0, n=2,048, seed 0 | 38.6036 | 0.0 |

Both runs used the same canonical 10,000-image CIFAR-10 test-set hash
`707547e3…1102`, the same seeded subset hash `c32e26fe…08a`, and fresh
generated images, real-image arrays, and activation arrays (`cache_hit=false`).
The complete identities and all 2,048 selected-image hashes are in each JSON
receipt. `SHA256SUMS.txt` binds the receipts and stdout logs.

These are repository-specific, small-sample diagnostics for fixed legacy
checkpoint bytes. They are not canonical 50,000-sample CIFAR-10 FID, not a
reproduction of the legacy training runs, and not a controlled conditioning
ablation.
