# Run history

## Version 1 — infrastructure failure after successful extraction

The frozen encoder completed all 4,407 studies in 1,661 seconds and wrote the
exact expected `[4407,6,4,16,384]` float16 cache. Local validation found:

- SHA-256 `d70c4340cdb044c1cebee885cb48a173f41765f4c72abd5a4a44437c730b9932`;
- all values finite;
- exactly 23,089 nonzero sequence slots.

The run then failed before any fitting or organizer-gold evaluation because
`h5py 3.14` could not convert the HDF5 mask dataset directly to boolean during
`np.asarray(dataset, dtype=bool)`. Version 2 changes only this reproducible read:
the mask is loaded in its native dtype and converted by NumPy afterward. The
representation, architecture, targets, seeds, epochs, residual weight and gates
are unchanged.

## Version 2 — complete, integrity pass, scientific rejection

Version 2 completed the unchanged contract. Strict local validation reproduced
every identity and integrity receipt:

- feature cache SHA-256
  `d70c4340cdb044c1cebee885cb48a173f41765f4c72abd5a4a44437c730b9932`,
  shape `[4407,6,4,16,384]`, dtype `float16`, all values finite, and exactly
  23,089 occupied slots;
- prediction archive SHA-256
  `f7eb28f449e5d8676c66f1ad6d8d79a4234640d8dbb549cd3d101d95907bb4a1`,
  loaded with `allow_pickle=False`, with 58 unique Unicode IDs and complete
  finite `[58,12]` / `[5,58,12]` prediction tensors in `[0,1]`;
- exact counts 4,349 fit, 2,610 corrected, 1,739 unchanged, 58 gold
  evaluation-only, and zero gold fit;
- all five checkpoint SHA-256 values matched the report;
- clean logs, zero CSV files, and `submission_created=false`.

The frozen anchor scored `0.9325528441` macro AUC. The spatial DINOv2 reader
scored `0.7499648385`; its fixed 10% residual blend scored `0.9313867300`, a
delta of `-0.0011661141`. It produced `0/5` positive seed blends and only five
strictly nonnegative targets. The paired bootstrap gave `p=0.2830` and
`q025=-0.0051809979`. Every predeclared scientific gate failed.

Decision: reject and close this exact representation. Do not tune its targets,
blend weight, architecture, epochs, or checkpoints; do not build hidden
inference or submit it.

