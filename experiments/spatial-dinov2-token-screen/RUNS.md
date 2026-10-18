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

