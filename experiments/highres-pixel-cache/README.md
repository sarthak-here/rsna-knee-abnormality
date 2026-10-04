# High-resolution physical pixel cache

CPU-only preprocessing for a materially richer MRI representation:

- 288 × 288 pixels
- 140 mm physical crop
- 12 physically ordered slices per each of six clinical sequence slots
- central 84% slice window
- laterality normalization inherited from the attributed public implementation
- random-access HDF5 with per-study/per-slot compressed chunks

The implementation derives from Steven Lee Hans' public
`rsna-knee-500gb-to-11gib-cpu-pixel-cache` notebook and preserves its upstream credits.
This artifact contains competition training pixels only—no labels, model, or submission.
Its hash and study ordering are written beside the HDF5 file.
