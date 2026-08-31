# Run history

## Version 1

The leakage gate and MedicalNet checkpoint authentication passed. Execution then failed before feature extraction because normalized DICOM tensors were promoted to `float64` while CUDA autocast used `float16` model weights.

## Version 2

The volume transfer was corrected to explicit `float32`. The notebook completed private validation successfully but rejected the candidate:

- Screening-control macro AUC: `0.854324`
- MedicalNet-only macro AUC: `0.643037`
- Fixed 20% blend macro AUC: `0.847129`
- Delta: `-0.007195`
- Positive seeds: `0/5`
- Bootstrap improvement probability: `0.051`
- Bootstrap 2.5th percentile: `-0.016108`

No competition submission was created.
