# RSNA Knee Abnormality Detection

GPU inference pipeline for detecting 12 knee MRI findings from multi-series DICOM studies.

## Approach

- Selects sagittal, coronal, and axial MRI series using the competition metadata.
- Orders slices from DICOM geometry and normalizes each selected series independently.
- Samples a fixed 64-slice study representation with a physical 140 mm crop.
- Scores complementary 24-window and 56-window views with one CoAtNet checkpoint and finding-specific attention.
- Blends both views with equal-weight percentile ranks, selected by nested Optuna evaluation.
- Produces rank-normalized probabilities for all 12 targets.
- Preserves every test study with neutral fallback predictions if an individual study cannot be decoded.

## Kaggle execution

The notebook metadata declares the competition data and public checkpoint dependencies. Internet access is disabled and GPU execution is enabled.

```powershell
kaggle kernels push -p .
```

The run writes `/kaggle/working/submission.csv` and validates its columns, row order, values, and identifiers before completion.

## Targets

ACL, MCL, medial meniscus, lateral meniscus, medial OA, lateral OA, PF OA, effusion, synovitis, Baker's cyst, contusion, and fracture.
