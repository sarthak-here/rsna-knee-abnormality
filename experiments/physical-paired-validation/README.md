# Paired physical preprocessing validation

This private experiment compares incumbent percentile-based preprocessing with physical-millimetre sampling, independent row/column spacing normalization, robust per-series intensity normalization, clinical PD/PDFS/T1/T2 sequence selection, and geometry-backed laterality normalization.

Both arms use the same frozen official RadImageNet ResNet-50 and identical attention heads. Steven Lee Hans' public v4 report-derived labels supervise only the 4,349 non-gold studies. The 58 organizer-labelled studies are excluded from fitting and used once for final paired evaluation across five deterministic seeds.

The RadImageNet asset is pinned by SHA-256 and retains its public CC-BY-NC-SA-4.0 provenance. The notebook cannot create a competition submission.
