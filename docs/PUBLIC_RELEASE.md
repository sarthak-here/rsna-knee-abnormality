# Public-release safety

## Status

Public visibility is blocked pending sensitive-artifact/history cleanup. The tracked tree contains per-study report-derived split/target material and prediction artifacts. Removing current copies alone does not remove historical copies.

The targeted all-ref audit at research revision248974f scanned755 unique blobs and matched4407 official IDs locally. It flagged13 historical blob versions across data/prediction files, split supervision and embedded notebook identifiers. No selected credential-pattern hits were found; this is not a comprehensive credential or medical-text clearance. See `release-audit.json` and `tools/audit_public_release.py`. The report publishes paths/counts/hashes, not study IDs or target values. New ignore rules prevent some future accidental additions but do not untrack or sanitize existing history.

## Intended public scope

Model implementations, builders without embedded private payloads, synthetic tests, aggregate results, checkpoint hashes, technology documentation, upstream attribution and failure analyses.

## Keep private

- DICOM pixels, medical reports and per-study identifiers/derived targets/splits.
- Study-level prediction arrays/tables and notebook outputs containing them.
- Credentials, authentication files and local caches.
- Private weights or assets with unverified redistribution rights.
- Noncommercial-only assets presented as competition-ready releases.

Publishing code does not license upstream checkpoints or medical data. Architecture descriptions and checkpoint hashes can be shared without distributing weight bytes.

## Cleanup decision needed

Either authorize a sanitized public snapshot in a separate repository while preserving this private history, or authorize coordinated history rewriting of this repository after an all-ref audit. Rewriting changes commit IDs and affects clones; do not force-push implicitly. Then verify public visibility. The GitHub description can be updated while the repository remains private.
