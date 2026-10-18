# Submission launch and correction

The user explicitly approved submitting the exact audited public kernel after
disclosure of its OAI-trained Raptor provenance and the unrounded-score identity
caveat.

- Competition: `rsna-knee-abnormality-detection`
- Kaggle submission ID: `57336570`
- Kernel: `heliosli/this-is-best-public-dataset-new-version-0-9586`
- Exact kernel version: `1`
- Output: `submission.csv`
- Submitted at: `2026-10-18T03:25:29.543000Z`
- Initial state: `SubmissionStatus.PENDING`
- Terminal state: `SubmissionStatus.COMPLETE`
- Terminal public macro AUC: **`0.957`**
- Remaining daily submissions after launch: `4`
- Description: `Exact public Heliosli 0.9586-title v1; Heliosli/nartaa/dreaddevelopment attribution; OAI-trained Raptor provenance acknowledged`

The run tied the displayed score of locked submission `57197157` at public
`0.957`. It supplies no evidence of an improvement, so `57197157` remains the
locked baseline. Do not launch a duplicate.

## Identity correction

After launch, comparison with the earlier reproduction proved that the URL pins
the same `scriptVersionId=358332966` used by submission `57228245`. The exact
model/data/checkpoint family is unchanged; only five per-finding blend weights
arrived through a mutable upstream dependency. A fixed offline evaluation on
the immutable 58-study cache found macro delta `-0.0004427250` versus the
earlier 50/50 contract.

This launch consumed a slot because the identity gate failed to compare the
script version before submission. Kaggle offers no cancellation endpoint for
the pending code submission. Its terminal `0.957` receipt confirms that it did
not improve the locked baseline. Do not call it a new model, resubmit it, or
promote it as research progress.
