# ConvNeXt reader OOF-corrected head validation

This private paired experiment applies the validated OOF supervision to the
strong public ConvNeXt reader representation. Both arms fit all 4,349 non-gold
studies with the author's exact target-specific Transformer-attention head,
identical initialization, optimizer, batch order, five seeds, and fixed 20
epochs. The candidate changes only the 2,610 legitimately covered targets.

The original public fold-0 head is independently evaluated on the untouched 58
organizer-labelled studies as an absolute reference. No label or metric selects
a checkpoint. The notebook cannot create a submission.

Passing requires +0.005 paired ensemble AUC, four positive seeds, bootstrap
p-improve at least 0.95, positive q025, and candidate absolute AUC at least 0.90.

## Attribution

- `goodpjw2008/rsna-knee-2-5d-convnext-reader`, Apache-2.0
- `stevenleehans/rsna-knee-llm-report-labels`

## Result

Version 1 completed cleanly but was rejected. The corrected ensemble reached
`0.900914` versus `0.897221` for its paired control (`+0.003693`), with four of
five positive seeds. Bootstrap confidence was insufficient (`p_improve=0.7702`,
q025 `-0.005978`), and the candidate remained below the original public fold-0
head at `0.909258`. No submission was created.
