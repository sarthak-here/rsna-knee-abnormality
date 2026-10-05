# High-resolution DINOv2 validation

Private one-fold screen of the validated 288px/12-slice physical cache. A pinned
DINOv2-S encoder is kept frozen and extracts four adjacent 2.5D depth tokens for
each of six clinical sequence slots. A target-specific depth/slot attention head
is trained with Steven Lee Hans' v4 report-derived labels across three seeds.

All 58 organizer-labelled studies are excluded from fitting, splitting, stopping,
and model selection and are used only for frozen final evaluation. The notebook
cannot create a competition submission. Scaling is allowed only if the candidate
materially exceeds the prior private 0.860457 gold control; the locked public
0.943 submission remains untouched.
