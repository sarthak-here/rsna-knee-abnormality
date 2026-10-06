# Result

The optimized private run completed successfully and the downloaded artifacts
passed strict local validation. The feature tensor has shape
`[4407, 6, 4, 768]`; its SHA-256 is
`f70be5723452513826a35788bb4179472773fb05e351b7e9e51b64539149488f`.
The three-head checkpoint SHA-256 is
`fdf535f4569e944bfb25781a57abe4cddaae2b0ada0748044ba69ea67fe57001`.
The execution log contains no runtime traceback.

The candidate was rejected decisively. Gold macro AUCs were `0.756158`,
`0.746875`, and `0.746726`; their ensemble reached `0.752404`, which is
`-0.108053` below the prior private `0.860457` control. All 58 organizer-labelled
studies remained evaluation-only, and no submission was created. Frozen
DINOv2-S features from this high-resolution contract should not be scaled or
blended into the locked public `0.943` baseline.
