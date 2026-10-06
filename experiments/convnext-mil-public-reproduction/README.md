# Public ConvNeXt-MIL reproduction

Private, exact reproduction of Good P. J. W.'s public notebook
`goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944`.

The upstream notebook adds an independently trained 2.5D ConvNeXt-Tiny MIL
reader to the attributed community stack with one predeclared 30% rank-blend
weight. Its public report records `0.929` for the reader alone and `0.944` for
the blend, compared with `0.943` for the stack. The reader checkpoints and code
are published as `goodpjw2008/rsna-knee-2-5d-convnext-reader` under Apache-2.0;
the notebook retains all upstream credits and input references.

The copied upstream notebook SHA-256 is
`941802c45fbc21a763c259d54d35f5fe1600f3a269a39d6773edcbf4185f457e`.
No source cell or blend constant is changed. Any generated `submission.csv`
must pass strict schema, test-ID, finiteness, range, and runtime-receipt checks
before it can be submitted.

## Result

The exact private notebook version 1 passed every validation gate and was
submitted as Kaggle submission `56877854`. It scored **0.944** publicly on
6 October 2026, improving the previous locked `0.943` result and becoming the
new validated baseline. See `RESULTS.md` for the complete audit trail.
