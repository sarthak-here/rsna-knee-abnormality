# Version 2 validated cache

Version 2 completed in 676.0 seconds and passed strict local validation with
`allow_pickle=False`.

- Output SHA-256: `40768bf7c57478732c1fb14c77510b4e650634222337364c109747d0d3e9c434`
- Shape/dtype: `[4407,6,4,512]`, `float16`
- IDs: 4,407 unique fixed-Unicode values
- Occupied slots: 23,089
- Embeddings: finite; occupied-token norms `0.999783` to `1.000099`
- Exact model, config, physical-cache, and manifest hashes matched
- Logs contained no error/fallback markers
- No CSV or submission was created

Version 1 computed identical representations but serialized IDs as an object
array. Version 2 changed only that packaging defect. The validated cache is the
sole input authorized for the downstream BiomedCLIP residual screen.
