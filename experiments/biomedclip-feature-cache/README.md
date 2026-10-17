# BiomedCLIP physical feature cache

This no-submission preprocessing stage introduces a genuinely independent image
representation: Microsoft's official MIT-licensed BiomedCLIP ViT-B/16 visual
tower. It consumes the already validated 288 px / 140 mm physical cache.

For each of six clinical sequence slots, the fixed twelve slices become four
non-overlapping 2.5D RGB windows. Each window is antialiased to the model's
native 224 px grid and normalized with the official BiomedCLIP constants. The
frozen visual tower and frozen 768-to-512 projection produce a
`[4407,6,4,512]` float16 cache. No labels are loaded and no CSV can be created.

Pinned inputs:

- physical HDF5 SHA-256: `490e5ccc07143699d05a9eb3ef9583e08795f1c2bacc6d9cf2bd0a9268158526`
- study manifest SHA-256: `efc63afee30345f94a916f98ba6b30f266fccfe2976b8475fa86653b2c510513`
- BiomedCLIP checkpoint SHA-256: `52cc993c5c5ff962bd0c60931874bc001e7e9b41666a385530f4a036294576be`
- official config SHA-256: `9a41f334a8c444678772c0ebb9ab854c97ab350bced3a17b803e258d39c23dc0`

The downstream screen will be separately predeclared and will keep all 58
organizer-labelled studies evaluation-only.
