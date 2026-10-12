# Public CoAtNet-2 gold reproduction

Private, inference-only reconstruction of the public Apache-2.0 model
`patimakornsrinamen/rsna-67knee-coatnet2` variation `t4`.

The release contains a complete state dict and run configuration but does not contain
the referenced `src/model.py` or preprocessing source. This experiment therefore makes
one predeclared reconstruction attempt from the documented contract and the checkpoint
structure:

- 280 px isotropic images from a 150 mm crop;
- six clinical slots and five physically ordered adjacent 2.5D triplets per slot
  (15 slices per occupied slot);
- the exact checkpoint head recovered from its strict state-dict keys;
- plain and horizontal-flip predictions, with their 50/50 average predeclared as the
  primary result because the model card specifies hflip TTA.

Only the 58 organizer-labelled studies are decoded. They are used for inference-only
source verification; there is no fitting, checkpoint selection, blending, or submission
capability. The reconstruction is rejected unless strict loading succeeds, every identity
and hash check passes, and the predeclared average reaches macro AUC 0.89 (close enough to
the release's stated 0.9029/0.908 gold result to support building full test inference).

## Attribution and licence

- Model: `patimakornsrinamen/rsna-67knee-coatnet2`, Apache-2.0.
- Pixel pipeline: derived from Steven Lee Hans' public
  `rsna-knee-500gb-to-11gib-cpu-pixel-cache`, preserving its attribution and physical
  DICOM ordering implementation.

