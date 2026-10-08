# Public MAViT 288 reproduction

This experiment is a strict private reproduction of the public notebook
`hengck23/mavit-288-baseline-01-1-5sec-study-1gpu` using the author's public
MIT-licensed `hengck23/hengck23-rnsa-knee` dataset.

It is materially independent from the locked DINO/CoAtNet/ConvNeXt ensemble:
the model uses a MAViT encoder with six clinical views, eight slices per view,
and 288px inputs. The public notebook does not provide a leaderboard score, so
this is an exploratory diversity submission rather than a claimed improvement.

The reproduction pins the public notebook, checkpoint, and source hashes. It
preserves the public notebook's per-study all-zero fallback for atypical hidden
DICOMs, but makes every fallback visible in the completion receipt and refuses
to publish if more than 1% of hidden studies require it. The final file must
still match the official schema, exact test-ID order, unique IDs, and finite
probabilities in `[0,1]`. Full credit remains with Hengck23 and the MAViT/MALA
authors.
