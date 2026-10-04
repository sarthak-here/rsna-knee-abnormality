# End-to-end DINOv2 validation

Private one-fold screen for a fully fine-tuned DINOv2-S encoder. Unlike the public
last-six-block members and prior frozen-feature experiments, this trains every encoder block using six
MRI sequence slots and adjacent 2.5D slice triplets. It uses the attributed public CC0
JEV hybrid report labels and the attributed public 11 GiB pixel cache.

The input is the attributed 224px/9-slice cache, so this isolates complete encoder
adaptation before spending GPU hours on a larger 288-336px cache. The 58
organizer-labelled studies are excluded from fitting and checkpoint selection.
This notebook cannot create a competition submission. A five-fold training run is only
justified if the one-fold screen is materially stronger than the prior control.
