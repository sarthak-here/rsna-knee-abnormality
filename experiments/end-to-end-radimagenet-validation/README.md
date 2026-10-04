# End-to-end RadImageNet validation

Private one-fold screen for a fully fine-tuned RadImageNet ResNet-50. Unlike the prior
frozen-feature and final-stage experiments, this trains the complete encoder using six
MRI sequence slots and adjacent 2.5D slice triplets. It uses the attributed public CC0
JEV hybrid report labels and the attributed public 11 GiB pixel cache.

The 58 organizer-labelled studies are excluded from fitting and checkpoint selection.
This notebook cannot create a competition submission. A five-fold training run is only
justified if the one-fold screen is materially stronger than the prior control.
