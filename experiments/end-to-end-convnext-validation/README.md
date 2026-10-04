# End-to-end ConvNeXt-Tiny validation

Private one-fold screen for a fully fine-tuned ConvNeXt-Tiny encoder pretrained on
ImageNet-22k and fine-tuned on ImageNet-1k. This backbone is not represented by the
public DINO/RadImageNet/CoAtNet baseline and therefore tests an independent error family
using six MRI sequence slots and adjacent 2.5D slice triplets. It uses the attributed
Steven v4 labels and the attributed public 11 GiB pixel cache.

The input is the attributed 224px/9-slice cache, so this isolates complete encoder
adaptation with the empirically stronger Steven v4 labels. The 58
organizer-labelled studies are excluded from fitting and checkpoint selection.
This notebook cannot create a competition submission. A five-fold training run is only
justified if the one-fold screen is materially stronger than the prior control.
