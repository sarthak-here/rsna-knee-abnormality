# ConvNeXt reader anatomical-mirror validation

This private diagnostic tests a fixed 50/50 anatomical-mirror TTA on the
public Apache-2.0 ConvNeXt fold-0 reader. It uses the public reader's exact
checkpoint, preprocessing, and 12-window inference contract.

The mirror transform is adapted from the public `nartaa/rsna-knee-0949-anatomical-mirror`
method: sagittal triplets reverse their slice-channel order, while coronal and
axial triplets flip image width. The OAI-trained checkpoint from that notebook
is deliberately not used because the competition host's external-data ruling
requires general access without institutional review/sign-off.

All 58 organizer-labelled studies are evaluation-only. The notebook performs
no fitting and cannot create a competition submission. The predeclared pass
gate is +0.005 macro AUC over the unmirrored public reader, bootstrap
`p_improve >= 0.95`, positive bootstrap q025, and absolute macro AUC >= 0.914.
