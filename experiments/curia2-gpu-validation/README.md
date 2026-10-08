# Curia-2 GPU private validation

This experiment reproduces the fully public Curia-2 V64 branch from
`prvsiyan/rsna-knee-read-the-report-then-the-knee` and changes only its
accelerator from PyTorch/XLA TPU to the CUDA path already implemented by the
author. The source notebook is pinned at SHA-256
`cd00cfe1c13ef24aa740a620c8242ddc1c2c0c49ac57fe5e369a2bb70e5cd220`.
The original public run stopped before model loading because
`torch_xla` was unavailable.

The candidate uses the radiology-pretrained Curia-2 DINOv2-Base encoder, eight
slices from each of six clinical slots, five report-group folds, and the
author's nested OOF blend gate against the embedded V52 control. The 58 official
labels are used only by the source's declared diagnostic gate. This notebook is
private and its final safety cell deletes every `submission*.csv` artifact.

No leaderboard submission is authorized by this run. A pass must first show
complete artifact identity, five-fold OOF coverage, a positive nested blend,
clean inference, and an independently reviewed license/provenance record.
