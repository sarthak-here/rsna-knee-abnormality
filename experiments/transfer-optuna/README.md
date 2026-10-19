# Transfer-learning comparison — 19 October 2026

Status: version 1 completed; all three completed checkpoints and source/split
receipts validated. See RESULTS.md. **No pathology improvement or submission**.
The user requested a wider web search and a bounded Optuna comparison. The locked
public 0.957 submission 57197157 is unchanged. Cancelled R6 jobs remain closed.

## Shortlist from primary sources

| Candidate | Why investigate | Current limitation |
|---|---|---|
| [OrthoFoundation-MSK-5M](https://github.com/ytrsk/OrthoFoundation-MSK-5M) | Released DINOv3-L checkpoint continued on musculoskeletal images, including knee MRI; closest domain transfer found | Official repository reports OAI/fastMRI/private pretraining. GitHub API returned `license: null`; no verified checkpoint-use license. HOLD: terms and exact-new-OAI approval, not a silent replacement for an approved checkpoint. Published ACL results are not RSNA macro AUC. |
| [VideoMAE Base](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics) | Masked spatiotemporal pretraining; fine-tune final two transformer blocks rather than another frozen feature head | Natural-video, not knee pretraining. CC-BY-NC-4.0; private research only pending competition-use review. Exact HF revision `488eb9a0565f257b32866000305c8178965eb9f6`, safetensors SHA-256 `065cfde56f97671da4d196e1a3a6d3cf9304dcad92eb671e3632a6c6cdd04f73`, 346,161,616 bytes. |
| [Video Swin Tiny](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.video.swin3d_t.html) | Different hierarchical 3D local-window architecture; about 28M parameters; adapt final spatial/temporal stage | Kinetics action labels are not pathology evidence. Official checkpoint `https://download.pytorch.org/models/swin3d_t-7615ae03.pth`; full SHA must be obtained and pinned before a search. Torchvision code BSD-3-Clause; pretrained data/model terms remain distinct from code license. |
| [V-JEPA2/2.1](https://github.com/facebookresearch/vjepa2) | Predictive video pretraining rather than masked pixel reconstruction | Deferred, not in this two-model search. Latest raw hub source inspected during research used a localhost testing URL; do not execute mutable torch.hub main. Pin working source, weight license and checkpoint before adding. |
| [KneePreM](https://arxiv.org/abs/2609.31461) | Knee-specific 3D masked pretraining on OAI | No verified public checkpoint found in this search; paper is not a runnable asset. |
| [Decipher-MR](https://pmc.ncbi.nlm.nih.gov/articles/PMC13230676/) | 3D MRI/report transfer | Paper describes restrictions on sharing pretrained weights; not an immediately executable candidate. |

These are hypotheses, not evidence that any will beat 0.957. Prior failed frozen
BiomedCLIP/MedImageInsight/DINO heads are not reopened. This experiment changes
the trainable representation and treats ordered slices jointly.

## Fixed first screen

`search.py` performs real backbone adaptation with Optuna TPE + median pruning.
It accepts the existing competition-only physical HDF5, its exact study-order
CSV, official `train.csv`, and Steven v4 labels. SHA pins are mandatory for all
inputs and the two local checkpoints in an external `assets.json`. It never
downloads weights, executes remote code, loads an R6 checkpoint or submits.

- Exactly 4,407 unique official IDs and 58 complete organizer-labelled rows are
  required. Gold labels are never accessed by the objective. All gold IDs and
  any non-gold rows with an identical normalized report are excluded.
- GroupShuffleSplit: 20% report groups sealed as confirmation, then 25% of the
  remaining groups for tuning (~60/20/20 studies, actual sizes vary). Split seed
  20261019. Empty reports fail closed. Every partition/ID/hash is persisted.
- Steven-only soft targets: deliberately no existing competition teacher whose
  whole outer-fold exclusion cannot be proved. These targets are a noisy report
  proxy, NOT an independently labelled pathology validation set.
- Objective = negative soft-label BCE on tuning groups. AUC against thresholded
  proxy labels is not used to select a model. No organizer-gold optimization,
  target-specific selection, blend search or public-LB tuning.
- Both models receive one identical baseline trial before TPE explores LR
  [1e-5,1e-4] and weight decay [1e-4,1e-2]. Architecture is categorical;
  **one epoch for this coarse screen**, eight trials maximum, total active search
  budget two hours. This is a representation/LR prefilter, not a fully trained
  model contest; a slower-converging candidate can lose a short screen.
  Training/model exceptions abort the run, never become a fallback reader.
- Six occupied sequence slots, target-specific masked pooling, 224px, 16 frames.
  The source has only **12 real slices**. Both arms use the same fixed rounded
  linspace indices to repeat four frames; this adds no spatial information.
  RGB is replicated grayscale. Official ImageNet mean/std; resize 256 then
  center-crop 224, no anatomical flips or invented view geometry.
- Frozen early blocks are in eval mode (no stochastic corruption of features).
  The final Swin stage / final two VideoMAE blocks and the new pooling head train.
- Halfway and end-of-epoch tuning BCE are the two pruning rungs. A default trial
  for each architecture precedes TPE selection. Before training,
  `preflight.py` measures synthetic step cost/VRAM for both exact weights. This
  checks execution only and is never called a model result.
- SQLite persists trials and cumulative consumed time across restarts. Save
  model/optimizer/RNG/epoch after every complete epoch; partial work is not
  presented as complete and no selected checkpoint is silently substituted.

CPU inspection of the actual official metadata found four non-gold studies with
the same normalized report as organizer-gold rows. They are excluded rather than
silently allowing group leakage. The resulting fixed counts are 2,637 fit,
856 tuning, 852 sealed confirmation, 58 gold, and four excluded duplicates.
Official train CSV SHA-256:
`8ca2203c0e9d61c080c7a314c7cdb51c1b03a1d9eb4770819f7f34af53ef4e33`.
Downloaded Steven v4 CSV SHA-256:
`eb4c2b29e7e84adbb4b5ccee825ca20ade6a43dbe295f13be442d549e4f5dca4`.
All 4,349 non-gold proxy rows passed coverage/range/finiteness checks; no organizer
target values were scored. Actual private notebook source embeds these pins.

There is intentionally no automatic promotion, gold evaluator or Kaggle
submission. A winning proxy BCE is only a screening result. Freeze the selected
configuration, compare it once with the fixed default on sealed confirmation
groups, and establish independent pathology evidence/whole-pipeline lineage
before any submission or larger GPU expenditure. The repeatedly inspected 58
organizer rows are not a fresh test, and Raptor upstream used them for selection.

## Commands

```powershell
python experiments/transfer-optuna/preflight.py --assets assets.json --out preflight.json
python experiments/transfer-optuna/search.py --assets assets.json --out search-output
python -m unittest discover -s experiments/transfer-optuna -p "test_*.py"
python experiments/transfer-optuna/build_notebook.py --label-sha eb4c2b29e7e84adbb4b5ccee825ca20ade6a43dbe295f13be442d549e4f5dca4
```

Required assets keys: `h5`, `studies`, `train`, `labels`, `preflight` each with `path` and
`sha256`; `swin3d_t` with `path`, `sha256`; `videomae` with local HF directory
`path`, `revision`, `config_sha256`, and `sha256` for its `model.safetensors`.
Local model configuration must match the pinned HF revision. A trial is refused
if the measured compute for the epoch and two evaluations exceeds the remaining budget;
pruning must not conceal an impossible schedule. `terms_reviewed` must be true to run real research
training. It acknowledges a completed review, not a license override.

Initial synthetic measurements on the local RTX 3060 Laptop (6 GB), with exact
pretrained weights and full-precision backward through all six occupied slots:
Swin ~0.359 s/step, peak 1,040,794,112 bytes; VideoMAE ~0.750–0.766 s/step,
peak 1,796,154,880 bytes. Both passed. These timings exclude HDF5 and are NOT T4
measurements or pathology results. The initial four-epoch plan would exceed two
hours for VideoMAE alone. The one-epoch prefilter was set before any target loss
or prediction was computed. A new preflight is mandatory on the training GPU.
Transformers 5.5 changed VideoMAE bias names; the adapter maps released Q/V biases
and supplies exactly-zero K bias, verifies the full state-key set, and loads
strictly. It never ignores missing/unexpected weights. Five CPU tests pass.

Data preprocessing attribution: Steven Lee Hans' competition-only physical
cache implementation, preserved in `experiments/highres-pixel-cache`.
Model attribution: MCG-NJU VideoMAE authors; Video Swin authors and PyTorch
contributors. No controlled-access OAI/MRNet data is acquired by this code.
