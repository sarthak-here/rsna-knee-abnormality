# Public discussion and notebook audit - 2026-09-02

Status: no submission candidate promoted.

## Audited public work

- `romantamrazov/rsna-knee-dinosaur-v4` V5.5: exact reproduction completed, but its candidate and 0.936 anchor were byte-identical on the visible test rows; its V16 gate used an approximate OOF anchor. Rejected and documented separately.
- `anhadmahajan06/rsna-knee-take-care-of-your-knee`: describes the same DINO/DINOv3, RadImageNet and Raptor/CoAtNet 0.936 stack. No new complementary checkpoint was found.
- `prvsiyan/head-and-shoulders-knees-and-toes`: explicitly restores the 0.936 parent after OrthoDiffusion variants scored 0.935. No promotable addition.
- `dreaddevelopment/knee-mri-twelve-findings-from-a-single-model`: public 0.924 CoAtNet checkpoint family is already represented by the Raptor branch in the locked ensemble.
- `paiky1995/rsna-knee-0-926-lb-coatnet-fine-tune-blend`: arm 1 reproduced at 0.9128 on the 58-study gate, but the fine-tuned arm used a label file that contains gold labels and reports `arm2_gold_contaminated=true`. Its weak holdout AUC also fell from 0.9083 to 0.9026. The final 0.926 model is not a leakage-safe improvement candidate.
- `fleongg/rsna-knee-dinov2-base-physical-scale-soup`: physical 130 mm crop, 280 px DINOv2-Base, EMA/checkpoint soup and ranking loss were inspected. Executed weak holdout AUC reached only 0.7175, gold was included in selection, and the notebook later fitted all 4,407 studies. Rejected without reproduction.

## Discussion evidence

- Public ablation `discussion/738096` reports DINOv2-Base converges faster than Small at 224 px but reaches the same final ceiling. Increasing encoder size alone is therefore unsupported.
- Public bottleneck discussion `discussion/735826` identifies report-label quality and physical MRI preprocessing as dominant constraints. Both directions already received leakage-safe private tests in this repository and failed their promotion gates.
- Current high-vote public notebooks are largely copies or recombinations of the same DINO, RadImageNet and Raptor assets. This is consistent with `discussion/736268` and means vote count or displayed score is not evidence of a distinct model.

## Decision

Do not spend submissions on copied 0.936 notebooks, the contaminated 0.926 fine-tune, the weak physical-scale soup, or another DINO size swap. Continue monitoring for newly released checkpoints or public OOF predictions that are genuinely outside the locked ensemble and permit an honest paired comparison.

Public discussion links:

- https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/738096
- https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735826
- https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/736268
