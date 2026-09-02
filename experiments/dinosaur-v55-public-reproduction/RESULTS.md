# DINOsaur V5.5 reproduction result

Status: completed and rejected as an improvement candidate.

The private Kaggle version 1 run completed cleanly and found the public V16 SlotHead specialist. The three generated files passed strict validation against the official sample submission: exact IDs and row order, exact 13-column schema, finite values, and values within `[0, 1]`.

However, on the three visible test studies, `submission.csv`, `v45_anchor_0936.csv`, and `v55_gold_oof_slothead_candidate.csv` are byte-identical (SHA-256 `daaefd7836e6b0786bde7f53134edead6a33a2e293e8650d03dcaf64c17fba46`). The specialist therefore caused no observable rank change on the public test rows. Its training gate also used an approximate 58-study OOF anchor rather than the exact V4.5 leaderboard ensemble.

Conclusion: successful reproduction does not establish an improvement over 0.936. No competition submission was made. Further work must investigate a genuinely complementary discussion-derived public checkpoint or prediction family and validate it independently before submission.

Source attribution remains in the unchanged public notebook and Kaggle metadata: `romantamrazov/rsna-knee-dinosaur-v4` and `romantamrazov/rsna-knee-dinosaur-v4-train`.
