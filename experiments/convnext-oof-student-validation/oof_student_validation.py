"""Paired fixed-epoch student test of public ConvNeXt OOF-corrected targets."""

SEEDS = (20261021, 20261022, 20261023, 20261024, 20261025)


def aligned_frame(train, filename):
    frame = pd.read_csv(unique_input(filename), dtype={"StudyInstanceUID": str})
    return train[["StudyInstanceUID"]].merge(
        frame, on="StudyInstanceUID", how="left", validate="one_to_one")


def train_fixed(features, masks, labels, fit_idx, seed, epochs=20):
    seed_all(seed)
    model = AttentionHead().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    tx = torch.from_numpy(features[fit_idx])
    tm = torch.from_numpy(masks[fit_idx])
    ty = torch.from_numpy(labels[fit_idx])
    loader = DataLoader(
        TensorDataset(tx, tm, ty), batch_size=64, shuffle=True,
        generator=torch.Generator().manual_seed(seed))
    losses = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        seen = 0
        for x, mask, y in loader:
            x, mask, y = x.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(x, mask), y.float())
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(x)
            seen += len(x)
        losses.append(total / seen)
    return model, losses


def main_oof_student():
    if DEVICE.type != "cuda":
        raise RuntimeError("GPU required")
    root = competition_root()
    train = pd.read_csv(root / "train.csv", dtype={"StudyInstanceUID": str})
    ids = train.StudyInstanceUID.astype(str).to_numpy()
    gold = train[TARGETS].notna().all(axis=1).to_numpy()
    if int(gold.sum()) != 58:
        raise RuntimeError("official gold count drift")

    steven = aligned_frame(train, "llm_labels_v4_blend.csv")[TARGETS]
    if not np.isfinite(steven.to_numpy(float)).all():
        raise RuntimeError("Steven labels incomplete")
    steven = steven.to_numpy(np.float32)

    oof = pd.read_csv(unique_input("convnext_oof_predictions.csv"),
                      dtype={"StudyInstanceUID": str})
    expected_columns = ["StudyInstanceUID", "teacher_fold"] + TARGETS
    if oof.columns.tolist() != expected_columns:
        raise RuntimeError("OOF schema drift")
    if len(oof) != 2610 or oof.StudyInstanceUID.nunique() != len(oof):
        raise RuntimeError("OOF coverage drift")
    if set(oof.teacher_fold.unique()) != {0, 1, 2}:
        raise RuntimeError("OOF fold drift")
    if set(oof.StudyInstanceUID) & set(train.loc[gold, "StudyInstanceUID"]):
        raise RuntimeError("gold leakage in OOF teacher table")
    teacher = train[["StudyInstanceUID"]].merge(
        oof, on="StudyInstanceUID", how="left", validate="one_to_one")
    covered = teacher[TARGETS].notna().all(axis=1).to_numpy()
    fit_idx = np.flatnonzero(covered & ~gold)
    if len(fit_idx) != 2610 or covered[gold].any():
        raise RuntimeError("teacher alignment failure")
    teacher_values = teacher[TARGETS].to_numpy(np.float32)
    corrected = steven.copy()
    corrected[fit_idx] = 0.5 * steven[fit_idx] + 0.5 * teacher_values[fit_idx]

    package = np.load(unique_input("incumbent_radimagenet_features.npz"),
                      allow_pickle=False)
    if package["ids"].astype(str).tolist() != ids.tolist():
        raise RuntimeError("feature identity drift")
    if str(package["encoder_sha256"]) != RAD_SHA:
        raise RuntimeError("feature encoder hash drift")
    features, masks = package["features"], package["masks"]
    gold_idx = np.flatnonzero(gold)
    gold_truth = train.loc[gold, TARGETS].to_numpy(np.float64)

    rows, control_gold, candidate_gold = [], [], []
    for seed in SEEDS:
        control_model, control_loss = train_fixed(
            features, masks, steven, fit_idx, seed)
        candidate_model, candidate_loss = train_fixed(
            features, masks, corrected, fit_idx, seed)
        control_pred = predict(control_model, features, masks, gold_idx)
        candidate_pred = predict(candidate_model, features, masks, gold_idx)
        control_auc = macro_auc(gold_truth, control_pred)
        candidate_auc = macro_auc(gold_truth, candidate_pred)
        rows.append({
            "seed": seed,
            "fit_studies": int(len(fit_idx)),
            "epochs": 20,
            "control_final_loss": control_loss[-1],
            "candidate_final_loss": candidate_loss[-1],
            "control_gold_auc": control_auc,
            "candidate_gold_auc": candidate_auc,
            "delta_candidate_minus_control": candidate_auc - control_auc,
        })
        control_gold.append(control_pred)
        candidate_gold.append(candidate_pred)
        log(f"seed {seed}: corrected-control gold delta "
            f"{candidate_auc-control_auc:+.6f}")
        del control_model, candidate_model
        torch.cuda.empty_cache()

    control_mean = np.mean(control_gold, axis=0)
    candidate_mean = np.mean(candidate_gold, axis=0)
    control_auc = macro_auc(gold_truth, control_mean)
    candidate_auc = macro_auc(gold_truth, candidate_mean)
    bootstrap = paired_bootstrap(
        gold_truth, control_mean, candidate_mean, seed=20261026)
    deltas = np.asarray([r["delta_candidate_minus_control"] for r in rows])
    report = {
        "contract": "private fixed-epoch paired OOF-teacher target audit; no submission",
        "encoder": "RadImageNet ResNet-50",
        "encoder_sha256": RAD_SHA,
        "control_targets": "Steven v4 soft labels",
        "candidate_targets": "fixed 50/50 Steven v4 and true fold-held-out ConvNeXt teacher",
        "fit_population": "same 2610 OOF-covered non-gold studies for both arms",
        "checkpoint_selection": "none; fixed 20 epochs",
        "teacher_folds": [0, 1, 2],
        "teacher_oof_count": 2610,
        "gold_fit_count": 0,
        "gold_evaluation_count": 58,
        "seeds": rows,
        "ensemble": {
            "control_gold_auc": control_auc,
            "candidate_gold_auc": candidate_auc,
            "delta_candidate_minus_control": candidate_auc - control_auc,
        },
        "bootstrap": bootstrap,
        "gate": {
            "delta_at_least_0_005": candidate_auc - control_auc >= 0.005,
            "positive_seeds_at_least_4": int((deltas > 0).sum()) >= 4,
            "bootstrap_p_at_least_0_95": bootstrap["p_improve"] >= 0.95,
            "bootstrap_q025_positive": bootstrap["q025"] > 0,
        },
        "submission_created": False,
    }
    report["passed"] = bool(all(report["gate"].values()))
    Path("/kaggle/working/oof_student_validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n")
    if Path("/kaggle/working/submission.csv").exists():
        raise RuntimeError("forbidden submission artifact")
    print(json.dumps(report, indent=2))


main_oof_student()

