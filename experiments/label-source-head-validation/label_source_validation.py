"""Paired image-head audit of Steven v4 versus JEV hybrid supervision."""

SEEDS = (20261011, 20261012, 20261013, 20261014, 20261015)


def normalized_report_groups(series):
    text = (series.fillna("").astype(str).str.lower()
            .str.replace(r"\s+", " ", regex=True).str.strip())
    return np.asarray([hashlib.sha256(value.encode()).hexdigest()[:24] for value in text])


def aligned_table(train, filename):
    frame = pd.read_csv(unique_input(filename), dtype={"StudyInstanceUID": str})
    return train[["StudyInstanceUID"]].merge(
        frame, on="StudyInstanceUID", how="left", validate="one_to_one")


def load_sources(train):
    steven = aligned_table(train, "llm_labels_v4_blend.csv")[TARGETS]
    if not np.isfinite(steven.to_numpy(float)).all():
        raise RuntimeError("Steven v4 labels incomplete")
    jev = aligned_table(train, "HYBRID_labels_scores.csv")[TARGETS]
    fallback = aligned_table(train, "FINAL2_labels_scores.csv")[TARGETS]
    missing = jev.isna()
    if int(missing.sum().sum()) != 3:
        raise RuntimeError(f"unexpected JEV missing-cell count {int(missing.sum().sum())}")
    jev = jev.where(~missing, fallback)
    if not np.isfinite(jev.to_numpy(float)).all():
        raise RuntimeError("JEV labels incomplete after fallback")
    return steven.to_numpy(np.float32), jev.to_numpy(np.float32)


def main_label_source():
    if DEVICE.type != "cuda":
        raise RuntimeError("GPU required")
    root = competition_root()
    train = pd.read_csv(root / "train.csv", dtype={"StudyInstanceUID": str})
    ids = train.StudyInstanceUID.astype(str).to_numpy()
    gold = train[TARGETS].notna().all(axis=1).to_numpy()
    if int(gold.sum()) != 58:
        raise RuntimeError("official gold count drift")
    steven, jev = load_sources(train)
    package = np.load(unique_input("incumbent_radimagenet_features.npz"), allow_pickle=False)
    if package["ids"].astype(str).tolist() != ids.tolist():
        raise RuntimeError("feature identity drift")
    if str(package["encoder_sha256"]) != RAD_SHA:
        raise RuntimeError("feature encoder hash drift")
    features, masks = package["features"], package["masks"]
    nongold, gold_index = np.flatnonzero(~gold), np.flatnonzero(gold)
    groups = normalized_report_groups(train.Report)
    gold_truth = train.loc[gold, TARGETS].to_numpy(np.float64)
    rows, steven_gold, jev_gold = [], [], []
    for seed in SEEDS:
        unique = np.unique(groups[nongold])
        shuffled = np.random.default_rng(seed).permutation(unique)
        validation_groups = set(shuffled[:max(1, int(.10 * len(shuffled)))])
        validation = nongold[np.asarray([g in validation_groups for g in groups[nongold]])]
        fitting = nongold[np.asarray([g not in validation_groups for g in groups[nongold]])]
        if set(groups[fitting]).intersection(groups[validation]):
            raise RuntimeError("report-group leakage")
        steven_prediction, steven_weak = train_head(
            features, masks, steven, fitting, validation, seed)
        jev_prediction, jev_weak = train_head(
            features, masks, jev, fitting, validation, seed)
        steven_auc = macro_auc(gold_truth, steven_prediction[gold_index])
        jev_auc = macro_auc(gold_truth, jev_prediction[gold_index])
        rows.append({"seed": seed, "fit_studies": int(len(fitting)),
                     "validation_studies": int(len(validation)),
                     "steven_weak_auc": steven_weak, "jev_weak_auc": jev_weak,
                     "steven_gold_auc": steven_auc, "jev_gold_auc": jev_auc,
                     "delta_jev_minus_steven": jev_auc - steven_auc})
        steven_gold.append(steven_prediction[gold_index])
        jev_gold.append(jev_prediction[gold_index])
        log(f"seed {seed}: JEV-Steven gold delta {jev_auc-steven_auc:+.6f}")
    steven_mean = np.mean(steven_gold, axis=0)
    jev_mean = np.mean(jev_gold, axis=0)
    steven_auc = macro_auc(gold_truth, steven_mean)
    jev_auc = macro_auc(gold_truth, jev_mean)
    bootstrap = paired_bootstrap(gold_truth, steven_mean, jev_mean,
                                 seed=20261020)
    deltas = np.asarray([row["delta_jev_minus_steven"] for row in rows])
    report = {
        "contract": "private paired supervision audit; fixed authenticated image features; no submission",
        "encoder": "RadImageNet ResNet-50",
        "encoder_sha256": RAD_SHA,
        "control_labels": "stevenleehans llm_labels_v4_blend.csv",
        "candidate_labels": "riadmohamed42 JEV HYBRID scores with 3-cell FINAL2 fallback",
        "gold_fit_count": 0, "gold_evaluation_count": 58,
        "seeds": rows,
        "ensemble": {"steven_gold_auc": steven_auc, "jev_gold_auc": jev_auc,
                     "delta_jev_minus_steven": jev_auc - steven_auc},
        "bootstrap": bootstrap,
        "gate": {"delta_at_least_0_005": jev_auc - steven_auc >= .005,
                 "positive_seeds_at_least_4": int((deltas > 0).sum()) >= 4,
                 "bootstrap_p_at_least_0_95": bootstrap["p_improve"] >= .95,
                 "bootstrap_q025_positive": bootstrap["q025"] > 0},
        "submission_created": False,
    }
    report["passed"] = bool(all(report["gate"].values()))
    Path("/kaggle/working/label_source_validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


main_label_source()
