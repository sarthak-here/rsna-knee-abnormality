"""Training-only report cross-supervision for knee MRI classification.

This file is appended to the shared, audited RadImageNet/DICOM runtime by
build_notebook.py. The 58 organizer-labelled studies are never used for text
fitting, image-head fitting, validation, stopping, or model selection.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


REPORT_DIM = 128
REPORT_LOSS_WEIGHT = 0.10


def normalize_reports(series):
    return (series.fillna("").astype(str).str.lower()
            .str.replace(r"\s+", " ", regex=True).str.strip())


def report_groups(series):
    return np.asarray([hashlib.sha256(x.encode()).hexdigest()[:24]
                       for x in normalize_reports(series)])


def build_report_latent(reports, nongold):
    """Fit every text transform on non-gold reports only."""
    clean = normalize_reports(reports)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=3, max_df=.98, max_features=30000,
        sublinear_tf=True, strip_accents="unicode",
    )
    train_matrix = vectorizer.fit_transform(clean.iloc[nongold])
    svd = TruncatedSVD(n_components=REPORT_DIM, random_state=20260901)
    train_latent = svd.fit_transform(train_matrix).astype(np.float32)
    train_latent /= np.maximum(np.linalg.norm(train_latent, axis=1, keepdims=True), 1e-6)
    latent = np.zeros((len(clean), REPORT_DIM), np.float32)
    latent[nongold] = train_latent
    return latent, {
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "explained_variance": float(svd.explained_variance_ratio_.sum()),
        "fit_studies": int(len(nongold)),
        "gold_reports_transformed": 0,
    }


class ReportSupervisedHead(nn.Module):
    """Identical classifier path for control and report-supervised arms."""
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(TOKEN_DIM)
        self.proj = nn.Linear(TOKEN_DIM, 256)
        self.att = nn.Linear(256, len(TARGETS))
        self.cls = nn.Parameter(torch.empty(len(TARGETS), 256))
        self.bias = nn.Parameter(torch.zeros(len(TARGETS)))
        self.report = nn.Linear(256, REPORT_DIM)
        nn.init.trunc_normal_(self.cls, std=.02)

    def forward(self, x, mask):
        hidden = torch.nn.functional.gelu(self.proj(self.norm(x.float())))
        attention = self.att(hidden).masked_fill(mask[:, :, None] < .5, -1e4)
        attention = torch.softmax(attention, dim=1)
        pooled = torch.einsum("btn,btf->bnf", attention, hidden)
        logits = (pooled * self.cls).sum(-1) + self.bias
        weight = mask / mask.sum(1, keepdim=True).clamp_min(1)
        global_hidden = torch.einsum("bt,btf->bf", weight, hidden)
        report_embedding = torch.nn.functional.normalize(self.report(global_hidden), dim=1)
        return logits, report_embedding


@torch.no_grad()
def predict_report_head(model, features, masks, indices):
    model.eval(); output = []
    for start in range(0, len(indices), 128):
        index = indices[start:start+128]
        x = torch.from_numpy(features[index]).to(DEVICE)
        mask = torch.from_numpy(masks[index]).to(DEVICE)
        logits, _ = model(x, mask)
        output.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(output)


def train_report_head(features, masks, labels, report_latent,
                      train_index, validation_index, seed, report_weight):
    seed_all(seed)
    model = ReportSupervisedHead().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    dataset = TensorDataset(
        torch.from_numpy(features[train_index]),
        torch.from_numpy(masks[train_index]),
        torch.from_numpy(labels[train_index]),
        torch.from_numpy(report_latent[train_index]),
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    best_state, best_score = None, -1.0
    for epoch in range(20):
        model.train()
        for x, mask, target, text_target in loader:
            x, mask = x.to(DEVICE), mask.to(DEVICE)
            target, text_target = target.to(DEVICE), text_target.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits, text_prediction = model(x, mask)
            classification = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, target.float()
            )
            report_loss = (1.0 - (text_prediction * text_target).sum(1)).mean()
            loss = classification + float(report_weight) * report_loss
            loss.backward(); optimizer.step()
        prediction = predict_report_head(
            model, features, masks, validation_index
        )
        score = macro_auc(labels[validation_index], prediction)
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    prediction = predict_report_head(
        model, features, masks, np.arange(len(features))
    )
    del model; torch.cuda.empty_cache()
    return prediction, float(best_score)


def cross_supervision_main():
    if DEVICE.type != "cuda":
        raise RuntimeError("cross-supervision validation requires GPU")
    root = competition_root()
    train = pd.read_csv(root / "train.csv", dtype={"StudyInstanceUID": str})
    gold = train[TARGETS].notna().all(axis=1).to_numpy()
    if int(gold.sum()) != 58:
        raise RuntimeError(f"expected 58 gold studies, found {gold.sum()}")
    gold_index = np.flatnonzero(gold)
    nongold = np.flatnonzero(~gold)

    weak = pd.read_csv(unique_input("llm_labels_v4_blend.csv"),
                       dtype={"StudyInstanceUID": str})
    aligned = train[["StudyInstanceUID"]].merge(
        weak, on="StudyInstanceUID", how="left", validate="one_to_one"
    )
    labels = aligned[TARGETS].to_numpy(np.float32)
    if not np.isfinite(labels).all():
        raise RuntimeError("weak labels incomplete")

    groups = report_groups(train["Report"])
    if len(np.unique(groups[nongold])) < 4000:
        raise RuntimeError("unexpected report-group collapse")

    series = pd.read_csv(root / "train_series.csv", dtype=str)
    series_root = root / ("train_series" if (root / "train_series").is_dir()
                          else "train_images")
    items = [(row.StudyInstanceUID, row.SeriesInstanceUID,
              row.Anatomical_Plane,
              series_root / row.StudyInstanceUID / row.SeriesInstanceUID)
             for row in series.itertuples(index=False)]
    with ThreadPoolExecutor(max_workers=16) as pool:
        records = [record for record in pool.map(inspect_series, items) if record]
    by_study = {}
    for record in records:
        by_study.setdefault(record["study"], []).append(record)
    studies = train.StudyInstanceUID.tolist()
    features, masks = encode_arm(studies, by_study, False)
    np.savez_compressed(
        "/kaggle/working/incumbent_radimagenet_features.npz",
        ids=np.asarray(studies), features=features, masks=masks,
        encoder_sha256=RAD_SHA,
    )

    y_gold = train.loc[gold, TARGETS].to_numpy(float)
    results, control_gold, cross_gold, text_audits = [], [], [], []
    for seed in SEEDS:
        # Split report groups, never individual duplicate reports.
        unique_groups = np.unique(groups[nongold])
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(unique_groups)
        validation_groups = set(shuffled[:max(1, int(.10 * len(shuffled)))])
        validation_index = nongold[np.asarray(
            [group in validation_groups for group in groups[nongold]]
        )]
        train_index = nongold[np.asarray(
            [group not in validation_groups for group in groups[nongold]]
        )]
        if set(groups[train_index]).intersection(groups[validation_index]):
            raise RuntimeError("report-group leakage")
        # Fit text transforms inside this seed's training partition only.
        latent, text_audit = build_report_latent(train["Report"], train_index)
        text_audit["seed"] = seed
        text_audit["weak_validation_reports_transformed"] = 0
        text_audits.append(text_audit)
        control, control_weak = train_report_head(
            features, masks, labels, latent, train_index, validation_index,
            seed, report_weight=0.0,
        )
        cross, cross_weak = train_report_head(
            features, masks, labels, latent, train_index, validation_index,
            seed, report_weight=REPORT_LOSS_WEIGHT,
        )
        control_auc = macro_auc(y_gold, control[gold_index])
        cross_auc = macro_auc(y_gold, cross[gold_index])
        results.append({
            "seed": seed,
            "control_weak_validation_auc": control_weak,
            "cross_weak_validation_auc": cross_weak,
            "control_gold_auc": control_auc,
            "cross_gold_auc": cross_auc,
            "delta": cross_auc - control_auc,
        })
        control_gold.append(control[gold_index])
        cross_gold.append(cross[gold_index])
        log(f"seed {seed}: report cross-supervision delta {cross_auc-control_auc:+.6f}")

    control_mean = np.mean(control_gold, axis=0)
    cross_mean = np.mean(cross_gold, axis=0)
    control_auc = macro_auc(y_gold, control_mean)
    cross_auc = macro_auc(y_gold, cross_mean)
    bootstrap = paired_bootstrap(y_gold, control_mean, cross_mean)
    deltas = np.asarray([record["delta"] for record in results])
    gate = {
        "delta_at_least_0_005": cross_auc-control_auc >= .005,
        "positive_seeds_at_least_4": int((deltas > 0).sum()) >= 4,
        "bootstrap_p_at_least_0_95": bootstrap["p_improve"] >= .95,
        "bootstrap_q025_positive": bootstrap["q025"] > 0,
    }
    report = {
        "contract": "private image-only evaluation; training-only report cross-supervision",
        "encoder": "RadImageNet ResNet-50",
        "encoder_sha256": RAD_SHA,
        "weak_labels": "stevenleehans llm_labels_v4_blend.csv",
        "report_representation": "non-gold TF-IDF bigrams + 128D truncated SVD",
        "report_loss_weight": REPORT_LOSS_WEIGHT,
        "text_audits": text_audits,
        "gold_fit_count": 0,
        "gold_text_transform_count": 0,
        "gold_evaluation_count": 58,
        "seeds": results,
        "ensemble": {
            "control_gold_auc": control_auc,
            "cross_supervised_gold_auc": cross_auc,
            "delta": cross_auc-control_auc,
        },
        "bootstrap": bootstrap,
        "gate": gate,
        "passed": bool(all(gate.values())),
        "submission_created": False,
    }
    Path("/kaggle/working/report_image_validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


cross_supervision_main()


