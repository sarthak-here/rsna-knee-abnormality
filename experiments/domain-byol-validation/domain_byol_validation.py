"""Private domain-adaptive BYOL validation for RSNA knee MRI.

The candidate adapts the pinned RadImageNet ResNet-50 to non-gold RSNA knee
MRI slices with a fixed, label-free BYOL objective. The 58 organizer-labelled
studies are excluded from representation learning, classifier fitting,
validation, stopping, and model selection. This script cannot submit.
"""

import copy
from torchvision import transforms as tvt
from torch.utils.data import Dataset


SSL_SEED = 20260911
SSL_EPOCHS = 10
SSL_BATCH = 32
SSL_LR = 1e-4
SSL_EMA = 0.996


class StudySliceDataset(Dataset):
    """One deterministic random non-gold slice per study and epoch."""

    def __init__(self, studies, paths_by_study, epoch):
        self.studies = list(studies)
        self.paths_by_study = paths_by_study
        self.epoch = int(epoch)
        self.augment = tvt.Compose([
            tvt.ToPILImage(),
            tvt.RandomResizedCrop(IMG, scale=(0.75, 1.0), ratio=(0.90, 1.10)),
            tvt.RandomAffine(degrees=8, translate=(0.03, 0.03), scale=(0.95, 1.05)),
            tvt.ColorJitter(brightness=0.20, contrast=0.25),
            tvt.RandomApply([tvt.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.35),
            tvt.ToTensor(),
            tvt.Normalize((0.5,), (0.5,)),
        ])

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, index):
        uid = self.studies[index]
        paths = self.paths_by_study[uid]
        rng = np.random.default_rng(SSL_SEED + self.epoch * 1000003 + index)
        image, _ = read_pixels(paths[int(rng.integers(0, len(paths)))])
        low, high = np.percentile(image, (1, 99))
        image = np.clip((image - low) / max(high - low, 1e-6), 0, 1)
        image = cv2.resize(image, (IMG, IMG), interpolation=cv2.INTER_AREA)
        image = np.round(image * 255).astype(np.uint8)
        view_a = self.augment(image).expand(3, -1, -1)
        view_b = self.augment(image).expand(3, -1, -1)
        return view_a, view_b


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_trainable_encoder():
    path = unique_input("ResNet50.pt")
    if sha256(path) != RAD_SHA:
        raise RuntimeError("RadImageNet checkpoint hash drift")
    model = Encoder()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.to(DEVICE)


def update_ema(target, online, decay):
    with torch.no_grad():
        for target_parameter, online_parameter in zip(target.parameters(), online.parameters()):
            target_parameter.data.mul_(decay).add_(online_parameter.data, alpha=1.0 - decay)


def byol_loss(prediction, target):
    prediction = torch.nn.functional.normalize(prediction, dim=1)
    target = torch.nn.functional.normalize(target.detach(), dim=1)
    return 2.0 - 2.0 * (prediction * target).sum(dim=1).mean()


def pretrain_byol(non_gold_studies, paths_by_study):
    seed_all(SSL_SEED)
    online_encoder = load_trainable_encoder()
    online_projector = MLP(TOKEN_DIM).to(DEVICE)
    predictor = MLP(256).to(DEVICE)
    target_encoder = copy.deepcopy(online_encoder).eval()
    target_projector = copy.deepcopy(online_projector).eval()
    for parameter in list(target_encoder.parameters()) + list(target_projector.parameters()):
        parameter.requires_grad_(False)
    parameters = list(online_encoder.parameters()) + list(online_projector.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=SSL_LR, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")
    history = []
    for epoch in range(SSL_EPOCHS):
        dataset = StudySliceDataset(non_gold_studies, paths_by_study, epoch)
        loader = DataLoader(
            dataset, batch_size=SSL_BATCH, shuffle=True, num_workers=4,
            pin_memory=True, persistent_workers=False,
            generator=torch.Generator().manual_seed(SSL_SEED + epoch),
        )
        online_encoder.train(); online_projector.train(); predictor.train()
        losses = []
        for view_a, view_b in loader:
            view_a = view_a.to(DEVICE, non_blocking=True)
            view_b = view_b.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=DEVICE.type == "cuda"):
                pred_a = predictor(online_projector(online_encoder(view_a)))
                pred_b = predictor(online_projector(online_encoder(view_b)))
                with torch.no_grad():
                    target_a = target_projector(target_encoder(view_a))
                    target_b = target_projector(target_encoder(view_b))
                loss = 0.5 * (byol_loss(pred_a, target_b) + byol_loss(pred_b, target_a))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 2.0)
            scaler.step(optimizer); scaler.update()
            update_ema(target_encoder, online_encoder, SSL_EMA)
            update_ema(target_projector, online_projector, SSL_EMA)
            losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses))
        if not math.isfinite(mean_loss):
            raise RuntimeError("non-finite BYOL loss")
        history.append({"epoch": epoch + 1, "loss": mean_loss})
        log(f"BYOL epoch {epoch + 1}/{SSL_EPOCHS}: loss={mean_loss:.6f}")
    del target_encoder, target_projector, online_projector, predictor, optimizer, scaler
    gc.collect(); torch.cuda.empty_cache()
    online_encoder.eval()
    for parameter in online_encoder.parameters():
        parameter.requires_grad_(False)
    return online_encoder, history


def selected_ssl_paths(records):
    paths = []
    for plane, fluid in SLOTS:
        record = choose(records, plane, fluid, False)
        if record is not None:
            paths.extend(row[0] for row in selected_rows(record, False))
    return sorted(set(paths))


@torch.no_grad()
def encode_with_model(studies, by_study, encoder):
    features = np.zeros((len(studies), len(SLOTS) * N_SLICE, TOKEN_DIM), np.float16)
    masks = np.zeros((len(studies), len(SLOTS) * N_SLICE), np.float32)
    batch_study = 24
    for start in range(0, len(studies), batch_study):
        ids = studies[start:start + batch_study]
        jobs = []
        for uid in ids:
            records = by_study.get(uid, [])
            jobs.extend((choose(records, plane, fluid, False), False) for plane, fluid in SLOTS)
        with ThreadPoolExecutor(max_workers=12) as pool:
            rendered = list(pool.map(lambda item: render_slot(*item), jobs))
        pixels = np.stack([item[0] for item in rendered]).reshape(len(ids), len(SLOTS) * N_SLICE, IMG, IMG)
        slot_ok = np.asarray([item[1] for item in rendered], np.float32).reshape(len(ids), len(SLOTS))
        token_ok = np.repeat(slot_ok, N_SLICE, axis=1)
        flat = pixels.reshape(-1, IMG, IMG)
        valid = np.flatnonzero(token_ok.reshape(-1) > 0)
        encoded = np.zeros((len(flat), TOKEN_DIM), np.float16)
        for batch_start in range(0, len(valid), 128):
            indices = valid[batch_start:batch_start + 128]
            x = torch.from_numpy(flat[indices]).to(DEVICE).float().div_(127.5).sub_(1).unsqueeze(1).expand(-1, 3, -1, -1)
            with torch.autocast("cuda", enabled=DEVICE.type == "cuda"):
                encoded[indices] = encoder(x).float().cpu().numpy().astype(np.float16)
        features[start:start + len(ids)] = encoded.reshape(len(ids), -1, TOKEN_DIM)
        masks[start:start + len(ids)] = token_ok
        if start % (batch_study * 10) == 0:
            log(f"BYOL-adapted encoder encoded {start}/{len(studies)}")
    return features, masks


def normalized_reports_byol(series):
    return (series.fillna("").astype(str).str.lower()
            .str.replace(r"\s+", " ", regex=True).str.strip())


def report_groups_byol(series):
    return np.asarray([hashlib.sha256(text.encode()).hexdigest()[:24]
                       for text in normalized_reports_byol(series)])


def domain_byol_main():
    if DEVICE.type != "cuda":
        raise RuntimeError("domain BYOL validation requires GPU")
    root = competition_root()
    train = pd.read_csv(root / "train.csv", dtype={"StudyInstanceUID": str})
    studies = train.StudyInstanceUID.tolist()
    gold = train[TARGETS].notna().all(axis=1).to_numpy()
    if int(gold.sum()) != 58:
        raise RuntimeError(f"expected 58 gold studies, found {gold.sum()}")
    gold_index = np.flatnonzero(gold)
    non_gold = np.flatnonzero(~gold)
    non_gold_studies = [studies[index] for index in non_gold]

    weak = pd.read_csv(unique_input("llm_labels_v4_blend.csv"), dtype={"StudyInstanceUID": str})
    aligned = train[["StudyInstanceUID"]].merge(weak, on="StudyInstanceUID", how="left", validate="one_to_one")
    labels = aligned[TARGETS].to_numpy(np.float32)
    if not np.isfinite(labels).all():
        raise RuntimeError("weak labels incomplete")

    series = pd.read_csv(root / "train_series.csv", dtype=str)
    series_root = root / ("train_series" if (root / "train_series").is_dir() else "train_images")
    items = [(row.StudyInstanceUID, row.SeriesInstanceUID, row.Anatomical_Plane,
              series_root / row.StudyInstanceUID / row.SeriesInstanceUID)
             for row in series.itertuples(index=False)]
    with ThreadPoolExecutor(max_workers=16) as pool:
        records = [record for record in pool.map(inspect_series, items) if record]
    by_study = {}
    for record in records:
        by_study.setdefault(record["study"], []).append(record)
    paths_by_study = {uid: selected_ssl_paths(by_study.get(uid, [])) for uid in non_gold_studies}
    missing = [uid for uid, paths in paths_by_study.items() if not paths]
    if missing:
        raise RuntimeError(f"non-gold studies without SSL slices: {len(missing)}")
    if set(paths_by_study).intersection(set(train.loc[gold, "StudyInstanceUID"])):
        raise RuntimeError("gold study entered BYOL pretraining set")

    adapted_encoder, ssl_history = pretrain_byol(non_gold_studies, paths_by_study)
    checkpoint_path = Path("/kaggle/working/byol_adapted_encoder.pt")
    torch.save(adapted_encoder.state_dict(), checkpoint_path)
    adapted_sha = sha256(checkpoint_path)
    candidate_features, candidate_masks = encode_with_model(studies, by_study, adapted_encoder)
    del adapted_encoder; gc.collect(); torch.cuda.empty_cache()

    package = np.load(unique_input("incumbent_radimagenet_features.npz"), allow_pickle=False)
    if package["ids"].astype(str).tolist() != studies:
        raise RuntimeError("incumbent feature identity/order drift")
    if str(package["encoder_sha256"]) != RAD_SHA:
        raise RuntimeError("incumbent encoder hash drift")
    control_features = package["features"]
    control_masks = package["masks"]
    if control_features.shape != candidate_features.shape:
        raise RuntimeError("candidate/control feature shape mismatch")
    if not np.array_equal(control_masks, candidate_masks):
        raise RuntimeError("candidate/control token mask mismatch")

    groups = report_groups_byol(train["Report"])
    y_gold = train.loc[gold, TARGETS].to_numpy(float)
    results, control_gold, candidate_gold = [], [], []
    for seed in SEEDS:
        unique_groups = np.unique(groups[non_gold])
        shuffled = np.random.default_rng(seed).permutation(unique_groups)
        validation_groups = set(shuffled[:max(1, int(0.10 * len(shuffled)))])
        validation_index = non_gold[np.asarray([group in validation_groups for group in groups[non_gold]])]
        train_index = non_gold[np.asarray([group not in validation_groups for group in groups[non_gold]])]
        if set(groups[train_index]).intersection(groups[validation_index]):
            raise RuntimeError("report-group leakage")
        control_prediction, control_weak = train_head(
            control_features, control_masks, labels, train_index, validation_index, seed
        )
        candidate_prediction, candidate_weak = train_head(
            candidate_features, candidate_masks, labels, train_index, validation_index, seed
        )
        control_auc = macro_auc(y_gold, control_prediction[gold_index])
        candidate_auc = macro_auc(y_gold, candidate_prediction[gold_index])
        results.append({
            "seed": seed,
            "control_weak_validation_auc": control_weak,
            "candidate_weak_validation_auc": candidate_weak,
            "control_gold_auc": control_auc,
            "candidate_gold_auc": candidate_auc,
            "delta": candidate_auc - control_auc,
        })
        control_gold.append(control_prediction[gold_index])
        candidate_gold.append(candidate_prediction[gold_index])
        log(f"seed {seed}: domain-BYOL delta {candidate_auc - control_auc:+.6f}")

    control_mean = np.mean(control_gold, axis=0)
    candidate_mean = np.mean(candidate_gold, axis=0)
    control_auc = macro_auc(y_gold, control_mean)
    candidate_auc = macro_auc(y_gold, candidate_mean)
    bootstrap = paired_bootstrap(y_gold, control_mean, candidate_mean)
    deltas = np.asarray([result["delta"] for result in results])
    gate = {
        "delta_at_least_0_005": candidate_auc - control_auc >= 0.005,
        "positive_seeds_at_least_4": int((deltas > 0).sum()) >= 4,
        "bootstrap_p_at_least_0_95": bootstrap["p_improve"] >= 0.95,
        "bootstrap_q025_positive": bootstrap["q025"] > 0,
    }
    report = {
        "contract": "private domain-adaptive BYOL validation; gold evaluation only; no submission",
        "initial_encoder": "RadImageNet ResNet-50",
        "initial_encoder_sha256": RAD_SHA,
        "adapted_encoder_sha256": adapted_sha,
        "feature_control_source": "rsna-knee-report-image-cross-supervision authenticated output",
        "weak_labels": "stevenleehans llm_labels_v4_blend.csv",
        "ssl": {
            "method": "BYOL",
            "seed": SSL_SEED,
            "epochs": SSL_EPOCHS,
            "batch_size": SSL_BATCH,
            "learning_rate": SSL_LR,
            "ema": SSL_EMA,
            "study_count": len(non_gold_studies),
            "gold_study_count": 0,
            "history": ssl_history,
        },
        "gold_fit_count": 0,
        "gold_evaluation_count": 58,
        "seeds": results,
        "ensemble": {
            "control_gold_auc": control_auc,
            "candidate_gold_auc": candidate_auc,
            "delta": candidate_auc - control_auc,
        },
        "bootstrap": bootstrap,
        "gate": gate,
        "passed": bool(all(gate.values())),
        "submission_created": False,
    }
    Path("/kaggle/working/domain_byol_validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


domain_byol_main()
