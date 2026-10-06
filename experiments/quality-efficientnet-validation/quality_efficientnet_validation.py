"""Leakage-safe quality-filtered EfficientNet-B0 knee-MRI screen.

This candidate is deliberately different from the frozen-feature and six-slot
readers already rejected. It trains a compact convolutional encoder end to end
on 24 quality-filtered 2.5D windows from the public 44-slice Raptor corpus.
Organizer gold labels are used once, after checkpoint selection is frozen.
"""

from sklearn.model_selection import GroupKFold

SEED = 20261007
FOLD = 0
EPOCHS = 12
BATCH = 2
N_WINDOW_TRAIN = 24
N_WINDOW_EVAL = 24
N_TARGET = 12


def report_groups(series):
    normalized = (series.fillna("").astype(str).str.lower()
                  .str.replace(r"\s+", " ", regex=True).str.strip())
    return np.asarray([hashlib.sha256(text.encode()).hexdigest()[:24]
                       for text in normalized])


def one_input(name):
    paths = list(Path("/kaggle/input").rglob(name))
    if len(paths) != 1:
        raise RuntimeError(f"expected one {name}, found {paths}")
    return paths[0]


class TwoPartVolumes:
    def __init__(self, first, second):
        self.first, self.second = first, second
        self.cut = int(first.shape[0])
        self.shape = (self.cut + int(second.shape[0]),) + tuple(first.shape[1:])

    def __getitem__(self, index):
        return self.first[index] if index < self.cut else self.second[index - self.cut]


def open_corpus():
    first = np.load(one_input("all_vols.npy"), mmap_mode="r")
    second = np.load(one_input("extra_vols.npy"), mmap_mode="r")
    first_mask = np.load(one_input("all_masks.npy"), mmap_mode="r")
    second_mask = np.load(one_input("extra_masks.npy"), mmap_mode="r")
    first_ids = np.load(one_input("all_ids.npy"), allow_pickle=True).astype(str)
    second_ids = np.load(one_input("extra_ids.npy"), allow_pickle=True).astype(str)
    volumes = TwoPartVolumes(first, second)
    masks = np.concatenate([first_mask, second_mask], axis=0)
    ids = np.concatenate([first_ids, second_ids])
    if volumes.shape != (4407, 44, 336, 336):
        raise RuntimeError(f"unexpected corpus shape {volumes.shape}")
    if masks.shape != (4407, 44) or len(ids) != 4407 or len(set(ids)) != 4407:
        raise RuntimeError("corpus identity or mask contract failed")
    return volumes, masks, ids


def quality_centers(volume, mask):
    valid = np.flatnonzero(mask > 0)
    valid_set = set(valid.tolist())
    centers = np.asarray([i for i in valid if i - 1 in valid_set and i + 1 in valid_set], int)
    if len(centers) < 3:
        lo, hi = int(valid.min()), int(valid.max())
        centers = np.arange(max(1, lo + 1), min(volume.shape[0] - 1, hi), dtype=int)
    images = np.asarray(volume[centers], np.float32) / 255.0
    contrast = images.std(axis=(1, 2))
    gradient = (np.abs(np.diff(images, axis=1)).mean(axis=(1, 2))
                + np.abs(np.diff(images, axis=2)).mean(axis=(1, 2)))
    score = contrast + gradient
    threshold = np.quantile(score, .25)
    kept = centers[score >= threshold]
    return kept if len(kept) >= 3 else centers


def choose_centers(eligible, count, train):
    if train:
        if len(eligible) >= count:
            return np.sort(np.random.choice(eligible, count, replace=False))
        return np.sort(np.random.choice(eligible, count, replace=True))
    positions = np.linspace(0, len(eligible) - 1, count).round().astype(int)
    return eligible[positions]


class QualityWindowStudies(torch.utils.data.Dataset):
    def __init__(self, ids, labels, id_to_row, volumes, masks, train):
        self.ids = np.asarray(ids)
        self.labels = np.asarray(labels, np.float32)
        self.id_to_row = id_to_row
        self.volumes = volumes
        self.masks = masks
        self.train = bool(train)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        uid = str(self.ids[index]); row = self.id_to_row[uid]
        volume = np.asarray(self.volumes[row])
        eligible = quality_centers(volume, self.masks[row])
        count = N_WINDOW_TRAIN if self.train else N_WINDOW_EVAL
        centers = choose_centers(eligible, count, self.train)
        windows = np.stack([volume[c - 1:c + 2] for c in centers]).astype(np.float32) / 255.0
        tensor = torch.from_numpy(windows)
        tensor = torch.nn.functional.interpolate(
            tensor, size=(224, 224), mode="bilinear", align_corners=False)
        return tensor, torch.from_numpy(self.labels[index])


class TargetAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Parameter(torch.randn(N_TARGET, dim) * .02)
        self.out = nn.Linear(dim, N_TARGET)
        self.dropout = nn.Dropout(.25)

    def forward(self, feature):
        feature = self.norm(feature)
        score = torch.einsum("bkd,td->btk", feature, self.query) / feature.shape[-1] ** .5
        pooled = self.dropout(torch.einsum("btk,bkd->btd", score.softmax(-1), feature))
        return (pooled * self.out.weight[None]).sum(-1) + self.out.bias


class QualityEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        weights = [p for p in Path("/kaggle/input").rglob("model.safetensors")
                   if "efficient" in str(p).lower()]
        if len(weights) != 1:
            bins = [p for p in Path("/kaggle/input").rglob("*.bin")
                    if "efficient" in str(p).lower()]
            weights = weights + bins
        if len(weights) != 1:
            raise RuntimeError(f"expected one EfficientNet weight file, found {weights}")
        self.weight_path = weights[0]
        self.encoder = timm.create_model(
            "tf_efficientnet_b0.ns_jft_in1k", pretrained=True,
            pretrained_cfg_overlay={"file": str(self.weight_path)},
            num_classes=0, global_pool="avg")
        self.head = TargetAttention(self.encoder.num_features)
        self.register_buffer("mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([.229, .224, .225])[None, :, None, None])

    def forward(self, windows):
        batch, count, channels, height, width = windows.shape
        image = windows.reshape(batch * count, channels, height, width)
        image = (image - self.mean) / self.std
        feature = self.encoder(image).reshape(batch, count, -1)
        return self.head(feature)


def augment(windows):
    batch = len(windows)
    gain = .75 + .50 * torch.rand((batch, 1, 1, 1, 1), device=windows.device)
    gamma = .80 + .40 * torch.rand((batch, 1, 1, 1, 1), device=windows.device)
    windows = (windows.clamp_min(1e-4).pow(gamma) * gain).clamp(0, 1)
    if torch.rand(()) < .75:
        windows = (windows + torch.randn_like(windows) * .025).clamp(0, 1)
    if torch.rand(()) < .50:
        h, w = windows.shape[-2:]
        cut_h, cut_w = int(h * .10), int(w * .10)
        y = int(torch.randint(0, h - cut_h + 1, ()).item())
        x = int(torch.randint(0, w - cut_w + 1, ()).item())
        windows[..., y:y + cut_h, x:x + cut_w] = 0
    return windows


@torch.no_grad()
def predict(model, ids, labels, id_to_row, volumes, masks):
    model.eval(); output = []
    dataset = QualityWindowStudies(ids, labels, id_to_row, volumes, masks, False)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2,
                        pin_memory=True, persistent_workers=True)
    for windows, _ in loader:
        windows = windows.to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            output.append(torch.sigmoid(model(windows)).float().cpu().numpy())
    return np.concatenate(output)


def load_soft_labels(train):
    source = pd.read_csv(unique_input("llm_labels_v4_blend.csv"),
                         dtype={"StudyInstanceUID": str})
    aligned = train[["StudyInstanceUID"]].merge(source, on="StudyInstanceUID", how="left",
                                                  validate="one_to_one")
    labels = aligned[TARGETS].to_numpy(np.float32)
    if not np.isfinite(labels).all():
        raise RuntimeError("Steven v4 labels do not cover the training studies")
    return labels


def main_quality_efficientnet():
    if DEVICE.type != "cuda":
        raise RuntimeError("GPU required")
    seed_all(SEED)
    root = competition_root()
    train = pd.read_csv(root / "train.csv", dtype={"StudyInstanceUID": str})
    gold = train[TARGETS].notna().all(axis=1).to_numpy()
    if int(gold.sum()) != 58:
        raise RuntimeError("official gold count drift")
    labels = load_soft_labels(train)
    volumes, masks, corpus_ids = open_corpus()
    id_to_row = {uid: row for row, uid in enumerate(corpus_ids)}
    ids = train.StudyInstanceUID.astype(str).to_numpy()
    missing = sorted(set(ids) - set(id_to_row))
    if missing:
        raise RuntimeError(f"corpus misses {len(missing)} studies")

    non_gold = np.flatnonzero(~gold)
    groups = report_groups(train.Report)
    fit_local, validation_local = list(GroupKFold(5).split(
        non_gold, groups=groups[non_gold]))[FOLD]
    fit_index = non_gold[fit_local]; validation_index = non_gold[validation_local]
    if set(groups[fit_index]).intersection(groups[validation_index]):
        raise RuntimeError("report-group leakage")

    model = QualityEfficientNet().to(DEVICE)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": 3e-5},
        {"params": model.head.parameters(), "lr": 6e-4},
    ], weight_decay=.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    dataset = QualityWindowStudies(ids[fit_index], labels[fit_index], id_to_row,
                                   volumes, masks, True)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=2,
                        pin_memory=True, persistent_workers=True, drop_last=True,
                        generator=torch.Generator().manual_seed(SEED))
    prevalence = np.clip(labels[fit_index].mean(0), .03, .70)
    positive_weight = torch.tensor(np.clip((1 - prevalence) / prevalence, 1, 8),
                                   dtype=torch.float32, device=DEVICE)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    best_auc = -np.inf; best_state = None; history = []

    for epoch in range(EPOCHS):
        model.train(); total = 0.; seen = 0
        for windows, target in loader:
            windows = augment(windows.to(DEVICE, non_blocking=True))
            target = target.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    model(windows), target, pos_weight=positive_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(optimizer); scaler.update()
            total += float(loss.detach()) * len(windows); seen += len(windows)
        scheduler.step()
        weak_prediction = predict(model, ids[validation_index], labels[validation_index],
                                  id_to_row, volumes, masks)
        weak_truth = (labels[validation_index] >= .5).astype(np.int8)
        weak_auc = macro_auc(weak_truth, weak_prediction)
        history.append({"epoch": epoch + 1, "loss": total / seen,
                        "weak_validation_auc": float(weak_auc)})
        log(f"epoch {epoch+1}/{EPOCHS}: loss {total/seen:.5f}, weak AUC {weak_auc:.6f}")
        if weak_auc > best_auc:
            best_auc = weak_auc
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    gold_index = np.flatnonzero(gold)
    gold_prediction = predict(model, ids[gold_index], labels[gold_index],
                              id_to_row, volumes, masks)
    gold_truth = train.loc[gold, TARGETS].to_numpy(np.float32)
    gold_auc = macro_auc(gold_truth, gold_prediction)
    checkpoint = Path("/kaggle/working/quality_efficientnet_fold0.pt")
    torch.save({"model": best_state, "fold": FOLD, "seed": SEED,
                "targets": TARGETS, "weak_auc": best_auc, "gold_auc": gold_auc}, checkpoint)
    report = {
        "contract": "private one-fold screen; no competition submission",
        "candidate": "quality-filtered 24-window EfficientNet-B0 target-attention MIL",
        "attribution": [
            "dreaddevelopment/knee-mri-training-the-twelve-finding-model",
            "dreaddevelopment/knee-raptor-corpus",
            "dreaddevelopment/knee-raptor-corpus-ext",
            "timm/tf-efficientnet/PyTorch/tf-efficientnet-b0/1",
            "stevenleehans/rsna-knee-llm-report-labels",
        ],
        "fold": FOLD, "seed": SEED, "epochs": EPOCHS,
        "fit_studies": int(len(fit_index)),
        "weak_validation_studies": int(len(validation_index)),
        "gold_fit_count": 0, "gold_evaluation_count": 58,
        "best_weak_auc": float(best_auc), "gold_auc": float(gold_auc),
        "prior_private_control_auc": 0.860457,
        "screen_gate_auc": 0.865457,
        "screen_passed": bool(gold_auc >= 0.865457),
        "history": history,
        "checkpoint_sha256": sha256(checkpoint),
        "pretrained_weight_sha256": sha256(model.weight_path),
        "submission_created": False,
    }
    Path("/kaggle/working/quality_efficientnet_validation_report.json").write_text(
        json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


main_quality_efficientnet()
