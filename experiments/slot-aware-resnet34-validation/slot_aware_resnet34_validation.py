"""Leakage-safe slot-aware ResNet34 knee-MRI screen.

The rejected flat 44-slice reader mixed windows across clinical series slots.
This candidate preserves the public Raptor corpus' five fixed slot boundaries
and pools each slot independently before study classification.
"""

SLOT_SEED = 20261007
SLOT_FOLD = 0
SLOT_EPOCHS = 14
SLOT_BATCH = 2
SLOT_WINDOWS = 4
SLOT_WEAK_GATE = 0.87
SLOT_GOLD_FLOOR = 0.860457
SLOT_WEIGHT_SHA256 = "333f7ec4c6338da2cbed37f1fc0445f9624f1355633fa1d7eab79a91084c6cef"
SLOT_RANGES = ((0, 12), (12, 22), (22, 30), (30, 36), (36, 44))
SLOT_NAMES = ("sagittal_fs", "sagittal_nonfs", "coronal_fs", "coronal_nonfs", "axial")


def slot_centers(volume, mask, start, stop):
    valid = np.flatnonzero(mask[start:stop] > 0) + start
    valid_set = set(valid.tolist())
    centers = np.asarray([
        index for index in valid
        if index - 1 in valid_set and index + 1 in valid_set
    ], dtype=int)
    if not len(centers):
        return centers
    images = np.asarray(volume[centers], np.float32) / 255.0
    contrast = images.std(axis=(1, 2))
    gradient = (
        np.abs(np.diff(images, axis=1)).mean(axis=(1, 2))
        + np.abs(np.diff(images, axis=2)).mean(axis=(1, 2)))
    score = contrast + gradient
    kept = centers[score >= np.quantile(score, .25)]
    return kept if len(kept) else centers


class SlotWindowStudies(torch.utils.data.Dataset):
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
        uid = str(self.ids[index])
        row = self.id_to_row[uid]
        volume = np.asarray(self.volumes[row])
        mask = self.masks[row]
        output = np.zeros((5, SLOT_WINDOWS, 3, 224, 224), np.float32)
        slot_mask = np.zeros(5, np.bool_)
        for slot, (start, stop) in enumerate(SLOT_RANGES):
            eligible = slot_centers(volume, mask, start, stop)
            if not len(eligible):
                continue
            centers = choose_centers(eligible, SLOT_WINDOWS, self.train)
            windows = np.stack([volume[c - 1:c + 2] for c in centers]).astype(np.float32) / 255.0
            output[slot] = torch.nn.functional.interpolate(
                torch.from_numpy(windows), size=(224, 224), mode="bilinear",
                align_corners=False).numpy()
            slot_mask[slot] = True
        return (
            torch.from_numpy(output),
            torch.from_numpy(slot_mask),
            torch.from_numpy(self.labels[index]),
        )


class SlotAwareResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet34

        candidates = list(Path("/kaggle/input").rglob("resnet34.pth"))
        if len(candidates) != 1:
            raise RuntimeError(f"expected one resnet34.pth, found {candidates}")
        self.weight_path = candidates[0]
        weight_sha = sha256(self.weight_path)
        if weight_sha != SLOT_WEIGHT_SHA256:
            raise RuntimeError(f"ResNet34 weight identity mismatch: {weight_sha}")
        network = resnet34(weights=None)
        state = torch.load(self.weight_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {str(key).removeprefix("module."): value for key, value in state.items()}
        missing, unexpected = network.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"ResNet34 state mismatch: {missing}, {unexpected}")
        self.encoder = nn.Sequential(*list(network.children())[:-2])
        self.head = nn.Sequential(
            nn.LayerNorm(5 * 1024),
            nn.Dropout(.35),
            nn.Linear(5 * 1024, 512),
            nn.GELU(),
            nn.Dropout(.25),
            nn.Linear(512, N_TARGET),
        )
        self.register_buffer("mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([.229, .224, .225])[None, :, None, None])

    def forward(self, windows, slot_mask):
        batch, slots, count, channels, height, width = windows.shape
        image = windows.reshape(batch * slots * count, channels, height, width)
        image = (image - self.mean) / self.std
        feature = self.encoder(image).mean((-2, -1)).reshape(batch, slots, count, 512)
        pooled = torch.cat([feature.mean(2), feature.amax(2)], dim=-1)
        pooled = pooled * slot_mask.unsqueeze(-1).to(pooled.dtype)
        return self.head(pooled.flatten(1))


@torch.no_grad()
def predict_slot_model(model, ids, labels, id_to_row, volumes, masks):
    model.eval()
    output = []
    dataset = SlotWindowStudies(ids, labels, id_to_row, volumes, masks, False)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2,
                        pin_memory=True, persistent_workers=True)
    for windows, slot_mask, _ in loader:
        windows = windows.to(DEVICE, non_blocking=True)
        slot_mask = slot_mask.to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            output.append(torch.sigmoid(model(windows, slot_mask)).float().cpu().numpy())
    return np.concatenate(output)


def main_slot_aware_resnet34():
    if DEVICE.type != "cuda":
        raise RuntimeError("GPU required")
    seed_all(SLOT_SEED)
    root = competition_root()
    train = pd.read_csv(root / "train.csv", dtype={"StudyInstanceUID": str})
    gold = train[TARGETS].notna().all(axis=1).to_numpy()
    if int(gold.sum()) != 58:
        raise RuntimeError("official gold count drift")
    labels = load_soft_labels(train)
    volumes, masks, corpus_ids = open_corpus()
    id_to_row = {uid: row for row, uid in enumerate(corpus_ids)}
    ids = train.StudyInstanceUID.astype(str).to_numpy()
    if set(ids) - set(id_to_row):
        raise RuntimeError("corpus identity mismatch")

    non_gold = np.flatnonzero(~gold)
    groups = report_groups(train.Report)
    fit_local, validation_local = list(GroupKFold(5).split(
        non_gold, groups=groups[non_gold]))[SLOT_FOLD]
    fit_index = non_gold[fit_local]
    validation_index = non_gold[validation_local]
    if set(groups[fit_index]).intersection(groups[validation_index]):
        raise RuntimeError("report-group leakage")

    model = SlotAwareResNet34().to(DEVICE)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": 2e-5},
        {"params": model.head.parameters(), "lr": 5e-4},
    ], weight_decay=.03)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SLOT_EPOCHS)
    dataset = SlotWindowStudies(ids[fit_index], labels[fit_index], id_to_row,
                                volumes, masks, True)
    loader = DataLoader(dataset, batch_size=SLOT_BATCH, shuffle=True, num_workers=2,
                        pin_memory=True, persistent_workers=True, drop_last=True,
                        generator=torch.Generator().manual_seed(SLOT_SEED))
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    best_auc = -np.inf
    best_state = None
    history = []

    for epoch in range(SLOT_EPOCHS):
        model.train()
        total = 0.0
        seen = 0
        for windows, slot_mask, target in loader:
            shape = windows.shape
            windows = windows.to(DEVICE, non_blocking=True)
            windows = augment(windows.reshape(shape[0], shape[1] * shape[2], *shape[3:]))
            windows = windows.reshape(shape)
            slot_mask = slot_mask.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    model(windows, slot_mask), target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(windows)
            seen += len(windows)
        scheduler.step()
        weak_prediction = predict_slot_model(
            model, ids[validation_index], labels[validation_index], id_to_row, volumes, masks)
        weak_truth = (labels[validation_index] >= .5).astype(np.int8)
        weak_auc = macro_auc(weak_truth, weak_prediction)
        history.append({"epoch": epoch + 1, "loss": total / seen,
                        "weak_validation_auc": float(weak_auc)})
        log(f"slot epoch {epoch + 1}/{SLOT_EPOCHS}: loss {total / seen:.5f}, "
            f"weak AUC {weak_auc:.6f}")
        if weak_auc > best_auc:
            best_auc = weak_auc
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    gold_index = np.flatnonzero(gold)
    gold_prediction = predict_slot_model(
        model, ids[gold_index], labels[gold_index], id_to_row, volumes, masks)
    gold_auc = macro_auc(train.loc[gold, TARGETS].to_numpy(np.float32), gold_prediction)
    checkpoint = Path("/kaggle/working/slot_aware_resnet34_fold0.pt")
    torch.save({"model": best_state, "fold": SLOT_FOLD, "seed": SLOT_SEED,
                "targets": TARGETS, "weak_auc": best_auc, "gold_auc": gold_auc}, checkpoint)
    report = {
        "contract": "private one-fold screen; no competition submission",
        "candidate": "five-slot 20-window ResNet34 mean-max pooling with plain BCE",
        "slot_names": SLOT_NAMES,
        "slot_ranges": SLOT_RANGES,
        "fit_studies": int(len(fit_index)),
        "weak_validation_studies": int(len(validation_index)),
        "gold_fit_count": 0,
        "gold_evaluation_count": 58,
        "best_weak_auc": float(best_auc),
        "weak_validation_gate": SLOT_WEAK_GATE,
        "gold_auc": float(gold_auc),
        "gold_diagnostic_floor": SLOT_GOLD_FLOOR,
        "screen_passed": bool(best_auc >= SLOT_WEAK_GATE and gold_auc >= SLOT_GOLD_FLOOR),
        "history": history,
        "checkpoint_sha256": sha256(checkpoint),
        "pretrained_weight_sha256": sha256(model.weight_path),
        "submission_created": False,
    }
    Path("/kaggle/working/slot_aware_resnet34_validation_report.json").write_text(
        json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


main_slot_aware_resnet34()
