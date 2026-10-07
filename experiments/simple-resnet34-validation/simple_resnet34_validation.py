"""Leakage-safe plain ResNet34 knee-MRI screen.

This experiment directly tests the public competition hypothesis that a small
224px CNN with simple pooling and plain BCE can outperform more elaborate MIL
heads under noisy report-derived supervision. Organizer gold labels are used
only once after checkpoint selection is frozen.
"""

RESNET_SEED = 20261007
RESNET_FOLD = 0
RESNET_EPOCHS = 14
RESNET_BATCH = 2
RESNET_WEAK_GATE = 0.87
RESNET_GOLD_FLOOR = 0.860457


class PlainResNet34(nn.Module):
    """ImageNet ResNet34 with spatial GAP and study-level mean/max pooling."""

    def __init__(self):
        super().__init__()
        from torchvision.models import resnet34

        candidates = list(Path("/kaggle/input").rglob("resnet34.pth"))
        if len(candidates) != 1:
            raise RuntimeError(f"expected one resnet34.pth, found {candidates}")
        self.weight_path = candidates[0]
        network = resnet34(weights=None)
        state = torch.load(self.weight_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {str(key).removeprefix("module."): value for key, value in state.items()}
        missing, unexpected = network.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"ResNet34 state mismatch: missing={missing}, unexpected={unexpected}")
        self.encoder = nn.Sequential(*list(network.children())[:-2])
        self.head = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Dropout(.30),
            nn.Linear(1024, N_TARGET),
        )
        self.register_buffer("mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([.229, .224, .225])[None, :, None, None])

    def forward(self, windows):
        batch, count, channels, height, width = windows.shape
        image = windows.reshape(batch * count, channels, height, width)
        image = (image - self.mean) / self.std
        feature = self.encoder(image).mean((-2, -1)).reshape(batch, count, 512)
        study = torch.cat([feature.mean(1), feature.amax(1)], dim=1)
        return self.head(study)


def main_simple_resnet34():
    if DEVICE.type != "cuda":
        raise RuntimeError("GPU required")
    seed_all(RESNET_SEED)
    root = competition_root()
    train = pd.read_csv(root / "train.csv", dtype={"StudyInstanceUID": str})
    gold = train[TARGETS].notna().all(axis=1).to_numpy()
    if int(gold.sum()) != 58:
        raise RuntimeError("official gold count drift")

    labels = load_soft_labels(train)
    volumes, masks, corpus_ids = open_corpus()
    id_to_row = {uid: row for row, uid in enumerate(corpus_ids)}
    ids = train.StudyInstanceUID.astype(str).to_numpy()
    missing_ids = sorted(set(ids) - set(id_to_row))
    if missing_ids:
        raise RuntimeError(f"corpus misses {len(missing_ids)} studies")

    non_gold = np.flatnonzero(~gold)
    groups = report_groups(train.Report)
    fit_local, validation_local = list(GroupKFold(5).split(
        non_gold, groups=groups[non_gold]))[RESNET_FOLD]
    fit_index = non_gold[fit_local]
    validation_index = non_gold[validation_local]
    if set(groups[fit_index]).intersection(groups[validation_index]):
        raise RuntimeError("report-group leakage")

    model = PlainResNet34().to(DEVICE)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": 2e-5},
        {"params": model.head.parameters(), "lr": 5e-4},
    ], weight_decay=.03)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=RESNET_EPOCHS)
    dataset = QualityWindowStudies(
        ids[fit_index], labels[fit_index], id_to_row, volumes, masks, True)
    loader = DataLoader(
        dataset, batch_size=RESNET_BATCH, shuffle=True, num_workers=2,
        pin_memory=True, persistent_workers=True, drop_last=True,
        generator=torch.Generator().manual_seed(RESNET_SEED))
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    best_auc = -np.inf
    best_state = None
    history = []

    for epoch in range(RESNET_EPOCHS):
        model.train()
        total = 0.0
        seen = 0
        for windows, target in loader:
            windows = augment(windows.to(DEVICE, non_blocking=True))
            target = target.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    model(windows), target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(windows)
            seen += len(windows)
        scheduler.step()

        weak_prediction = predict(
            model, ids[validation_index], labels[validation_index],
            id_to_row, volumes, masks)
        weak_truth = (labels[validation_index] >= .5).astype(np.int8)
        weak_auc = macro_auc(weak_truth, weak_prediction)
        history.append({
            "epoch": epoch + 1,
            "loss": total / seen,
            "weak_validation_auc": float(weak_auc),
        })
        log(f"epoch {epoch + 1}/{RESNET_EPOCHS}: loss {total / seen:.5f}, "
            f"weak AUC {weak_auc:.6f}")
        if weak_auc > best_auc:
            best_auc = weak_auc
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    gold_index = np.flatnonzero(gold)
    gold_prediction = predict(
        model, ids[gold_index], labels[gold_index], id_to_row, volumes, masks)
    gold_truth = train.loc[gold, TARGETS].to_numpy(np.float32)
    gold_auc = macro_auc(gold_truth, gold_prediction)

    checkpoint = Path("/kaggle/working/simple_resnet34_fold0.pt")
    torch.save({
        "model": best_state,
        "fold": RESNET_FOLD,
        "seed": RESNET_SEED,
        "targets": TARGETS,
        "weak_auc": best_auc,
        "gold_auc": gold_auc,
    }, checkpoint)
    report = {
        "contract": "private one-fold screen; no competition submission",
        "candidate": "24-window ResNet34 mean-max pooling with plain BCE",
        "attribution": [
            "pytorch/resnet34 (CC0-1.0 ImageNet weights)",
            "dreaddevelopment/knee-raptor-corpus",
            "dreaddevelopment/knee-raptor-corpus-ext",
            "stevenleehans/rsna-knee-llm-report-labels",
        ],
        "fold": RESNET_FOLD,
        "seed": RESNET_SEED,
        "epochs": RESNET_EPOCHS,
        "fit_studies": int(len(fit_index)),
        "weak_validation_studies": int(len(validation_index)),
        "gold_fit_count": 0,
        "gold_evaluation_count": 58,
        "best_weak_auc": float(best_auc),
        "weak_validation_gate": RESNET_WEAK_GATE,
        "gold_auc": float(gold_auc),
        "gold_diagnostic_floor": RESNET_GOLD_FLOOR,
        "screen_passed": bool(
            best_auc >= RESNET_WEAK_GATE and gold_auc >= RESNET_GOLD_FLOOR),
        "history": history,
        "checkpoint_sha256": sha256(checkpoint),
        "pretrained_weight_sha256": sha256(model.weight_path),
        "submission_created": False,
    }
    Path("/kaggle/working/simple_resnet34_validation_report.json").write_text(
        json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


main_simple_resnet34()
