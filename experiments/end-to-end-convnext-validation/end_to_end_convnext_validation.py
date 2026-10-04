"""One-fold screen for end-to-end ConvNeXt-Tiny knee-MRI fine-tuning.

This is deliberately not another frozen-feature head. It trains the complete encoder
on six MRI sequence slots, samples adjacent 2.5D slice triplets, and averages three
physical-depth views at validation. The 58 organizer-labelled studies are never fitted.
"""

from sklearn.model_selection import GroupKFold

SEED = 20261004
FOLD = 0
EPOCHS = 10
BATCH = 4
N_SLOT = 6
N_DEPTH = 9
N_TARGET = 12
FEATURE_DIM = 768


def report_groups(series):
    normalized = (series.fillna("").astype(str).str.lower()
                  .str.replace(r"\s+", " ", regex=True).str.strip())
    return np.asarray([hashlib.sha256(text.encode()).hexdigest()[:24]
                       for text in normalized])


def find_cache_root():
    base = Path("/kaggle/input")
    matches = []
    for path in base.rglob("cache_meta.json"):
        parent = path.parent
        if list(parent.glob("train.s*of*_cache.npy")):
            matches.append(parent)
    if len(matches) != 1:
        raise RuntimeError(f"expected one mounted pixel cache, found {matches}")
    return matches[0]


class ShardedCache:
    def __init__(self, root):
        self.root = Path(root)
        self.shards = []
        self.lookup = {}
        for cache_path in sorted(self.root.glob("train.s*of*_cache.npy")):
            stem = cache_path.name.removesuffix("_cache.npy")
            studies_path = self.root / f"{stem}_studies.csv"
            mask_path = self.root / f"{stem}_mask.npy"
            if not studies_path.is_file() or not mask_path.is_file():
                raise RuntimeError(f"incomplete cache shard {stem}")
            ids = pd.read_csv(studies_path, dtype=str)["StudyInstanceUID"].tolist()
            pixels = np.load(cache_path, mmap_mode="r")
            masks = np.load(mask_path, mmap_mode="r")
            if pixels.shape != (len(ids), N_SLOT, N_DEPTH, 224, 224):
                raise RuntimeError(f"unexpected {stem} shape {pixels.shape}")
            shard_id = len(self.shards)
            self.shards.append((pixels, masks))
            for row, uid in enumerate(ids):
                if uid in self.lookup:
                    raise RuntimeError(f"duplicate cached study {uid}")
                self.lookup[uid] = (shard_id, row)

    def get(self, uid):
        shard_id, row = self.lookup[str(uid)]
        pixels, masks = self.shards[shard_id]
        return np.asarray(pixels[row]), np.asarray(masks[row], np.float32)


class StudyRows(torch.utils.data.Dataset):
    def __init__(self, ids, labels, weights, cache, train):
        self.ids = np.asarray(ids)
        self.labels = np.asarray(labels, np.float32)
        self.weights = np.asarray(weights, np.float32)
        self.cache = cache
        self.train = bool(train)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        pixels, mask = self.cache.get(self.ids[index])
        if self.train:
            group = np.random.randint(0, 3)
            view = pixels[:, group * 3:(group + 1) * 3].copy()
        else:
            view = pixels.copy()
        return (torch.from_numpy(view), torch.from_numpy(mask.copy()),
                torch.from_numpy(self.labels[index]), torch.from_numpy(self.weights[index]))


class TargetSlotHead(nn.Module):
    def __init__(self, dim=FEATURE_DIM, hidden=384):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot = nn.Parameter(torch.randn(N_SLOT, hidden) * .02)
        self.query = nn.Parameter(torch.randn(N_TARGET, hidden) * .02)
        self.out = nn.Linear(hidden, N_TARGET)
        self.dropout = nn.Dropout(.30)

    def forward(self, feature, mask):
        hidden = self.proj(feature) + self.slot
        score = torch.einsum("bsh,th->bts", hidden, self.query) / hidden.shape[-1] ** .5
        score = score.masked_fill(mask[:, None, :] < .5, -1e4).softmax(-1)
        pooled = self.dropout(torch.einsum("bts,bsh->bth", score, hidden))
        return (pooled * self.out.weight[None]).sum(-1) + self.out.bias


class EndToEndModel(nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        weights = sorted(Path("/kaggle/input").rglob("model.safetensors"))
        weights = [path for path in weights if "convnext" in str(path).lower()]
        if len(weights) != 1:
            raise RuntimeError(f"expected one mounted ConvNeXt checkpoint, found {weights}")
        self.encoder = timm.create_model(
            "convnext_tiny.fb_in22k_ft_in1k", pretrained=True,
            pretrained_cfg_overlay={"file": str(weights[0])},
            num_classes=0, global_pool="avg")
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(True)
        self.head = TargetSlotHead()
        self.register_buffer("mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([.229, .224, .225])[None, :, None, None])

    def encode(self, images):
        images = images.float().div_(255.0)
        images = (images - self.mean) / self.std
        return self.encoder(images)

    def forward(self, pixels, mask):
        batch, slots, channels, height, width = pixels.shape
        image = pixels.reshape(batch * slots, channels, height, width)
        feature = self.encode(image).reshape(batch, slots, FEATURE_DIM)
        return self.head(feature, mask)


def augment(pixels):
    if torch.rand(()) < .8:
        gain = .90 + .20 * torch.rand((len(pixels), 1, 1, 1, 1), device=pixels.device)
        shift = (-8. + 16. * torch.rand((len(pixels), 1, 1, 1, 1), device=pixels.device))
        pixels = (pixels.float() * gain + shift).clamp_(0, 255)
    if torch.rand(()) < .35:
        pixels = pixels + torch.randn_like(pixels.float()) * 3.0
    return pixels.clamp_(0, 255)


@torch.no_grad()
def predict(model, ids, cache):
    model.eval()
    output = []
    neutral = np.zeros((len(ids), N_TARGET), np.float32)
    unit = np.ones_like(neutral)
    loader = DataLoader(StudyRows(ids, neutral, unit, cache, False), batch_size=6,
                        shuffle=False, num_workers=2, pin_memory=True)
    for pixels, mask, _, _ in loader:
        pixels = pixels.to(DEVICE, non_blocking=True)
        mask = mask.to(DEVICE, non_blocking=True)
        logits = []
        for group in range(3):
            view = pixels[:, :, group * 3:(group + 1) * 3]
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                logits.append(model(view, mask).float())
        output.append(torch.sigmoid(torch.stack(logits).mean(0)).cpu().numpy())
    return np.concatenate(output)


def load_training_targets(train):
    source = pd.read_csv(unique_input("llm_labels_v4_blend.csv"),
                         dtype={"StudyInstanceUID": str})
    aligned = train[["StudyInstanceUID"]].merge(source, on="StudyInstanceUID", how="left",
                                                  validate="one_to_one")
    labels = aligned[TARGETS].to_numpy(np.float32)
    if not np.isfinite(labels).all():
        raise RuntimeError("Steven v4 labels do not cover every training study")
    weights = np.ones_like(labels)
    return labels, weights


def main_end_to_end():
    if DEVICE.type != "cuda":
        raise RuntimeError("GPU required")
    seed_all(SEED)
    root = competition_root()
    train = pd.read_csv(root / "train.csv", dtype={"StudyInstanceUID": str})
    gold = train[TARGETS].notna().all(axis=1).to_numpy()
    if int(gold.sum()) != 58:
        raise RuntimeError("official gold count drift")
    labels, weights = load_training_targets(train)
    cache = ShardedCache(find_cache_root())
    ids = train.StudyInstanceUID.astype(str).to_numpy()
    missing = sorted(set(ids) - set(cache.lookup))
    if missing:
        raise RuntimeError(f"cache misses {len(missing)} training studies")

    non_gold = np.flatnonzero(~gold)
    groups = report_groups(train.Report)
    splits = list(GroupKFold(5).split(non_gold, groups=groups[non_gold]))
    train_local, validation_local = splits[FOLD]
    train_index = non_gold[train_local]
    validation_index = non_gold[validation_local]
    if set(groups[train_index]).intersection(groups[validation_index]):
        raise RuntimeError("report-group leakage")

    model = EndToEndModel().to(DEVICE)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": 1e-5},
        {"params": model.head.parameters(), "lr": 4e-4},
    ], weight_decay=2e-4)
    loader = DataLoader(StudyRows(ids[train_index], labels[train_index], weights[train_index],
                                  cache, True), batch_size=BATCH, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True,
                        generator=torch.Generator().manual_seed(SEED))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    best_state, best_auc, history = None, -np.inf, []

    for epoch in range(EPOCHS):
        model.train(); running = 0.; seen = 0
        for pixels, mask, target, weight in loader:
            pixels = augment(pixels.to(DEVICE, non_blocking=True))
            mask = mask.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
            weight = weight.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                raw = torch.nn.functional.binary_cross_entropy_with_logits(
                    model(pixels, mask), target, reduction="none")
                loss = (raw * weight).sum() / weight.sum().clamp_min(1.)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            running += float(loss.detach()) * len(pixels); seen += len(pixels)
        scheduler.step()
        weak_prediction = predict(model, ids[validation_index], cache)
        weak_truth = (labels[validation_index] >= .5).astype(np.int8)
        weak_auc = macro_auc(weak_truth, weak_prediction)
        history.append({"epoch": epoch + 1, "loss": running / seen,
                        "weak_validation_auc": weak_auc})
        log(f"epoch {epoch+1}/{EPOCHS}: loss {running/seen:.5f}, weak AUC {weak_auc:.6f}")
        if weak_auc > best_auc:
            best_auc = weak_auc
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    gold_index = np.flatnonzero(gold)
    gold_prediction = predict(model, ids[gold_index], cache)
    gold_truth = train.loc[gold, TARGETS].to_numpy(np.float32)
    gold_auc = macro_auc(gold_truth, gold_prediction)
    checkpoint = Path("/kaggle/working/convnext_tiny_full_finetune_fold0.pt")
    torch.save({"model": best_state, "fold": FOLD, "seed": SEED,
                "targets": TARGETS, "weak_auc": best_auc, "gold_auc": gold_auc}, checkpoint)
    report = {
        "contract": "private one-fold screen; no competition submission",
        "candidate": "ConvNeXt-Tiny ImageNet-22k/1k 2.5D six-slot fine-tuning",
        "label_source": "stevenleehans llm_labels_v4_blend.csv",
        "attribution": [
            "ambrosm/convnext-tiny-fb-in22k-ft-in1k/PyTorch/default/1",
            "stevenleehans/rsna-knee-500gb-to-11gib-cpu-pixel-cache",
            "stevenleehans/rsna-knee-llm-report-labels",
        ],
        "fold": FOLD, "seed": SEED, "epochs": EPOCHS,
        "fit_studies": int(len(train_index)),
        "weak_validation_studies": int(len(validation_index)),
        "gold_fit_count": 0, "gold_evaluation_count": 58,
        "best_weak_auc": float(best_auc), "gold_auc": float(gold_auc),
        "history": history, "checkpoint_sha256": sha256(checkpoint),
    }
    Path("/kaggle/working/end_to_end_validation_report.json").write_text(
        json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


main_end_to_end()
