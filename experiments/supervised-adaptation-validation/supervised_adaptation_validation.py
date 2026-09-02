"""Private supervised final-stage adaptation of RadImageNet."""

ADAPT_SEED = 20260902
ADAPT_EPOCHS = 3
ADAPT_BATCH = 4
HEAD_SEEDS = (20260911, 20260912, 20260913, 20260914, 20260915)


def report_groups_adapt(series):
    normalized = (series.fillna("").astype(str).str.lower()
                  .str.replace(r"\s+", " ", regex=True).str.strip())
    return np.asarray([hashlib.sha256(text.encode()).hexdigest()[:24]
                       for text in normalized])


def fixed_group_split(train, gold):
    non_gold = np.flatnonzero(~gold)
    groups = report_groups_adapt(train["Report"])
    unique_groups = np.unique(groups[non_gold])
    shuffled = np.random.default_rng(ADAPT_SEED).permutation(unique_groups)
    validation_groups = set(shuffled[:max(1, int(.10 * len(shuffled)))])
    validation = non_gold[np.asarray([g in validation_groups for g in groups[non_gold]])]
    fitting = non_gold[np.asarray([g not in validation_groups for g in groups[non_gold]])]
    if set(groups[fitting]).intersection(groups[validation]):
        raise RuntimeError("report-group leakage")
    return fitting, validation, groups


def render_bags(indices, studies, by_study):
    bags = np.zeros((len(indices), len(SLOTS), IMG, IMG), np.uint8)
    masks = np.zeros((len(indices), len(SLOTS)), np.float32)
    for offset in range(0, len(indices), 32):
        batch_indices = indices[offset:offset + 32]
        jobs = []
        for index in batch_indices:
            records = by_study.get(studies[index], [])
            jobs.extend((choose(records, plane, fluid, False), False)
                        for plane, fluid in SLOTS)
        with ThreadPoolExecutor(max_workers=12) as pool:
            rendered = list(pool.map(lambda item: render_slot(*item), jobs))
        bags[offset:offset + len(batch_indices)] = np.stack(
            [item[0][N_SLICE // 2] for item in rendered]
        ).reshape(len(batch_indices), len(SLOTS), IMG, IMG)
        masks[offset:offset + len(batch_indices)] = np.asarray(
            [item[1] for item in rendered], np.float32
        ).reshape(len(batch_indices), len(SLOTS))
        if offset % 320 == 0:
            log(f"adaptation bags {offset}/{len(indices)}")
    if int((masks.sum(1) == 0).sum()):
        raise RuntimeError("study without a valid adaptation slice")
    return bags, masks


class StudyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(TOKEN_DIM)
        self.proj = nn.Linear(TOKEN_DIM, 256)
        self.att = nn.Linear(256, len(TARGETS))
        self.cls = nn.Parameter(torch.empty(len(TARGETS), 256))
        self.bias = nn.Parameter(torch.zeros(len(TARGETS)))
        nn.init.trunc_normal_(self.cls, std=.02)

    def forward(self, features, mask):
        hidden = torch.nn.functional.gelu(self.proj(self.norm(features.float())))
        attention = self.att(hidden).masked_fill(mask[:, :, None] < .5, -1e4)
        attention = torch.softmax(attention, dim=1)
        pooled = torch.einsum("btn,btf->bnf", attention, hidden)
        return (pooled * self.cls).sum(-1) + self.bias


class AdaptationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = load_encoder()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.encoder.backbone[7].parameters():
            parameter.requires_grad_(True)
        self.head = StudyAttention()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.backbone[:7].eval()
        self.encoder.backbone[7].train(mode)
        return self

    def forward(self, pixels, mask):
        batch, tokens, height, width = pixels.shape
        image = pixels.reshape(batch * tokens, height, width).float().div_(127.5).sub_(1)
        image = image[:, None].expand(-1, 3, -1, -1)
        features = self.encoder(image).reshape(batch, tokens, TOKEN_DIM)
        return self.head(features, mask)


@torch.no_grad()
def predict_bags(model, bags, masks):
    model.eval(); output = []
    for start in range(0, len(bags), 16):
        pixels = torch.from_numpy(bags[start:start + 16]).to(DEVICE)
        mask = torch.from_numpy(masks[start:start + 16]).to(DEVICE)
        with torch.autocast("cuda", enabled=DEVICE.type == "cuda"):
            output.append(torch.sigmoid(model(pixels, mask)).float().cpu().numpy())
    return np.concatenate(output)


def adapt_encoder(train_bags, train_masks, train_labels,
                  validation_bags, validation_masks, validation_labels):
    seed_all(ADAPT_SEED)
    model = AdaptationModel().to(DEVICE)
    layer4 = list(model.encoder.backbone[7].parameters())
    optimizer = torch.optim.AdamW([
        {"params": layer4, "lr": 1e-5},
        {"params": model.head.parameters(), "lr": 2e-4},
    ], weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    loader = DataLoader(TensorDataset(torch.from_numpy(train_bags),
                                      torch.from_numpy(train_masks),
                                      torch.from_numpy(train_labels)),
                        batch_size=ADAPT_BATCH, shuffle=True, num_workers=2,
                        pin_memory=True,
                        generator=torch.Generator().manual_seed(ADAPT_SEED))
    history = []
    for epoch in range(ADAPT_EPOCHS):
        model.train(); running = 0.; seen = 0
        for pixels, mask, target in loader:
            pixels = pixels.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=DEVICE.type == "cuda"):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    model(pixels, mask), target
                )
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(layer4, 1.)
            scaler.step(optimizer); scaler.update()
            running += float(loss.detach()) * len(pixels); seen += len(pixels)
        weak_prediction = predict_bags(model, validation_bags, validation_masks)
        weak_auc = macro_auc(validation_labels, weak_prediction)
        history.append({"epoch": epoch + 1, "train_bce": running / seen,
                        "weak_validation_auc": weak_auc})
        log(f"adapt epoch {epoch+1}: loss={running/seen:.5f}, weak AUC={weak_auc:.6f}")
    encoder = model.encoder; encoder.eval()
    for parameter in encoder.parameters(): parameter.requires_grad_(False)
    del model; gc.collect(); torch.cuda.empty_cache()
    return encoder, history


@torch.no_grad()
def encode_adapted(studies, by_study, encoder):
    features = np.zeros((len(studies), len(SLOTS) * N_SLICE, TOKEN_DIM), np.float16)
    masks = np.zeros((len(studies), len(SLOTS) * N_SLICE), np.float32)
    encoder.eval(); batch_studies = 24
    for start in range(0, len(studies), batch_studies):
        ids = studies[start:start + batch_studies]; jobs = []
        for uid in ids:
            records = by_study.get(uid, [])
            jobs.extend((choose(records, plane, fluid, False), False)
                        for plane, fluid in SLOTS)
        with ThreadPoolExecutor(max_workers=12) as pool:
            rendered = list(pool.map(lambda item: render_slot(*item), jobs))
        pixels = np.stack([item[0] for item in rendered]).reshape(
            len(ids), len(SLOTS) * N_SLICE, IMG, IMG
        )
        slot_valid = np.asarray([item[1] for item in rendered], np.float32).reshape(
            len(ids), len(SLOTS)
        )
        token_valid = np.repeat(slot_valid, N_SLICE, axis=1)
        flat = pixels.reshape(-1, IMG, IMG)
        valid_indices = np.flatnonzero(token_valid.reshape(-1) > 0)
        encoded = np.zeros((len(flat), TOKEN_DIM), np.float16)
        for offset in range(0, len(valid_indices), 96):
            index = valid_indices[offset:offset + 96]
            image = torch.from_numpy(flat[index]).to(DEVICE).float().div_(127.5).sub_(1)
            image = image[:, None].expand(-1, 3, -1, -1)
            with torch.autocast("cuda", enabled=DEVICE.type == "cuda"):
                encoded[index] = encoder(image).float().cpu().numpy().astype(np.float16)
        features[start:start + len(ids)] = encoded.reshape(len(ids), -1, TOKEN_DIM)
        masks[start:start + len(ids)] = token_valid
        if start % 240 == 0: log(f"adapted features {start}/{len(studies)}")
    return features, masks


def supervised_adaptation_main():
    if DEVICE.type != "cuda": raise RuntimeError("GPU required")
    root = competition_root()
    train = pd.read_csv(root / "train.csv", dtype={"StudyInstanceUID": str})
    studies = train.StudyInstanceUID.tolist()
    gold = train[TARGETS].notna().all(axis=1).to_numpy()
    if int(gold.sum()) != 58: raise RuntimeError("gold count drift")
    gold_index = np.flatnonzero(gold)
    fitting_index, validation_index, groups = fixed_group_split(train, gold)
    weak = pd.read_csv(unique_input("llm_labels_v4_blend.csv"), dtype={"StudyInstanceUID": str})
    aligned = train[["StudyInstanceUID"]].merge(weak, on="StudyInstanceUID", how="left", validate="one_to_one")
    labels = aligned[TARGETS].to_numpy(np.float32)
    if not np.isfinite(labels).all(): raise RuntimeError("weak labels incomplete")
    series = pd.read_csv(root / "train_series.csv", dtype=str)
    series_root = root / ("train_series" if (root / "train_series").is_dir() else "train_images")
    items = [(row.StudyInstanceUID, row.SeriesInstanceUID, row.Anatomical_Plane,
              series_root / row.StudyInstanceUID / row.SeriesInstanceUID)
             for row in series.itertuples(index=False)]
    with ThreadPoolExecutor(max_workers=16) as pool:
        records = [record for record in pool.map(inspect_series, items) if record]
    by_study = {}
    for record in records: by_study.setdefault(record["study"], []).append(record)
    train_bags, train_masks = render_bags(fitting_index, studies, by_study)
    validation_bags, validation_masks = render_bags(validation_index, studies, by_study)
    encoder, history = adapt_encoder(train_bags, train_masks, labels[fitting_index],
                                     validation_bags, validation_masks, labels[validation_index])
    del train_bags, train_masks, validation_bags, validation_masks
    gc.collect()
    checkpoint_path = Path("/kaggle/working/adapted_radimagenet_layer4.pt")
    torch.save({key: value.detach().cpu() for key, value in encoder.state_dict().items()}, checkpoint_path)
    adapted_sha = sha256(checkpoint_path)
    adapted_features, adapted_masks = encode_adapted(studies, by_study, encoder)
    del encoder; gc.collect(); torch.cuda.empty_cache()
    package = np.load(unique_input("incumbent_radimagenet_features.npz"), allow_pickle=False)
    if package["ids"].astype(str).tolist() != studies: raise RuntimeError("incumbent identity drift")
    if str(package["encoder_sha256"]) != RAD_SHA: raise RuntimeError("incumbent hash drift")
    control_features, control_masks = package["features"], package["masks"]
    if not np.array_equal(adapted_masks, control_masks): raise RuntimeError("token mask contract drift")
    y_gold = train.loc[gold, TARGETS].to_numpy(float)
    results, control_gold, adapted_gold = [], [], []
    for seed in HEAD_SEEDS:
        control_prediction, control_weak = train_head(control_features, control_masks, labels,
                                                       fitting_index, validation_index, seed)
        adapted_prediction, adapted_weak = train_head(adapted_features, adapted_masks, labels,
                                                       fitting_index, validation_index, seed)
        control_auc = macro_auc(y_gold, control_prediction[gold_index])
        adapted_auc = macro_auc(y_gold, adapted_prediction[gold_index])
        results.append({"seed": seed, "control_weak_validation_auc": control_weak,
                        "adapted_weak_validation_auc": adapted_weak,
                        "control_gold_auc": control_auc, "adapted_gold_auc": adapted_auc,
                        "delta": adapted_auc - control_auc})
        control_gold.append(control_prediction[gold_index]); adapted_gold.append(adapted_prediction[gold_index])
        log(f"seed {seed}: delta {adapted_auc-control_auc:+.6f}")
    control_mean, adapted_mean = np.mean(control_gold, axis=0), np.mean(adapted_gold, axis=0)
    control_auc, adapted_auc = macro_auc(y_gold, control_mean), macro_auc(y_gold, adapted_mean)
    bootstrap = paired_bootstrap(y_gold, control_mean, adapted_mean)
    deltas = np.asarray([result["delta"] for result in results])
    gate = {"delta_at_least_0_005": adapted_auc - control_auc >= .005,
            "positive_seeds_at_least_4": int((deltas > 0).sum()) >= 4,
            "bootstrap_p_at_least_0_95": bootstrap["p_improve"] >= .95,
            "bootstrap_q025_positive": bootstrap["q025"] > 0}
    report = {"contract": "private supervised layer4 adaptation; gold evaluation only; no submission",
              "candidate": "RadImageNet ResNet-50 final-stage supervised multi-plane adaptation",
              "initial_checkpoint_sha256": RAD_SHA, "adapted_checkpoint_sha256": adapted_sha,
              "adapted_checkpoint_size": checkpoint_path.stat().st_size,
              "adaptation_seed": ADAPT_SEED, "adaptation_epochs": ADAPT_EPOCHS,
              "adaptation_slice_contract": "six central clinically selected slices",
              "final_feature_contract": "same fixed 18-token contract as control",
              "fitting_studies": int(len(fitting_index)), "weak_validation_studies": int(len(validation_index)),
              "report_group_overlap": int(len(set(groups[fitting_index]).intersection(groups[validation_index]))),
              "adaptation_history": history, "gold_fit_count": 0, "gold_evaluation_count": 58,
              "seeds": results,
              "ensemble": {"control_gold_auc": control_auc, "adapted_gold_auc": adapted_auc,
                           "delta": adapted_auc - control_auc},
              "bootstrap": bootstrap, "gate": gate, "passed": bool(all(gate.values())),
              "submission_created": False}
    Path("/kaggle/working/supervised_adaptation_validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


supervised_adaptation_main()
