"""Private adjacent-slice sequence-context self-supervision."""

SSL_SEED = 20260921
SSL_EPOCHS = 5
SSL_BATCH = 16
HEAD_SEEDS = (20260921, 20260922, 20260923, 20260924, 20260925)


def sequence_groups(series):
    normalized = (series.fillna("").astype(str).str.lower()
                  .str.replace(r"\s+", " ", regex=True).str.strip())
    return np.asarray([hashlib.sha256(text.encode()).hexdigest()[:24]
                       for text in normalized])


def sequence_split(train, gold):
    non_gold = np.flatnonzero(~gold); groups = sequence_groups(train["Report"])
    unique = np.unique(groups[non_gold])
    shuffled = np.random.default_rng(SSL_SEED).permutation(unique)
    validation_groups = set(shuffled[:max(1, int(.10 * len(shuffled)))])
    validation = non_gold[np.asarray([g in validation_groups for g in groups[non_gold]])]
    fitting = non_gold[np.asarray([g not in validation_groups for g in groups[non_gold]])]
    if set(groups[fitting]).intersection(groups[validation]):
        raise RuntimeError("report-group leakage")
    return fitting, validation, groups


def choose_sequence_record(records, offset):
    planes = ("Sagittal", "Coronal", "Axial")
    for shift in range(3):
        plane = planes[(offset + shift) % 3]
        record = choose(records, plane, True, True)
        if record is not None: return record
        candidates = [item for item in records if item["plane"] == plane]
        if candidates: return max(candidates, key=lambda item: item["n"])
    return max(records, key=lambda item: item["n"]) if records else None


def render_pair(record):
    if record is None: return None
    rows = ordered_headers(record)
    if len(rows) < 4: return None
    center = len(rows) // 2
    first_index = max(0, min(len(rows) - 2, center - 1))
    images = []
    for index in (first_index, first_index + 1):
        try:
            image, _ = read_pixels(rows[index][0])
        except Exception:
            return None
        low, high = np.percentile(image, (1, 99))
        image = np.clip((image - low) / max(high - low, 1e-6), 0, 1)
        h, w = image.shape; side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        image = cv2.resize(image[y0:y0 + side, x0:x0 + side], (IMG, IMG),
                           interpolation=cv2.INTER_AREA)
        images.append(np.round(image * 255).astype(np.uint8))
    return np.stack(images)


def build_pairs(indices, studies, by_study):
    pairs, kept = [], []
    jobs = []
    for offset, index in enumerate(indices):
        record = choose_sequence_record(by_study.get(studies[index], []), offset)
        jobs.append(record)
    with ThreadPoolExecutor(max_workers=12) as pool:
        rendered = list(pool.map(render_pair, jobs))
    for index, pair in zip(indices, rendered):
        if pair is not None:
            kept.append(index); pairs.append(pair)
    if len(pairs) < .95 * len(indices):
        raise RuntimeError(f"adjacent-pair coverage too low: {len(pairs)}/{len(indices)}")
    return np.stack(pairs), np.asarray(kept, dtype=np.int64)


class SequenceSSL(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder = load_encoder()
        for parameter in self.encoder.parameters(): parameter.requires_grad_(False)
        for parameter in self.encoder.backbone[7].parameters(): parameter.requires_grad_(True)
        self.projector = nn.Sequential(nn.Linear(TOKEN_DIM, 512), nn.GELU(),
                                       nn.Linear(512, 128))
        self.order = nn.Sequential(nn.Linear(TOKEN_DIM * 2, 256), nn.GELU(),
                                   nn.Linear(256, 2))

    def train(self, mode=True):
        super().train(mode); self.encoder.backbone[:7].eval()
        self.encoder.backbone[7].train(mode); return self

    def encode(self, image):
        image = image.float().div_(127.5).sub_(1)[:, None].expand(-1, 3, -1, -1)
        return self.encoder(image)


def symmetric_infonce(z1, z2, temperature=.10):
    z1 = torch.nn.functional.normalize(z1, dim=1)
    z2 = torch.nn.functional.normalize(z2, dim=1)
    logits = z1 @ z2.T / temperature
    target = torch.arange(len(z1), device=z1.device)
    return .5 * (torch.nn.functional.cross_entropy(logits, target) +
                 torch.nn.functional.cross_entropy(logits.T, target))


def pretrain_sequence(pairs):
    seed_all(SSL_SEED); model = SequenceSSL().to(DEVICE)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=5e-6, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    loader = DataLoader(TensorDataset(torch.from_numpy(pairs)), batch_size=SSL_BATCH,
                        shuffle=True, num_workers=2, pin_memory=True,
                        generator=torch.Generator().manual_seed(SSL_SEED))
    history = []
    for epoch in range(SSL_EPOCHS):
        model.train(); total = contrast_total = order_total = 0.; seen = 0
        for (pair,) in loader:
            pair = pair.to(DEVICE, non_blocking=True); batch = len(pair)
            reverse = torch.rand(batch, device=DEVICE) < .5
            first = pair[:, 0].clone(); second = pair[:, 1].clone()
            ordered_first = torch.where(reverse[:, None, None], second, first)
            ordered_second = torch.where(reverse[:, None, None], first, second)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=DEVICE.type == "cuda"):
                f1, f2 = model.encode(ordered_first), model.encode(ordered_second)
                contrast = symmetric_infonce(model.projector(f1), model.projector(f2))
                order_loss = torch.nn.functional.cross_entropy(
                    model.order(torch.cat((f1, f2), dim=1)), reverse.long())
                loss = contrast + .25 * order_loss
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.)
            scaler.step(optimizer); scaler.update()
            total += float(loss.detach()) * batch
            contrast_total += float(contrast.detach()) * batch
            order_total += float(order_loss.detach()) * batch; seen += batch
        history.append({"epoch": epoch + 1, "loss": total / seen,
                        "infonce": contrast_total / seen,
                        "relative_order_ce": order_total / seen})
        log(f"sequence SSL epoch {epoch+1}: loss={total/seen:.5f}")
    encoder = model.encoder; encoder.eval()
    for parameter in encoder.parameters(): parameter.requires_grad_(False)
    del model; gc.collect(); torch.cuda.empty_cache()
    return encoder, history


@torch.no_grad()
def encode_sequence_encoder(studies, by_study, encoder):
    features = np.zeros((len(studies), len(SLOTS)*N_SLICE, TOKEN_DIM), np.float16)
    masks = np.zeros((len(studies), len(SLOTS)*N_SLICE), np.float32)
    for start in range(0, len(studies), 24):
        ids = studies[start:start+24]; jobs = []
        for uid in ids:
            records = by_study.get(uid, [])
            jobs.extend((choose(records, plane, fluid, False), False) for plane, fluid in SLOTS)
        with ThreadPoolExecutor(max_workers=12) as pool:
            rendered = list(pool.map(lambda item: render_slot(*item), jobs))
        pixels = np.stack([item[0] for item in rendered]).reshape(len(ids), len(SLOTS)*N_SLICE, IMG, IMG)
        slot_valid = np.asarray([item[1] for item in rendered], np.float32).reshape(len(ids), len(SLOTS))
        token_valid = np.repeat(slot_valid, N_SLICE, axis=1)
        flat = pixels.reshape(-1, IMG, IMG); valid = np.flatnonzero(token_valid.reshape(-1) > 0)
        encoded = np.zeros((len(flat), TOKEN_DIM), np.float16)
        for offset in range(0, len(valid), 96):
            index = valid[offset:offset+96]
            image = torch.from_numpy(flat[index]).to(DEVICE).float().div_(127.5).sub_(1)
            image = image[:, None].expand(-1, 3, -1, -1)
            with torch.autocast("cuda", enabled=DEVICE.type == "cuda"):
                encoded[index] = encoder(image).float().cpu().numpy().astype(np.float16)
        features[start:start+len(ids)] = encoded.reshape(len(ids), -1, TOKEN_DIM)
        masks[start:start+len(ids)] = token_valid
        if start % 240 == 0: log(f"sequence encoder features {start}/{len(studies)}")
    return features, masks


def sequence_context_main():
    if DEVICE.type != "cuda": raise RuntimeError("GPU required")
    root = competition_root()
    train = pd.read_csv(root / "train.csv", dtype={"StudyInstanceUID": str})
    studies = train.StudyInstanceUID.tolist(); gold = train[TARGETS].notna().all(axis=1).to_numpy()
    if int(gold.sum()) != 58: raise RuntimeError("gold count drift")
    gold_index = np.flatnonzero(gold)
    fitting_index, validation_index, groups = sequence_split(train, gold)
    weak = pd.read_csv(unique_input("llm_labels_v4_blend.csv"), dtype={"StudyInstanceUID": str})
    aligned = train[["StudyInstanceUID"]].merge(weak, on="StudyInstanceUID", how="left", validate="one_to_one")
    labels = aligned[TARGETS].to_numpy(np.float32)
    if not np.isfinite(labels).all(): raise RuntimeError("weak labels incomplete")
    series = pd.read_csv(root / "train_series.csv", dtype=str)
    series_root = root / ("train_series" if (root / "train_series").is_dir() else "train_images")
    items = [(row.StudyInstanceUID, row.SeriesInstanceUID, row.Anatomical_Plane,
              series_root / row.StudyInstanceUID / row.SeriesInstanceUID) for row in series.itertuples(index=False)]
    with ThreadPoolExecutor(max_workers=16) as pool:
        records = [record for record in pool.map(inspect_series, items) if record]
    by_study = {}
    for record in records: by_study.setdefault(record["study"], []).append(record)
    pairs, pair_indices = build_pairs(fitting_index, studies, by_study)
    encoder, history = pretrain_sequence(pairs)
    del pairs; gc.collect()
    checkpoint = Path("/kaggle/working/sequence_context_radimagenet.pt")
    torch.save({key: value.detach().cpu() for key, value in encoder.state_dict().items()}, checkpoint)
    adapted_sha = sha256(checkpoint)
    adapted_features, adapted_masks = encode_sequence_encoder(studies, by_study, encoder)
    del encoder; gc.collect(); torch.cuda.empty_cache()
    package = np.load(unique_input("incumbent_radimagenet_features.npz"), allow_pickle=False)
    if package["ids"].astype(str).tolist() != studies: raise RuntimeError("identity drift")
    if str(package["encoder_sha256"]) != RAD_SHA: raise RuntimeError("control hash drift")
    control_features, control_masks = package["features"], package["masks"]
    if not np.array_equal(adapted_masks, control_masks): raise RuntimeError("mask drift")
    y_gold = train.loc[gold, TARGETS].to_numpy(float)
    results, control_gold, adapted_gold = [], [], []
    for seed in HEAD_SEEDS:
        cp, cw = train_head(control_features, control_masks, labels, fitting_index, validation_index, seed)
        ap, aw = train_head(adapted_features, adapted_masks, labels, fitting_index, validation_index, seed)
        ca, aa = macro_auc(y_gold, cp[gold_index]), macro_auc(y_gold, ap[gold_index])
        results.append({"seed": seed, "control_weak_validation_auc": cw,
                        "sequence_weak_validation_auc": aw, "control_gold_auc": ca,
                        "sequence_gold_auc": aa, "delta": aa-ca})
        control_gold.append(cp[gold_index]); adapted_gold.append(ap[gold_index])
        log(f"seed {seed}: sequence delta {aa-ca:+.6f}")
    control_mean, adapted_mean = np.mean(control_gold, 0), np.mean(adapted_gold, 0)
    control_auc, adapted_auc = macro_auc(y_gold, control_mean), macro_auc(y_gold, adapted_mean)
    bootstrap = paired_bootstrap(y_gold, control_mean, adapted_mean)
    deltas = np.asarray([result["delta"] for result in results])
    gate = {"delta_at_least_0_005": adapted_auc-control_auc >= .005,
            "positive_seeds_at_least_4": int((deltas > 0).sum()) >= 4,
            "bootstrap_p_at_least_0_95": bootstrap["p_improve"] >= .95,
            "bootstrap_q025_positive": bootstrap["q025"] > 0}
    report = {"contract": "private adjacent-slice sequence-context SSL; no submission",
              "candidate": "RadImageNet final-stage adjacent-slice InfoNCE and order pretraining",
              "initial_checkpoint_sha256": RAD_SHA, "adapted_checkpoint_sha256": adapted_sha,
              "adapted_checkpoint_size": checkpoint.stat().st_size, "ssl_seed": SSL_SEED,
              "ssl_epochs": SSL_EPOCHS, "pretraining_pairs": int(len(pair_indices)),
              "pretraining_pair_coverage": float(len(pair_indices)/len(fitting_index)),
              "pretraining_uses_report_labels": False, "fitting_studies": int(len(fitting_index)),
              "weak_validation_studies": int(len(validation_index)),
              "report_group_overlap": int(len(set(groups[fitting_index]).intersection(groups[validation_index]))),
              "ssl_history": history, "gold_fit_count": 0, "gold_evaluation_count": 58,
              "seeds": results, "ensemble": {"control_gold_auc": control_auc,
              "sequence_gold_auc": adapted_auc, "delta": adapted_auc-control_auc},
              "bootstrap": bootstrap, "gate": gate, "passed": bool(all(gate.values())),
              "submission_created": False}
    Path("/kaggle/working/sequence_context_validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


sequence_context_main()
