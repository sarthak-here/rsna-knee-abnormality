"""Private MARS 3D MRI foundation-model validation.

The candidate replaces 2D RadImageNet tokens with three volumetric embeddings
from the official Apache-2.0 MARS 3D Swin Transformer. All fitting uses only
non-gold report groups; the 58 organizer-labelled studies are evaluation-only.
"""

import subprocess
import sys
import urllib.request

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "monai==1.4.0"])
from monai.networks.nets.swin_unetr import SwinTransformer


MARS_URL = "https://drive.usercontent.google.com/download?id=1__lWJfBaCSQqkyPvxpQqK-MWH_d__bWz&export=download&confirm=t"
MARS_SHA256 = "88388a59d92935a15d30bd41c0e9dbece99ba78db73cb374d44290fd6566ccc3"
MARS_SIZE = 82121520
MARS_REPOSITORY = "zqiuak/MARS"
MARS_FEATURE_DIM = 768
VOLUME_DEPTH = 32
VOLUME_SIZE = 96


class MARSBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.swinViT = SwinTransformer(
            in_chans=1,
            embed_dim=48,
            window_size=(7, 7, 7),
            patch_size=(2, 2, 2),
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=nn.LayerNorm,
            use_checkpoint=False,
            spatial_dims=3,
            use_v2=True,
        )

    def forward(self, volume):
        feature = self.swinViT(volume)[4]
        return torch.nn.functional.adaptive_avg_pool3d(feature, 1).flatten(1)


def load_mars():
    path = Path("/kaggle/working/MARS.pt")
    if not path.is_file():
        urllib.request.urlretrieve(MARS_URL, path)
    if path.stat().st_size != MARS_SIZE:
        raise RuntimeError("MARS checkpoint size drift")
    if sha256(path) != MARS_SHA256:
        raise RuntimeError("MARS checkpoint hash drift")
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = MARSBackbone()
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"MARS state mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.eval().to(DEVICE)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def render_volume(record):
    if record is None:
        return np.zeros((VOLUME_DEPTH, VOLUME_SIZE, VOLUME_SIZE), np.float32), 0
    rows = ordered_headers(record)
    if not rows:
        return np.zeros((VOLUME_DEPTH, VOLUME_SIZE, VOLUME_SIZE), np.float32), 0
    source_indices = np.linspace(0, len(rows) - 1, VOLUME_DEPTH).round().astype(int)
    images = []
    for index in source_indices:
        try:
            image, spacing = read_pixels(rows[index][0])
            height, width = image.shape
            if spacing is None:
                spacing = np.asarray((0.5, 0.5))
            crop_h = min(height, max(16, int(round(140.0 / max(spacing[0], 1e-3)))))
            crop_w = min(width, max(16, int(round(140.0 / max(spacing[1], 1e-3)))))
            y0, x0 = (height - crop_h) // 2, (width - crop_w) // 2
            image = image[y0:y0 + crop_h, x0:x0 + crop_w]
            image = cv2.resize(image, (VOLUME_SIZE, VOLUME_SIZE), interpolation=cv2.INTER_AREA)
            images.append(image.astype(np.float32))
        except Exception:
            return np.zeros((VOLUME_DEPTH, VOLUME_SIZE, VOLUME_SIZE), np.float32), 0
    volume = np.stack(images)
    nonzero = volume[np.isfinite(volume) & (volume != 0)]
    if len(nonzero) < 100:
        return np.zeros_like(volume), 0
    mean, standard_deviation = float(nonzero.mean()), float(nonzero.std())
    if standard_deviation < 1e-6:
        return np.zeros_like(volume), 0
    foreground = np.isfinite(volume) & (volume != 0)
    normalized = np.zeros_like(volume, dtype=np.float32)
    normalized[foreground] = (volume[foreground] - mean) / standard_deviation
    normalized = np.clip(normalized, -8.0, 8.0).astype(np.float32)
    return normalized, 1


@torch.no_grad()
def encode_mars(studies, by_study):
    model = load_mars()
    features = np.zeros((len(studies), len(PLANES), MARS_FEATURE_DIM), np.float16)
    masks = np.zeros((len(studies), len(PLANES)), np.float32)
    batch_studies = 8
    for start in range(0, len(studies), batch_studies):
        ids = studies[start:start + batch_studies]
        jobs = []
        for uid in ids:
            records = by_study.get(uid, [])
            jobs.extend(choose(records, plane, True, True) for plane in PLANES)
        with ThreadPoolExecutor(max_workers=8) as pool:
            rendered = list(pool.map(render_volume, jobs))
        volumes = np.stack([item[0] for item in rendered])
        valid = np.asarray([item[1] for item in rendered], np.float32)
        encoded = np.zeros((len(volumes), MARS_FEATURE_DIM), np.float16)
        valid_indices = np.flatnonzero(valid > 0)
        for offset in range(0, len(valid_indices), 4):
            indices = valid_indices[offset:offset + 4]
            tensor = torch.from_numpy(volumes[indices]).to(DEVICE).unsqueeze(1)
            with torch.autocast("cuda", enabled=DEVICE.type == "cuda"):
                encoded[indices] = model(tensor).float().cpu().numpy().astype(np.float16)
        features[start:start + len(ids)] = encoded.reshape(len(ids), len(PLANES), MARS_FEATURE_DIM)
        masks[start:start + len(ids)] = valid.reshape(len(ids), len(PLANES))
        if start % (batch_studies * 20) == 0:
            log(f"MARS encoded {start}/{len(studies)}")
    del model
    gc.collect(); torch.cuda.empty_cache()
    return features, masks


class MARSAttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(MARS_FEATURE_DIM)
        self.proj = nn.Linear(MARS_FEATURE_DIM, 256)
        self.attention = nn.Linear(256, len(TARGETS))
        self.classifier = nn.Parameter(torch.empty(len(TARGETS), 256))
        self.bias = nn.Parameter(torch.zeros(len(TARGETS)))
        nn.init.trunc_normal_(self.classifier, std=0.02)

    def forward(self, features, mask):
        hidden = torch.nn.functional.gelu(self.proj(self.norm(features.float())))
        attention = self.attention(hidden)
        attention = attention.masked_fill(mask[:, :, None] < 0.5, -1e4)
        attention = torch.softmax(attention, dim=1)
        pooled = torch.einsum("btn,btf->bnf", attention, hidden)
        return (pooled * self.classifier).sum(-1) + self.bias


@torch.no_grad()
def mars_predict(model, features, masks, indices):
    model.eval(); output = []
    for start in range(0, len(indices), 128):
        index = indices[start:start + 128]
        x = torch.from_numpy(features[index]).to(DEVICE)
        mask = torch.from_numpy(masks[index]).to(DEVICE)
        output.append(torch.sigmoid(model(x, mask)).cpu().numpy())
    return np.concatenate(output)


def train_mars_head(features, masks, labels, train_index, validation_index, seed):
    seed_all(seed)
    model = MARSAttentionHead().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    dataset = TensorDataset(
        torch.from_numpy(features[train_index]),
        torch.from_numpy(masks[train_index]),
        torch.from_numpy(labels[train_index]),
    )
    loader = DataLoader(
        dataset, batch_size=64, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    best_state, best_score = None, -1.0
    for _ in range(20):
        model.train()
        for x, mask, target in loader:
            x, mask, target = x.to(DEVICE), mask.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(x, mask), target.float()
            )
            loss.backward(); optimizer.step()
        prediction = mars_predict(model, features, masks, validation_index)
        score = macro_auc(labels[validation_index], prediction)
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    prediction = mars_predict(model, features, masks, np.arange(len(features)))
    del model
    torch.cuda.empty_cache()
    return prediction, float(best_score)


def normalized_reports_mars(series):
    return (series.fillna("").astype(str).str.lower()
            .str.replace(r"\s+", " ", regex=True).str.strip())


def report_groups_mars(series):
    return np.asarray([hashlib.sha256(text.encode()).hexdigest()[:24]
                       for text in normalized_reports_mars(series)])


def mars_3d_main():
    if DEVICE.type != "cuda":
        raise RuntimeError("MARS 3D validation requires GPU")
    root = competition_root()
    train = pd.read_csv(root / "train.csv", dtype={"StudyInstanceUID": str})
    studies = train.StudyInstanceUID.tolist()
    gold = train[TARGETS].notna().all(axis=1).to_numpy()
    if int(gold.sum()) != 58:
        raise RuntimeError(f"expected 58 gold studies, found {gold.sum()}")
    gold_index = np.flatnonzero(gold)
    non_gold = np.flatnonzero(~gold)

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
    mars_features, mars_masks = encode_mars(studies, by_study)
    if float(mars_masks.mean()) < 0.75:
        raise RuntimeError(f"MARS sequence coverage too low: {mars_masks.mean()}")
    missing_studies = int((mars_masks.sum(axis=1) == 0).sum())
    if missing_studies:
        raise RuntimeError(f"MARS has no valid sequence for {missing_studies} studies")

    package = np.load(unique_input("incumbent_radimagenet_features.npz"), allow_pickle=False)
    if package["ids"].astype(str).tolist() != studies:
        raise RuntimeError("incumbent feature identity/order drift")
    if str(package["encoder_sha256"]) != RAD_SHA:
        raise RuntimeError("incumbent encoder hash drift")
    control_features, control_masks = package["features"], package["masks"]

    groups = report_groups_mars(train["Report"])
    y_gold = train.loc[gold, TARGETS].to_numpy(float)
    results, control_gold, mars_gold = [], [], []
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
        mars_prediction, mars_weak = train_mars_head(
            mars_features, mars_masks, labels, train_index, validation_index, seed
        )
        control_auc = macro_auc(y_gold, control_prediction[gold_index])
        mars_auc = macro_auc(y_gold, mars_prediction[gold_index])
        results.append({
            "seed": seed,
            "control_weak_validation_auc": control_weak,
            "mars_weak_validation_auc": mars_weak,
            "control_gold_auc": control_auc,
            "mars_gold_auc": mars_auc,
            "delta": mars_auc - control_auc,
        })
        control_gold.append(control_prediction[gold_index])
        mars_gold.append(mars_prediction[gold_index])
        log(f"seed {seed}: MARS 3D delta {mars_auc - control_auc:+.6f}")

    control_mean = np.mean(control_gold, axis=0)
    mars_mean = np.mean(mars_gold, axis=0)
    control_auc = macro_auc(y_gold, control_mean)
    mars_auc = macro_auc(y_gold, mars_mean)
    bootstrap = paired_bootstrap(y_gold, control_mean, mars_mean)
    deltas = np.asarray([result["delta"] for result in results])
    gate = {
        "delta_at_least_0_005": mars_auc - control_auc >= 0.005,
        "positive_seeds_at_least_4": int((deltas > 0).sum()) >= 4,
        "bootstrap_p_at_least_0_95": bootstrap["p_improve"] >= 0.95,
        "bootstrap_q025_positive": bootstrap["q025"] > 0,
    }
    report = {
        "contract": "private MARS 3D foundation validation; gold evaluation only; no submission",
        "candidate": "MARS 3D Swin Transformer Base",
        "candidate_repository": MARS_REPOSITORY,
        "candidate_license": "Apache-2.0",
        "candidate_citation": "Large-scale multi-sequence pretraining for generalizable MRI analysis, Nature Biomedical Engineering (2026)",
        "candidate_checkpoint_sha256": MARS_SHA256,
        "candidate_checkpoint_size": MARS_SIZE,
        "candidate_volume_shape": [VOLUME_DEPTH, VOLUME_SIZE, VOLUME_SIZE],
        "candidate_sequence_tokens": list(PLANES),
        "sequence_coverage": float(mars_masks.mean()),
        "control": "RadImageNet ResNet-50",
        "control_sha256": RAD_SHA,
        "weak_labels": "stevenleehans llm_labels_v4_blend.csv",
        "gold_fit_count": 0,
        "gold_evaluation_count": 58,
        "seeds": results,
        "ensemble": {
            "control_gold_auc": control_auc,
            "mars_gold_auc": mars_auc,
            "delta": mars_auc - control_auc,
        },
        "bootstrap": bootstrap,
        "gate": gate,
        "passed": bool(all(gate.values())),
        "submission_created": False,
    }
    Path("/kaggle/working/mars_3d_validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


mars_3d_main()

