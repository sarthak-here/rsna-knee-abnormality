"""Private KneeXNet-guided anatomy ROI validation.

The candidate uses an attributed knee cartilage/meniscus segmentation model to
replace only confident sagittal RadImageNet tokens with anatomy-focused crops.
The fixed hybrid uses those predictions only for meniscus and OA targets.
All 58 organizer-labelled studies remain final evaluation-only.
"""

import subprocess
import sys
import urllib.request

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "segmentation-models-pytorch==0.5.0",
])
import segmentation_models_pytorch as smp


SEGMENTATION_LFS_BATCH_URL = (
    "https://github.com/pitthexai/"
    "Knee_MRI_Segmentation_2.5D.git/info/lfs/objects/batch"
)
SEGMENTATION_SHA256 = "0c2b5fe87f86e78c7d90e9dfc3de1307aaef57c36d44ebb1251eecdcffc3c663"
SEGMENTATION_SIZE = 97924670
SEGMENTATION_COMMIT = "f2452feb602f2bd9e790550c8a5a07f62e7d6e55"
SEGMENTATION_TARGETS = (2, 3, 4, 5, 6)
SAGITTAL_SLOT_INDICES = (0, 3)
SEG_INPUT = 256


def load_segmentation_model():
    path = Path("/kaggle/working/kneexnet_baseline_best_model.pth")
    if not path.is_file():
        body = json.dumps({
            "operation": "download",
            "transfers": ["basic"],
            "objects": [{"oid": SEGMENTATION_SHA256, "size": SEGMENTATION_SIZE}],
        }).encode("utf-8")
        request = urllib.request.Request(
            SEGMENTATION_LFS_BATCH_URL,
            data=body,
            headers={
                "Accept": "application/vnd.git-lfs+json",
                "Content-Type": "application/vnd.git-lfs+json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            metadata = json.loads(response.read().decode("utf-8"))
        download_url = metadata["objects"][0]["actions"]["download"]["href"]
        urllib.request.urlretrieve(download_url, path)
    if path.stat().st_size != SEGMENTATION_SIZE:
        raise RuntimeError("KneeXNet checkpoint size drift")
    if sha256(path) != SEGMENTATION_SHA256:
        raise RuntimeError("KneeXNet checkpoint hash drift")
    model = smp.Unet(
        encoder_name="resnet34", encoder_weights=None,
        in_channels=3, classes=5,
    )
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval().to(DEVICE)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_frozen_radimagenet():
    model = load_encoder()
    model.eval()
    return model


def normalized_triplets(record):
    rows = ordered_headers(record)
    if not rows:
        return []
    centres = np.linspace(0.20 * (len(rows) - 1), 0.80 * (len(rows) - 1), N_SLICE).round().astype(int)
    output = []
    for centre in centres:
        indices = [max(0, centre - 1), centre, min(len(rows) - 1, centre + 1)]
        arrays = []
        try:
            for index in indices:
                image, _ = read_pixels(rows[index][0])
                height, width = image.shape
                side = min(height, width)
                y0, x0 = (height - side) // 2, (width - side) // 2
                arrays.append(image[y0:y0 + side, x0:x0 + side])
        except Exception:
            continue
        values = np.concatenate([image.ravel()[::max(1, image.size // 20000)] for image in arrays])
        low, high = np.percentile(values, (1, 99))
        stack = []
        for image in arrays:
            image = np.clip((image - low) / max(high - low, 1e-6), 0, 1)
            image = cv2.resize(image, (SEG_INPUT, SEG_INPUT), interpolation=cv2.INTER_AREA)
            stack.append(image.astype(np.float32))
        output.append(np.stack(stack))
    return output


def confident_roi(probability, centre_image):
    foreground = probability[1:].max(axis=0)
    mask = foreground >= 0.50
    area = float(mask.mean())
    rows, columns = np.where(mask)
    if len(rows) < 32 or area < 0.005 or area > 0.60:
        return None, area
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(columns.min()), int(columns.max()) + 1
    height, width = y1 - y0, x1 - x0
    if min(height, width) < 12:
        return None, area
    margin_y, margin_x = max(8, int(0.25 * height)), max(8, int(0.25 * width))
    y0, y1 = max(0, y0 - margin_y), min(SEG_INPUT, y1 + margin_y)
    x0, x1 = max(0, x0 - margin_x), min(SEG_INPUT, x1 + margin_x)
    crop = centre_image[y0:y1, x0:x1]
    if min(crop.shape) < 16:
        return None, area
    crop = cv2.resize(crop, (IMG, IMG), interpolation=cv2.INTER_AREA)
    return np.round(np.clip(crop, 0, 1) * 255).astype(np.uint8), area


@torch.no_grad()
def build_segmentation_roi_features(studies, by_study, control_features, control_masks):
    candidate = np.array(control_features, copy=True)
    segmentation = load_segmentation_model()
    encoder = load_frozen_radimagenet()
    audit = {"attempted": 0, "confident": 0, "fallback": 0, "foreground_areas": []}
    batch_studies = 32
    for start in range(0, len(studies), batch_studies):
        ids = studies[start:start + batch_studies]
        stacks, assignments = [], []
        for local_index, uid in enumerate(ids):
            records = by_study.get(uid, [])
            for slot_index in SAGITTAL_SLOT_INDICES:
                plane, fluid = SLOTS[slot_index]
                record = choose(records, plane, fluid, False)
                if record is None:
                    continue
                for slice_index, stack in enumerate(normalized_triplets(record)):
                    stacks.append(stack)
                    assignments.append((local_index, slot_index, slice_index))
        if stacks:
            probabilities = []
            array = np.stack(stacks).astype(np.float32)
            for offset in range(0, len(array), 64):
                tensor = torch.from_numpy(array[offset:offset + 64]).to(DEVICE)
                with torch.autocast("cuda", enabled=DEVICE.type == "cuda"):
                    probability = torch.softmax(segmentation(tensor), dim=1)
                probabilities.append(probability.float().cpu().numpy())
            probabilities = np.concatenate(probabilities)
            roi_images, roi_assignments = [], []
            for probability, stack, assignment in zip(probabilities, stacks, assignments):
                audit["attempted"] += 1
                roi, area = confident_roi(probability, stack[1])
                audit["foreground_areas"].append(area)
                if roi is None:
                    audit["fallback"] += 1
                    continue
                audit["confident"] += 1
                roi_images.append(roi)
                roi_assignments.append(assignment)
            if roi_images:
                encoded = []
                array = np.stack(roi_images)
                for offset in range(0, len(array), 128):
                    tensor = torch.from_numpy(array[offset:offset + 128]).to(DEVICE).float()
                    tensor = tensor.div_(127.5).sub_(1).unsqueeze(1).expand(-1, 3, -1, -1)
                    with torch.autocast("cuda", enabled=DEVICE.type == "cuda"):
                        encoded.append(encoder(tensor).float().cpu().numpy().astype(np.float16))
                encoded = np.concatenate(encoded)
                for feature, (local_index, slot_index, slice_index) in zip(encoded, roi_assignments):
                    study_index = start + local_index
                    token_index = slot_index * N_SLICE + slice_index
                    if control_masks[study_index, token_index] > 0:
                        candidate[study_index, token_index] = feature
        if start % (batch_studies * 10) == 0:
            log(f"segmentation ROI encoded {start}/{len(studies)}")
    areas = np.asarray(audit.pop("foreground_areas"), dtype=float)
    audit.update({
        "confidence_rate": audit["confident"] / max(1, audit["attempted"]),
        "foreground_area_median": float(np.median(areas)) if len(areas) else None,
        "foreground_area_q05": float(np.quantile(areas, 0.05)) if len(areas) else None,
        "foreground_area_q95": float(np.quantile(areas, 0.95)) if len(areas) else None,
    })
    del segmentation, encoder
    gc.collect(); torch.cuda.empty_cache()
    return candidate, audit


def normalized_reports_roi(series):
    return (series.fillna("").astype(str).str.lower()
            .str.replace(r"\s+", " ", regex=True).str.strip())


def report_groups_roi(series):
    return np.asarray([hashlib.sha256(text.encode()).hexdigest()[:24]
                       for text in normalized_reports_roi(series)])


def segmentation_roi_main():
    if DEVICE.type != "cuda":
        raise RuntimeError("segmentation ROI validation requires GPU")
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

    package = np.load(unique_input("incumbent_radimagenet_features.npz"), allow_pickle=False)
    if package["ids"].astype(str).tolist() != studies:
        raise RuntimeError("incumbent feature identity/order drift")
    if str(package["encoder_sha256"]) != RAD_SHA:
        raise RuntimeError("incumbent encoder hash drift")
    control_features, control_masks = package["features"], package["masks"]

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
    candidate_features, roi_audit = build_segmentation_roi_features(
        studies, by_study, control_features, control_masks
    )
    if roi_audit["confidence_rate"] < 0.25:
        raise RuntimeError(f"segmentation confidence coverage too low: {roi_audit}")

    groups = report_groups_roi(train["Report"])
    y_gold = train.loc[gold, TARGETS].to_numpy(float)
    results, control_gold, hybrid_gold = [], [], []
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
        roi_prediction, roi_weak = train_head(
            candidate_features, control_masks, labels, train_index, validation_index, seed
        )
        hybrid_prediction = np.array(control_prediction, copy=True)
        hybrid_prediction[:, SEGMENTATION_TARGETS] = roi_prediction[:, SEGMENTATION_TARGETS]
        hybrid_weak = macro_auc(labels[validation_index], hybrid_prediction[validation_index])
        control_auc = macro_auc(y_gold, control_prediction[gold_index])
        hybrid_auc = macro_auc(y_gold, hybrid_prediction[gold_index])
        results.append({
            "seed": seed,
            "control_weak_validation_auc": control_weak,
            "roi_head_weak_validation_auc": roi_weak,
            "hybrid_weak_validation_auc": hybrid_weak,
            "control_gold_auc": control_auc,
            "hybrid_gold_auc": hybrid_auc,
            "delta": hybrid_auc - control_auc,
        })
        control_gold.append(control_prediction[gold_index])
        hybrid_gold.append(hybrid_prediction[gold_index])
        log(f"seed {seed}: segmentation-ROI hybrid delta {hybrid_auc - control_auc:+.6f}")

    control_mean = np.mean(control_gold, axis=0)
    hybrid_mean = np.mean(hybrid_gold, axis=0)
    control_auc = macro_auc(y_gold, control_mean)
    hybrid_auc = macro_auc(y_gold, hybrid_mean)
    bootstrap = paired_bootstrap(y_gold, control_mean, hybrid_mean)
    deltas = np.asarray([result["delta"] for result in results])
    per_target_delta = {
        TARGETS[index]: float(roc_auc_score(y_gold[:, index], hybrid_mean[:, index]) -
                              roc_auc_score(y_gold[:, index], control_mean[:, index]))
        for index in SEGMENTATION_TARGETS
    }
    gate = {
        "delta_at_least_0_005": hybrid_auc - control_auc >= 0.005,
        "positive_seeds_at_least_4": int((deltas > 0).sum()) >= 4,
        "bootstrap_p_at_least_0_95": bootstrap["p_improve"] >= 0.95,
        "bootstrap_q025_positive": bootstrap["q025"] > 0,
    }
    report = {
        "contract": "private attributed segmentation-guided ROI validation; gold evaluation only; no submission",
        "encoder": "RadImageNet ResNet-50",
        "encoder_sha256": RAD_SHA,
        "segmentation_model": "KneeXNet-2.5D baseline ResNet34 U-Net",
        "segmentation_repository": "pitthexai/Knee_MRI_Segmentation_2.5D",
        "segmentation_commit": SEGMENTATION_COMMIT,
        "segmentation_checkpoint_sha256": SEGMENTATION_SHA256,
        "segmentation_citation": "Sanogo et al., KneeXNet-2.5D, npj Health Systems (2026), doi:10.1038/s44401-026-00072-5",
        "segmentation_use": "research/educational private validation",
        "hybrid_targets": [TARGETS[index] for index in SEGMENTATION_TARGETS],
        "roi_audit": roi_audit,
        "weak_labels": "stevenleehans llm_labels_v4_blend.csv",
        "gold_fit_count": 0,
        "gold_evaluation_count": 58,
        "seeds": results,
        "ensemble": {
            "control_gold_auc": control_auc,
            "hybrid_gold_auc": hybrid_auc,
            "delta": hybrid_auc - control_auc,
            "per_target_delta": per_target_delta,
        },
        "bootstrap": bootstrap,
        "gate": gate,
        "passed": bool(all(gate.values())),
        "submission_created": False,
    }
    Path("/kaggle/working/segmentation_roi_validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


segmentation_roi_main()
