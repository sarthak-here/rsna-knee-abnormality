"""Target-specific report entity cross-supervision with image-only evaluation.

Appended to the shared RadImageNet runtime by build_notebook.py. Public reports
are used only inside each seed's non-gold training partition. The 58 organizer-
labelled studies and their reports are evaluation-only and never transformed.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


ENTITY_DIM = 16
ENTITY_LOSS_WEIGHT = 0.10
ENTITY_PATTERNS = {
    "ACL": r"\bACL\b|anterior cruciate|cruciate ligament",
    "MCL": r"\bMCL\b|medial collateral",
    "Medial Meniscus": r"medial menisc",
    "Lateral Meniscus": r"lateral menisc",
    "Medial OA": r"medial compartment|medial femoro?tibial|medial tibiofemoral|medial joint space",
    "Lateral OA": r"lateral compartment|lateral femoro?tibial|lateral tibiofemoral|lateral joint space",
    "PF OA": r"patellofemoral|patello-femoral|patellar cartilage|trochlear cartilage|trochlea",
    "Effusion": r"effusion|joint fluid",
    "Synovitis": r"synovitis|synovial",
    "Baker's": r"baker.?s|popliteal cyst",
    "Contusion": r"contusion|bone bruise|marrow edema|marrow oedema",
    "Fracture": r"fracture|cortical break|trabecular break",
}


def clean_reports(series):
    return (series.fillna("").astype(str).str.lower()
            .str.replace(r"\s+", " ", regex=True).str.strip())


def report_groups(series):
    return np.asarray([hashlib.sha256(text.encode()).hexdigest()[:24]
                       for text in clean_reports(series)])


def target_sentences(text, pattern):
    sentences = re.split(r"(?<=[.!?;])\s+|\n+", text)
    selected = [sentence.strip() for sentence in sentences
                if re.search(pattern, sentence, flags=re.I)]
    return " ".join(selected)


def build_entity_latents(reports, train_index):
    clean = clean_reports(reports).tolist()
    latent = np.zeros((len(clean), len(TARGETS), ENTITY_DIM), np.float32)
    mask = np.zeros((len(clean), len(TARGETS)), np.float32)
    audit = []
    for target_index, target in enumerate(TARGETS):
        snippets = [target_sentences(text, ENTITY_PATTERNS[target]) for text in clean]
        fit_index = np.asarray([i for i in train_index if snippets[i]], dtype=np.int64)
        if len(fit_index) < 25:
            raise RuntimeError(f"too few training snippets for {target}: {len(fit_index)}")
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_df=.995,
            max_features=5000, sublinear_tf=True, strip_accents="unicode",
        )
        matrix = vectorizer.fit_transform([snippets[i] for i in fit_index])
        components = min(ENTITY_DIM, matrix.shape[0]-1, matrix.shape[1]-1)
        if components < 2:
            raise RuntimeError(f"insufficient text rank for {target}: {matrix.shape}")
        svd = TruncatedSVD(n_components=components, random_state=20260901+target_index)
        values = svd.fit_transform(matrix).astype(np.float32)
        values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-6)
        latent[fit_index, target_index, :components] = values
        mask[fit_index, target_index] = 1.0
        audit.append({
            "target": target,
            "fit_snippets": int(len(fit_index)),
            "vocabulary_size": int(len(vectorizer.vocabulary_)),
            "components": int(components),
            "explained_variance": float(svd.explained_variance_ratio_.sum()),
        })
    return latent, mask, audit


class EntitySupervisedHead(nn.Module):
    """Control and candidate have the same classifier and parameterization."""
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(TOKEN_DIM)
        self.proj = nn.Linear(TOKEN_DIM, 256)
        self.att = nn.Linear(256, len(TARGETS))
        self.cls = nn.Parameter(torch.empty(len(TARGETS), 256))
        self.bias = nn.Parameter(torch.zeros(len(TARGETS)))
        self.entity_weight = nn.Parameter(
            torch.empty(len(TARGETS), 256, ENTITY_DIM)
        )
        self.entity_bias = nn.Parameter(torch.zeros(len(TARGETS), ENTITY_DIM))
        nn.init.trunc_normal_(self.cls, std=.02)
        nn.init.trunc_normal_(self.entity_weight, std=.02)

    def forward(self, x, mask):
        hidden = torch.nn.functional.gelu(self.proj(self.norm(x.float())))
        attention = self.att(hidden).masked_fill(mask[:, :, None] < .5, -1e4)
        attention = torch.softmax(attention, dim=1)
        pooled = torch.einsum("btn,btf->bnf", attention, hidden)
        logits = (pooled * self.cls).sum(-1) + self.bias
        entity = torch.einsum("bnf,nfd->bnd", pooled, self.entity_weight)
        entity = torch.nn.functional.normalize(entity + self.entity_bias, dim=2)
        return logits, entity


@torch.no_grad()
def entity_predict(model, features, masks, indices):
    model.eval(); output=[]
    for start in range(0, len(indices), 128):
        index=indices[start:start+128]
        x=torch.from_numpy(features[index]).to(DEVICE)
        mask=torch.from_numpy(masks[index]).to(DEVICE)
        logits,_=model(x,mask)
        output.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(output)


def train_entity_head(features, masks, labels, entity_latent, entity_mask,
                      train_index, validation_index, seed, entity_weight):
    seed_all(seed)
    model=EntitySupervisedHead().to(DEVICE)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-3)
    dataset=TensorDataset(
        torch.from_numpy(features[train_index]),
        torch.from_numpy(masks[train_index]),
        torch.from_numpy(labels[train_index]),
        torch.from_numpy(entity_latent[train_index]),
        torch.from_numpy(entity_mask[train_index]),
    )
    loader=DataLoader(dataset,batch_size=64,shuffle=True,
                      generator=torch.Generator().manual_seed(seed))
    best_state,best_score=None,-1.0
    for epoch in range(20):
        model.train()
        for x,mask,target,text_target,text_mask in loader:
            x,mask=x.to(DEVICE),mask.to(DEVICE)
            target,text_target,text_mask=target.to(DEVICE),text_target.to(DEVICE),text_mask.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits,text_prediction=model(x,mask)
            classification=torch.nn.functional.binary_cross_entropy_with_logits(logits,target.float())
            distance=1.0-(text_prediction*text_target).sum(2)
            report_loss=(distance*text_mask).sum()/text_mask.sum().clamp_min(1)
            loss=classification+float(entity_weight)*report_loss
            loss.backward(); optimizer.step()
        prediction=entity_predict(model,features,masks,validation_index)
        score=macro_auc(labels[validation_index],prediction)
        if score>best_score:
            best_score=score
            best_state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    model.load_state_dict(best_state)
    prediction=entity_predict(model,features,masks,np.arange(len(features)))
    del model; torch.cuda.empty_cache()
    return prediction,float(best_score)


def target_entity_main():
    if DEVICE.type!="cuda": raise RuntimeError("target entity validation requires GPU")
    root=competition_root()
    train=pd.read_csv(root/"train.csv",dtype={"StudyInstanceUID":str})
    gold=train[TARGETS].notna().all(axis=1).to_numpy()
    if int(gold.sum())!=58: raise RuntimeError(f"expected 58 gold studies, found {gold.sum()}")
    gold_index=np.flatnonzero(gold); nongold=np.flatnonzero(~gold)
    weak=pd.read_csv(unique_input("llm_labels_v4_blend.csv"),dtype={"StudyInstanceUID":str})
    aligned=train[["StudyInstanceUID"]].merge(weak,on="StudyInstanceUID",how="left",validate="one_to_one")
    labels=aligned[TARGETS].to_numpy(np.float32)
    if not np.isfinite(labels).all(): raise RuntimeError("weak labels incomplete")

    feature_path=unique_input("incumbent_radimagenet_features.npz")
    package=np.load(feature_path,allow_pickle=False)
    ids=package["ids"].astype(str).tolist()
    if ids!=train.StudyInstanceUID.tolist(): raise RuntimeError("feature study identity/order drift")
    features=package["features"]; masks=package["masks"]
    if features.shape!=(4407,len(SLOTS)*N_SLICE,TOKEN_DIM): raise RuntimeError(f"feature shape drift: {features.shape}")
    if str(package["encoder_sha256"])!=RAD_SHA: raise RuntimeError("feature encoder hash drift")

    groups=report_groups(train["Report"])
    y_gold=train.loc[gold,TARGETS].to_numpy(float)
    results,control_gold,entity_gold,text_audits=[],[],[],[]
    for seed in SEEDS:
        unique_groups=np.unique(groups[nongold]); rng=np.random.default_rng(seed)
        shuffled=rng.permutation(unique_groups)
        validation_groups=set(shuffled[:max(1,int(.10*len(shuffled)))])
        validation_index=nongold[np.asarray([g in validation_groups for g in groups[nongold]])]
        train_index=nongold[np.asarray([g not in validation_groups for g in groups[nongold]])]
        if set(groups[train_index]).intersection(groups[validation_index]): raise RuntimeError("report-group leakage")
        latent,text_mask,audit=build_entity_latents(train["Report"],train_index)
        text_audits.append({"seed":seed,"targets":audit,"gold_reports_transformed":0,"weak_validation_reports_transformed":0})
        control,control_weak=train_entity_head(features,masks,labels,latent,text_mask,train_index,validation_index,seed,0.0)
        entity,entity_weak=train_entity_head(features,masks,labels,latent,text_mask,train_index,validation_index,seed,ENTITY_LOSS_WEIGHT)
        control_auc=macro_auc(y_gold,control[gold_index]); entity_auc=macro_auc(y_gold,entity[gold_index])
        results.append({"seed":seed,"control_weak_validation_auc":control_weak,"entity_weak_validation_auc":entity_weak,"control_gold_auc":control_auc,"entity_gold_auc":entity_auc,"delta":entity_auc-control_auc})
        control_gold.append(control[gold_index]); entity_gold.append(entity[gold_index])
        log(f"seed {seed}: target-entity delta {entity_auc-control_auc:+.6f}")
    control_mean=np.mean(control_gold,axis=0); entity_mean=np.mean(entity_gold,axis=0)
    control_auc=macro_auc(y_gold,control_mean); entity_auc=macro_auc(y_gold,entity_mean)
    bootstrap=paired_bootstrap(y_gold,control_mean,entity_mean)
    deltas=np.asarray([x["delta"] for x in results])
    gate={"delta_at_least_0_005":entity_auc-control_auc>=.005,"positive_seeds_at_least_4":int((deltas>0).sum())>=4,"bootstrap_p_at_least_0_95":bootstrap["p_improve"]>=.95,"bootstrap_q025_positive":bootstrap["q025"]>0}
    report={"contract":"private image-only evaluation; target-specific report sentence cross-supervision","encoder":"RadImageNet ResNet-50","encoder_sha256":RAD_SHA,"feature_source":"rsna-knee-report-image-cross-supervision v1 authenticated output","weak_labels":"stevenleehans llm_labels_v4_blend.csv","entity_patterns":ENTITY_PATTERNS,"entity_loss_weight":ENTITY_LOSS_WEIGHT,"text_audits":text_audits,"gold_fit_count":0,"gold_text_transform_count":0,"gold_evaluation_count":58,"seeds":results,"ensemble":{"control_gold_auc":control_auc,"entity_gold_auc":entity_auc,"delta":entity_auc-control_auc},"bootstrap":bootstrap,"gate":gate,"passed":bool(all(gate.values())),"submission_created":False}
    Path("/kaggle/working/target_entity_validation_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


target_entity_main()
