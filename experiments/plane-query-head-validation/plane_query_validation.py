"""Plane- and slice-aware target-query head on frozen RadImageNet tokens.

The candidate adds explicit six-slot/three-slice identities, contextual token
encoding, and twelve learned finding queries. All fitting uses non-gold report
groups; the 58 organizer-labelled studies remain final evaluation-only.
"""


HEAD_DIM = 256
HEAD_EPOCHS = 24


def normalized_reports(series):
    return (series.fillna("").astype(str).str.lower()
            .str.replace(r"\s+", " ", regex=True).str.strip())


def report_groups(series):
    return np.asarray([hashlib.sha256(text.encode()).hexdigest()[:24]
                       for text in normalized_reports(series)])


class PlaneQueryHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(TOKEN_DIM)
        self.project = nn.Linear(TOKEN_DIM, HEAD_DIM)
        self.slot = nn.Parameter(torch.randn(len(SLOTS), HEAD_DIM)*.01)
        self.slice = nn.Parameter(torch.randn(N_SLICE, HEAD_DIM)*.01)
        layer = nn.TransformerEncoderLayer(
            d_model=HEAD_DIM, nhead=8, dim_feedforward=512,
            dropout=.10, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2,
                                             enable_nested_tensor=False)
        self.queries = nn.Parameter(torch.randn(len(TARGETS), HEAD_DIM)*.02)
        self.cross = nn.MultiheadAttention(
            HEAD_DIM, 8, dropout=.10, batch_first=True
        )
        self.query_norm = nn.LayerNorm(HEAD_DIM)
        self.classifier = nn.Parameter(torch.empty(len(TARGETS), HEAD_DIM))
        self.bias = nn.Parameter(torch.zeros(len(TARGETS)))
        nn.init.trunc_normal_(self.classifier, std=.02)
        slot_index = torch.arange(len(SLOTS)).repeat_interleave(N_SLICE)
        slice_index = torch.arange(N_SLICE).repeat(len(SLOTS))
        self.register_buffer("slot_index", slot_index, persistent=False)
        self.register_buffer("slice_index", slice_index, persistent=False)

    def forward(self, x, mask):
        hidden = torch.nn.functional.gelu(self.project(self.norm(x.float())))
        hidden = hidden + self.slot[self.slot_index] + self.slice[self.slice_index]
        padding = mask < .5
        hidden = self.encoder(hidden, src_key_padding_mask=padding)
        query = self.queries.unsqueeze(0).expand(x.shape[0], -1, -1)
        query, _ = self.cross(
            query, hidden, hidden, key_padding_mask=padding,
            need_weights=False,
        )
        query = self.query_norm(query)
        return (query*self.classifier).sum(-1)+self.bias


@torch.no_grad()
def model_predict(model, features, masks, indices):
    model.eval(); output=[]
    for start in range(0,len(indices),128):
        index=indices[start:start+128]
        x=torch.from_numpy(features[index]).to(DEVICE)
        mask=torch.from_numpy(masks[index]).to(DEVICE)
        output.append(torch.sigmoid(model(x,mask)).cpu().numpy())
    return np.concatenate(output)


def fit_head(model_type, features, masks, labels,
             train_index, validation_index, seed):
    seed_all(seed)
    model=model_type().to(DEVICE)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-3)
    dataset=TensorDataset(
        torch.from_numpy(features[train_index]),
        torch.from_numpy(masks[train_index]),
        torch.from_numpy(labels[train_index]),
    )
    loader=DataLoader(dataset,batch_size=64,shuffle=True,
                      generator=torch.Generator().manual_seed(seed))
    best_state,best_score=None,-1.0
    for _ in range(HEAD_EPOCHS):
        model.train()
        for x,mask,target in loader:
            x,mask,target=x.to(DEVICE),mask.to(DEVICE),target.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(
                model(x,mask),target.float()
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
        prediction=model_predict(model,features,masks,validation_index)
        score=macro_auc(labels[validation_index],prediction)
        if score>best_score:
            best_score=score
            best_state={key:value.detach().cpu().clone()
                        for key,value in model.state_dict().items()}
    model.load_state_dict(best_state)
    prediction=model_predict(model,features,masks,np.arange(len(features)))
    del model; torch.cuda.empty_cache()
    return prediction,float(best_score)


def plane_query_main():
    if DEVICE.type!="cuda": raise RuntimeError("plane-query validation requires GPU")
    root=competition_root()
    train=pd.read_csv(root/"train.csv",dtype={"StudyInstanceUID":str})
    gold=train[TARGETS].notna().all(axis=1).to_numpy()
    if int(gold.sum())!=58: raise RuntimeError(f"expected 58 gold studies, found {gold.sum()}")
    gold_index=np.flatnonzero(gold); nongold=np.flatnonzero(~gold)
    weak=pd.read_csv(unique_input("llm_labels_v4_blend.csv"),dtype={"StudyInstanceUID":str})
    aligned=train[["StudyInstanceUID"]].merge(weak,on="StudyInstanceUID",how="left",validate="one_to_one")
    labels=aligned[TARGETS].to_numpy(np.float32)
    if not np.isfinite(labels).all(): raise RuntimeError("weak labels incomplete")
    package=np.load(unique_input("incumbent_radimagenet_features.npz"),allow_pickle=False)
    if package["ids"].astype(str).tolist()!=train.StudyInstanceUID.tolist(): raise RuntimeError("feature identity/order drift")
    features,masks=package["features"],package["masks"]
    if features.shape!=(4407,len(SLOTS)*N_SLICE,TOKEN_DIM): raise RuntimeError(f"feature shape drift: {features.shape}")
    if str(package["encoder_sha256"])!=RAD_SHA: raise RuntimeError("feature encoder hash drift")

    groups=report_groups(train["Report"]); y_gold=train.loc[gold,TARGETS].to_numpy(float)
    results,control_gold,query_gold=[],[],[]
    for seed in SEEDS:
        unique_groups=np.unique(groups[nongold]); shuffled=np.random.default_rng(seed).permutation(unique_groups)
        validation_groups=set(shuffled[:max(1,int(.10*len(shuffled)))])
        validation_index=nongold[np.asarray([g in validation_groups for g in groups[nongold]])]
        train_index=nongold[np.asarray([g not in validation_groups for g in groups[nongold]])]
        if set(groups[train_index]).intersection(groups[validation_index]): raise RuntimeError("report-group leakage")
        control,control_weak=fit_head(AttentionHead,features,masks,labels,train_index,validation_index,seed)
        query,query_weak=fit_head(PlaneQueryHead,features,masks,labels,train_index,validation_index,seed)
        control_auc=macro_auc(y_gold,control[gold_index]); query_auc=macro_auc(y_gold,query[gold_index])
        results.append({"seed":seed,"control_weak_validation_auc":control_weak,"query_weak_validation_auc":query_weak,"control_gold_auc":control_auc,"query_gold_auc":query_auc,"delta":query_auc-control_auc})
        control_gold.append(control[gold_index]); query_gold.append(query[gold_index])
        log(f"seed {seed}: plane-query delta {query_auc-control_auc:+.6f}")
    control_mean=np.mean(control_gold,axis=0); query_mean=np.mean(query_gold,axis=0)
    control_auc=macro_auc(y_gold,control_mean); query_auc=macro_auc(y_gold,query_mean)
    bootstrap=paired_bootstrap(y_gold,control_mean,query_mean)
    deltas=np.asarray([x["delta"] for x in results])
    gate={"delta_at_least_0_005":query_auc-control_auc>=.005,"positive_seeds_at_least_4":int((deltas>0).sum())>=4,"bootstrap_p_at_least_0_95":bootstrap["p_improve"]>=.95,"bootstrap_q025_positive":bootstrap["q025"]>0}
    report={"contract":"plane/slice-aware target-query head; image-only gold evaluation","encoder":"RadImageNet ResNet-50","encoder_sha256":RAD_SHA,"feature_source":"rsna-knee-report-image-cross-supervision v1 authenticated output","weak_labels":"stevenleehans llm_labels_v4_blend.csv","candidate":{"head_dim":HEAD_DIM,"encoder_layers":2,"attention_heads":8,"epochs":HEAD_EPOCHS,"slot_embeddings":len(SLOTS),"slice_embeddings":N_SLICE,"target_queries":len(TARGETS)},"gold_fit_count":0,"gold_evaluation_count":58,"seeds":results,"ensemble":{"control_gold_auc":control_auc,"query_gold_auc":query_auc,"delta":query_auc-control_auc},"bootstrap":bootstrap,"gate":gate,"passed":bool(all(gate.values())),"submission_created":False}
    Path("/kaggle/working/plane_query_validation_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


plane_query_main()
