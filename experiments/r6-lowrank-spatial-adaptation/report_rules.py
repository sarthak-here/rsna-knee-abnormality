"""New conservative report-state supervision; no teacher or published labels.

These literal rules are a weak report proxy, not expert pathology annotations.
Absence of a mention NEVER means normal. No organizer report is inspected.
"""
import re
import unicodedata

VERSION = "own_literal_states_v1_1"
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
ENTITY = {
    "ACL": r"\bacl\b|\blca\b|anterior cruciate|cruzado anterior|croise anterieur|vorder\w* kreuzband|voorste kruisband",
    "MCL": r"\bmcl\b|medial collateral|colateral (?:medial|interno)|mediale kollateral|mediale collaterale",
    "Medial Meniscus": r"medial menisc\w*|menisc\w* (?:medial|intern[oa])|menisco intern[oa]|innenmenisk\w*|mediale meniscus",
    "Lateral Meniscus": r"lateral menisc\w*|menisc\w* (?:lateral|extern[oa])|menisco extern[oa]|aussenmenisk\w*|laterale meniscus",
    "Effusion": r"\beffusion\b|\bderrame\b|gelenkerguss|hydrops|epanchement",
    "Synovitis": r"\bsynovitis\b|\bsinovitis\b|\bsinovite\b",
    "Baker's": r"baker|popliteal cyst|quiste popliteo|cisto popliteo",
    "Fracture": r"\bfractur\w*|\bfraktur\w*",
}
UNCERTAIN = r"\?|\b(?:possible|possibly|probable|suspect\w*|equivocal|cannot exclude|rule out|query|sospech\w*|posible|probabl\w*|verdacht|mogelijk|eventueel)\b"
NORMAL = r"\b(?:normal|intact|preserved|intakt\w*|integro|integra|conservad[oa]|unauffallig|normaal|intacte)\b"
NO = r"\b(?:no|without|absent|sin|sem|kein\w*|geen|aucun\w*|pas de)\b"
TEAR = r"\b(?:tear|torn|ruptur\w*|rotur\w*|desgarr\w*|riss|scheur)\b"
COMPLETE = r"\b(?:complete|completa|completo|full.thickness|vollstandig\w*|komplett\w*|volledige)\b"
SMALL = r"\b(?:small|pequen\w*|klein\w*|petit\w*)\b"
MODERATE = r"\b(?:moderate|moderad\w*|massig\w*|matig\w*|modere\w*)\b"
LARGE = r"\b(?:large|grande|gross\w*|groot|important)\b"
POSTOP = r"\b(?:postop\w*|post.operativ\w*|meniscect\w*|reconstruct\w*|graft|injerto|plastik)\b"
BODY = r"(?:findings|impression|results|resultados|befund|beurteilung|bevindingen|conclusion|conclusiones|conclusie|impressie|diagnostico)\s*[:\n]"
ABNORMAL_STATE = r"\b(?:degener\w*|sprain|esguince|edema|oedema|partial|parcial|partiell|injur\w*|lesion\w*|abnormal)\b|increased signal"


def norm(text):
    text = unicodedata.normalize("NFKD", text.casefold()).replace("ß", "ss")
    return "".join(c for c in text if not unicodedata.combining(c))


def state_for(label, clause):
    if re.search(UNCERTAIN, clause):
        return None
    entity = re.search(ENTITY[label], clause)
    before = clause[max(0, entity.start()-55):entity.start()]
    negators = list(re.finditer(NO, before))
    neg = bool(negators and len(before[negators[-1].end():].split()) <= 5)
    neg = neg or bool(re.match(r".{0,12}\b(?:is absent|are absent|not seen)\b", clause[entity.end():]))
    normal = bool(re.search(NORMAL, clause))
    tear = bool(re.search(TEAR, clause))
    if label in ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus"):
        if re.search(POSTOP, clause):
            return None
        # Do not associate a generic negation with an unrelated positive tear.
        if re.search(NO, clause) and tear:
            # Absence of a tear does NOT establish the fine-grained normal
            # state (e.g. degeneration without tear). Leave it unlabelled.
            return None
        if normal and tear:
            return None
        if normal and not re.search(ABNORMAL_STATE, clause):
            return "normal"
        if not tear or neg:
            return None
        if label in ("Medial Meniscus", "Lateral Meniscus"):
            return "tear"
        if re.search(COMPLETE, clause):
            return "complete_tear"
        # No forced grade mapping from an unspecified MCL injury.
        return "tear_unspecified" if label == "ACL" and not re.search(r"partial|parcial|partiell|grade|grado", clause) else None
    if label in ("Effusion", "Baker's"):
        if neg:
            if any(re.search(p, clause) for p in (SMALL, MODERATE, LARGE)) or re.search(r"significant|sizeable", clause):
                return None  # "no large effusion" does not mean no effusion.
            return "none_or_trace" if label == "Effusion" else "none"
        if label == "Baker's" and re.search(r"ruptur\w*|rotur\w*|leak\w*", clause):
            if re.search(r"\b(?:not|nicht|no|geen)\b", clause):
                return None
            return "ruptured_or_leaking"
        grades = [name for name, pattern in (("small", SMALL), ("moderate", MODERATE), ("large", LARGE)) if re.search(pattern, clause)]
        if len(grades) == 1:
            return grades[0]
        if label == "Effusion" and re.search(r"\b(?:trace|trivial|minimal)\b", clause) and not grades:
            return "none_or_trace"
        return None
    if label == "Synovitis":
        return "none" if neg else "synovitis"
    if label == "Fracture":
        if neg:
            if re.search(r"\b(?:acute|agud\w*|akut\w*|old|healed|chronic)\b", clause):
                return None
            return "none"
        if re.search(r"\b(?:old|healed|chronic|antigu\w*|consolidat\w*|sequela\w*)\b", clause):
            return "old_or_healed_fracture"
        if re.search(r"\b(?:acute|agud\w*|akut\w*|recent)\b", clause):
            return "acute_fracture"
    return None


def extract(report):
    text = norm(report)
    start = re.search(BODY, text)
    if start is None:
        return {}  # Never train on a clinical question masquerading as findings.
    text = text[start.end():]
    found = {}
    for clause in re.split(r"[.;\n]+", text):
        labels = [l for l, p in ENTITY.items() if re.search(p, clause)]
        if len(labels) != 1:
            continue  # No cross-entity attribution of severity/negation.
        label = labels[0]
        state = state_for(label, clause)
        if state is not None:
            found.setdefault(label, set()).add(state)
    return {label: next(iter(states)) for label, states in found.items() if len(states) == 1}
