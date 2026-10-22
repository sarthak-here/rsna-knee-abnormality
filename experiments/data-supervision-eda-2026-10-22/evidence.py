"""Review-only multilingual evidence ledger; NEVER emits training targets.

Offsets refer to normalized text, not raw report bytes. Cue flags are screening
signals, not adjudicated negation/uncertainty or clinical conclusions.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'r6-own-report-adaptation'))
from report_rules import ENTITY, norm

FINDINGS = r'findings|impression|results|resultados|befund|beurteilung|bevindingen|conclusie|conclusion|conclusiones|diagnostico|hallazgos|resultats|compte rendu|description|descrizione|reperti|conclusioni|achados|conclusao'
HISTORY = r'clinical history|clinical question|indication|history|anamnesis|antecedentes|renseignements cliniques|indication clinique|fragestellung|anamnese|klinische gegevens|quesito clinico|indicacao'
TECHNIQUE = r'technique|tecnica|techniek|untersuchungstechnik|protocol'
HEADER = re.compile(r'(?im)^\s*(' + FINDINGS + '|' + HISTORY + '|' + TECHNIQUE + r')\s*(?::|\n)')
PATTERNS = dict(ENTITY)
# Broad entity/compartment co-mentions are retained for review, NOT interpreted
# as compartment-specific OA or pathology-positive labels.
PATTERNS.update({
    'Medial OA': r'medial (?:compartment|cartilage|chondr\w*)|compartiment\w* (?:medial|intern\w*)|mediales? kompartiment',
    'Lateral OA': r'lateral (?:compartment|cartilage|chondr\w*)|compartiment\w* (?:lateral|extern\w*)|laterales? kompartiment',
    'PF OA': r'patellofemoral|femoropatell\w*|femoro[- ]patell\w*|retropatell\w*|femoro[- ]rotul\w*',
    'Contusion': r'\bcontusion\w*|bone bruise|osseous contusion|edema oseo|oedeme osseux|knochenmarkodem',
})
NEGATION = re.compile(r'\b(?:no|not|without|absent|aucun\w*|pas de|sans|kein\w*|ohne|geen|sin|sem|non)\b')
UNCERTAINTY = re.compile(r'\b(?:possible|possibly|suspect\w*|cannot exclude|may represent|probabl\w*|verdacht|mogeli\w*|sospech\w*|dubbio)\b')
HISTORICAL = re.compile(r'\b(?:history|previous|prior|old|healed|post[- ]?op\w*|status post|antecedent\w*|antigu\w*|chroni\w*|zustand nach)\b')


def ledger(report):
    text = norm(report)
    headers = list(HEADER.finditer(text))
    sections = []
    if not headers:
        sections.append(('unsectioned_requires_review', 0, len(text)))
    else:
        if headers[0].start() > 0:
            sections.append(('preamble_requires_review', 0, headers[0].start()))
        for i, h in enumerate(headers):
            name = h.group(1)
            kind = 'history' if re.fullmatch(HISTORY, name) else 'technique' if re.fullmatch(TECHNIQUE, name) else 'findings'
            sections.append((kind, h.end(), headers[i+1].start() if i+1 < len(headers) else len(text)))
    entries = []
    for kind, start, end in sections:
        for clause in re.finditer(r'[^.;\n]+', text[start:end]):
            sentence = clause.group()
            mentions = []
            for label, pattern in PATTERNS.items():
                for match in re.finditer(pattern, sentence):
                    mentions.append({'finding': label, 'start': start+clause.start()+match.start(),
                                     'end': start+clause.start()+match.end()})
            if not mentions:
                continue
            entries.append({'section': kind, 'start': start+clause.start(), 'end': start+clause.end(),
                'mentions': sorted(mentions, key=lambda m: (m['start'], m['finding'])),
                'negation_cue': bool(NEGATION.search(sentence)),
                'uncertainty_cue': bool(UNCERTAINTY.search(sentence)),
                'historical_cue': bool(HISTORICAL.search(sentence)),
                'multi_entity': len({m['finding'] for m in mentions}) > 1,
                'requires_review': True})
    return {'schema': 'review_evidence_v1', 'offset_space': 'NFKD_casefold_without_combining_marks',
            'entries': entries, 'training_targets': None, 'clinically_validated': False}
