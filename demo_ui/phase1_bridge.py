"""Phase 1 pragmatic strategies on the demo's pragmatic axis.

Generation and validation are Phase 1's own code (src/pragmatic_perturbation.py): `_generate_candidates`
(QWERTY typo, instruction prefix, distracting suffix) and `validate` (SBERT >= 0.75, then ONE-WAY NLI:
rewrite => original, entailment must be the top label). Phase 1's default models (MiniLM, nli-deberta-v3-base)
are not in the local cache, so the engine object gets the models the demo already loaded (SBERT mpnet, DeBERTa-v3
large NLI) instead of loading its own. Added here: a seed from the input text (same output on every click), keeping
the original line breaks around a typo, a sanity check per strategy, and hiding one suffix that reads oddly.
"""
from __future__ import annotations

import difflib
import random
import re
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pragmatic_perturbation import PragmaticPerturbationEngine  # noqa: E402

LABEL = {"typo_injection": "Typo", "instruction_reframing": "Instruction prefix",
         "context_distraction": "Distracting sentence"}
RULE = {"typo_injection": "p1.typo", "instruction_reframing": "p1.instruction_prefix",
        "context_distraction": "p1.distractor_suffix"}
DESC = {"typo_injection": "One keyboard-neighbour typo (Phase 1)",
        "instruction_reframing": "Instruction placed before the question (Phase 1)",
        "context_distraction": "Unrelated instruction placed after the question (Phase 1)"}
# Refers to "details provided previously", which a one-line question does not have: reads as junk on stage.
HIDE = {"Ignore any irrelevant details provided previously."}
CHECKSET = "one-way NLI (Phase 1 pragmatic)"


class _NLIAdapter:
    """Gives Phase 1's `self.nli.predict([(premise, hypothesis)])` the demo's DeBERTa NLI; keeps the last probs."""

    def __init__(self, scorer):
        self.s = scorer
        self.last = None

    def predict(self, pairs):
        t = self.s.torch
        with t.inference_mode():
            enc = self.s.tok([p for p, _ in pairs], [h for _, h in pairs], padding=True, truncation=True,
                             max_length=256, return_tensors="pt").to(self.s.device)
            logits = self.s.nli(**enc).logits.float()
        self.last = t.softmax(logits, -1).cpu().numpy()
        return logits.cpu().numpy()


def keep_spacing(orig: str, joined: str) -> str:
    """Phase 1 rebuilds text with ' '.join(words); put the typo'd words back into the original spacing."""
    parts = re.split(r"(\s+)", orig)
    new_words = joined.split()
    if len(new_words) != len([p for p in parts if p and not p.isspace()]):
        return joined
    it = iter(new_words)
    return "".join(p if (not p or p.isspace()) else next(it) for p in parts)


def char_diff(a: str, b: str):
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            out.append(["eq", a[i1:i2], b[j1:j2]])
        else:
            if i2 > i1:
                out.append(["del", a[i1:i2]])
            if j2 > j1:
                out.append(["ins", b[j1:j2]])
    return out


def sanity(kind: str, orig: str, new: str):
    if kind == "typo_injection":
        wa, wb = orig.split(), new.split()
        diff = [(x, y) for x, y in zip(wa, wb) if x != y]
        ok = (len(wa) == len(wb) and len(orig) == len(new) and len(diff) == 1
              and sum(c1 != c2 for c1, c2 in zip(*diff[0])) == 1)
        return ok, "one character in one word, same length"
    ok = new.startswith(orig) or new.endswith(orig)
    return ok, "original question kept word for word"


class Phase1:
    def __init__(self, scorer):
        eng = object.__new__(PragmaticPerturbationEngine)   # skip its own model loading
        eng.device = scorer.device
        eng.similarity_threshold = 0.75                      # Phase 1 default
        eng.sbert = scorer.sbert
        eng.nli = _NLIAdapter(scorer)
        eng.nli_labels = [scorer.nli.config.id2label[i].lower() for i in range(len(scorer.nli.config.id2label))]
        self.eng, self.i_ent = eng, scorer.i_ent

    def run(self, text: str):
        eng = self.eng
        eng.rng = random.Random(zlib.crc32(text.encode("utf-8")))   # deterministic per input
        cands = eng._generate_candidates(text)
        done, cards, rejected = set(), [], []
        for new, tr in cands:
            kind = tr.transform_type
            if kind not in LABEL or kind in done:     # skip the prefix+suffix mix; one card per strategy
                continue
            if any(h in new for h in HIDE):
                continue
            if kind == "typo_injection":
                new = keep_spacing(text, new)
            if new == text:
                continue
            ok_s, need = sanity(kind, text, new)
            sim, ent, ok_nli = None, None, None
            if ok_s:
                eng.nli.last = None
                ok_all, sim, ok_nli = eng.validate(text, new)
                ent = float(eng.nli.last[0][self.i_ent]) if eng.nli.last is not None else None
                ok_sim = sim >= eng.similarity_threshold
            else:
                ok_all = ok_sim = False
            checks = [["Sanity check", "pass" if ok_s else "fail", need, ok_s],
                      ["SBERT whole text", sim, ">= 0.75", ok_sim if ok_s else None],
                      ["Entailment, rewrite to original", ent, "entailment is the top label",
                       ok_nli if ent is not None else None],
                      ["Entailment, original to rewrite", None, "not used (one-way, Phase 1)", None]]
            diff = char_diff(text, new) if kind == "typo_injection" else [
                ["ins", new[:len(new) - len(text)]], ["eq", text, text]] if new.endswith(text) else [
                ["eq", text, text], ["ins", new[len(text):]]]
            entry = {"rule": RULE[kind], "rule_text": DESC[kind], "label": LABEL[kind], "rewrite": new, "diff": diff,
                     "checks": checks, "checkset": CHECKSET, "passed": bool(ok_all)}
            if ok_all:
                cards.append(entry)
                done.add(kind)
            else:
                code, reason = (("phase1:sanity", "The rewrite changed more than the strategy allows") if not ok_s else
                                ("phase1:sbert", "Whole text drifted too far in meaning (SBERT)") if not ok_sim else
                                ("phase1:entail", "The rewrite does not imply the original question (NLI, one way)"))
                entry.update(code=code, reason=reason, detail=code)
                rejected.append(entry)
        return cards, rejected
