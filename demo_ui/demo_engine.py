"""Thin wrapper around perturb_v3 for the demo.

No rewrite rule lives here. Candidates come from perturb_v3.run.options (which calls the engine's
lexical / syntactic / discourse rule functions), hard checks from perturb_v3.checks.hard_check, scores from
perturb_v3.checks.Scorer and thresholds from perturb_v3.run.th_for. The only difference from
perturb_v3.run.process is that every hard-check survivor is scored, so the page can show rejected
candidates with their scores. The accepted rewrite is still the first one that passes, as in the engine.
"""
from __future__ import annotations

import difflib
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perturb_v3 import checks as C  # noqa: E402
from perturb_v3 import engine as E  # noqa: E402
from perturb_v3 import run as R  # noqa: E402
import demo_rules as DR  # noqa: E402

# demo-only thresholds for demo rules (v3 rules keep their own, from config.json)
DEMO_TH = {"lexical": {"sbert_global_min": 0.80, "sbert_local_min": 0.80, "entail_min": 0.5, "contra_max": 0.10},
           "syntactic": {"sbert_global_min": 0.80, "sbert_local_min": 0.80, "entail_min": 0.6, "contra_max": 0.10}}
NLI_CAP = {"lexical": 2, "syntactic": 2, "discourse": 2}   # candidates per axis that get the (slow) NLI check
MAX_CARDS = 3

AXES = ("lexical", "syntactic", "discourse")

# Same driver-level guard as train_data/engine_driver.py ("an" before a consonant sound, e.g. "an usual").
# Copied here because train_data/ is not in the repo; perturb_v3 itself is untouched.
AN_BAD = re.compile(r"\b(an)\s+(u[sn]i?\w*|use\w*|usu\w*|eu\w*|one\b|once\b)", re.I)


def hard_check(text, v):
    f = C.hard_check(text, v)
    if f:
        return f
    if len(AN_BAD.findall(v["rewrite"])) > len(AN_BAD.findall(text)):
        return "article_an_before_consonant_sound"
    return None


RULE_TEXT = {
    "lex.word": "Words swapped for exact synonyms",
    "lex.phrase": "Fixed phrase swapped for an equivalent",
    "lex.contraction": "Contraction expanded",
    "syn.wh_to_imperative": "Question turned into an instruction (Explain ...)",
    "syn.polar_to_whether": "Yes/no question turned into 'Explain whether ...'",
    "syn.why_to_reason": "'Why ...?' turned into 'What is the reason that ...?'",
    "syn.can_to_possible": "'Can X ...?' turned into 'Is it possible for X to ...?'",
    "syn.how_in_what_way": "'How ...?' turned into 'In what way ...?'",
    "syn.move_clause": "If / when / because clause moved to the other end",
    "dis.title_to_end": "Title moved after the body",
    "dis.question_first": "Final question moved before its context",
    "dis.drop_pleasantry": "'Thanks!' line removed",
    "dis.frame": "Neutral opening line added",
}


CARD_LABEL = {"dis.frame": "Neutral opener", "dis.question_first": "Question moved first",
              "dis.drop_pleasantry": "Thank-you line removed", "dis.title_to_end": "Title moved to the end"}


DEMO_TEXT = {
    "demo.lex.word": "A word swapped for a synonym", "demo.lex.phrase": "A phrase swapped for an equivalent",
    "demo.lex.multi": "Several words swapped for synonyms",
    "demo.syn.tell_me": "Turned into 'Could you tell me ...?'",
    "demo.syn.like_to_know": "Turned into 'I would like to know whether ...'",
    "demo.syn.is_it_true": "Turned into 'Is it true that ...?'",
    "demo.syn.why_cause": "'Why does X ...?' turned into 'What causes X to ...?'",
    "demo.syn.way_in_which": "'How ...?' turned into 'What is the way in which ...?'",
    "demo.syn.can_you": "Instruction turned into a question", "demo.syn.imperative": "Question turned into an instruction",
    "demo.syn.front_adverbial": "Time phrase moved to the front", "demo.syn.end_adverbial": "Time word moved to the end",
    "demo.syn.swap_and": "Order of 'X and Y' swapped",
    "demo.syn.why_reason": "'Why ...?' turned into 'What is the reason that ...?'",
    "demo.syn.explain_what": "'What is X?' turned into 'Explain what X is.'",
    "demo.syn.split_sentence": "Split into two sentences",
}


def rule_text(rule):
    if rule in DEMO_TEXT:
        return DEMO_TEXT[rule]
    if rule in RULE_TEXT:
        return RULE_TEXT[rule]
    if rule.startswith("lex."):
        return "Words swapped for exact synonyms"
    return rule


def plain_reason(code: str) -> str:
    kind, _, rest = code.partition(":")
    rest = rest.strip()
    if kind == "semantic":
        name = rest.split(" ")[0]
        return {
            "sbert_global": "Whole text drifted too far in meaning (SBERT, whole text)",
            "sbert_local": "Changed sentence drifted too far in meaning (SBERT, changed part)",
            "entail": "The two versions do not clearly imply each other (NLI)",
            "contradiction": "The model sees a contradiction between the versions (NLI)",
        }.get(name, "Meaning check failed")
    r = rest
    if r == "unchanged":
        return "The rule produced the same text"
    if r.startswith("protected span"):
        return "It would touch protected text (numbers, formulas, quotes, acronyms)"
    if r.startswith("numbers"):
        return "A number would change"
    if r.startswith("scope words changed") or r.startswith("scope words lost"):
        return "A logic word (not, all, only, may, ...) would change"
    if r.startswith("word content"):
        return "Words other than the rule's own would be added or lost"
    if r.startswith("edit count"):
        return "Too many or too few word swaps"
    if r.startswith("length ratio"):
        return "The rewrite would be much longer or shorter"
    if r.startswith("punctuation"):
        return "It would create a punctuation error"
    if r.startswith("doubled"):
        return "It would create a doubled word"
    if r.startswith("article_an"):
        return "It would create 'an' before a consonant sound"
    return "Rejected by a rule check"


def short_code(code: str) -> str:
    kind, _, rest = code.partition(":")
    head = re.split(r"[ {=]", rest.strip())[0]
    if head == "scope":
        head = "scope_words"
    elif head == "word":
        head = "word_content"
    elif head == "length":
        head = "length_ratio"
    elif head == "protected":
        head = "protected"
    return f"{kind}:{head}"


TOK = re.compile(r"\s+|[A-Za-z0-9'’-]+|[^\sA-Za-z0-9]")


def word_diff(a: str, b: str):
    """Word-level diff as segments: [op, text] with op in eq / del / ins."""
    ta, tb = TOK.findall(a), TOK.findall(b)
    ka = [t.lower() if not t.isspace() else " " for t in ta]
    kb = [t.lower() if not t.isspace() else " " for t in tb]
    out = []
    sm = difflib.SequenceMatcher(None, ka, kb, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            out.append(["eq", "".join(ta[i1:i2]), "".join(tb[j1:j2])])
        else:
            if i2 > i1:
                out.append(["del", "".join(ta[i1:i2])])
            if j2 > j1:
                out.append(["ins", "".join(tb[j1:j2])])
    return out


def diff_for(text, v):
    if v["rule"] == "dis.frame" and v["rewrite"].startswith(v["frame"]):
        rest = v["rewrite"][len(v["frame"]):]
        return [["ins", v["frame"]], ["eq", text if rest.strip() == text else rest, rest]]
    return word_diff(text, v["rewrite"])


def checks_table(s, th, rule, hard_ok=True):
    """Ledger rows: [label, value, threshold, ok]. ok None = not used for this rule."""
    reorder = rule in R.REORDER
    g_min = th["sbert_global_min"]
    if s.get("single_sentence") and th["sbert_local_min"] > 0:
        g_min = min(g_min, th["sbert_local_min"])
    rows = [["Hard checks", "pass" if hard_ok else "fail", "all rules", hard_ok],
            ["SBERT whole text", s["sbert_global"], f">= {g_min:.2f}", s["sbert_global"] >= g_min]]
    if reorder:
        rows.append(["SBERT changed part", s["sbert_local"], "not used", None])
        rows.append(["Entailment, original to rewrite", None, "not used (no content words change)", None])
        rows.append(["Entailment, rewrite to original", None, "not used", None])
    else:
        rows.append(["SBERT changed part", s["sbert_local"], f">= {th['sbert_local_min']:.2f}",
                     s["sbert_local"] >= th["sbert_local_min"]])
        if s["ent_fwd"] is None:      # stopped at SBERT, NLI not run
            rows.append(["Entailment, original to rewrite", None, "not reached", None])
            rows.append(["Entailment, rewrite to original", None, "not reached", None])
            return rows
        rows.append(["Entailment, original to rewrite", s["ent_fwd"], f">= {th['entail_min']:.2f}",
                     s["ent_fwd"] >= th["entail_min"]])
        rows.append(["Entailment, rewrite to original", s["ent_bwd"], f">= {th['entail_min']:.2f}",
                     s["ent_bwd"] >= th["entail_min"]])
        c = max(s["con_fwd"], s["con_bwd"])
        rows.append(["Contradiction (higher way)", round(c, 4), f"<= {th['contra_max']:.2f}", c <= th["contra_max"]])
    return rows


class Demo:
    def __init__(self, device="cpu"):
        t0 = time.time()
        self.scorer = C.Scorer(R.CFG, device=device, batch_size=16)
        E.nlp()
        from phase1_bridge import Phase1
        self.phase1 = Phase1(self.scorer)
        self._cache = {}
        self.load_seconds = time.time() - t0
        self.device = device

    def run(self, text: str):
        """Output is deterministic per text, so repeat clicks are served from a small cache."""
        key = text.strip()
        if key in self._cache:
            out = dict(self._cache[key])
            out["timing"] = dict(out["timing"], total_s=0.0, cached=True)
            return out
        out = self._run(text)
        if len(self._cache) > 200:
            self._cache.clear()
        self._cache[key] = out
        return out

    def _run(self, text: str):
        t0 = time.time()
        text = text.strip()
        item = {"id": "demo:" + text[:200], "question": text, "source": "demo"}
        opts = R.options(item)
        opts["lexical"] += DR.lexical_options(text)
        opts["syntactic"] += DR.syntactic_options(text)
        axes, to_score = {}, []
        for axis in AXES:
            rec = {"axis": axis, "accepted": None, "also_passed": [], "rejected": [], "cards": [],
                   "candidates": 0, "none_reason": None}
            seen = set()
            for v in opts[axis]:
                if v["rewrite"] in seen or v["rewrite"].strip() == text:
                    continue
                seen.add(v["rewrite"])
                rec["candidates"] += 1
                fail = DR.hard_check(text, v, hard_check) if v.get("demo") else hard_check(text, v)
                if fail:
                    code = "hard:" + fail
                    rec["rejected"].append({"rule": v["rule"], "rule_text": rule_text(v["rule"]),
                                            "rewrite": v["rewrite"], "diff": diff_for(text, v),
                                            "code": short_code(code), "detail": fail, "reason": plain_reason(code),
                                            "checks": None})
                else:
                    to_score.append((rec, v))
            axes[axis] = rec
        t1 = time.time()
        # stage 1: SBERT for every candidate (cheap)
        locs = [C.local_pair(text, v) for _, v in to_score]
        g = self.scorer.sbert_cos([(text, v["rewrite"]) for _, v in to_score]) if to_score else []
        l = self.scorer.sbert_cos(locs) if to_score else []
        staged = {a: [] for a in AXES}
        for (rec, v), lp, gg, ll in zip(to_score, locs, g, l):
            s = {"sbert_global": round(gg, 4), "sbert_local": round(ll, 4), "ent_fwd": 1.0, "ent_bwd": 1.0,
                 "con_fwd": 0.0, "con_bwd": 0.0}
            if text.strip() == lp[0].strip():
                s["single_sentence"] = True
            th = DEMO_TH[v["axis"]] if v.get("demo") else R.th_for(v)
            fail = C.semantic_fail(s, "x", {"x": th})
            if fail:
                s.update(ent_fwd=None, ent_bwd=None, con_fwd=None, con_bwd=None)
                self._reject(text, rec, v, s, th, fail)
            else:
                staged[v["axis"]].append((rec, v, s, th, lp))
        # stage 2: NLI only on the best few per axis (v3 first, then demo rules by SBERT), in one batch
        nli_jobs = []
        for axis, lst in staged.items():
            v3 = [x for x in lst if not x[1].get("demo")]
            dm = sorted([x for x in lst if x[1].get("demo")], key=lambda x: -x[2]["sbert_global"])
            n = 0
            for x in v3 + dm:
                if x[1]["rule"] in R.REORDER:
                    self._decide(text, *x[:4])
                elif n < NLI_CAP[axis] - (1 if len(text) > 200 and axis != "discourse" else 0):
                    nli_jobs.append(x)
                    n += 1
        if nli_jobs:
            w = lambda t: R.CFG["models"]["nli"]["wrapper"].format(text=t)
            fwd = [(w(x[4][0]), w(x[4][1])) for x in nli_jobs]
            bwd = [(w(x[4][1]), w(x[4][0])) for x in nli_jobs]
            probs = self.scorer.nli_probs(fwd + bwd)
            k = len(nli_jobs)
            for j, (rec, v, s, th, lp) in enumerate(nli_jobs):
                (ef, cf), (eb, cb) = probs[j], probs[k + j]
                s.update(ent_fwd=round(ef, 4), ent_bwd=round(eb, 4), con_fwd=round(cf, 4), con_bwd=round(cb, 4))
                self._decide(text, rec, v, s, th)
        t2 = time.time()
        for axis in ("lexical", "syntactic"):
            rec = axes[axis]
            rec["cards"].sort(key=lambda e: e["demo"])          # v3 first (stable)
            rec["cards"] = rec["cards"][:MAX_CARDS]
            rec["accepted"] = rec["cards"][0] if rec["cards"] else None
        # Pragmatic axis (demo name for discourse): every passing v3 discourse rewrite becomes a card,
        # then one Phase 1 rewrite per strategy (typo, instruction prefix, distracting sentence).
        d = axes["discourse"]
        for e in d["cards"]:
            e.update(label=CARD_LABEL.get(e["rule"], e["rule_text"]), checkset="v3 discourse checks")
        d["accepted"] = d["cards"][0] if d["cards"] else None
        t3 = time.time()
        cards, rej = self.phase1.run(text)
        d["cards"] += cards
        d["rejected"] += rej
        t4 = time.time()
        for axis, rec in axes.items():
            if rec["cards"]:
                continue
            if rec["candidates"] == 0:
                why = {"lexical": "no word or phrase here is in the synonym lists",
                       "syntactic": "no matching sentence pattern",
                       "discourse": "no matching pattern"}[axis]
                rec["none_reason"] = f"No {axis} rewrite applicable: {why}"
            else:
                rec["none_reason"] = f"No {axis} rewrite kept: every candidate failed a check"
        return {"text": text, "axes": [axes[a] for a in AXES],
                "timing": {"rules_s": round(t1 - t0, 3), "models_s": round(t2 - t1, 3),
                           "pragmatic_s": round(t4 - t3, 3), "total_s": round(time.time() - t0, 3),
                           "scored": len(to_score), "nli": len(nli_jobs), "device": self.device}}

    def _reject(self, text, rec, v, s, th, fail):
        code = "semantic:" + fail
        rec["rejected"].append({"rule": v["rule"], "rule_text": rule_text(v["rule"]), "rewrite": v["rewrite"],
                                "diff": diff_for(text, v), "checks": checks_table(s, th, v["rule"]),
                                "passed": False, "code": short_code(code), "detail": fail,
                                "reason": plain_reason(code), "demo": bool(v.get("demo"))})

    def _decide(self, text, rec, v, s, th):
        fail = C.semantic_fail(s, "x", {"x": th})
        if fail:
            return self._reject(text, rec, v, s, th, fail)
        demo = bool(v.get("demo"))
        rec["cards"].append({"rule": v["rule"], "rule_text": rule_text(v["rule"]), "label": rule_text(v["rule"]),
                             "rewrite": v["rewrite"], "diff": diff_for(text, v), "demo": demo,
                             "checks": checks_table(s, th, v["rule"]), "passed": True,
                             "checkset": "demo rule \u00b7 relaxed thresholds" if demo else "v3 rule \u00b7 v3 thresholds"})
