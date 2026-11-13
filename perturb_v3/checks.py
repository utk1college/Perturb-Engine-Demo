"""Checks for v3 rewrites. Hard checks are rule-based; meaning checks use encoder models that only
SCORE text (SBERT cosine + two-way NLI). Nothing here writes or edits text."""
from __future__ import annotations

import re
from collections import Counter

from perturb_v3 import engine as E

SCOPE = {"not", "no", "never", "without", "except", "only", "all", "every", "each", "any", "some", "none", "nothing",
         "both", "either", "neither", "most", "few", "many", "more", "less", "least", "always", "sometimes", "nor",
         "unless", "if", "or", "and", "but", "must", "should", "might", "may", "cannot", "can", "could", "would",
         "will", "true", "false", "same", "different", "than"}
BAD_PUNCT = re.compile(r"[.?!,;:]\s*[.?!,;:]|[ \t][,.;:?!]|  +")
EXPAND = {k.lower(): v.lower() for k, v in E.CONTRACTIONS.items()}

# words each rule may add / remove (beyond the moved text itself)
ALLOWED = {
    "syn.wh_to_imperative": ({"explain"}, {"does", "do", "did"}),
    "syn.polar_to_whether": ({"explain", "whether"}, {"does", "do", "did"}),
    "syn.why_to_reason": ({"what", "is", "the", "reason", "that"}, {"why", "does", "do", "did"}),
    "syn.can_to_possible": ({"is", "it", "possible", "for", "to", "me", "us", "them", "him", "her"},
                            {"can", "i", "we", "they", "he", "she"}),
    "syn.how_in_what_way": ({"in", "what", "way", "how"}, {"how", "in", "what", "way", "ways"}),
    "syn.move_clause": (set(), set()),
    "dis.title_to_end": (set(), set()),
    "dis.question_first": (set(), set()),
    "dis.frame": ({w for f in E.FRAMES for w in re.findall(r"[a-z]+", f.lower())}, set()),
}
# the scope-word list includes 'can'/'could'..., which these rules legitimately turn into other words
_WONDER = ["i was wondering", "i am wondering", "i'm wondering", "i am curious", "i'm curious"]
LEX_SCOPE_OK = {**{(w, "i would like to know"): {"would"} for w in _WONDER},
                ("various", "different"): {"different"}, ("whether or not", "whether"): {"or", "not"}}
SCOPE_EXEMPT ={"syn.can_to_possible": {"can"}, "syn.polar_to_whether": set(), "syn.why_to_reason": set()}


def norm_words(text: str) -> Counter:
    t = text.lower()
    for k, v in EXPAND.items():
        t = re.sub(r"(?<![\w'])" + re.escape(k) + r"(?![\w'])", v, t)
    return Counter(re.findall(r"[a-z]+", t))


def hard_check(original: str, v: dict):
    new, rule = v["rewrite"], v["rule"]
    if new.strip() == original.strip():
        return "unchanged"
    if E.protected_texts(original) != E.protected_texts(new) and rule != "dis.drop_pleasantry":
        return "protected span changed"
    nums = lambda s: Counter(re.findall(r"\d+(?:\.\d+)?", s))
    if nums(original) != nums(new):
        return "numbers changed"
    wo, wn = norm_words(original), norm_words(new)
    ex = SCOPE_EXEMPT.get(rule, set())
    so = Counter({w: c for w, c in wo.items() if w in SCOPE - ex})
    sn = Counter({w: c for w, c in wn.items() if w in SCOPE - ex})
    if v["axis"] == "lexical":
        # scope words a specific swap legitimately adds/removes (and nothing else)
        for e in v.get("edits", []):
            f_, t_ = e["from"].lower(), e["to"].lower()
            for w in LEX_SCOPE_OK.get((f_, t_), set()):
                so[w] = max(0, so[w] - norm_words(f_)[w]); sn[w] = max(0, sn[w] - norm_words(t_)[w])
        so, sn = +so, +sn
    if v["axis"] != "discourse" and so != sn:
        return f"scope words changed {dict((so - sn) + (sn - so))}"
    if v["axis"] == "discourse" and rule != "dis.drop_pleasantry" and (so - sn):
        return "scope words lost"
    if v["axis"] in ("syntactic", "discourse") and rule in ALLOWED:
        add, rem = ALLOWED[rule]
        lost = {w: c for w, c in (wo - wn).items() if w not in rem}
        gained = {w: c for w, c in (wn - wo).items() if w not in add}
        # verb re-inflection under do-support (affects->affect) is allowed: compare stems
        if rule in ("syn.wh_to_imperative", "syn.polar_to_whether", "syn.why_to_reason"):
            from lemminflect import getLemma
            stem = lambda w: (getLemma(w, upos="VERB") or (w,))[0]
            ls, gs = Counter(stem(w) for w in lost for _ in range(lost[w])), Counter(stem(w) for w in gained for _ in range(gained[w]))
            if ls == gs:
                lost, gained = {}, {}
        if lost or gained:
            return f"word content changed lost={lost} gained={gained}"
    if v["axis"] == "lexical":
        n = len(v.get("edits", []))
        if not 1 <= n <= E.MAX_SUBS:
            return "edit count"
    ratio = len(new) / max(1, len(original))
    if not 0.75 <= ratio <= 1.5 and rule != "dis.frame":  # frames add one fixed neutral sentence (checked via ALLOWED)
        return f"length ratio {ratio:.2f}"
    lo, changed = local_pair(original, v)
    glitches = lambda s: len(BAD_PUNCT.findall(s.replace("\n", " ")))
    if glitches(changed) > glitches(lo):  # only glitches the rewrite introduced
        return "punctuation glitch"
    dbl = lambda s: len(re.findall(r"\b(\w+) \1\b", s.lower()))
    if dbl(changed) > dbl(lo):
        return "doubled word"
    return None


def local_pair(original: str, v: dict):
    """The smallest (original, rewrite) text pair that contains the change, for NLI."""
    if v["axis"] == "syntactic":
        return v["from"].strip(), v["to"].strip()
    if v["axis"] == "lexical":
        spans = E.sentences(original)
        hit = [(a, b) for a, b in spans if any(a <= e["start"] < b for e in v["edits"])]
        if not hit:
            return original, v["rewrite"]
        a, b = min(x[0] for x in hit), max(x[1] for x in hit)
        delta = sum(len(e["to"]) - (e["end"] - e["start"]) for e in v["edits"] if a <= e["start"] < b)
        # article fixes shift text too; recompute from the rewrite by locating the same suffix
        tail = len(original) - b
        return original[a:b], v["rewrite"][a:len(v["rewrite"]) - tail]
    if v["rule"] == "dis.frame":
        first = E.sentences(original)[0]
        return original[first[0]:first[1]], v["frame"] + " " + original[first[0]:first[1]]
    return original, v["rewrite"]


class Scorer:
    def __init__(self, cfg, device="cpu", batch_size=32):
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        m = cfg["models"]
        self.torch = torch
        torch.set_num_threads(max(1, (torch.get_num_threads())))
        self.sbert = SentenceTransformer(m["sbert"]["name"], revision=m["sbert"]["revision"], device=device)
        self.tok = AutoTokenizer.from_pretrained(m["nli"]["name"], revision=m["nli"]["revision"])
        self.nli = AutoModelForSequenceClassification.from_pretrained(m["nli"]["name"], revision=m["nli"]["revision"]).to(device).eval()
        lab = {v.lower(): int(k) for k, v in self.nli.config.id2label.items()}
        self.i_ent, self.i_con = lab["entailment"], lab["contradiction"]
        self.wrap, self.bs, self.device = m["nli"]["wrapper"], batch_size, device

    def sbert_cos(self, pairs):
        a = self.sbert.encode([p[0] for p in pairs], batch_size=self.bs, normalize_embeddings=True, convert_to_tensor=True)
        b = self.sbert.encode([p[1] for p in pairs], batch_size=self.bs, normalize_embeddings=True, convert_to_tensor=True)
        return (a * b).sum(-1).float().cpu().tolist()

    def nli_probs(self, prem_hyp):
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(prem_hyp), self.bs):
                ch = prem_hyp[i:i + self.bs]
                enc = self.tok([p for p, _ in ch], [q for _, q in ch], padding=True, truncation=True, max_length=256,
                               return_tensors="pt").to(self.device)
                pr = self.torch.softmax(self.nli(**enc).logits.float(), -1).cpu()
                out += [(float(r[self.i_ent]), float(r[self.i_con])) for r in pr]
        return out

    def score(self, items):
        """items: list of (original_full, rewrite_full, local_orig, local_new, skip_nli). Returns score dicts."""
        if not items:
            return []
        g = self.sbert_cos([(o, r) for o, r, _, _, _ in items])
        l = self.sbert_cos([(a, b) for _, _, a, b, _ in items])
        w = lambda t: self.wrap.format(text=t)
        idx = [i for i, x in enumerate(items) if not x[4]]
        f_ = self.nli_probs([(w(items[i][2]), w(items[i][3])) for i in idx])
        b_ = self.nli_probs([(w(items[i][3]), w(items[i][2])) for i in idx])
        fwd, bwd = [(1.0, 0.0)] * len(items), [(1.0, 0.0)] * len(items)
        for j, i in enumerate(idx):
            fwd[i], bwd[i] = f_[j], b_[j]
        return [{"sbert_global": round(x, 4), "sbert_local": round(y, 4), "ent_fwd": round(f[0], 4),
                 "ent_bwd": round(b_[0], 4), "con_fwd": round(f[1], 4), "con_bwd": round(b_[1], 4)}
                for x, y, f, b_ in zip(g, l, fwd, bwd)]


def semantic_fail(s, axis, th):
    t = th[axis]
    g_min = t["sbert_global_min"]
    if s.get("single_sentence"):  # whole text IS the changed sentence: global == local, use the local bar
        g_min = min(g_min, t["sbert_local_min"]) if t["sbert_local_min"] > 0 else g_min
    if s["sbert_global"] < g_min:
        return f"sbert_global {s['sbert_global']:.3f}"
    if s["sbert_local"] < t["sbert_local_min"]:
        return f"sbert_local {s['sbert_local']:.3f}"
    if min(s["ent_fwd"], s["ent_bwd"]) < t["entail_min"]:
        return f"entail {min(s['ent_fwd'], s['ent_bwd']):.3f}"
    if max(s["con_fwd"], s["con_bwd"]) > t["contra_max"]:
        return f"contradiction {max(s['con_fwd'], s['con_bwd']):.3f}"
    return None
