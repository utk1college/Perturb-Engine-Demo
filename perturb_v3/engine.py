"""Perturb-Engine v3: deterministic, meaning-preserving rewrites of real exam and forum questions.

Three axes, never mixed in one variant:
  lexical    - up to 3 word/phrase substitutions from curated tables (no domain terms)
  syntactic  - one structural change to one question sentence
  discourse  - one change to how the whole question is presented (order or framing)

No generative model writes or edits any text. Encoder models (checks.py) only *check* rewrites.
Everything inside protected spans (maths, code, URLs, quotes, numbers) is never touched.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from functools import lru_cache

SPACY_MODEL = "en_core_web_lg"
MAX_SUBS = 3
_NLP = None


def nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load(SPACY_MODEL)
    return _NLP


@lru_cache(maxsize=50000)
def parse(text: str):
    return nlp()(text)


def h(*parts) -> int:
    """Stable hash for deterministic choices (seeded by item id)."""
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def case_like(src: str, rep: str) -> str:
    if src.isupper() and len(src) > 1:
        return rep.upper()
    return cap(rep) if src[:1].isupper() else rep


# ============================================================ protected spans
PROTECT_RES = [
    re.compile(r"\$\$.+?\$\$", re.S), re.compile(r"(?<!\\)\$[^$\n]+?\$"), re.compile(r"\\\(.+?\\\)", re.S),
    re.compile(r"\\\[.+?\\\]", re.S), re.compile(r"\\begin\{(\w+\*?)\}.+?\\end\{\1\}", re.S),
    re.compile(r"\\[A-Za-z]+(?:\{[^}]*\})*"),                      # stray LaTeX commands
    re.compile(r"`[^`\n]+`"), re.compile(r"https?://\S+|www\.\S+"),
    re.compile(r'"[^"\n]{1,300}"|“[^”\n]{1,300}”|‘[^’\n]{1,300}’|(?<![\w])\'[^\'\n]{1,200}\'(?![\w])'),
    re.compile(r"\S*\d(?:\S*[^\s.,;:?!)\"'])?"),                    # anything containing a digit (minus trailing punct)
    re.compile(r"\b[A-Z]{2,}[A-Za-z]*\b"),                          # acronyms
    re.compile(r"\([^()\n]{0,80}\)"),                               # parentheticals
]
CODE_LINE = re.compile(r"[;{}=<>\[\]]|^\s*(def|class|import|for|while|if|return|int|var|let|const)\b")


def protected_spans(text: str):
    spans = []
    for rx in PROTECT_RES:
        spans += [(m.start(), m.end()) for m in rx.finditer(text)]
    pos = 0
    strip_rx = PROTECT_RES[:7] + [PROTECT_RES[-1]]  # LaTeX, backticks, URLs, quotes, parentheticals
    for line in text.split("\n"):
        clean = line
        for rx in strip_rx:
            clean = rx.sub(" ", clean)
        sym = len(re.findall(r"[;{}=<>\[\]]", clean))
        n_words = len(re.findall(r"[A-Za-z]{2,}", clean))
        # code: symbol-dense lines (prose with LaTeX/parentheticals is not code), or short keyword-led lines
        if line.strip() and ((sym >= 3 and sym >= 0.15 * max(1, n_words))
                             or (CODE_LINE.search(clean) and sym >= 1 and len(line.split()) < 12)):
            spans.append((pos, pos + len(line)))
        pos += len(line) + 1
    return merge(spans)


def merge(spans):
    out = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def overlaps(a, b, spans) -> bool:
    return any(a < y and x < b for x, y in spans)


def protected_texts(text: str) -> Counter:
    return Counter(text[a:b] for a, b in protected_spans(text))


# ============================================================ sentences
def paragraphs(text: str):
    """(start, end) of non-empty lines."""
    out, pos = [], 0
    for line in text.split("\n"):
        if line.strip():
            out.append((pos, pos + len(line)))
        pos += len(line) + 1
    return out


def sentences(text: str):
    """(start, end) char spans of sentences, split per line then by spaCy."""
    out = []
    for a, b in paragraphs(text):
        doc = parse(text[a:b])
        for s in doc.sents:
            st = s.text.strip()
            if st:
                off = a + s.start_char + (len(s.text) - len(s.text.lstrip()))
                out.append((off, off + len(st)))
    return out


INSTR_VERBS = {"explain", "describe", "discuss", "outline", "analyze", "analyse", "examine", "evaluate", "assess",
               "compare", "contrast", "illustrate", "justify", "elaborate", "argue", "summarize", "summarise",
               "define", "identify", "state", "show", "prove", "derive", "determine", "give", "list", "name"}


def target_sentences(text: str, title: str | None):
    """Question sentences: the title, sentences ending in '?', instruction sentences."""
    prot = protected_spans(text)
    out = []
    for a, b in sentences(text):
        s = text[a:b]
        first = re.match(r"[A-Za-z]+", s)
        is_q = s.rstrip().endswith("?")
        is_i = bool(first) and first.group(0).lower() in INSTR_VERBS
        is_title = bool(title) and a == 0
        if (is_q or is_i or is_title) and 5 <= len(s.split()) <= 45:
            out.append((a, b))
    return out


# ============================================================ LEXICAL axis
# Phrase table: (regex, replacement). Case of the first letter is preserved. Applied outside protected spans.
CONTRACTIONS = {"can't": "cannot", "don't": "do not", "doesn't": "does not", "didn't": "did not", "isn't": "is not",
                "aren't": "are not", "wasn't": "was not", "weren't": "were not", "won't": "will not",
                "wouldn't": "would not", "couldn't": "could not", "shouldn't": "should not", "haven't": "have not",
                "hasn't": "has not", "hadn't": "had not", "I'm": "I am", "I've": "I have", "we're": "we are",
                "they're": "they are", "you're": "you are", "I'd like": "I would like"}
PHRASES = [  # (source, target); listed both ways where the swap is symmetric
    ("for example", "for instance"), ("for instance", "for example"), ("e.g.", "for example"),
    ("i.e.", "that is"), ("in order to", "so as to"), ("so as to", "in order to"),
    ("additionally", "in addition"), ("in addition", "additionally"), ("moreover", "furthermore"),
    ("furthermore", "moreover"), ("in other words", "put differently"), ("basically", "essentially"),
    ("essentially", "basically"), ("thanks", "thank you"), ("I was wondering", "I would like to know"),
    ("I am wondering", "I would like to know"), ("I'm wondering", "I would like to know"),
    ("I am curious", "I would like to know"), ("I'm curious", "I would like to know"),
    ("whether or not", "whether"), ("as far as I know", "to my knowledge"), ("to my knowledge", "as far as I know"),
    ("in general", "generally"), ("generally speaking", "in general"), ("perhaps", "maybe"), ("maybe", "perhaps"),
    ("a bit", "slightly"),
    # instruction / template phrases (exam + MedQuAD)
    ("give an example", "provide an example"), ("provide an example", "give an example"),
    ("give examples", "provide examples"), ("provide examples", "give examples"),
    ("what is the outlook for", "what is the prognosis for"),
    ("genetic changes related to", "genetic changes associated with"),
]
# context guards: phrase -> (regex the following text must NOT match, regex the preceding text must NOT match)
PHRASE_GUARD = {
    "a bit": (r"\s+of\b", r"\bquite\s+$"),                 # 'a bit of X' / 'quite a bit' are quantities
    "thanks": (r"\s+(a|so|tons|heaps|loads)\b", None),       # 'thank you a lot' is unidiomatic
    **{w: (r"(?!\s*(if|why|what|how|whether|when|where|who|which|about)\b|\s*[,:])", None)
       for w in ("I was wondering", "I am wondering", "I'm wondering", "I am curious", "I'm curious")},
    # only before a question word: not 'I'm curious as to why' / 'curious to know' -> 'would like to know to know'
    "in addition": (r"\s+to\b", None),                     # 'in addition to X'
    "in general": (r"\s+[a-z]", None),                     # 'in general relativity'; only before punctuation
    "generally speaking": (r"\s+[a-z]", None),
    "e.g.": (None, r"[^(,\s]\s*$"),                       # only after '(' or ',' ('to e.g. X' -> 'to for example X')
    "i.e.": (None, r"[^(,\s]\s*$"),
}
# word tables (spaCy-checked POS; nouns only in an 'X of/in/...' frame, as in the earlier engine)
LEX_NOUN = {"advantages": "benefits", "benefits": "advantages", "disadvantages": "drawbacks",
            "drawbacks": "disadvantages", "challenges": "difficulties", "difficulties": "challenges",
            "limitations": "shortcomings", "shortcomings": "limitations", "differences": "distinctions",
            "concept": "notion", "notion": "concept", "strategies": "approaches", "purpose": "aim"}
NOUN_FRAME = {"of", "on", "in", "between", "for", "to", "behind"}
LEX_ADJ = {"various": "different", "crucial": "essential", "huge": "enormous", "enormous": "huge",
           "entire": "whole", "main": "principal"}
MAIN_OK = {"features", "feature", "reason", "reasons", "idea", "ideas", "difference", "differences", "point", "points", "goal", "goals", "purpose", "aim", "aims", "focus", "challenge", "challenges", "advantage", "advantages", "issue", "issues", "problem", "problems", "concern", "concerns", "argument", "arguments", "cause", "causes", "factor", "factors", "objective", "objectives", "characteristics", "aspects", "findings", "contribution", "contributions", "source", "sources", "types", "steps", "principles", "themes", "drawbacks", "limitations", "benefits", "risks", "elements", "components", "functions", "roles"}
FIXED_BIGRAMS = {("main", "memory"), ("main", "sequence"), ("main", "effect"), ("main", "effects"),
                 ("main", "diagonal"), ("main", "theorem"), ("entire", "function"), ("entire", "functions"),
                 ("various", "artists")}
LEX_VERB = {"affect": "influence", "influence": "affect", "enhance": "improve", "improve": "enhance",
            "utilize": "use", "utilise": "use", "suppose": "assume",
            "assume": "suppose", "allow": "permit", "permit": "allow", "choose": "select", "occur": "happen",
            "notice": "observe", "mention": "note", "calculate": "compute", "compute": "calculate",
            "assist": "help"}
# verbs that are only safe in a given frame: lemma -> (replacement, allowed next-token set or None=any non-ADP)
LEX_VERB_FRAMED = {"try": ("attempt", {"to"}), "want": ("wish", {"to"}), "seem": ("appear", {"to"}),
                   "happen": ("occur", None), "think": ("believe", {"that"}), "help": ("assist", None),
                   "start": ("begin", {"to", "with", "by"}), "begin": ("start", {"to", "with", "by"}),  # not "start the engine"
                   "attempt": ("try", {"to"})}                                  # not "attempted murder"
LEX_ADV = {"entirely": "completely", "completely": "entirely", "quickly": "rapidly", "rapidly": "quickly",
           "usually": "typically", "typically": "usually", "eventually": "ultimately", "ultimately": "eventually",
           "exactly": "precisely", "precisely": "exactly", "almost": "nearly", "nearly": "almost",
           "therefore": "thus", "thus": "therefore", "actually": "in fact",
           "obviously": "evidently", "apparently": "seemingly"}
LEX_ADJ.update({"obvious": "evident", "evident": "obvious", "strange": "peculiar", "weird": "strange",
                "confusing": "puzzling", "useful": "helpful", "helpful": "useful", "typical": "usual",
                "usual": "typical", "specific": "particular", "particular": "specific", "easiest": "simplest"})
FIXED_BIGRAMS |= {("particular", "solution"), ("particular", "integral"), ("specific", "heat"),
                  ("specific", "gravity"), ("specific", "volume"), ("specific", "energy"), ("specific", "impulse")}


def _inflect(lemma, tag):
    from lemminflect import getInflection
    if tag in {"VB", "VBP"}:
        return lemma
    f = getInflection(lemma, tag=tag)
    return f[0] if f else None


def nxt_is_punct(doc, t):
    return t.i + 1 >= len(doc) or doc[t.i + 1].is_punct


def participle_adj(doc, t):
    """A past participle used as an adjective before a noun ('the supposed exposure', 'attempted murder',
    'improved fit'): swapping the verb changes meaning or reads badly."""
    if t.tag_ != "VBN" or t.i + 1 >= len(doc):
        return False
    prev_aux = t.i > 0 and doc[t.i - 1].lemma_ in {"have", "be", "get"}
    return doc[t.i + 1].pos_ in {"NOUN", "PROPN", "ADJ"} and not prev_aux or t.dep_ == "amod"


def lexical_candidates(text: str):
    prot = protected_spans(text)
    cands = []  # (priority, start, end, replacement, kind)
    low = text.lower()
    for src, tgt in sorted(CONTRACTIONS.items()):
        for m in re.finditer(r"(?<![\w'])" + re.escape(src.lower()) + r"(?![\w'])", low):
            if overlaps(m.start(), m.end(), prot):
                continue
            # inverted questions keep the contraction: "Why don't they...", "Isn't it..." (not "Why do not they")
            if re.match(r"\s+[A-Z]", text[m.end():]) and text[m.start()].isupper():   # Title Case headings
                continue
            if src.lower().endswith("n't") and re.search(r"(^|[.?!:\n]\s*|\b(why|how|what|where|when|who|which|so|then)\s+)$",
                                                          low[:m.start()]):
                continue
            # any inversion: "..., but shouldn't the acid form?" (n't directly before its subject)
            if src.lower().endswith("n't") and re.match(
                    r"\s+(i|you|he|she|it|we|they|there|the|a|an|this|that|these|those|my|your|our|their|his|her|its|"
                    r"anyone|anybody|someone|somebody|everyone|people|one)\b", low[m.end():]):
                continue
            cands.append((3, m.start(), m.end(), case_like(text[m.start():m.end()], tgt), "contraction"))
    for src, tgt in PHRASES:
        for m in re.finditer(r"(?<![\w.])" + re.escape(src.lower()) + r"(?![\w])", low):
            if overlaps(m.start(), m.end(), prot):
                continue
            if src in ("thanks",) and re.match(r"\s+to\b", low[m.end():]):   # 'thanks to X' is not gratitude
                continue
            g_next, g_prev = PHRASE_GUARD.get(src, (None, None))
            if (g_next and re.match(g_next, low[m.end():])) or (g_prev and re.search(g_prev, low[:m.start()])):
                continue
            cands.append((2, m.start(), m.end(), case_like(text[m.start():m.end()], tgt), "phrase"))
    for a, b in paragraphs(text):
        doc = parse(text[a:b])
        for t in doc:
            s, e = a + t.idx, a + t.idx + len(t.text)
            if overlaps(s, e, prot) or text[s - 1:s] == "-" or text[e:e + 1] == "-":   # not inside 'above-mentioned'
                continue
            w, nxt =t.lower_, (doc[t.i + 1].lower_ if t.i + 1 < len(doc) else "")
            rep = None
            if t.pos_ == "NOUN" and w in LEX_NOUN and nxt in NOUN_FRAME \
                    and not (w == "purpose" and re.search(r"\b(for|on) the\s+$", low[:s])):   # 'for the purpose of'
                rep = LEX_NOUN[w]
            elif t.pos_ == "ADJ" and w in LEX_ADJ and (w, nxt) not in FIXED_BIGRAMS \
                    and not (t.i > 0 and doc[t.i - 1].lower_ == "in") \
                    and not (w == "main" and nxt not in MAIN_OK) \
                    and not (w in {"specific", "particular"} and (t.dep_ == "acomp" or nxt_is_punct(doc, t)
                             or (t.i > 0 and doc[t.i - 1].lower_ in {"more", "very", "too", "less"}))):
                rep = LEX_ADJ[w]                                                  # 'be more specific.' is not 'particular'
            elif t.pos_ == "VERB" and t.lemma_.lower() in LEX_VERB and t.tag_.startswith("VB") \
                    and not (t.lemma_.lower() in {"suppose", "assume"} and nxt == "to") \
                    and not (t.lemma_.lower() in {"occur", "happen"} and nxt == "to") \
                    and not (t.lemma_.lower() in {"allow", "permit"} and nxt == "for") \
                    and not participle_adj(doc, t):
                # 'supposed/assumed to' (expected), 'it occurred to me', 'allow for' (= account for), 'attempted murder'
                rep = _inflect(LEX_VERB[t.lemma_.lower()], t.tag_)
            elif t.pos_ == "VERB" and t.lemma_.lower() in LEX_VERB_FRAMED and t.tag_.startswith("VB"):
                tgt, frame = LEX_VERB_FRAMED[t.lemma_.lower()]
                nxt_tok = doc[t.i + 1] if t.i + 1 < len(doc) else None
                if frame:
                    ok = nxt in frame
                elif t.lemma_.lower() == "happen":   # 'happen upon', 'happens to be' are not 'occur'
                    ok = nxt_tok is None or nxt_tok.is_punct or nxt in HAPPEN_NEXT
                else:                                # help -> assist only with a direct object ('help me')
                    ok = nxt_tok is not None and nxt_tok.pos_ in {"PRON", "DET", "PROPN", "NOUN"} \
                        and not any(x.lower_ in {"not", "n't", "cannot"} for x in doc[max(0, t.i - 2):t.i]) \
                        and not any(x.tag_ in {"VB", "VBP"} and doc[x.i - 1].lower_ != "to"
                                    for x in doc[t.i + 2:t.i + 5])       # 'help us determine' -> no 'assist us determine'
                if ok and not (t.lemma_.lower() == "help" and nxt == "with"):
                    rep = _inflect(tgt, t.tag_)
            elif t.pos_ == "ADV" and w in LEX_ADV and not (w == "almost" and nxt in
                                                           {"no", "none", "nothing", "never", "nobody", "nowhere", "surely", "everywhere", "certainly"}):
                rep = LEX_ADV[w]                                                  # 'nearly no' is unidiomatic
            if rep:
                cands.append((1, s, e, case_like(t.text, rep), "word"))
    return cands


def apply_edits(text, edits):
    for s, e, rep in sorted(edits, reverse=True):
        text = text[:s] + rep + text[e:]
    return text


def fix_articles(text, edits):
    """'a' -> 'an' (and back) directly before a substituted word."""
    out = []
    for s, e, rep in edits:
        m = re.search(r"\b(a|an|A|An)\s+$", text[max(0, s - 4):s])
        if m:
            art_s = s - len(m.group(0))
            want = "an" if rep[:1].lower() in "aeiou" else "a"
            if m.group(1).lower() != want:
                out.append((art_s, art_s + len(m.group(1)), case_like(m.group(1), want)))
        out.append((s, e, rep))
    return out


def lexical(item_id: str, text: str):
    cands = lexical_candidates(text)
    if not cands:
        return None, "no lexical candidate"
    # priority: content words, then discourse phrases, then contractions; spread by stable hash
    cands.sort(key=lambda c: (c[0], h(item_id, c[1])))
    chosen = []
    for c in cands:
        if len(chosen) >= MAX_SUBS:
            break
        if any(not (c[2] <= x[1] or x[2] <= c[1]) for x in chosen):
            continue
        chosen.append(c)
    edits = fix_articles(text, [(c[1], c[2], c[3]) for c in chosen])
    new = apply_edits(text, edits)
    return {"axis": "lexical", "rule": "lex." + "+".join(sorted({c[4] for c in chosen})),
            "edits": [{"start": c[1], "end": c[2], "from": text[c[1]:c[2]], "to": c[3], "kind": c[4]} for c in chosen],
            "rewrite": new}, None


# ============================================================ SYNTACTIC axis
AUX_DO = {"does": "VBZ", "do": "VBP", "did": "VBD"}
AUX_OTHER = {"is", "are", "was", "were", "can", "could", "might", "may", "would", "will", "should", "has", "have",
             "had", "must", "shall"}
WH_START = {"how", "why", "what", "which", "where", "when", "who"}
OBJ_PRON = {"i": "me", "we": "us", "they": "them", "he": "him", "she": "her", "you": "you", "it": "it", "one": "one"}


def _ends(doc):
    end = len(doc)
    while end > 0 and doc[end - 1].text in {"?", ".", "!", "?!", "??", "..."}:  # keep ')' and quotes
        end -= 1
    return end


def _span(doc, a, b):
    return doc[a:b].text if a < b else ""


def _coordinated(doc, a, b) -> bool:
    for t in doc[a:b]:
        if t.dep_ == "cc" and t.i + 1 < b and doc[t.i + 1].lower_ in WH_START | {"if", "whether"}:
            return True
    return False


def split_question(sent: str):
    """Decompose a one-sentence question into (kind, wh, subject, aux, rest, declarative).
    kind = 'wh' or 'polar'. Returns None when the parse is not clean enough to rearrange safely."""
    s = sent.strip()
    if s.count("?") != 1 or not s.endswith("?"):
        return None
    doc = parse(s)
    if len(list(doc.sents)) != 1:
        return None
    low = [t.lower_ for t in doc]
    if low[:3] in (["in", "what", "way"], ["in", "what", "ways"], ["to", "what", "extent"]):
        k, kind = 3, "wh"
    elif low[0] in WH_START:
        k, kind = 1, "wh"
        while k < min(len(doc), 3) and low[k] not in AUX_DO and low[k] not in AUX_OTHER \
                and doc[k].pos_ in {"NOUN", "ADJ", "ADV", "DET"}:
            k += 1
    elif low[0] in AUX_DO or low[0] in AUX_OTHER:
        k, kind = 0, "polar"
    else:
        return None
    end = _ends(doc)
    if k >= end - 1 or _coordinated(doc, k, end):
        return None
    roots = [t for t in doc if t.dep_ == "ROOT"]
    if len(roots) != 1 or any(c.dep_ == "conj" and c.pos_ in {"VERB", "AUX"} for c in roots[0].children):
        return None
    wh = (doc[:k].text if doc[0].pos_ == "PROPN" else doc[:k].text.lower()) if k else ""
    aux = doc[k]
    if aux.lower_ in AUX_DO and aux.dep_ == "aux":
        main = aux.head
        if main.pos_ not in {"VERB", "AUX"} or main.dep_ != "ROOT" or main.tag_ not in {"VB", "VBP"}:  # do-support needs a base verb
            return None
        subj = [c for c in main.children if c.dep_ in {"nsubj", "nsubjpass", "expl"}]
        if len(subj) != 1 or subj[0].left_edge.i != k + 1 or subj[0].right_edge.i >= main.i:
            return None
        form = _inflect(main.lemma_, AUX_DO[aux.lower_])
        if not form:
            return None
        if aux.lower_ in ("does", "do") and main.lemma_ == "be":
            return None
        s_hi = subj[0].right_edge.i + 1
        if any(t.dep_ == "neg" for t in doc[s_hi:main.i]):
            # negated do-support keeps the auxiliary: 'did England not maintain' -> 'England did not maintain'
            words = [t.text + t.whitespace_ for t in doc[k + 1:s_hi]] + [aux.lower_ + " "] + \
                    [t.text + t.whitespace_ for t in doc[s_hi:end]]
        else:
            words = [(form if t.i == main.i else t.text) + t.whitespace_ for t in doc[k + 1:end]]
        rest_base = _span(doc, s_hi, end)  # verb in base form
        return {"kind": kind, "wh": wh, "aux": aux.text, "aux_low": aux.lower_, "subj": _span(doc, k + 1, s_hi),
                "subj_head": subj[0], "rest": rest_base, "decl": "".join(words).strip(), "doc": doc, "end": end}
    if aux.lower_ in AUX_OTHER and aux.dep_ in {"ROOT", "aux", "auxpass"}:
        head = aux if aux.dep_ == "ROOT" else aux.head
        subj = [c for c in head.children if c.dep_ in {"nsubj", "nsubjpass", "expl", "csubj"}]
        if not subj and aux.lemma_ == "be":   # spaCy often hangs the subject on the predicate: 'Are [these results] consistent'
            subj = [g for c in head.children if c.dep_ in {"acomp", "attr"} for g in c.children if g.dep_ == "nsubj"]
        if len(subj) != 1 or subj[0].left_edge.i != k + 1:
            return None
        s_hi = subj[0].right_edge.i + 1
        # 'this equality not valid' / 'smart pointers so popular': adverbs and negation belong after the verb
        while s_hi - 1 > subj[0].i and (doc[s_hi - 1].dep_ in {"neg", "advmod"} or doc[s_hi - 1].lower_ in {"not", "so"}):
            s_hi -= 1
        if aux.lemma_ == "be" and low[0] != "what":
            post = [t for t in doc[subj[0].i + 1:s_hi] if t.dep_ == "amod" and t.head.i == subj[0].i]
            if post:
                s_hi = post[0].left_edge.i      # include its adverb: 'pointers | so popular'
        if s_hi >= end:
            return None
        after = [t for t in doc[s_hi:end] if t.pos_ != "ADV"]
        if aux.tag_ == "MD" and (not after or after[0].tag_ != "VB"):
            return None
        if aux.lemma_ == "have" and (not after or after[0].tag_ not in {"VBN", "VBD"}):
            return None
        if doc[s_hi - 1].pos_ in {"ADJ", "DET", "ADP", "CCONJ", "SCONJ"}:
            return None
        if aux.lemma_ == "be" and doc[s_hi].pos_ in {"ADP", "ADV", "SCONJ"} and (
                kind == "wh" or aux.dep_ != "ROOT" or any(x.tag_ == "VBN" for x in doc[s_hi:end])):
            return None
        if doc[s_hi].dep_ == "mark":                                   # 'exceptions | as control flow considered'
            return None
        if aux.lemma_ == "be" and doc[s_hi].lower_ == "to":           # 'prospects of the award | to be enforced'
            return None
        if aux.lemma_ == "be" and any(x.dep_ == "prep" and x.lower_ != "of" for x in doc[k + 1:s_hi]):
            return None                                                # PP attachment ambiguous: 'approach in line | with'
        subj_txt, rest = _span(doc, k + 1, s_hi), _span(doc, s_hi, end)
        return {"kind": kind, "wh": wh, "aux": aux.text, "aux_low": aux.lower_, "subj": subj_txt,
                "subj_head": subj[0], "rest": rest, "decl": " ".join(p for p in (subj_txt, aux.lower_, rest) if p),
                "doc": doc, "end": end}
    if kind == "wh" and k >= 1:
        wh_head = doc[k - 1]
        if wh_head.dep_ in {"nsubj", "nsubjpass"} and wh_head.head.dep_ == "ROOT" and low[0] in {"what", "which", "who"}:
            return {"kind": "wh_subject", "wh": wh, "decl": _span(doc, k, end), "doc": doc, "end": end,
                    "aux_low": None}
    return None


def _lower_first(s: str, doc_first_tok=None) -> str:
    if doc_first_tok is not None and (doc_first_tok.pos_ == "PROPN" or doc_first_tok.text == "I"):
        return s
    return s[:1].lower() + s[1:]


MEDQUAD_WHAT = re.compile(r"^What (is|are) \(are\) (.+?)\s*\?$")


def syn_wh_to_imperative(sent):
    m = MEDQUAD_WHAT.match(sent.strip())
    if m:
        return f"Explain what {m.group(2)} {m.group(1)} (are)."
    q = split_question(sent)
    if not q or q["kind"] not in ("wh", "wh_subject"):
        return None
    if q["kind"] == "wh_subject":
        doc = q["doc"]
        body = _lower_first(doc[:q["end"]].text, doc[0])
        return f"Explain {body}."
    if q["wh"] in ("who",):
        return None
    if q["wh"] == "how" and q["aux_low"] in AUX_DO and q["subj"].lower() in {"i", "we", "you", "one"}:
        return None  # 'How do I tell...' -> 'Explain how I tell...' is unnatural
    return f"Explain {q['wh']} {q['decl']}."


READER = {"you", "anyone", "anybody", "someone", "somebody", "everyone", "y'all"}


def syn_polar_to_whether(sent):
    q = split_question(sent)
    if not q or q["kind"] != "polar" or q["subj"].lower() in READER:
        return None
    return f"Explain whether {q['decl']}."


def syn_why_to_reason(sent):
    q = split_question(sent)
    if not q or q["kind"] != "wh" or q["wh"] != "why":
        return None
    return f"What is the reason that {q['decl']}?"


def syn_can_to_possible(sent):
    q = split_question(sent)
    if not q or q["kind"] != "polar" or q["aux_low"] != "can":
        return None
    subj = q["subj"]
    if not subj or not q["rest"] or subj.lower() in {"it", "there", "this", "that"}:   # 'possible for it to happen'
        return None
    if subj.lower() == "i" and re.match(r"(ask|say|add|mention|get some)\b", q["rest"]):   # 'Can I ask...' = polite opener
        return None
    if subj.lower() in OBJ_PRON:
        subj = OBJ_PRON[subj.lower()]
    return f"Is it possible for {subj} to {q['rest']}?"


HOW_AUX = AUX_DO.keys() | AUX_OTHER


def syn_how_in_what_way(sent):
    if re.match(r"How (is|was|can|could) it (be |possible )?that\b", sent):   # idiom ~ 'why', not manner
        return None
    if re.match(r"How (is|are|was|were|can|could|do|does|did)\b[^?]*\bso \w+", sent):   # 'How is X so high?' ~ 'why'
        return None
    m = re.match(r"(How)\s+(\w+)\s+([\w']+)", sent)
    if m and m.group(2).lower() in HOW_AUX and sent.rstrip().endswith("?") \
            and m.group(3).lower() not in {"be", "been", "being", "not", "come", "n't", "i", "we", "you", "one"}:
        # 'How do I tell if...' -> 'In what way do I tell...' reads unnaturally; only third-person subjects
        return "In what way " + sent[m.start(2):]
    m = re.match(r"In what ways?\s+(\w+)\b", sent)
    if m and m.group(1).lower() in HOW_AUX and sent.rstrip().endswith("?"):
        return "How " + sent[m.start(1):]
    return None


HAPPEN_NEXT = {"in", "at", "during", "when", "if", "after", "before", "because", "again", "often", "here", "there", "now", "frequently", "naturally", "so", "first", "next", "later"}
MARKS = {"if", "when", "although", "though", "since", "because", "unless", "given"}  # not while/whereas (contrast + ellipsis)
COMPLEMENT_HEADS = {"wonder", "curious", "know", "ask", "sure", "unsure", "unclear", "clear", "see", "check", "tell",
                    "decide", "determine", "matter", "care", "doubt", "find", "test", "figure", "question", "certain",
                    "interested", "confused", "understand", "verify", "show", "prove", "say", "learn", "remember"}


def syn_move_clause(sent):
    """Move a sentence-final adverbial clause to the front, or a fronted one to the end."""
    doc = parse(sent.strip())
    if len(list(doc.sents)) != 1 or sent.count("?") > 1:
        return None
    end = _ends(doc)
    punct = sent.strip()[len(doc[:end].text):].strip() or "."
    for t in doc:
        if t.dep_ != "advcl":
            continue
        marks = [c for c in t.children if c.dep_ == "mark" and c.lower_ in MARKS]
        if not marks:
            continue
        a, b = t.left_edge.i, t.right_edge.i + 1
        if a != marks[0].i:
            continue
        clause = doc[a:b]
        if any(x.text in {",", ";", ":"} for x in clause) or len(clause) < 3:
            continue
        if sum(x.dep_ == "mark" or x.lower_ in MARKS for x in clause) != 1:      # nested 'because if ...'
            continue
        if marks[0].lower_ in {"because", "since"} and sent.strip().endswith("?"):
            continue
        if not any(c.dep_ in {"nsubj", "nsubjpass", "expl", "csubj"} for c in t.children):
            continue                                                   # 'while having...' (no subject) reads badly fronted
            continue
        # 'I am curious / wonder / not sure IF ...' is an embedded question (= whether), not a condition
        heads = {t.head.lemma_.lower()} | {c.lemma_.lower() for c in t.head.children if c.dep_ in {"acomp", "attr", "xcomp"}}
        if marks[0].lower_ in {"if", "when"} and heads & COMPLEMENT_HEADS:
            continue
        if b == end and a > 2 and doc[a - 1].text != ",":            # final -> front
            if any(x.text in {",", ";", ":"} for x in doc[:a]):         # main clause already has its own fronted part
                continue
            main = doc[:a].text.rstrip(" ,")
            return f"{cap(clause.text)}, {_lower_first(main, doc[0])}{punct}"
        if a == 0 and b < end and doc[b].text == "," and (b + 1 >= end or doc[b + 1].lower_ != "then"):  # front -> final
            main = doc[b + 1:end].text
            return f"{cap(main)} {clause.text[:1].lower() + clause.text[1:]}{punct}"
    return None


SYN_RULES = [("syn.wh_to_imperative", syn_wh_to_imperative), ("syn.polar_to_whether", syn_polar_to_whether),
             ("syn.why_to_reason", syn_why_to_reason), ("syn.can_to_possible", syn_can_to_possible),
             ("syn.how_in_what_way", syn_how_in_what_way), ("syn.move_clause", syn_move_clause)]


def syntactic_options(item_id, text, title):
    opts = []
    for a, b in target_sentences(text, title):
        sent = text[a:b]
        sent_core = re.sub(r"\s+\?$", "?", sent)  # MedQuAD 'X ?' style
        for name, f in SYN_RULES:
            try:
                new = f(sent_core)
            except Exception:
                new = None
            if new and new.strip() != sent.strip():
                opts.append({"axis": "syntactic", "rule": name, "span": [a, b], "from": sent, "to": new,
                             "rewrite": text[:a] + new + text[b:]})
    # deterministic order: rarer rules first (balance), then by stable hash
    order = {n: i for i, (n, _) in enumerate(reversed(SYN_RULES))}
    opts.sort(key=lambda o: (order[o["rule"]], h(item_id, o["rule"], o["span"][0])))
    return opts


# ============================================================ DISCOURSE axis
FRAMES = ["Here is my question:", "I have a question about the following.", "Please help me with the following question.",
          "I would like some help with this question."]
PLEASANTRY = re.compile(r"^\s*(thanks( a lot| in advance)?|thank you( in advance| very much)?|any help (would be|is) "
                        r"(much |greatly |really )?appreciated|cheers|regards)[.!]*\s*$", re.I)
BACKREF = re.compile(r"\b(above|aforementioned|previous|preceding|this|these|that|those|such|the following|below|here|"
                     r"it|they|them|its|their)\b", re.I)


def discourse_options(item_id, text, title):
    opts = []
    lines = text.split("\n")
    # D1: SE title moved to the end of the question
    if title and text.startswith(title) and text[len(title):].strip():
        body = text[len(title):].strip("\n")
        opts.append({"axis": "discourse", "rule": "dis.title_to_end", "rewrite": body.rstrip() + "\n\n" + title})
    # D2: remove pure pleasantry lines ('Thanks in advance!')
    keep = [l for l in lines if not PLEASANTRY.match(l)]
    if len(keep) < len(lines) and "\n".join(keep).strip():
        removed = [l.strip() for l in lines if PLEASANTRY.match(l)]
        opts.append({"axis": "discourse", "rule": "dis.drop_pleasantry", "removed": removed,
                     "rewrite": re.sub(r"\n{3,}", "\n\n", "\n".join(keep)).strip()})
    # D3: question first - a final question sentence moved before its context (no back-references)
    if not title:
        sents = sentences(text)
        if len(sents) >= 2:
            a, b = sents[-1]
            last = text[a:b]
            full_q = len(last.split()) >= 5 and re.match(r"[A-Z]", last) and not re.match(
                r"(\w+\s+)?(he|she|it|they|his|her|its|their|this|that|these|those|such)\b", last, re.I)
            if full_q and last.endswith("?") and not BACKREF.search(last) and not overlaps(a, b, protected_spans(text)):
                ctx = text[:a].rstrip()
                if ctx and not ctx.endswith(":"):
                    opts.append({"axis": "discourse", "rule": "dis.question_first", "rewrite": last + "\n\n" + ctx})
    # D4: neutral request frame (always applicable)
    f = FRAMES[h(item_id, "frame") % len(FRAMES)]
    opts.append({"axis": "discourse", "rule": "dis.frame", "frame": f, "rewrite": f + "\n\n" + text})
    specific = [o for o in opts if o["rule"] != "dis.frame"]
    specific.sort(key=lambda o: h(item_id, o["rule"]))
    return specific + [o for o in opts if o["rule"] == "dis.frame"]
