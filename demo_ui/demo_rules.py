"""Demo-only rules (website only, not the journal engine). They run next to the v3 rules and produce
candidates in the same format, so the v3 hard checks, SBERT and NLI still apply. Relaxed thresholds are set
in demo_engine.py and labelled on the page.

Lexical: curated synonym table (lemma based, re-inflected with lemminflect) + phrase table, with guards.
Syntactic: extra question patterns on EVERY sentence, plus v3's own rule functions run on non-question
sentences too. Each syntactic rule lists the only words it may add or remove; that is checked by lemma.
"""
from __future__ import annotations

import re
import zlib
from collections import Counter

from lemminflect import getInflection, getLemma

from perturb_v3 import checks as C
from perturb_v3 import engine as E

# ============================================================ lexical
# guards: (doc, token) -> True when the swap is safe here
def _nxt(doc, t, k=1):
    j = t.i + k
    return doc[j] if j < len(doc) else None


OBJ_PRON = {"me", "us", "him", "her", "them", "you", "it"}


def g_not_to(doc, t):
    n = _nxt(doc, t)
    return not (n is not None and n.lower_ == "to")


def g_obj(doc, t):               # needs a plain object next: 'find the value', not 'find it hard'
    n = _nxt(doc, t)
    return n is not None and (n.pos_ in {"DET", "NOUN", "NUM"} or n.lower_ in {"what", "how", "why", "whether", "where", "which", "who", "when"})


def g_not_person(doc, t):
    n = _nxt(doc, t)
    return n is None or n.lower_ not in OBJ_PRON | {"to"}


def g_get(doc, t):
    n = _nxt(doc, t)
    return n is not None and n.pos_ in {"DET", "NOUN", "NUM"} and n.lower_ not in {"rid"}


def g_think(doc, t):
    n = _nxt(doc, t)
    return n is not None and n.lower_ not in {"of", "about", "so", "through", "over", "up", "twice"} and n.pos_ != "PUNCT"


def g_happen(doc, t):
    n = _nxt(doc, t)
    return n is None or n.is_punct or n.lower_ in E.HAPPEN_NEXT


def g_begin(doc, t):
    n = _nxt(doc, t)
    return n is not None and n.lower_ in {"to", "with", "by"}


def g_not_for(doc, t):
    n = _nxt(doc, t)
    return not (n is not None and n.lower_ == "for")


def g_main(doc, t):
    n = _nxt(doc, t)
    return n is not None and n.lower_ in E.MAIN_OK


def g_pred_or_task(doc, t):      # hard/easy/difficult: predicative, or before a task-like noun
    n = _nxt(doc, t)
    return n is None or n.pos_ != "NOUN" or n.lemma_ in {"question", "problem", "task", "exam", "topic", "part", "concept", "test", "subject"}


def g_not_make(doc, t):          # 'make a difference' is an idiom
    return not any(x.lemma_ == "make" for x in t.sent)


def g_not_as_a(doc, t):
    return not (t.i >= 2 and doc[t.i - 1].lower_ == "a" and doc[t.i - 2].lower_ == "as")


def g_not_in(doc, t):
    return not (t.i >= 1 and doc[t.i - 1].lower_ == "in")


def g_not_stat(doc, t):
    return not (t.i >= 1 and doc[t.i - 1].lower_ == "statistically")


def g_almost(doc, t):
    n = _nxt(doc, t)
    return n is None or n.lower_ not in {"no", "surely", "never", "all", "always", "everywhere", "nothing", "none"}


def g_complex(doc, t):
    n = _nxt(doc, t)
    return n is None or n.lemma_ not in {"number", "plane", "analysis", "conjugate", "ion", "carbohydrate", "sentence", "root"}


def g_change(doc, t):
    n = _nxt(doc, t)
    return n is not None and n.pos_ in {"DET", "NOUN"} and n.lower_ not in {"into", "to"}


def g_check(doc, t):
    n = _nxt(doc, t)
    return n is not None and n.lower_ not in {"out", "in", "on", "up", "off"}


def g_believe(doc, t):
    n = _nxt(doc, t)
    return n is not None and n.lower_ not in {"in", "me", "you", "him", "her", "them", "us", "it"}


def g_likely(doc, t):
    n = _nxt(doc, t)
    return n is not None and n.lower_ not in {"to"}


def g_against(doc, t):
    n = _nxt(doc, t)
    return n is not None and n.lower_ == "against"


def g_ok(doc, t):
    return True


VERB = {  # lemma -> (target lemma, guard)
    "explain": ("describe", g_not_person), "show": ("demonstrate", g_not_person), "get": ("obtain", g_get),
    "need": ("require", g_not_to), "require": ("need", g_ok), "happen": ("occur", g_happen),
    "occur": ("happen", g_not_to), "find": ("determine", g_obj), "use": ("employ", g_not_to),
    "utilize": ("use", g_ok), "utilise": ("use", g_ok), "affect": ("influence", g_ok), "influence": ("affect", g_ok),
    "improve": ("enhance", g_ok), "enhance": ("improve", g_ok), "decrease": ("reduce", g_ok),
    "reduce": ("decrease", g_ok), "choose": ("select", g_ok), "select": ("choose", g_ok), "buy": ("purchase", g_ok),
    "purchase": ("buy", g_ok), "build": ("construct", g_ok), "construct": ("build", g_ok), "check": ("verify", g_check),
    "verify": ("check", g_ok), "change": ("alter", g_change), "alter": ("change", g_ok), "finish": ("complete", g_ok),
    "complete": ("finish", g_ok), "connect": ("link", g_ok), "link": ("connect", g_ok), "suggest": ("propose", g_ok),
    "propose": ("suggest", g_ok), "understand": ("comprehend", g_ok), "comprehend": ("understand", g_ok),
    "think": ("believe", g_think), "believe": ("think", g_believe), "begin": ("start", g_begin), "start": ("begin", g_begin),
    "allow": ("permit", g_not_for), "permit": ("allow", g_ok), "calculate": ("compute", g_ok),
    "compute": ("calculate", g_ok), "assess": ("evaluate", g_ok), "evaluate": ("assess", g_ok),
    "produce": ("generate", g_ok), "generate": ("produce", g_ok), "remember": ("recall", g_not_to),
    "recall": ("remember", g_ok), "obtain": ("get", g_ok), "acquire": ("gain", g_ok), "demonstrate": ("show", g_ok),
    "attempt": ("try", g_not_to), "assist": ("help", g_ok), "boost": ("increase", g_ok), "maintain": ("keep", g_obj),
    "eliminate": ("remove", g_ok), "remove": ("eliminate", g_obj), "examine": ("inspect", g_ok),
    "inspect": ("examine", g_ok), "depict": ("show", g_ok), "illustrate": ("show", g_obj), "establish": ("set up", g_ok),
    "investigate": ("examine", g_ok), "discover": ("find", g_ok), "prefer": ("favour", g_obj), "recommend": ("suggest", g_ok),
    "tackle": ("address", g_ok), "float": ("stay afloat", g_ok), "work": ("function", g_against), "resolve": ("settle", g_obj), "omit": ("leave out", g_ok), "seek": ("look for", g_ok),
}
NOUN = {
    "difference": ("distinction", g_not_make), "distinction": ("difference", g_ok), "problem": ("issue", g_ok),
    "method": ("approach", g_ok), "approach": ("method", g_ok), "result": ("outcome", g_not_as_a),
    "outcome": ("result", g_ok), "effect": ("impact", g_not_in), "impact": ("effect", g_ok), "type": ("kind", g_ok),
    "step": ("stage", g_ok), "stage": ("phase", g_ok), "phase": ("stage", g_ok), "amount": ("quantity", g_ok),
    "quantity": ("amount", g_ok), "question": ("query", g_ok), "query": ("question", g_ok),
    "exam": ("examination", g_ok), "examination": ("exam", g_ok), "student": ("learner", g_ok),
    "aim": ("goal", g_ok), "goal": ("aim", g_ok), "purpose": ("aim", g_ok), "objective": ("aim", g_ok),
    "feature": ("characteristic", g_ok), "characteristic": ("feature", g_ok), "idea": ("notion", g_ok),
    "concept": ("notion", g_ok), "notion": ("idea", g_ok), "advantage": ("benefit", g_ok), "benefit": ("advantage", g_ok),
    "disadvantage": ("drawback", g_ok), "drawback": ("disadvantage", g_ok), "option": ("choice", g_ok),
    "choice": ("option", g_ok), "trouble": ("difficulty", g_ok), "mistake": ("error", g_ok), "topic": ("subject", g_ok),
"example": ("instance", g_ok), "instance": ("example", g_ok),
    "importance": ("significance", g_ok), "significance": ("importance", g_ok), "requirement": ("condition", g_ok),
    "technique": ("method", g_ok), "strategy": ("approach", g_ok), "issue": ("problem", g_ok), "task": ("job", g_ok),
    "help": ("assistance", g_ok), "assistance": ("help", g_ok), "role": ("function", g_ok),
    "consequence": ("result", g_ok), "doctor": ("physician", g_ok), "physician": ("doctor", g_ok),
    "fever": ("high temperature", g_ok), "process": ("procedure", g_ok), "procedure": ("process", g_ok),
}
ADJ = {
    "important": ("significant", g_ok), "significant": ("important", g_not_stat), "main": ("principal", g_main),
    "principal": ("main", g_main), "big": ("large", g_ok), "large": ("big", g_ok), "huge": ("enormous", g_ok),
    "enormous": ("huge", g_ok), "hard": ("difficult", g_pred_or_task), "difficult": ("hard", g_pred_or_task),
    "easy": ("simple", g_pred_or_task), "usual": ("typical", g_ok), "typical": ("usual", g_ok),
    "wrong": ("incorrect", g_ok), "incorrect": ("wrong", g_ok), "whole": ("entire", g_ok), "entire": ("whole", g_ok),
    "quick": ("fast", g_ok), "rapid": ("quick", g_ok), "basic": ("fundamental", g_ok), "fundamental": ("basic", g_ok),
    "useful": ("helpful", g_ok), "helpful": ("useful", g_ok), "exact": ("precise", g_ok), "precise": ("exact", g_ok),
    "complex": ("complicated", g_complex), "complicated": ("complex", g_ok), "modern": ("contemporary", g_ok),
    "likely": ("probable", g_likely), "obvious": ("evident", g_ok), "evident": ("obvious", g_ok),
    "crucial": ("essential", g_ok), "essential": ("crucial", g_ok), "key": ("central", g_main),
    "major": ("main", g_main), "simple": ("straightforward", g_pred_or_task), "proper": ("appropriate", g_ok),
    "appropriate": ("suitable", g_ok), "suitable": ("appropriate", g_ok), "accurate": ("precise", g_ok),
    "confusing": ("unclear", g_ok), "unclear": ("confusing", g_ok), "strange": ("odd", g_ok), "odd": ("strange", g_ok),
    "tiny": ("very small", g_ok), "vital": ("essential", g_ok), "primary": ("main", g_main),
}
ADV = {
    "usually": ("typically", g_ok), "typically": ("usually", g_ok), "often": ("frequently", g_ok),
    "frequently": ("often", g_ok), "mainly": ("mostly", g_ok), "mostly": ("mainly", g_ok),
    "exactly": ("precisely", g_ok), "precisely": ("exactly", g_ok), "almost": ("nearly", g_almost),
    "nearly": ("almost", g_almost), "completely": ("entirely", g_ok), "entirely": ("completely", g_ok),
    "rarely": ("seldom", g_ok), "seldom": ("rarely", g_ok), "especially": ("particularly", g_ok),
    "particularly": ("especially", g_ok), "immediately": ("at once", g_ok), "correctly": ("properly", g_ok),
    "properly": ("correctly", g_ok), "approximately": ("roughly", g_ok), "roughly": ("approximately", g_ok),
    "quickly": ("rapidly", g_ok), "rapidly": ("quickly", g_ok), "therefore": ("thus", g_ok), "thus": ("therefore", g_ok),
    "really": ("truly", g_ok), "basically": ("essentially", g_ok), "essentially": ("basically", g_ok),
    "generally": ("in general", g_ok), "eventually": ("in the end", g_ok),
}
POS_TABLE = {"VERB": VERB, "NOUN": NOUN, "ADJ": ADJ, "ADV": ADV}

# phrases: (regex, replacement, guard on the text after the match)
PHRASES = [
    (r"a lot of(?= \w+s\b)", "many", None), (r"lots of(?= \w+s\b)", "many", None),
    (r"in order to", "to", None), (r"because of", "due to", None), (r"in spite of", "despite", None),
    (r"despite", "in spite of", None), (r"at the same time", "simultaneously", None),
    (r"what kind of", "what type of", None), (r"what type of", "what kind of", None),
    (r"in general(?! relativity)(?=[\s,.?!])", "generally", None), (r"in addition(?! to)", "additionally", None),
    (r"as a result(?! of)", "consequently", None), (r"first of all", "firstly", None),
    (r"in my opinion", "in my view", None), (r"with regard to", "regarding", None), (r"with respect to", "regarding", None),
    (r"prior to", "before", None), (r"so that", "in order that", None), (r"as well(?=[.?!])", "too", None),
    (r"for example", "for instance", None), (r"for instance", "for example", None),
    (r"make sure", "ensure", None), (r"makes sure", "ensures", None), (r"made sure", "ensured", None),
    (r"find out(?! about)", "discover", None), (r"found out(?! about)", "discovered", None),
    (r"carry out", "perform", None), (r"carried out", "performed", None), (r"carries out", "performs", None),
    (r"figure out", "work out", None), (r"point out", "indicate", None), (r"take place", "occur", None),
    (r"takes place", "occurs", None), (r"took place", "occurred", None), (r"deal with", "handle", None),
    (r"deals with", "handles", None), (r"come up with", "devise", None), (r"get rid of", "eliminate", None),
    (r"(?<=question )about", "regarding", None), (r"(?<=questions )about", "regarding", None),
(r"(?<=information )about", "regarding", None),
    (r"(?<=curious )about", "regarding", None), (r"a great deal of", "much", None),
    (r"in the case of", "for", None), (r"causes of", "reasons for", None), (r"cause of", "reason for", None), (r"is made up of", "consists of", None), (r"are made up of", "consist of", None),
    (r"is composed of", "consists of", None), (r"are composed of", "consist of", None),
    (r"as soon as", "once", None), (r"in terms of", "regarding", None), (r"has to do with", "relates to", None),
]
_PHRASE_RX = [(re.compile(r"(?<![\w-])" + p + r"(?![\w-])", re.I), r) for p, r, _ in PHRASES]


def _inflect(lemma, tag):
    words = lemma.split(" ", 1)
    form = getInflection(words[0], tag=tag)
    head = form[0] if form else words[0]
    return head + ((" " + words[1]) if len(words) > 1 else "")


def _form(t, target):
    tag = t.tag_
    if t.pos_ == "VERB":
        if tag in {"VB", "VBP"}:
            out = target
        else:
            out = _inflect(target, tag)
    elif t.pos_ == "NOUN":
        out = _inflect(target, "NNS") if tag == "NNS" else target
    else:
        if tag not in {"JJ", "RB"}:
            return None
        out = target
    return E.case_like(t.text, out)


def _h(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def lexical_edits(text):
    prot = E.protected_spans(text)
    doc = E.parse(text)
    edits = []
    for rx, rep in _PHRASE_RX:
        for m in rx.finditer(text):
            if not E.overlaps(m.start(), m.end(), prot):
                edits.append((m.start(), m.end(), E.case_like(m.group(0), rep), "demo_phrase"))
    for t in doc:
        table = POS_TABLE.get(t.pos_)
        if not table or t.lemma_.lower() not in table:
            continue
        s, e = t.idx, t.idx + len(t.text)
        head_ok = t.lemma_.lower() in {"exam", "examination", "question", "problem", "doctor", "student"}
        if E.overlaps(s, e, prot) or t.ent_type_ or t.dep_ == "compound" or \
                (not head_ok and any(c.dep_ == "compound" for c in t.children)):
            continue
        if (s > 0 and text[s - 1] in "-/") or (e < len(text) and text[e] in "-/"):
            continue
        if t.text[:1].isupper() and t.i > 0 and not t.is_sent_start:     # mid-sentence capital: a name or title
            continue
        target, guard = table[t.lemma_.lower()]
        if not guard(doc, t):
            continue
        rep = _form(t, target)
        if rep and rep.lower() != t.text.lower():
            edits.append((s, e, rep, "demo_word"))
    # keep non-overlapping, phrases win over words inside them
    edits.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    out = []
    for ed in edits:
        if out and ed[0] < out[-1][1]:
            continue
        out.append(ed)
    return out


def _lex_cand(text, chosen):
    fixed = E.fix_articles(text, [(s, e, r) for s, e, r, _ in chosen])
    return {"axis": "lexical", "rule": "demo.lex." + ("multi" if len(chosen) > 1 else chosen[0][3].split("_")[1]),
            "demo": True, "rewrite": E.apply_edits(text, fixed),
            "edits": [{"start": s, "end": e, "from": text[s:e], "to": r, "kind": k} for s, e, r, k in chosen]}


def lexical_options(text):
    eds = lexical_edits(text)
    if not eds:
        return []
    order = sorted(eds, key=lambda x: _h(text, x[0]))
    cands = [_lex_cand(text, [order[0]])]
    if len(order) > 1:
        cands.append(_lex_cand(text, [order[1]]))
        cands.append(_lex_cand(text, sorted(order[:3])))
    return cands


# in-memory registrations (no file edits): the scope words these swaps legitimately add
C.LEX_SCOPE_OK.update({("a lot of", "many"): {"many"}, ("lots of", "many"): {"many"}})


# ============================================================ syntactic
WH_ALL = {"what", "how", "why", "where", "when", "which", "who"}
INSTR = E.INSTR_VERBS - {"state", "name", "list", "give", "show", "prove", "derive", "determine"}
TIME_ADV = {"usually", "often", "today", "nowadays", "generally", "typically", "sometimes", "normally", "frequently"}
TIME_NOUN = {"summer", "winter", "spring", "autumn", "general", "practice", "theory", "reality", "humans", "plants",
             "animals", "children", "adults", "class", "school", "exams", "history", "past", "future", "nature"}
ORDERED = {("cause", "effect"), ("rise", "fall"), ("before", "after"), ("question", "answer"), ("pro", "con"),
           ("supply", "demand"), ("input", "output"), ("trial", "error"), ("black", "white"), ("past", "present"),
           ("theory", "practice"), ("bread", "butter"), ("read", "write"), ("husband", "wife")}

# words each demo syntactic rule may add / remove (compared by lemma); scope words they may add
SYN_ALLOW = {
    "demo.syn.tell_me": ({"could", "you", "tell", "me"}, {"do", "does", "did"}),
    "demo.syn.like_to_know": ({"i", "would", "like", "to", "know", "whether"}, {"do", "does", "did"}),
    "demo.syn.is_it_true": ({"is", "it", "true", "that"}, {"do", "does", "did"}),
    "demo.syn.why_reason": ({"what", "is", "the", "reason", "that"}, {"why", "do", "does", "did"}),
    "demo.syn.explain_what": ({"explain"}, set()),
    "demo.syn.split_sentence": (set(), set()),
    "demo.syn.why_cause": ({"what", "cause", "to", "me", "us", "them", "him", "her"},
                           {"why", "do", "does", "did", "i", "we", "they", "he", "she"}),
    "demo.syn.way_in_which": ({"what", "is", "the", "way", "in", "which"}, {"how", "do", "does", "did"}),
    "demo.syn.can_you": ({"can", "you"}, set()),
    "demo.syn.imperative": (set(), {"can", "could", "you"}),
    "demo.syn.front_adverbial": (set(), set()),
    "demo.syn.end_adverbial": (set(), set()),
    "demo.syn.swap_and": (set(), set()),
}
C.SCOPE_EXEMPT.update({"demo.syn.tell_me": {"could"}, "demo.syn.like_to_know": {"would"},
                       "demo.syn.is_it_true": {"true"}, "demo.syn.can_you": {"can"},
                       "demo.syn.imperative": {"can", "could"}})


def _lem(w):
    return (getLemma(w, upos="VERB") or (w,))[0]


def word_check(original, v):
    """Only the rule's own words may change (compared by lemma, so 'pump'/'pumps' is fine)."""
    add, rem = SYN_ALLOW[v["rule"]]
    lo = Counter(_lem(w) for w, c in C.norm_words(original).items() for _ in range(c))
    ln = Counter(_lem(w) for w, c in C.norm_words(v["rewrite"]).items() for _ in range(c))
    add_l, rem_l = {_lem(w) for w in add}, {_lem(w) for w in rem}
    lost = {w: c for w, c in (lo - ln).items() if w not in rem_l}
    gained = {w: c for w, c in (ln - lo).items() if w not in add_l}
    if lost or gained:
        return f"word content changed lost={lost} gained={gained}"
    return None


def _lower_first(s, doc=None):
    if doc is not None and len(doc) and (doc[0].pos_ == "PROPN" or doc[0].text == "I" or doc[0].text.isupper()):
        return s
    return s[:1].lower() + s[1:]


WHAT_IS = re.compile(r"^(What|Who) (is|are|was|were) ([^?]+?)\s*\?$")
DO_Q = re.compile(r"^(Why|How) (do|does|did) (.+?)\s*\?$")
COMMA_Q = re.compile(r"^(.+?), (what|how|why|where|when|which|who|should|can|could|is|are|do|does) (.+\?)$", re.I)


def _what_is(sent):
    m = WHAT_IS.match(sent.strip())
    if not m:
        return []
    x = m.group(3)
    doc = E.parse(x)
    if any(t.pos_ in {"VERB", "AUX"} for t in doc) or x.lower().startswith(("the reason", "the way")):
        return []
    wh, aux = m.group(1).lower(), m.group(2)
    return [("demo.syn.tell_me", f"Could you tell me {wh} {x} {aux}?"),
            ("demo.syn.explain_what", f"Explain {wh} {x} {aux}.")]


def _do_question(sent):
    """'Why does ice float on water?' when spaCy misreads the verb: try subject lengths 1-4 and keep the
    split whose declarative ('ice floats on water') parses with that word as the main verb."""
    m = DO_Q.match(sent.strip())
    if not m:
        return []
    wh, aux, body = m.group(1).lower(), m.group(2).lower(), m.group(3)
    words = body.split(" ")
    for k in range(1, min(5, len(words))):
        subj, verb, rest = " ".join(words[:k]), words[k], " ".join(words[k + 1:])
        if not verb.isalpha() or not verb.islower():
            continue
        tag = {"does": "VBZ", "did": "VBD", "do": "VBP"}[aux]
        vf = verb if tag == "VBP" else (getInflection(verb, tag=tag) or [None])[0]
        if not vf:
            continue
        decl = " ".join(p for p in (subj, vf, rest) if p)
        doc = E.parse(decl[:1].upper() + decl[1:] + ".")   # lowercase 'ice floats' parses as a compound
        vt = [t for t in doc if t.i == len(subj.split())]
        if not vt or vt[0].pos_ != "VERB" or vt[0].dep_ != "ROOT" or \
                not any(c.dep_ in {"nsubj", "nsubjpass"} for c in vt[0].children):
            continue
        out = [("demo.syn.tell_me", f"Could you tell me {wh} {decl}?")]
        if wh == "why":
            obj = E.OBJ_PRON.get(subj.lower(), subj)
            out.append(("demo.syn.why_cause", f"What {'caused' if aux == 'did' else 'causes'} {obj} to {verb}"
                                              f"{(' ' + rest) if rest else ''}?"))
            out.append(("demo.syn.why_reason", f"What is the reason that {decl}?"))
        elif subj.lower() not in {"i", "we", "you", "one"}:
            out.append(("demo.syn.way_in_which", f"What is the way in which {decl}?"))
        return out
    return []


def _split_comma(sent):
    """'I have a fever, what should I do?' -> 'I have a fever. What should I do?'"""
    m = COMMA_Q.match(sent.strip())
    if not m:
        return []
    first = E.parse(m.group(1))
    if first[0].pos_ == "SCONJ" or first[0].lower_ in E.MARKS | {"while", "after", "before", "as", "once", "unless"}:
        return []                                   # 'If a tenant does not pay rent.' would be a fragment
    if not any(t.pos_ in {"VERB", "AUX"} and any(c.dep_ == "nsubj" for c in t.children) for t in first):
        return []
    return [("demo.syn.split_sentence", f"{m.group(1)}. {m.group(2).capitalize()} {m.group(3)}")]


def _q_rules(sent):
    out = []
    q = E.split_question(sent)
    if not q:
        return _what_is(sent) + _do_question(sent) + _split_comma(sent)
    if q["kind"] == "wh" and q["wh"] in WH_ALL and q.get("decl"):
        out.append(("demo.syn.tell_me", f"Could you tell me {q['wh']} {q['decl']}?"))
        if q["wh"] == "why" and q["aux_low"] in E.AUX_DO and q.get("subj") and q.get("rest"):
            subj = E.OBJ_PRON.get(q["subj"].lower(), q["subj"]) if q["subj"].lower() in E.OBJ_PRON else q["subj"]
            verb = "caused" if q["aux_low"] == "did" else "causes"
            out.append(("demo.syn.why_cause", f"What {verb} {subj} to {q['rest']}?"))
        if q["wh"] == "how" and q["aux_low"] in E.AUX_DO:
            out.append(("demo.syn.way_in_which", f"What is the way in which {q['decl']}?"))
    if q["kind"] == "polar" and q.get("decl") and q.get("subj", "").lower() not in E.READER:
        out.append(("demo.syn.like_to_know", f"I would like to know whether {q['decl']}."))
        if q["subj"].lower() not in {"it", "there", "i", "we", "you"}:
            out.append(("demo.syn.is_it_true", f"Is it true that {q['decl']}?"))
    return out + _split_comma(sent)


def _imperative(sent):
    s = sent.strip()
    m = re.match(r"^(Can|Could) you (please )?(\w+)\b(.*)\?$", s)
    if m and m.group(3).lower() in INSTR and not m.group(2):
        return [("demo.syn.imperative", f"{m.group(3).capitalize()}{m.group(4)}.")]
    m = re.match(r"^(\w+)\b(.*)\.$", s)
    if m and m.group(1).lower() in INSTR:
        doc = E.parse(s)
        if doc[0].tag_ == "VB":
            return [("demo.syn.can_you", f"Can you {m.group(1).lower()}{m.group(2)}?")]
    return []


def _adverbial(sent):
    s = sent.strip()
    doc = E.parse(s)
    if len(doc) < 5 or not doc[-1].is_punct or s.endswith("?"):
        return []
    end = doc[-1]
    root = [t for t in doc if t.dep_ == "ROOT"]
    if not root:
        return []
    root = root[0]
    out = []
    # move a short sentence-final time/frequency adverbial to the front
    for c in root.children:
        sub = list(c.subtree)
        if sub[-1].i != end.i - 1 or len(sub) > 4 or any(t.like_num for t in sub):
            continue
        ok = (c.dep_ == "advmod" and c.lower_ in TIME_ADV and len(sub) == 1) or \
             (c.dep_ == "prep" and c.lower_ in {"in", "during"} and sub[-1].lemma_.lower() in TIME_NOUN)
        if not ok or sub[0].i < 3:
            continue
        a, b = sub[0].idx, sub[-1].idx + len(sub[-1].text)
        phrase = s[a:b]
        rest = s[:a].rstrip().rstrip(",")
        new = f"{phrase[:1].upper()}{phrase[1:]}, {_lower_first(rest, doc)}{end.text}"
        out.append(("demo.syn.front_adverbial", new))
    # move a fronted adverb ('Usually, X.') to the end
    if doc[0].lower_ in TIME_ADV and doc[1].text == "," and doc[0].dep_ == "advmod":
        rest = s[doc[2].idx:]
        body, p = rest[:-1].rstrip(), rest[-1]
        first = doc[2]
        body = body if (first.pos_ == "PROPN" or first.text == "I") else body[:1].upper() + body[1:]
        out.append(("demo.syn.end_adverbial", f"{body} {doc[0].lower_}{p}"))
    return out


def _swap_and(sent):
    s = sent.strip()
    doc = E.parse(s)
    out = []
    for t in doc:
        if t.dep_ != "conj" or t.head.dep_ == "ROOT" or t.pos_ not in {"NOUN", "PROPN", "ADJ"}:
            continue
        h = t.head
        ccs = [c for c in h.children if c.dep_ == "cc"]
        conjs = [c for c in h.children if c.dep_ == "conj"]
        if len(ccs) != 1 or ccs[0].lower_ != "and" or len(conjs) != 1 or h.pos_ != t.pos_:
            continue
        cc = ccs[0]
        h_start = h.left_edge.i
        h_end = cc.i - 1
        if any(doc[k].dep_ in {"prep", "relcl", "acl"} for k in range(h.i + 1, h_end + 1)):
            continue
        t_start, t_end = t.left_edge.i, t.right_edge.i
        if t_start != cc.i + 1 or h_end < h_start or (h_end - h_start) > 4 or (t_end - t_start) > 4:
            continue
        if any(doc[k].is_punct or doc[k].like_num for k in range(h_start, t_end + 1)):
            continue
        if (doc[h_start].pos_ == "DET") != (doc[t_start].pos_ == "DET"):
            continue
        if (h.lemma_.lower(), t.lemma_.lower()) in ORDERED or (t.lemma_.lower(), h.lemma_.lower()) in ORDERED:
            continue
        if doc[h_start - 1].lower_ in {"both", "either", "neither"} if h_start > 0 else False:
            continue
        A = s[doc[h_start].idx: doc[h_end].idx + len(doc[h_end].text)]
        B = s[doc[t_start].idx: doc[t_end].idx + len(doc[t_end].text)]
        if h_start == 0:
            A2, B2 = _lower_first(A, doc), B[:1].upper() + B[1:]
        else:
            A2, B2 = A, B
        a, b = doc[h_start].idx, doc[t_end].idx + len(doc[t_end].text)
        out.append(("demo.syn.swap_and", s[:a] + f"{B2} and {A2}" + s[b:]))
    return out


def syntactic_options(text):
    opts = []
    targets = {(a, b) for a, b in E.target_sentences(text, None)}
    prot = E.protected_spans(text)
    for a, b in E.sentences(text):
        if E.overlaps(a, b, prot):
            continue
        sent = text[a:b]
        core = re.sub(r"\s+\?$", "?", sent)
        rules = _q_rules(core) + _imperative(core) + _adverbial(core) + _swap_and(core)
        if (a, b) not in targets:         # v3 rules only look at question sentences; run them on the rest too
            for name, f in E.SYN_RULES:
                try:
                    new = f(core)
                except Exception:
                    new = None
                if new:
                    rules.append((name, new))
        for name, new in rules:
            if new and new.strip() != sent.strip():
                opts.append({"axis": "syntactic", "rule": name, "demo": True, "span": [a, b], "from": sent,
                             "to": new, "rewrite": text[:a] + new + text[b:]})
    seen, uniq = set(), []
    for o in sorted(opts, key=lambda o: _h(text, o["rule"], o["span"][0])):
        if o["rewrite"] not in seen:
            seen.add(o["rewrite"])
            uniq.append(o)
    return uniq


def hard_check(original, v, base_check):
    """v3 hard checks (base_check) plus, for demo syntactic rules, the per-rule word list."""
    f = base_check(original, v)
    if f and f.startswith("length ratio") and v["axis"] == "syntactic" and v["rule"] in SYN_ALLOW:
        # short questions: a fixed frame ('Could you tell me ...') breaks the 1.5x length cap; the per-rule
        # word list below already limits what may be added, so run the remaining v3 checks instead
        lo, ch = C.local_pair(original, v)
        glitches = lambda t: len(C.BAD_PUNCT.findall(t.replace("\n", " ")))
        dbl = lambda t: len(re.findall(r"\b(\w+) \1\b", t.lower()))
        f = "punctuation glitch" if glitches(ch) > glitches(lo) else ("doubled word" if dbl(ch) > dbl(lo) else None)
    if f:
        return f
    if v["axis"] == "syntactic" and v["rule"] in SYN_ALLOW:
        return word_check(original, v)
    return None
