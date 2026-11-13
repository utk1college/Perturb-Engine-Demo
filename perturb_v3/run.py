"""Run the v3 engine over the final eval set.

python -I -m perturb_v3.run --split dev [--sample 60] [--out data/perturb_v3/dev_pilot] [--device cpu]
Resumable: items already in variants.jsonl are skipped. Writes variants.jsonl (one row per item x axis),
summary.json and audit.csv (random passing variants + near-threshold ones) for by-eye review.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from perturb_v3 import checks as C
from perturb_v3 import engine as E

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((Path(__file__).parent / "config.json").read_text(encoding="utf-8"))
REORDER = {"dis.title_to_end", "dis.question_first", "dis.drop_pleasantry", "dis.frame"}  # no NLI: content words unchanged by construction
MAX_OPTS = 4


def th_for(v):
    t = CFG["thresholds"]
    if v["rule"] == "dis.frame":
        return t["dis.frame"]
    if v["rule"] in REORDER:
        return t["dis.reorder"]
    return t[v["axis"]]


def load(split):
    p = ROOT / "data" / "human_exam_final" / f"eval_{split}.jsonl"
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def stratified(items, n, seed=42):
    """Equal share per track first, then round-robin over source/cap_group inside each track."""
    rng = random.Random(seed)
    by = defaultdict(lambda: defaultdict(list))
    for r in items:
        by[r["track"]][r.get("cap_group") or r["source"]].append(r)
    tracks = sorted(by)
    pools = {t: {k: rng.sample(v, len(v)) for k, v in by[t].items()} for t in tracks}
    out, step = [], defaultdict(int)
    while len(out) < n and any(any(p.values()) for p in pools.values()):
        for t in tracks:
            keys = sorted(k for k, v in pools[t].items() if v)
            if keys and len(out) < n:
                out.append(pools[t][keys[step[t] % len(keys)]].pop())
                step[t] += 1
    return out


def options(r):
    text = r["question"]
    title = (r.get("source_details") or {}).get("title") if r["source"] == "stackexchange" else None
    lex, why = E.lexical(r["id"], text)
    out = {"lexical": [lex] if lex else [], "syntactic": E.syntactic_options(r["id"], text, title),
           "discourse": E.discourse_options(r["id"], text, title)}
    # lexical fallback: each single substitution on its own, in priority order
    if lex and len(lex["edits"]) > 1:
        for e in lex["edits"]:
            edits = E.fix_articles(text, [(e["start"], e["end"], e["to"])])
            out["lexical"].append({"axis": "lexical", "rule": "lex." + e["kind"], "edits": [e],
                                   "rewrite": E.apply_edits(text, edits)})
    return out


def process(batch, scorer):
    rows, to_score = [], []
    by_id = {r["id"]: r["question"] for r in batch}
    for r in batch:
        text = r["question"]
        for axis, opts in options(r).items():
            rec = {"id": r["id"], "split": r["split"], "track": r["track"], "source": r["source"],
                   "cap_group": r.get("cap_group"), "subject": r["subject"], "axis": axis, "status": "none",
                   "tried": []}
            cands = []
            for v in opts:
                fail = C.hard_check(text, v)
                if fail:
                    rec["tried"].append({"rule": v["rule"], "fail": "hard:" + fail})
                    continue
                cands.append(v)
                if len(cands) >= MAX_OPTS:
                    break
            rec["_cands"] = cands
            rows.append(rec)
    # score in rounds: only the first remaining candidate per (item, axis), so NLI runs on fallbacks only when needed
    for rec in rows:
        rec["_queue"] = list(rec["_cands"])
    while True:
        batch_items = [(rec, rec["_queue"].pop(0)) for rec in rows if rec["status"] != "ok" and rec["_queue"]]
        if not batch_items:
            break
        pairs = []
        for rec, v in batch_items:
            text = by_id[rec["id"]]
            lo, ln = C.local_pair(text, v)
            pairs.append((text, v["rewrite"], lo, ln, v["rule"] in REORDER))
        scores = scorer.score(pairs)
        for (rec, v), s, p in zip(batch_items, scores, pairs):
            if p[0].strip() == p[2].strip():
                s["single_sentence"] = True
            fail = C.semantic_fail(s, "x", {"x": th_for(v)})
            if fail:
                rec["tried"].append({"rule": v["rule"], "fail": "semantic:" + fail, "scores": s})
                continue
            rec.update({"status": "ok", "rule": v["rule"], "rewrite": v["rewrite"], "scores": s,
                        "detail": {k: v[k] for k in ("edits", "from", "to", "frame", "removed") if k in v}})
    for rec in rows:
        rec.pop("_cands", None); rec.pop("_queue", None)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    items = load(a.split)
    if a.sample:
        items = stratified(items, a.sample, a.seed)
    out = ROOT / (a.out or f"data/perturb_v3/{a.split}")
    out.mkdir(parents=True, exist_ok=True)
    vf = out / "variants.jsonl"
    done = set()
    if vf.exists():
        done = {json.loads(l)["id"] for l in open(vf, encoding="utf-8")}
    todo = [r for r in items if r["id"] not in done]
    print(f"{len(items)} items, {len(todo)} to do -> {out}", flush=True)
    scorer = C.Scorer(CFG, device=a.device)
    t0 = time.time()
    with open(vf, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), a.batch):
            for row in process(todo[i:i + a.batch], scorer):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {min(i + a.batch, len(todo))}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    summarize(out, {r["id"]: r for r in items})


def summarize(out, items):
    rows = [json.loads(l) for l in open(out / "variants.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["id"] in items]
    cov = Counter((r["axis"], r["status"]) for r in rows)
    rules = Counter(r["rule"] for r in rows if r["status"] == "ok")
    fails = Counter((r["axis"], t["fail"].split(" ")[0].split("{")[0]) for r in rows for t in r["tried"])
    per_item = defaultdict(set)
    for r in rows:
        if r["status"] == "ok":
            per_item[r["id"]].add(r["axis"])
    all3 = sum(len(v) == 3 for v in per_item.values())
    by_track = defaultdict(Counter)
    for r in rows:
        by_track[(r["track"], r["axis"])][r["status"]] += 1
    import hashlib
    code = b"".join((Path(__file__).parent / f).read_bytes() for f in ("engine.py", "checks.py", "config.json", "run.py"))
    summ = {"engine_version": CFG["version"], "engine_sha256": hashlib.sha256(code).hexdigest()[:16],
            "items": len(items), "all_three_axes": all3,
            "coverage": {f"{a}/{s}": n for (a, s), n in sorted(cov.items())},
            "by_track": {f"{t}/{a}": dict(c) for (t, a), c in sorted(by_track.items())},
            "rules": dict(rules.most_common()), "fail_reasons": {f"{a}/{k}": n for (a, k), n in fails.most_common(40)}}
    (out / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    ok = [r for r in rows if r["status"] == "ok"]
    rng = random.Random(7)
    with open(out / "audit.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "axis", "rule", "original_part", "rewritten_part", "sbert_local", "entail_min", "verdict", "note"])
        for r in sorted(rng.sample(ok, min(len(ok), 300)), key=lambda r: (r["axis"], r["rule"])):
            d, s = r["detail"], r["scores"]
            if r["axis"] == "syntactic":
                a_, b_ = d["from"], d["to"]
            elif r["axis"] == "lexical":
                a_ = "; ".join(f'{e["from"]} -> {e["to"]}' for e in d["edits"]); b_ = ""
            else:
                a_, b_ = r["rule"], (d.get("frame") or str(d.get("removed") or ""))
            w.writerow([r["id"], r["axis"], r["rule"], a_, b_, s["sbert_local"], min(s["ent_fwd"], s["ent_bwd"]), "", ""])
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
