# Perturb-Engine v3 (rule-based, meaning-preserving)

Makes up to **three rewrites per question**, one for each **axis**, and never mixes axes within one rewrite. The
study can then ask *which kind* of rewording hurts a model. Every rewrite is produced by a deterministic rule.
Models are used only to **score** a rewrite, never to write one.

| Axis | What changes | Rules |
|---|---|---|
| **lexical** | 1–3 words or phrases swapped for an exact equivalent | `lex.word` (spaCy POS-checked synonym tables), `lex.phrase` (for example↔for instance, I was wondering→I would like to know, …), `lex.contraction` (expand only) |
| **syntactic** | one question sentence restructured, same words | `syn.wh_to_imperative` (What/How/Why … ? → Explain …), `syn.polar_to_whether` (Does X …? → Explain whether X …), `syn.why_to_reason` (Why … ? → What is the reason that … ?), `syn.can_to_possible` (Can X …? → Is it possible for X to …?), `syn.how_in_what_way` (How … ? ↔ In what way … ?), `syn.move_clause` (if/when/because clause moved front↔end) |
| **discourse** | order and framing of the whole text, content untouched | `dis.title_to_end` (Stack Exchange title after the body), `dis.question_first` (final question moved before its context), `dis.drop_pleasantry` ("Thanks!" lines removed), `dis.frame` (one fixed neutral opener, e.g. "Here is my question:") as the fallback |

## Pipeline per item and axis

1. **Candidates.** Each applicable rule produces an option. The order is fixed: rarer rules first, then a stable hash of the item id. The run is fully deterministic.
2. **Hard checks** (rules, in `checks.py`). An option is rejected if:
   - any protected span changes: LaTeX, code, URLs, quoted text, anything containing a digit, acronyms, parentheticals;
   - any number changes;
   - any scope word changes: negation, quantifiers, modals, connectives (see `SCOPE`). There are three explicit exceptions, listed in `LEX_SCOPE_OK`: the polite "would" in "I would like to know", various→different, and "whether or not"→"whether";
   - the word content changes beyond the words that rule is allowed to add or remove (`ALLOWED`). Verb re-inflection under do-support is compared by lemma;
   - the lexical edit count falls outside 1–3;
   - the length ratio falls outside 0.75–1.5 (frames are exempt);
   - the rewrite *introduces* a punctuation glitch or a doubled word.
3. **Meaning checks** (models pinned by revision in `config.json`):
   - **SBERT** `all-mpnet-base-v2` cosine between the full texts and between the changed sentences;
   - **two-way NLI** with `DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, using the wrapper "Someone asks the following question: …". Both directions must entail, and neither may contradict.
4. **Fallback.** The first option that passes is kept. If none pass, the axis is recorded as `none` with every reason it was tried and rejected. No rewrite is forced.

## Thresholds (`config.json`)

| | SBERT global | SBERT local | entail (both ways) | contradiction |
|---|---|---|---|---|
| lexical, syntactic | ≥ 0.90 (≥ 0.85 if the text is a single sentence) | ≥ 0.85 | ≥ 0.80 | ≤ 0.10 |
| discourse reorder (title_to_end, question_first, drop_pleasantry) | ≥ 0.90 | – | – | – |
| discourse frame | ≥ 0.70 | – | – | – |

Reorder and frame rules skip NLI. Their hard check already proves that no content word changed (frame: only the fixed
frame words are added). NLI on long, truncated texts would test the truncation rather than the rewrite.

## Guards added from the audits (CP1–CP2, dev split)

Each guard below came from a rewrite I read, judged wrong, and fixed in the rule. The checks did not catch these cases:

- **do-support with negation** keeps the auxiliary: "Why did England not maintain…" → "…that England *did not maintain*…".
- **Base verb required under do-support.** This stops misparses such as "molecules *having* … have".
- **Be-questions:**
  - adverbs and negation are trimmed off the subject ("pointers | so popular");
  - no prepositional-phrase subjects other than "of";
  - nothing is moved when the remainder starts with a clause marker.
- **Contractions:**
  - not expanded in inverted questions ("Why don't they…", "Isn't it…");
  - not expanded in Title Case headings.
- **Lexical context:**
  - in particular, almost no / almost surely, for the purpose of, supposed to;
  - happen→occur only before punctuation or time/condition words;
  - help→assist only before an object;
  - main→principal only before abstract nouns;
  - no swaps inside hyphenated words (above-mentioned);
  - e.g./i.e. only after "(" or ",";
  - phrase guards: "a bit of", "in addition to", "in general relativity".
- **how_in_what_way:**
  - skipped for I/we/you subjects ("How do I tell if…" reads unnaturally);
  - skipped for be/come/not;
  - skipped for "How is it that…" (an idiom close to "why").
- **move_clause:**
  - no embedded-question "if" (I am curious / not sure / wonder if…);
  - exactly one subordinator in the clause;
  - the clause must have its own subject;
  - no while/whereas;
  - no because/since inside questions;
  - not when the main clause already has a fronted part or starts with "then".
- **question_first:** the moved question must be a full question (5+ words, capitalised) that does not start with a pronoun.
- **Added from the CP3 early checks on the test split:**
  - contractions are not expanded before a subject in any inversion ("but shouldn't the acid form?");
  - start↔begin only before to/with/by (never "start the engine");
  - currently→at present removed;
  - how_in_what_way is skipped for "How is X so ADJ?" (means "why");
  - can_to_possible is skipped for "Can I ask/say…" openers;
  - "be" is never placed before an infinitive ("prospects of the award | to be enforced").
- **Added from a review of all 202 lexical swap types in the full test output, read with the words after each swap:**
  - no allow/permit before "for" ("allow for" = account for);
  - no occur/happen before "to" ("it occurred to me");
  - no assume/suppose before "to" ("is assumed to be");
  - attempt→try only before "to" ("attempted murder");
  - no swap of a past participle used as an adjective ("the supposed exposure");
  - help→assist not before object + bare verb ("help us determine");
  - specific↔particular not when predicative ("be more specific.");
  - no "thanks a lot" → "thank you a lot";
  - "I was wondering / I'm curious" → "I would like to know" only before a question word or "about".
- **Protected-span detection:** prose with LaTeX or parentheticals is no longer treated as code, and digit spans no longer swallow trailing punctuation.

## Final coverage (engine `c4cec6ab37185a4c`, both splits)

Each cell is the share of items that got a passing rewrite on that axis.

| | items | lexical | syntactic | discourse | all three |
|---|---|---|---|---|---|
| test | 8,310 | 71.2% | 49.3% | 100% | 35.6% |
| dev | 924 | 70.0% | 49.2% | 100% | 33.8% |

Test split by track:

| Track | lexical | syntactic | discourse |
|---|---|---|---|
| conversational (Stack Exchange) | 77% | 52% | 100% |
| exam | 45% | 21% | 100% |
| reference (MedQuAD) | 13% | 46% | 100% |

MedQuAD template questions ("What is (are) X ?") rarely contain a word that can be swapped safely, so low lexical coverage there is expected, not a fault. When an axis has no passing rewrite it is stored as `none` with the reasons; nothing is forced.

**QA** (`python -m perturb_v3.qa`, written to `data/perturb_v3/qa_report.json`):
- Structure: every item has exactly 3 rows, with 0 missing and 0 duplicates, and dev and test share one engine hash.
- Error patterns: 28 known error classes are checked, and a hit counts only when the rewrite has more of the pattern than its original.
- Result: 2 hits remain, and both are reviewed false alarms. One is an original "is it supposed to" moved by a syntactic rule; the other is "kVA is not the same as kW?", which is correct.

**Reproducibility:** after the last guard (help→assist), the 412 items containing "help" were reprocessed instead of rerunning everything. As a control, 120 random untouched items were also rerun from scratch: 360 of 360 rows gave the same decision and the same rewrite. The only differences were GPU score jitter in the 4th decimal, which depends on how items are batched.

## Outputs (`data/perturb_v3/<split>/`)

- `variants.jsonl`: one row per item × axis.
  - Fields: `status` (ok/none), `rule`, `rewrite`, `scores`, `detail` (edits, from/to sentence, frame, removed lines) and `tried` (every rejected option and why).
- `summary.json`: coverage, per-track coverage, rule mix, rejection reasons, `engine_sha256`.
- `audit.csv`: 300 random passing rewrites with an empty `verdict` column for human review.

## Run

```
.venv311\Scripts\python.exe -I -c "import sys; sys.path.insert(0,r'<repo>'); import runpy; runpy.run_module('perturb_v3.run', run_name='__main__')" --split test --device cuda
```

Options: `--sample N --seed S` (stratified by track) and `--out DIR`. The run is resumable. It needs CUDA torch for reasonable speed: about 0.3 s per item on an RTX 3050 4 GB, against about 30 s per item on CPU.
