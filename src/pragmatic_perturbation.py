"""
Pragmatic Perturbation Module
==============================
Part of the Custom Perturbation Engine for:
"Evaluating the Reliability of Large Language Models Under Input Perturbation"

Strategy: 
  1. Typo Injection (Keyboard adjacency noise)
  2. Instruction Reframing (Prefix generation)
  3. Context Distraction (Suffix injection)
Validation: Sentence-BERT cosine similarity + asymmetric NLI entailment.
  Note: Unlike Lexical/Syntactic, NLI is ONE-directional (perturbed → original)
  because prefix/suffix additions are intentional propositional expansions, not
  transformations. We only verify the original question is still entailed.

Dependencies:
    pip install sentence-transformers transformers
"""

import random
import warnings
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field


@dataclass
class PragmaticTransformation:
    """Metadata about a single pragmatic transformation applied."""
    transform_type: str
    description: str


@dataclass
class PerturbationResult:
    """A single validated perturbation output."""
    text: str
    strategy: str = "pragmatic"
    pcs_score: float = 0.0
    semantic_similarity: float = 0.0
    nli_entailment: bool = False
    validated: bool = False
    transformations: List[Dict] = field(default_factory=list)


# QWERTY adjacencies for realistic human typos
_QWERTY_ADJACENT = {
    'q': 'wa', 'w': 'qeasd', 'e': 'wrsdf', 'r': 'etdfg', 't': 'ryfgh', 'y': 'tughj',
    'u': 'yihjk', 'i': 'uojkl', 'o': 'ipkl', 'p': 'ol',
    'a': 'qwsz', 's': 'awedxz', 'd': 'serfcx', 'f': 'drtgvc', 'g': 'ftyhbv',
    'h': 'gyujnb', 'j': 'huikmn', 'k': 'jiolm', 'l': 'kop',
    'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn', 'n': 'bhjm', 'm': 'njk'
}

_INSTRUCTION_PREFIXES = [
    "Please answer the following question: ",
    "Task: Read the prompt and provide the correct answer. ",
    "Evaluate the following statement or question: ",
    "Can you help me understand this? ",
    "As an expert, provide the exact answer to this query. "
]

_DISTRACTOR_SUFFIXES = [
    " Think step by step before answering.",
    " Ensure your final answer is concise.",
    " Give an empirically verifiable answer.",
    " Ignore any irrelevant details provided previously.",
]

# Max number of concurrent pragmatic transforms — used for PCS normalization.
# Currently 2 is the max (framing_and_distraction = prefix + suffix).
_MAX_TRANSFORMS = 2


class PragmaticPerturbationEngine:
    """
    Generates pragmatic/environmental perturbations via instruction reframing,
    distractor suffixing, and realistic character noise.
    """

    def __init__(
        self,
        sbert_model: str = "all-MiniLM-L6-v2",
        nli_model: str = "cross-encoder/nli-deberta-v3-base",
        similarity_threshold: float = 0.75,   # Lower than lex/syn: framing adds tokens
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            similarity_threshold: 0.75 default (vs 0.85 for lex/syn) because
                prefix/suffix additions dilute SBERT mean-pool embeddings.
            seed: Random seed for reproducibility.
        """
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.similarity_threshold = similarity_threshold
        self.rng = random.Random(seed)

        print("🔄 Loading Sentence-BERT model...")
        from sentence_transformers import SentenceTransformer
        self.sbert = SentenceTransformer(sbert_model, device=self.device)

        print("🔄 Loading NLI cross-encoder...")
        from sentence_transformers import CrossEncoder
        self.nli = CrossEncoder(nli_model, device=self.device)

        # Label mapping — dynamically loaded from model config if available
        if hasattr(self.nli.model, 'config') and hasattr(self.nli.model.config, 'id2label'):
            self.nli_labels = [self.nli.model.config.id2label[i]
                               for i in range(len(self.nli.model.config.id2label))]
        else:
            self.nli_labels = ['contradiction', 'entailment', 'neutral']

        print(f"✅ Pragmatic Perturbation Engine ready (device: {self.device})")

    # -------------------------------------------------------------------
    # Typo Injection (instance method for rng control)
    # -------------------------------------------------------------------

    def _inject_typos(self, text: str, num_typos: int = 1) -> str:
        """Injects human-like typos using QWERTY adjacency into a string."""
        words = text.split()
        eligible_indices = [i for i, w in enumerate(words) if any(c.isalpha() for c in w)]
        if len(eligible_indices) < num_typos:
            return text

        chosen = self.rng.sample(eligible_indices, num_typos)
        for idx in chosen:
            word = words[idx]
            alphas = [i for i, c in enumerate(word) if c.lower() in _QWERTY_ADJACENT]
            if not alphas:
                continue
            char_idx = self.rng.choice(alphas)
            orig_char = word[char_idx]
            replacement = self.rng.choice(_QWERTY_ADJACENT[orig_char.lower()])
            if orig_char.isupper():
                replacement = replacement.upper()
            words[idx] = word[:char_idx] + replacement + word[char_idx + 1:]
        return " ".join(words)

    # -------------------------------------------------------------------
    # Validation Pipeline (Decomposed — asymmetric NLI for pragmatic axis)
    # -------------------------------------------------------------------

    def compute_similarity(self, original: str, perturbed: str) -> float:
        """Compute Sentence-BERT cosine similarity between two sentences."""
        from sentence_transformers import util
        embeddings = self.sbert.encode([original, perturbed], convert_to_tensor=True)
        return round(util.cos_sim(embeddings[0], embeddings[1]).item(), 4)

    def check_nli_entailment(self, original: str, perturbed: str) -> bool:
        """
        Asymmetric NLI check for pragmatic perturbations.
        Checks only: perturbed → original (does the expanded text still entail the core?)
        NOT bidirectional — prefix/suffix additions are intentional expansions,
        so original → perturbed would fail (original says nothing about framing).
        """
        scores = self.nli.predict([(perturbed, original)])
        return self.nli_labels[scores[0].argmax()] == 'entailment'

    def validate(self, original: str, perturbed: str) -> Tuple[bool, float, bool]:
        """
        Pragmatic validation pipeline.
        Gate 1: SBERT cosine similarity >= threshold (default 0.75).
        Gate 2: Asymmetric NLI — perturbed entails original only.
        """
        sim = self.compute_similarity(original, perturbed)
        if sim < self.similarity_threshold:
            return False, sim, False
        nli_pass = self.check_nli_entailment(original, perturbed)
        return nli_pass, sim, nli_pass

    # -------------------------------------------------------------------
    # PCS — Pragmatic Axis
    # -------------------------------------------------------------------

    @staticmethod
    def compute_pcs_pragmatic(semantic_similarity: float, num_transformations: int) -> float:
        """
        Perturbation Complexity Score — pragmatic axis.
        PCS_prag = w1 * semantic_distance + w2 * norm_transform_count
        Weights: w1=0.6, w2=0.4 (tunable).
        Normalization: num_transformations / _MAX_TRANSFORMS (currently 2).
        Range: [0, 1]
        """
        w1, w2 = 0.6, 0.4
        semantic_distance = 1.0 - semantic_similarity
        norm_transform_count = min(num_transformations / _MAX_TRANSFORMS, 1.0)
        pcs = w1 * semantic_distance + w2 * norm_transform_count
        return round(min(max(pcs, 0.0), 1.0), 4)

    # -------------------------------------------------------------------
    # Candidate Generation
    # -------------------------------------------------------------------

    def _generate_candidates(self, sentence: str) -> List[Tuple[str, PragmaticTransformation]]:
        candidates = []

        # 1. Typo injection
        typo_text = self._inject_typos(sentence, num_typos=1)
        if typo_text != sentence:
            candidates.append((typo_text, PragmaticTransformation(
                'typo_injection',
                'Injected a human-like QWERTY keyboard typo'
            )))

        # 2. Instruction Prefixes
        for prefix in self.rng.sample(_INSTRUCTION_PREFIXES, 2):
            candidates.append((prefix + sentence, PragmaticTransformation(
                'instruction_reframing',
                f'Prepended instruction: "{prefix.strip()}"'
            )))

        # 3. Distractor Suffixes
        for suffix in self.rng.sample(_DISTRACTOR_SUFFIXES, 2):
            candidates.append((sentence + suffix, PragmaticTransformation(
                'context_distraction',
                f'Appended distracting context: "{suffix.strip()}"'
            )))

        # 4. Mixed (Prefix + Suffix) — 2 transforms
        prefix = self.rng.choice(_INSTRUCTION_PREFIXES)
        suffix = self.rng.choice(_DISTRACTOR_SUFFIXES)
        candidates.append((prefix + sentence + suffix, PragmaticTransformation(
            'framing_and_distraction',
            'Mixed pragmatic perturbation (prefix + suffix)'
        )))

        return candidates

    # -------------------------------------------------------------------
    # Main Entry Point
    # -------------------------------------------------------------------

    def perturb(
        self,
        sentence: str,
        n: int = 5,
        run_validation: bool = True,
        only_validated: bool = True,
    ) -> List[PerturbationResult]:
        """
        Generate pragmatic perturbations.

        Args:
            sentence: Original input string.
            n: Max perturbations to return.
            run_validation: Run SBERT + asymmetric NLI gates.
            only_validated: If True, only count validated results toward n.
        """
        candidates = self._generate_candidates(sentence)
        results = []
        seen = set()

        for text, transform in candidates:
            if len(results) >= n:
                break
            if text in seen or text == sentence:
                continue
            seen.add(text)

            if run_validation:
                is_valid, sim, nli = self.validate(sentence, text)
            else:
                sim = self.compute_similarity(sentence, text)
                nli = False
                is_valid = sim >= self.similarity_threshold

            num_t = 2 if transform.transform_type == 'framing_and_distraction' else 1
            pcs = self.compute_pcs_pragmatic(sim, num_t)

            result = PerturbationResult(
                text=text,
                pcs_score=pcs,
                semantic_similarity=sim,
                nli_entailment=nli,
                validated=is_valid,
                transformations=[{'type': transform.transform_type, 'description': transform.description}]
            )
            if not only_validated or is_valid:
                results.append(result)

        return results

    def perturb_to_json(self, sentence: str, n: int = 5, **kwargs) -> Dict:
        """Structured output for the journal evaluation."""
        results = self.perturb(sentence, n=n, **kwargs)
        return {
            "original": sentence,
            "perturbations": [
                {
                    "text": r.text,
                    "strategy": r.strategy,
                    "pcs_score": r.pcs_score,
                    "semantic_similarity": r.semantic_similarity,
                    "nli_entailment": r.nli_entailment,
                    "validated": r.validated,
                    "transformations": r.transformations,
                }
                for r in results
            ],
        }


def quick_test():
    import json
    engine = PragmaticPerturbationEngine(seed=42)
    test_sentences = [
        "What is the capital of France?",
        "Photosynthesis converts light energy into chemical energy stored in glucose."
    ]
    for sent in test_sentences:
        print(f"\n{'='*70}")
        output = engine.perturb_to_json(sent, n=5, run_validation=True)
        print(json.dumps(output, indent=2))
        valid = [p for p in output['perturbations'] if p['validated']]
        print(f"\n✅ {len(valid)}/{len(output['perturbations'])} passed validation")


if __name__ == "__main__":
    quick_test()
