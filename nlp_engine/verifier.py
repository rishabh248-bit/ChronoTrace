"""
ChronoTrace / SynapseState NLI Entailment Grounding & Verification Engine
Implements zero-shot Natural Language Inference (NLI) using DeBERTa-v3 to calculate
the Claim Verification Rate (CVR) by validating extracted claims against transcript evidence.
"""

from typing import Any, Dict, Optional, Tuple
from nlp_engine.schemas import DecisionSchema


class NLIGroundingVerifier:
    """
    Evaluates whether an extracted decision/claim logically follows from
    a verbatim transcript evidence span (Entailment vs Neutral vs Contradiction).
    """

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small", threshold: float = 0.60):
        self.model_name = model_name
        self.threshold = threshold
        self._classifier = None
        self._tokenizer = None
        self._model = None

    def _load_model(self):
        """Lazy loader for DeBERTa-v3 NLI model."""
        if self._model is None:
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                import torch
                print(f"[NLIVerifier] Loading NLI model '{self.model_name}'...")
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                if torch.cuda.is_available():
                    self._model.to("cuda")
                self._model.eval()
            except Exception as e:
                print(f"[NLIVerifier] DeBERTa model loading note: {e}. Using semantic token-grounding verifier.")
                self._model = None
        return self._model

    def verify_claim(self, premise: str, hypothesis: str) -> Dict[str, Any]:
        """
        Calculates Claim Verification Rate (CVR) between verbatim premise and extracted hypothesis.
        Returns:
            - cvr_score (float 0.0 - 1.0)
            - label ('entailment', 'neutral', 'contradiction')
            - is_verified (bool: cvr_score >= threshold)
        """
        premise_clean = premise.strip()
        hypothesis_clean = hypothesis.strip()

        if not premise_clean or not hypothesis_clean:
            return {"cvr_score": 0.0, "label": "neutral", "is_verified": False}

        model = self._load_model()
        if model is not None and self._tokenizer is not None:
            try:
                import torch
                inputs = self._tokenizer(
                    premise_clean,
                    hypothesis_clean,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}

                with torch.no_grad():
                    logits = model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1).squeeze().tolist()

                # For cross-encoder/nli-deberta-v3-small, classes are:
                # 0: contradiction, 1: entailment, 2: neutral (or check config.id2label)
                id2label = getattr(model.config, "id2label", {0: "contradiction", 1: "entailment", 2: "neutral"})
                entailment_idx = next((i for i, lbl in id2label.items() if "entail" in lbl.lower()), 1)
                contradiction_idx = next((i for i, lbl in id2label.items() if "contra" in lbl.lower()), 0)

                entailment_prob = probs[entailment_idx]
                contradiction_prob = probs[contradiction_idx]

                if entailment_prob >= contradiction_prob and entailment_prob > 0.4:
                    label = "entailment"
                    score = entailment_prob
                elif contradiction_prob > 0.5:
                    label = "contradiction"
                    score = 1.0 - contradiction_prob
                else:
                    label = "neutral"
                    score = entailment_prob

                return {
                    "cvr_score": round(float(score), 3),
                    "label": label,
                    "is_verified": score >= self.threshold
                }
            except Exception as e:
                print(f"[NLIVerifier] Transformers inference exception: {e}. Falling back to lexical grounding.")

        # High-precision token grounding fallback (Jaccard + Entity Substring Alignment)
        return self._fallback_grounding_score(premise_clean, hypothesis_clean)

    def _fallback_grounding_score(self, premise: str, hypothesis: str) -> Dict[str, Any]:
        """
        Deterministic lexical/semantic token overlap grounding engine for offline environments.
        Computes precision of hypothesis keywords anchored in premise text.
        """
        p_tokens = set(re_tokenize(premise.lower()))
        h_tokens = set(re_tokenize(hypothesis.lower()))

        if not h_tokens:
            return {"cvr_score": 0.0, "label": "neutral", "is_verified": False}

        # Stop words to ignore in grounding ratio
        stopwords = {
            "the", "a", "an", "and", "or", "to", "in", "on", "of", "for", "with",
            "we", "our", "is", "are", "will", "let", "that", "this", "by", "from"
        }
        filtered_h = h_tokens - stopwords
        if not filtered_h:
            filtered_h = h_tokens

        overlap = filtered_h.intersection(p_tokens)
        grounding_ratio = len(overlap) / len(filtered_h)

        # Bonus if hypothesis is direct substring or quotes key technical identifiers
        if hypothesis.lower() in premise.lower():
            grounding_ratio = max(grounding_ratio, 0.95)

        # Detect contradiction cues
        negations = ["not", "never", "cannot", "cancel", "abandon", "reject"]
        has_negation_diff = any(neg in premise.lower() for neg in negations) != any(neg in hypothesis.lower() for neg in negations)

        if has_negation_diff and grounding_ratio > 0.6:
            label = "contradiction"
            cvr = round(0.15, 2)
        elif grounding_ratio >= 0.55:
            label = "entailment"
            cvr = round(min(0.98, max(0.65, grounding_ratio)), 2)
        else:
            label = "neutral"
            cvr = round(grounding_ratio, 2)

        return {
            "cvr_score": cvr,
            "label": label,
            "is_verified": cvr >= self.threshold
        }

    def verify_decision(self, decision: DecisionSchema) -> DecisionSchema:
        """Computes and populates CVR confidence score on DecisionSchema instance."""
        claim = f"The team agreed to: {decision.decision_text}"
        res = self.verify_claim(premise=decision.evidence_quote, hypothesis=claim)
        decision.cvr_confidence_score = res["cvr_score"]
        return decision


def re_tokenize(text: str) -> list:
    """Splits alphanumeric tokens."""
    import re
    return re.findall(r"\b\w+\b", text)


if __name__ == "__main__":
    verifier = NLIGroundingVerifier()
    premise = "Therefore, we will officially pivot from MongoDB to PostgreSQL 16 with Row-Level Security for ACID ledger compliance."
    hypothesis_true = "Pivot from MongoDB to PostgreSQL 16 for ACID compliance"
    hypothesis_unsupported = "We decided to build an in-memory graph database using Neo4j"

    r1 = verifier.verify_claim(premise, hypothesis_true)
    r2 = verifier.verify_claim(premise, hypothesis_unsupported)

    print("Verifying True Claim:")
    print(r1)
    print("\nVerifying Unsupported Claim:")
    print(r2)
