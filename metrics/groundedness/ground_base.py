import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict

import numpy as np

class GroundednessScore(Enum):
    HIGH = "HIGH"       # >0.8 - Well grounded
    MEDIUM = "MEDIUM"   # 0.5-0.8 - Partially grounded
    LOW = "LOW"         # 0.2-0.5 - Poorly grounded
    NONE = "NONE"       # <0.2 - Hallucination

@dataclass
class GroundednessResult:
    score: float
    level: GroundednessScore
    total_claims: int
    supported_claims: int
    unsupported_claims: List[str]
    supported_claims_list: List[str]
    details: Dict
    confidence: float


class RuleBasedGroundedness:
    """
    Fast, lightweight groundedness checking using string matching
    Good for quick filtering and initial screening
    """

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold

    def extract_claims(self, text: str) -> List[str]:
        """Extract claims using simple heuristics"""
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        claims = []

        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10:  # Skip very short sentences
                continue

            # Extract noun phrases and key information
            words = sent.lower().split()
            if len(words) > 3:
                claims.append(sent)

        return claims

    def check_claim_in_context(self, claim: str, context: List[str]) -> bool:
        """
        Check if a claim is supported by context using overlap metrics
        """
        claim_words = set(claim.lower().split())
        claim_words = {w for w in claim_words if len(w) > 3}  # Filter stopwords

        if not claim_words:
            return True

        best_overlap = 0
        for ctx in context:
            ctx_words = set(ctx.lower().split())
            ctx_words = {w for w in ctx_words if len(w) > 3}

            if not ctx_words:
                continue

            # Calculate Jaccard similarity
            intersection = claim_words.intersection(ctx_words)
            union = claim_words.union(ctx_words)
            overlap = len(intersection) / len(union)

            best_overlap = max(best_overlap, overlap)

        return best_overlap >= self.threshold

    def evaluate(self,
                 response: str,
                 contexts: List[str]) -> GroundednessResult:
        """
        Evaluate groundedness without LLM
        """
        # Extract claims from answer
        claims = self.extract_claims(response)

        if not claims:
            return GroundednessResult(
                score=1.0,
                level=GroundednessScore.HIGH,
                total_claims=0,
                supported_claims=0,
                unsupported_claims=[],
                supported_claims_list=[],
                details={"note": "No claims to evaluate"},
                confidence=0.5
            )

        supported = []
        unsupported = []

        for claim in claims:
            if self.check_claim_in_context(claim, contexts):
                supported.append(claim)
            else:
                unsupported.append(claim)

        # Calculate score
        score = len(supported) / len(claims) if claims else 1.0

        # Determine level
        if score >= 0.8:
            level = GroundednessScore.HIGH
        elif score >= 0.5:
            level = GroundednessScore.MEDIUM
        elif score >= 0.2:
            level = GroundednessScore.LOW
        else:
            level = GroundednessScore.NONE

        return GroundednessResult(
            score=score,
            level=level,
            total_claims=len(claims),
            supported_claims=len(supported),
            unsupported_claims=unsupported,
            supported_claims_list=supported,
            details={
                "method": "rule_based",
                "threshold": self.threshold,
                "avg_claim_length": np.mean([len(c.split()) for c in claims])
            },
            confidence=0.7  # Lower confidence for rule-based
        )


